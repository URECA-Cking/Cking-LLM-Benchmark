"""Private offline human criteria and prompt evaluation; never invokes a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from src.detail_tagging import (POLICY_VERSION, RevisionStore, build_prompt,
                                candidate_schema, classify, digest)
from src.detail_tagging_batch import read, write_new
from src.detail_tagging_compare import load_summary
from src.sample_review import new_artifacts

FIELDS = ("channel_id", "input_hash", "axis", "code", "decision", "reviewer", "evidence", "reason")
COHORTS = ("v3_unclassified_transition", "baseline_final_failure", "baseline_retry_success")
COMMON = ("inputs_hash", "criteria_hash", "model_version", "policy_version",
          "output_schema_hash", "temperature", "sdk_max_retries", "timeout_seconds", "max_attempts")


def prepare(inputs, taxonomy, cohorts=None):
    """Enumerate every tag, including negative decisions, without importing AI labels."""
    ids = [r["channel_id"] for r in inputs]
    if not ids or any(not isinstance(cid, str) or not cid.strip() for cid in ids):
        raise ValueError("Nonempty input channels required")
    if len(set(ids)) != len(ids) or any(not isinstance(r["text"], str) for r in inputs):
        raise ValueError("Duplicate channel or invalid text")
    cohorts = cohorts or {}
    if set(cohorts) - set(COHORTS):
        raise ValueError("Unknown cohort")
    for members in cohorts.values():
        if not isinstance(members, list) or len(set(members)) != len(members) or set(members) - set(ids):
            raise ValueError("Duplicate or unknown cohort channel")
    manifest = {"schema_version": 1, "inputs": inputs, "taxonomy": taxonomy,
                "inputs_hash": digest(inputs), "criteria_hash": digest(taxonomy),
                "cohorts": {name: cohorts.get(name, []) for name in COHORTS},
                "cohort_source": "Explicit historical channel lists; membership is not human truth"}
    return {"manifest": manifest, "hash": digest(manifest)}


def review_rows(pack):
    manifest = pack["manifest"]
    taxonomy = manifest["taxonomy"]
    tags = {"parents": taxonomy["parents"], "topics": [t["code"] for t in taxonomy["subtopics"]]}
    return [{"channel_id": r["channel_id"], "input_hash": digest(r["text"]),
             "axis": axis, "code": code, "decision": "pending", "reviewer": "",
             "evidence": "", "reason": ""}
            for r in manifest["inputs"] for axis, codes in tags.items() for code in codes]


def validate(pack, rows):
    manifest = pack["manifest"]
    if pack["hash"] != digest(manifest):
        raise ValueError("Review manifest changed")
    if (manifest["inputs_hash"] != digest(manifest["inputs"])
            or manifest["criteria_hash"] != digest(manifest["taxonomy"])):
        raise ValueError("Review source changed")
    key = lambda r: (r["channel_id"], r["axis"], r["code"])
    expected = {key(r): r for r in review_rows(pack)}
    seen = set()
    texts = {r["channel_id"]: r["text"] for r in manifest["inputs"]}
    decisions = {}
    for row in rows:
        if set(row) != set(FIELDS) or any(not isinstance(row[k], str) for k in FIELDS):
            raise ValueError("Invalid review columns/cells")
        identity = key(row)
        if identity not in expected or identity in seen:
            raise ValueError("Duplicate or unknown review tag")
        seen.add(identity)
        if row["input_hash"] != expected[identity]["input_hash"]:
            raise ValueError("Frozen review input changed")
        decision = row["decision"]
        if decision not in {"pending", "keep", "delete", "hold"}:
            raise ValueError("Invalid human decision")
        if decision == "pending":
            if any(row[k] for k in ("reviewer", "evidence", "reason")):
                raise ValueError("Pending row contains review")
        else:
            if not row["reviewer"].strip():
                raise ValueError("Human reviewer required")
            if decision in {"keep", "hold"} and not row["reason"].strip():
                raise ValueError("Human reviewer and reason required")
            evidence = row["evidence"]
            if (decision == "keep" and not evidence.strip()) or (evidence and evidence not in texts[row["channel_id"]]):
                raise ValueError("Human evidence must be an exact source quote")
        decisions[identity] = decision
    if seen != expected.keys():
        raise ValueError("Missing human review tags")
    taxonomy = manifest["taxonomy"]
    for cid in texts:
        parents = [code for (channel, axis, code), d in decisions.items()
                   if channel == cid and axis == "parents" and d == "keep"]
        topics = [t for t in taxonomy["subtopics"] if decisions[(cid, "topics", t["code"])] == "keep"]
        if len(parents) > 3 or any(sum(t["parent"] == p for t in topics) > 3 for p in parents):
            raise ValueError("Human tags exceed fixed policy")
        if any(t["parent"] not in parents for t in topics):
            raise ValueError("Kept human topic requires a kept parent")
    return decisions


def freeze(pack, csv_path):
    # One immutable byte snapshot for parsing and provenance.
    raw = Path(csv_path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Save human reviews as CSV UTF-8; cp949 is unsupported") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(FIELDS):
        raise ValueError("Invalid human review header")
    rows = list(reader)
    decisions = validate(pack, rows)
    if "pending" in decisions.values():
        raise ValueError("Human review pending; explicitly decide or hold every tag")
    value = {"pack": pack, "rows": rows, "csv_sha256": hashlib.sha256(raw).hexdigest(),
             "origin": "human_declared", "counts": dict(Counter(decisions.values()))}
    return {"criteria": value, "hash": digest(value)}


def _validate_attempts(directory, row, text, taxonomy, conditions):
    """Derive summary claims from complete sequential attempt records."""
    history = [read(p) for p in (directory / "attempts" / digest(row["channel_id"])).glob("*.json")]
    if (type(row.get("attempts")) is not int or not history or len(history) != row["attempts"]
            or type(conditions["max_attempts"]) is not int
            or not 1 <= len(history) <= conditions["max_attempts"]
            or any(type(e.get("attempt")) is not int for e in history)):
        raise ValueError("Missing or invalid attempt history")
    history.sort(key=lambda e: e["attempt"])
    if [e["attempt"] for e in history] != list(range(1, len(history) + 1)):
        raise ValueError("Nonsequential attempt history")
    messages = [{"role": "system", "content": build_prompt(taxonomy, conditions["prompt_version"])},
                {"role": "user", "content": json.dumps({"creator": text}, ensure_ascii=False)}]
    for entry in history:
        if (entry.get("channel_id") != row["channel_id"] or entry.get("input_hash") != digest(text)
                or entry.get("conditions") != conditions):
            raise ValueError("Attempt conditions differ")
        request = entry.get("request", {})
        if (request.get("messages") != messages or request.get("model") != conditions["model_version"]
                or request.get("temperature") != conditions["temperature"]
                or request.get("response_format", {}).get("json_schema", {}).get("schema") != candidate_schema(taxonomy)):
            raise ValueError("Attempt request conditions differ")
        if (entry.get("status") not in {"passed", "validation_error", "provider_error"}
                or bool(entry.get("revision")) != (entry["status"] == "passed")):
            raise ValueError("Invalid attempt status/revision")
    successful = [e for e in history if e["status"] == "passed"]
    if successful and (len(successful) != 1 or successful[0] is not history[-1]):
        raise ValueError("Attempt history continues after success")
    if (row["first_pass"] != (history[0]["status"] == "passed")
            or row["passed"] != bool(successful)
            or row["revision"] != (successful[0]["revision"] if successful else None)):
        raise ValueError("Summary differs from attempt history")


def _load_run(path, pack):
    report, rows = load_summary(path)
    conditions = report["conditions"]
    manifest = pack["manifest"]
    taxonomy = manifest["taxonomy"]
    required = COMMON + ("prompt_version", "prompt_hash")
    if any(k not in conditions for k in required):
        raise ValueError("Unverified run conditions; historical aggregates cannot be imputed")
    for key, value in {"inputs_hash": manifest["inputs_hash"], "criteria_hash": digest(taxonomy),
                       "policy_version": POLICY_VERSION, "output_schema_hash": digest(candidate_schema(taxonomy)),
                       "prompt_hash": digest(build_prompt(taxonomy, conditions["prompt_version"]))}.items():
        if conditions[key] != value:
            raise ValueError("Run conditions differ: " + key)
    inputs = {r["channel_id"]: r["text"] for r in manifest["inputs"]}
    if rows.keys() != inputs.keys():
        raise ValueError("Evaluation input channel sets differ")
    if (report.get("total") != len(rows)
            or report.get("first_pass") != sum(r["first_pass"] for r in rows.values())
            or report.get("passed_after_retries") != sum(r["passed"] for r in rows.values())):
        raise ValueError("Summary totals differ from rows")
    store = RevisionStore(Path(path).parent / "revisions")
    results = {}
    for cid, row in rows.items():
        _validate_attempts(Path(path).parent, row, inputs[cid], taxonomy, conditions)
        if not row["passed"]:
            results[cid] = None
            continue
        record = store.read(row["revision"])
        if (record["channel_id"] != cid or record["kind"] != "automatic"
                or record["input_hash"] != digest(inputs[cid])):
            raise ValueError("Invalid evaluation revision")
        for key in ("criteria_hash", "model_version", "policy_version", "prompt_version",
                    "prompt_hash", "output_schema_hash"):
            if record[key] != conditions[key]:
                raise ValueError("Revision conditions differ: " + key)
        result = classify(inputs[cid], record["candidates"], taxonomy)
        if result != record["result"]:
            raise ValueError("Revision result differs from validated candidates")
        results[cid] = result
    return report, rows, results


def _metrics(ids, rows, results, decisions):
    total = len(ids)
    success = [cid for cid in ids if results[cid] is not None]
    report = {"channels": total, "first_pass": sum(rows[cid]["first_pass"] for cid in ids),
              "passed_after_retries": len(success), "failed_channels": total - len(success),
              "format_denominator": total, "classification_denominator": len(success),
              "unclassified": sum(not results[cid]["parents"] for cid in success),
              "parent_only": sum(bool(results[cid]["parents"]) and not results[cid]["topics"] for cid in success)}
    report["first_pass_rate"] = report["first_pass"] / total if total else None
    report["after_retries_rate"] = len(success) / total if total else None
    for name in ("unclassified", "parent_only"):
        report[name + "_rate"] = report[name] / len(success) if success else None
    human_complete = [cid for cid in ids if all(d != "hold" for (channel, _, _), d in decisions.items() if channel == cid)]
    kept = lambda cid, axis: any(d == "keep" for (channel, a, _), d in decisions.items() if channel == cid and a == axis)
    report["human_classification_denominator"] = len(human_complete)
    report["human_unclassified"] = sum(not kept(cid, "parents") for cid in human_complete)
    report["human_parent_only"] = sum(kept(cid, "parents") and not kept(cid, "topics") for cid in human_complete)
    for axis in ("parents", "topics"):
        counts = Counter({k: 0 for k in ("reviewed_tags", "held_tags", "failed_reviewed_tags", "missing", "excess",
                                        "human_positive_tags", "predicted_reviewed_tags", "fully_reviewed_channels")})
        details = []
        for cid in ids:
            judged = {code: d for (channel, a, code), d in decisions.items() if channel == cid and a == axis}
            counts["held_tags"] += sum(d == "hold" for d in judged.values())
            decided = {code: d for code, d in judged.items() if d in {"keep", "delete"}}
            if results[cid] is None:
                counts["failed_reviewed_tags"] += len(decided)
                continue
            counts["fully_reviewed_channels"] += all(d != "hold" for d in judged.values())
            predicted = {t["code"] for t in results[cid][axis]}
            truth = {code for code, d in decided.items() if d == "keep"}
            missing, excess = sorted(truth - predicted), sorted((predicted & decided.keys()) - truth)
            counts["reviewed_tags"] += len(decided)
            counts["human_positive_tags"] += len(truth)
            counts["predicted_reviewed_tags"] += len(predicted & decided.keys())
            counts["missing"] += len(missing)
            counts["excess"] += len(excess)
            details.append({"channel_id": cid, "missing": missing, "excess": excess})
        report[axis] = {**counts, "rows": details}
    return report


def validated_criteria(frozen):
    criteria = frozen["criteria"]
    if frozen["hash"] != digest(criteria) or criteria["origin"] != "human_declared":
        raise ValueError("Frozen human criteria changed")
    pack = criteria["pack"]
    decisions = validate(pack, criteria["rows"])
    if "pending" in decisions.values():
        raise ValueError("Human review pending")
    return pack, decisions


def evaluate(frozen, summaries):
    pack, decisions = validated_criteria(frozen)
    if not summaries:
        raise ValueError("At least one run required")
    ids = [r["channel_id"] for r in pack["manifest"]["inputs"]]
    cohorts = pack["manifest"]["cohorts"]
    runs, reference = [], None
    for path in summaries:
        report, rows, results = _load_run(path, pack)
        if ("human_criteria_hash" in report["conditions"]
                and report["conditions"]["human_criteria_hash"] != frozen["hash"]):
            raise ValueError("Run human criteria differ")
        common = {k: report["conditions"][k] for k in COMMON}
        if reference is not None and reference != common:
            raise ValueError("Comparison run conditions differ")
        reference = common
        runs.append({"conditions": report["conditions"], "summary_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     "all": _metrics(ids, rows, results, decisions),
                     "cohorts": {name: _metrics(members, rows, results, decisions) for name, members in cohorts.items()}})
    return {"human_criteria_hash": frozen["hash"], "sample_size": len(ids),
            "human_review": "reviewed_with_holds" if "hold" in decisions.values() else "review_complete",
            "human_decisions": dict(Counter(decisions.values())), "runs": runs,
            "cohort_overlap": {cid: [name for name, members in cohorts.items() if cid in members]
                               for cid in ids if sum(cid in members for members in cohorts.values()) > 1},
            "adoption": "v2_retained_pending_human_adoption_decision",
            "limitations": ["Human-declared reviewers are not authenticated; AI drafts are not human truth.",
                            "v4 prompt examples derive from the original 50-item evaluation sample; reuse is development-set evaluation with optimistic bias. Adoption requires an independent holdout without example overlap.",
                            "Held tags are excluded from missing/excess denominators; failed runs are reported separately.",
                            "Classification rates use successful outputs; selected-success bias remains. Empty arrays can improve format validity.",
                            "Cohorts overlap and must not be summed; explicit historical memberships require source verification.",
                            "temperature=0 does not guarantee identical output. No ranking or service-quality claim."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--inputs", type=Path, required=True)
    prep.add_argument("--taxonomy", type=Path, default=Path("data/creator-subtopics.json"))
    prep.add_argument("--cohorts", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    lock = commands.add_parser("freeze")
    lock.add_argument("--pack", type=Path, required=True)
    lock.add_argument("--reviews", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("evaluate")
    score.add_argument("--criteria", type=Path, required=True)
    score.add_argument("--summary", type=Path, action="append", required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            pack = prepare(read(args.inputs), read(args.taxonomy), read(args.cohorts) if args.cohorts else None)
            with new_artifacts(args.output) as staging:
                write_new(staging / "pack.private.json", pack)
                with (staging / "human-review.private.csv").open("x", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(review_rows(pack))
        elif args.command == "freeze":
            write_new(args.output, freeze(read(args.pack), args.reviews))
        else:
            write_new(args.output, evaluate(read(args.criteria), args.summary))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Human evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
