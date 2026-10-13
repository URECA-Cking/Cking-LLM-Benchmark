"""Offline, frozen subtopic classification sampling and human review aggregation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.flow_eval import digest, load_source
from src.subtopic_tags import load_taxonomy
from src.detail_tagging import RevisionStore, classify, digest as revision_digest, POLICY_VERSION

BOUNDARIES = ("duplicate", "unclassified", "over_limit", "counterexample")
FIELDS = ("channel_id", "input_hash", "auto_topics", "auto_evidence", "status",
          "human_topics", "verdict", "cause", "reason", "reviewer")


def issues(codes, parents, taxonomy, limit):
    known = {t["code"]: t["parent"] for t in taxonomy["subtopics"]}
    found = []
    if len(codes) != len(set(codes)):
        found.append("duplicate")
    if any(sum(known.get(c) == p for c in codes) > limit for p in parents) or len(parents) > 3:
        found.append("over_limit")
    if any(c not in known for c in codes):
        found.append("unknown_tag")
    if any(c in known and known[c] not in parents for c in codes):
        found.append("parent_mismatch")
    return found


def prepare(records, taxonomy, size=50, limit=3, counterexamples=()):
    if size < 1 or limit < 1 or not records:
        raise ValueError("Positive size/limit and nonempty records required")
    by_id = {r["channel_id"]: r for r in records}
    if len(by_id) != len(records) or not set(counterexamples) <= by_id.keys():
        raise ValueError("Duplicate channel or unknown counterexample")
    candidates = []
    for row in sorted(records, key=lambda r: r["channel_id"]):
        if not set(row["parents"]) <= set(taxonomy["parents"]):
            raise ValueError("Unknown parent")
        codes = [t["code"] for t in row["topics"]]
        errors = issues(codes, row["parents"], taxonomy, limit)
        if any(not isinstance(t.get("evidence"), str) or not t["evidence"].strip()
               or t["evidence"] not in row["text"] for t in row["topics"]):
            errors.append("invalid_evidence")
        boundaries = [b for b in BOUNDARIES if b in errors]
        if not codes:
            boundaries.append("unclassified")
        if row["channel_id"] in counterexamples:
            boundaries.append("counterexample")
        candidates.append({"channel_id": row["channel_id"], "input_hash": digest(row["text"]),
                           "parents": sorted(set(row["parents"])), "topics": row["topics"],
                           "boundaries": boundaries, "auto_issues": errors})
        if "revision" in row:
            candidates[-1]["source_revision"] = row["revision"]
    selected, covered = [], set()
    required = set(taxonomy["parents"]) | set(BOUNDARIES)
    while candidates and len(selected) < size:
        winner = max(candidates, key=lambda r: len((set(r["parents"]) | set(r["boundaries"])) - covered))
        labels = set(winner["parents"]) | set(winner["boundaries"])
        winner["selection_reason"] = sorted(labels - covered) or ["deterministic_fill"]
        covered |= labels
        selected.append(winner)
        candidates.remove(winner)
    manifest = {"schema_version": 1, "taxonomy_hash": digest(taxonomy), "limit": limit,
                "limit_scope": "per_parent", "policy_version": POLICY_VERSION,
                "requested_size": size, "population_size": len(records),
                "source_hash": digest(sorted(records, key=lambda r: r["channel_id"])),
                "missing_coverage": sorted(required - covered), "size_shortfall": max(0, size-len(selected)),
                "rows": selected}
    return {"manifest": manifest, "hash": digest(manifest)}


def review_rows(pack):
    return [{"channel_id": r["channel_id"], "input_hash": r["input_hash"],
             "auto_topics": "|".join(t["code"] for t in r["topics"]),
             "auto_evidence": json.dumps(r["topics"], ensure_ascii=False),
             "status": "pending", "human_topics": "", "verdict": "", "cause": "",
             "reason": "", "reviewer": ""} for r in pack["manifest"]["rows"]]


def aggregate(pack, rows, taxonomy):
    manifest = pack["manifest"]
    if pack["hash"] != digest(manifest) or manifest["taxonomy_hash"] != digest(taxonomy):
        raise ValueError("Manifest or taxonomy changed")
    expected = {r["channel_id"]: r for r in review_rows(pack)}
    if len(rows) != len(expected) or {r["channel_id"] for r in rows} != expected.keys():
        raise ValueError("Missing, duplicate or unknown review row")
    sources = {r["channel_id"]: r for r in manifest["rows"]}
    counts, corrections, unresolved = Counter(), [], []
    for row in rows:
        cid = row["channel_id"]
        if any(row[k] != expected[cid][k] for k in FIELDS[:4]):
            raise ValueError("Frozen review input changed")
        status = row["status"]
        if status not in {"pending", "reviewed", "undecided"}:
            raise ValueError("Invalid review status")
        counts[status] += 1
        if status == "pending":
            if any(row[k] for k in FIELDS[5:]):
                raise ValueError("Pending row contains review; choose reviewed or undecided")
            unresolved.append({"channel_id": cid, "status": status})
            continue
        if not row["reviewer"].strip() or not row["reason"].strip():
            raise ValueError("Reviewer and reason required")
        if status == "undecided":
            unresolved.append({"channel_id": cid, "status": status, "reason": row["reason"]})
            continue
        human = row["human_topics"].split("|") if row["human_topics"] else []
        if issues(human, sources[cid]["parents"], taxonomy, manifest["limit"]):
            raise ValueError("Invalid human tags, limit or parent relation")
        auto = set(t["code"] for t in sources[cid]["topics"])
        missing, excess = sorted(set(human)-auto), sorted(auto-set(human))
        changed = bool(missing or excess or sources[cid]["auto_issues"])
        if row["verdict"] != ("corrected" if changed else "agree"):
            raise ValueError("Verdict does not match comparison")
        allowed = {"model_error", "criteria_problem", "both"} if changed else {"none"}
        if row["cause"] not in allowed:
            raise ValueError("Explicit error cause required")
        counts[row["cause"]] += 1
        counts["missing_channels"] += bool(missing)
        counts["excess_channels"] += bool(excess)
        counts["auto_unclassified"] += not auto
        counts["human_unclassified"] += not human
        if changed:
            corrections.append({"channel_id": cid, "missing": missing, "excess": excess,
                                "human_topics": human, "cause": row["cause"], "reason": row["reason"]})
    return {"manifest_hash": pack["hash"], "status": "review_complete" if not unresolved else "human_review_pending",
            "sample_size": len(rows), "counts": dict(counts), "corrections": corrections,
            "unresolved": unresolved, "missing_coverage": manifest["missing_coverage"],
            "size_shortfall": manifest["size_shortfall"],
            "parent_counts": dict(Counter(p for r in manifest["rows"] for p in r["parents"])),
            "boundary_counts": dict(Counter(b for r in manifest["rows"] for b in r["boundaries"])),
            "auto_issue_counts": dict(Counter(b for r in manifest["rows"] for b in r["auto_issues"])),
            "limitations": ["Purposive sample; no population accuracy estimate", "Classification only; no ranking improvement claim",
                            "Counterexamples require explicit human nomination", "Schedule sharing and demo approval require human action"]}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "aggregate"))
    parser.add_argument("--taxonomy", type=Path, default=Path("data/creator-subtopics.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("inputs/real_v4"))
    parser.add_argument("--source-dir", type=Path, default=Path("results/real_v4"))
    parser.add_argument("--subtopic-dir", type=Path, default=Path("results/subtopics_v1"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--counterexamples", type=Path, help="JSON array of human-nominated channel IDs")
    parser.add_argument("--revision-index", type=Path,
                        help="Optional JSON object channel_id -> automatic RevisionStore file")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)
    taxonomy = read_json(args.taxonomy)
    if args.step == "prepare":
        data, _, _, parents, _ = load_source(args.data_dir, args.source_dir)
        taxonomy = load_taxonomy(args.taxonomy, data.codes)
        plan = read_json(args.subtopic_dir / "subtopic_plan.json") if not args.revision_index else {}
        tags = read_json(args.subtopic_dir / "subtopics_llm.json") if not args.revision_index else {"tags": {}}
        source_hash = digest({"texts": [(c.id, c.text()) for c in data.pool],
                              "llm_parents": {cid: sorted(v) for cid, v in parents.items()}})
        if not args.revision_index and (plan["taxonomy_hash"] != digest(taxonomy)
            or tags["contract_hash"] != digest(plan) or plan["source_hash"] != source_hash):
            raise ValueError("Stale classification contract")
        records = []
        for c in data.pool:
            record = tags["tags"].get(c.id)
            if not args.revision_index and ((record is None and parents[c.id])
                or (record is not None and record["text_hash"] != digest(c.text()))):
                raise ValueError("Missing or stale automatic classification")
            records.append({"channel_id": c.id, "text": c.text(), "parents": sorted(parents[c.id]),
                            "topics": record["topics"] if record else []})
        if args.revision_index:
            indexed = read_json(args.revision_index)
            by_id = {r["channel_id"]: r for r in records}
            if not isinstance(indexed, dict) or not indexed or not indexed.keys() <= by_id.keys():
                raise ValueError("Invalid revision index")
            records = []
            for cid, filename in sorted(indexed.items()):
                path = args.revision_index.parent / filename
                record = RevisionStore(path.parent).read(path.stem)
                original = by_id[cid]
                if (record["kind"] != "automatic" or record["channel_id"] != cid
                    or record["revision"] != path.stem
                    or record["input_hash"] != revision_digest(original["text"])
                    or record["criteria_hash"] != revision_digest(taxonomy)
                    or record["policy_version"] != POLICY_VERSION
                    or record["result"] != classify(original["text"], record["candidates"], taxonomy)):
                    raise ValueError("Stale or incompatible automatic revision")
                records.append({**original, "parents": [r["code"] for r in record["result"]["parents"]],
                                "topics": record["result"]["topics"], "revision": record["revision"]})
        pack = prepare(records, taxonomy, args.size, args.limit,
                       read_json(args.counterexamples) if args.counterexamples else ())
        pack["manifest"]["classification_contract"] = {
            "legacy_plan_hash": digest(plan), "legacy_model": plan.get("tagger_model"),
            "legacy_prompt": plan.get("tag_prompt_version"),
            "mode": "automatic_revisions" if args.revision_index else "legacy_selected_subtopic_cache"}
        pack["hash"] = digest(pack["manifest"])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "manifest.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
        with (args.output_dir / "reviews.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(review_rows(pack))
        # Local-only reference, separate from the frozen manifest and aggregate report.
        selected = {r["channel_id"] for r in pack["manifest"]["rows"]}
        (args.output_dir / "private_inputs.json").write_text(json.dumps(
            [r for r in records if r["channel_id"] in selected], ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if args.manifest is None or args.reviews is None:
            parser.error("aggregate requires --manifest and --reviews")
        with args.reviews.open(encoding="utf-8-sig", newline="") as f:
            reviews = list(csv.DictReader(f))
            report = aggregate(read_json(args.manifest), reviews, taxonomy)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / "report.md").write_text("# 세부 태그 검수 보고서\n\n```json\n" +
            json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
        RevisionStore(args.output_dir / "revisions").save({"kind": "sample_review_aggregate",
            "manifest_hash": report["manifest_hash"], "report": report,
            "reviews": reviews, "reviews_hash": digest(args.reviews.read_text(encoding="utf-8-sig"))})
    print("Artifacts saved locally; human review and schedule decisions remain separate.")


if __name__ == "__main__":
    main()
