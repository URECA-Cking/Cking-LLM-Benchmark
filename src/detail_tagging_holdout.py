"""Private preparation and preregistered v2/v4 evaluation. Offline by default."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.detail_tagging import (PROMPT_VERSION, V4_PROMPT_VERSION, RevisionStore,
                                build_prompt, candidate_schema, classify, digest)
from src.detail_tagging_batch import read, run_batch, write_new
from src.detail_tagging_batch import conditions as batch_conditions
from src.detail_tagging_human_eval import (validate_attempt_history, evaluate, prepare, review_rows,
                                          validated_criteria, FIELDS)
from src.detail_tagging_compare import load_summary
from src.sample_review import new_artifacts

VERSIONS = (PROMPT_VERSION, V4_PROMPT_VERSION)
SETTINGS = {"model": "gpt-5.4-nano-2026-03-17", "temperature": 0,
            "sdk_max_retries": 0, "timeout_seconds": 60, "max_attempts": 3}


class HoldoutValidationError(ValueError):
    """Safe CLI diagnostic: use fixed messages, never input values or paths."""


def seal(value):
    return {"manifest": value, "hash": digest(value)}


def unseal(value):
    if value["hash"] != digest(value["manifest"]):
        raise HoldoutValidationError("Manifest changed")
    return value["manifest"]


def historical_v3(path, inputs, taxonomy):
    """Audit existing artifacts for cohort extraction only, allowing unknown timeout."""
    report, rows = load_summary(path)
    conditions = report["conditions"]
    for key, expected in {"inputs_hash": digest(inputs), "criteria_hash": digest(taxonomy),
                          "prompt_version": "grounded-detail-candidates-v3",
                          "prompt_hash": digest(build_prompt(taxonomy, "grounded-detail-candidates-v3")),
                          "output_schema_hash": digest(candidate_schema(taxonomy))}.items():
        if conditions.get(key) != expected:
            raise HoldoutValidationError("Historical v3 source differs")
    if (set(rows) != {r["channel_id"] for r in inputs} or report["total"] != len(inputs)
            or report["first_pass"] != sum(r["first_pass"] for r in rows.values())
            or report["passed_after_retries"] != sum(r["passed"] for r in rows.values())):
        raise HoldoutValidationError("Historical v3 summary differs")
    store = RevisionStore(Path(path).parent / "revisions")
    results = {}
    for row in inputs:
        cid = row["channel_id"]
        validate_attempt_history(Path(path).parent, rows[cid], row["text"], taxonomy, conditions)
        results[cid] = None
        if rows[cid]["passed"]:
            record = store.read(rows[cid]["revision"])
            if (record["channel_id"] != cid or record["input_hash"] != digest(row["text"])
                    or record["kind"] != "automatic"
                    or any(record[k] != conditions[k] for k in ("criteria_hash", "model_version", "policy_version",
                                                                 "prompt_version", "prompt_hash", "output_schema_hash"))
                    or classify(row["text"], record["candidates"], taxonomy) != record["result"]):
                raise HoldoutValidationError("Historical v3 revision differs")
            results[cid] = record["result"]
    return report, results


def historical_cohorts(inputs, taxonomy, first, final, index, revisions, v3_summary):
    """Use preserved historical snapshots, without inventing missing attempt data."""
    prepare(inputs, taxonomy)
    ids = [r["channel_id"] for r in inputs]
    sources = [read(p) for p in (first, final, index)]
    old_first, old_final, revision_index = sources
    failures = []
    for snapshot in (old_first, old_final):
        failed = [r["channel_id"] for r in snapshot["failures"]]
        if (len(set(failed)) != len(failed) or set(failed) - set(ids)
                or snapshot["completed"] != len(ids) - len(failed)
                or snapshot["criteria_hash"] != digest(taxonomy)
                or snapshot["model"] != old_final["model"]
                or snapshot["policy"] != old_final["policy"]
                or snapshot["source_manifest_hash"] != old_final["source_manifest_hash"]):
            raise HoldoutValidationError("Historical snapshot differs")
        failures.append(set(failed))
    if not failures[1] <= failures[0] or set(revision_index) != set(ids) - failures[1]:
        raise HoldoutValidationError("Historical success/failure sets differ")
    report, results = historical_v3(v3_summary, inputs, taxonomy)
    if (report["conditions"]["prompt_version"] != "grounded-detail-candidates-v3"
            or report["conditions"]["model_version"] != old_final["model"]
            or report["conditions"]["policy_version"] != old_final["policy"]):
        raise HoldoutValidationError("Expected matching historical v3 run")
    before, revision_hashes = {}, {}
    store = RevisionStore(revisions)
    for row in inputs:
        cid = row["channel_id"]
        if cid not in revision_index:
            continue
        # Never follow paths in a historical index outside the revision store.
        name = Path(revision_index[cid]).name
        record = store.read(Path(name).stem)
        if (record["channel_id"] != cid or record["input_hash"] != digest(row["text"])
                or record["criteria_hash"] != digest(taxonomy)
                or record["model_version"] != old_final["model"]
                or record["policy_version"] != old_final["policy"]
                or record["prompt_version"] != PROMPT_VERSION or record["kind"] != "automatic"
                or classify(row["text"], record["candidates"], taxonomy) != record["result"]):
            raise HoldoutValidationError("Historical revision differs")
        before[cid] = record["result"]
        revision_hashes[cid] = digest(record)
    cohorts = {
        "v3_unclassified_transition": [cid for cid in ids if cid in before and before[cid]["parents"]
                                        and results[cid] is not None and not results[cid]["parents"]],
        "baseline_final_failure": [cid for cid in ids if cid in failures[1]],
        "baseline_retry_success": [cid for cid in ids if cid in failures[0] - failures[1]],
    }
    return {"cohorts": cohorts, "counts": {k: len(v) for k, v in cohorts.items()},
            "overlap": {cid: [k for k, v in cohorts.items() if cid in v] for cid in ids
                        if sum(cid in v for v in cohorts.values()) > 1},
            "source_hashes": [digest(v) for v in sources] + [digest(read(v3_summary))],
            "baseline_revision_hashes": revision_hashes,
            "historical_only": True,
            "limitations": "Snapshot-derived cohorts only; missing v2 conditions and attempts remain unknown."}


def select_holdout(candidates, development, examples, quotas, seed):
    """Deterministic stratified selection from human-declared strata, before models."""
    if not isinstance(seed, str) or not seed.strip() or not quotas:
        raise HoldoutValidationError("Selection seed and strata quotas required")
    if any(not isinstance(k, str) or not k.strip() or type(v) is not int or v < 1
           for k, v in quotas.items()):
        raise HoldoutValidationError("Positive integer quotas required")
    for rows in (candidates, development, examples):
        ids = [r["channel_id"] for r in rows]
        if (len(ids) != len(set(ids)) or any(not isinstance(cid, str) or not cid.strip() for cid in ids)
                or any(not isinstance(r["text"], str) or not r["text"].strip() for r in rows)):
            raise HoldoutValidationError("Invalid or duplicate source inputs")
    excluded_ids = {r["channel_id"] for r in development + examples}
    excluded_texts = {digest(r["text"]) for r in development + examples}
    eligible = []
    excluded = 0
    for row in candidates:
        if row["stratum"] not in quotas:
            raise HoldoutValidationError("Unknown candidate stratum")
        if row["channel_id"] in excluded_ids or digest(row["text"]) in excluded_texts:
            excluded += 1
        else:
            eligible.append(row)
    available = Counter(r["stratum"] for r in eligible)
    shortage = {k: v - available[k] for k, v in quotas.items() if available[k] < v}
    if shortage:
        raise HoldoutValidationError("Insufficient independent candidates by stratum; provide more candidates or revise preregistered quotas")
    selected = []
    for stratum in sorted(quotas):
        pool = sorted((r for r in eligible if r["stratum"] == stratum),
                      key=lambda r: (digest([seed, r["channel_id"]]), r["channel_id"]))
        selected.extend(pool[:quotas[stratum]])
    if len({digest(r["text"]) for r in selected}) != len(selected):
        raise HoldoutValidationError("Duplicate holdout text; curate candidate pool before selection")
    inputs = [{"channel_id": r["channel_id"], "text": r["text"]} for r in selected]
    return seal({"inputs": inputs, "inputs_hash": digest(inputs), "quotas": quotas,
                 "seed": seed, "method": "sha256(seed, channel_id) within declared strata",
                 "selected_strata": {r["channel_id"]: r["stratum"] for r in selected},
                 "available": dict(available), "excluded_overlap": excluded,
                 "development_hash": digest(development), "examples_hash": digest(examples),
                 "candidate_pool_hash": digest(sorted(candidates, key=lambda r: r["channel_id"])),
                 "excluded_ids": sorted(excluded_ids), "excluded_text_hashes": sorted(excluded_texts),
                 "limitations": "Human-declared strata and example inventory require team verification; paraphrases are not detected."})


def protocol(development, holdout, selection, acceptance):
    dev, _ = validated_criteria(development)
    test, _ = validated_criteria(holdout)
    selected = unseal(selection)
    dev_inputs, test_inputs = dev["manifest"]["inputs"], test["manifest"]["inputs"]
    if (selected["development_hash"] != digest(dev_inputs)
            or selected["inputs"] != test_inputs or selected["inputs_hash"] != digest(test_inputs)
            or dev["manifest"]["taxonomy"] != test["manifest"]["taxonomy"]):
        raise HoldoutValidationError("Selection/criteria source differs")
    if any(r["channel_id"] in selected["excluded_ids"] or digest(r["text"]) in selected["excluded_text_hashes"]
           for r in test_inputs):
        raise HoldoutValidationError("Holdout overlaps development/examples")
    keys = {"min_holdout_channels", "min_topic_missing_reduction", "max_topic_excess_increase",
            "max_parent_missing_increase", "max_parent_excess_increase"}
    if (set(acceptance) != keys or any(type(v) is not int or v < 0 for v in acceptance.values())
            or acceptance["min_holdout_channels"] < 1 or acceptance["min_topic_missing_reduction"] < 1):
        raise HoldoutValidationError("Explicit nonnegative acceptance counts and positive minimums required")
    if len(test_inputs) < acceptance["min_holdout_channels"]:
        raise HoldoutValidationError("Holdout below preregistered minimum size")
    taxonomy = dev["manifest"]["taxonomy"]
    return seal({"schema_version": 1, "development": development, "holdout": holdout,
                 "selection": selection, "acceptance": acceptance, "settings": dict(SETTINGS),
                 "prompt_hashes": {v: digest(build_prompt(taxonomy, v)) for v in VERSIONS},
                 "adoption": "v2_retained_pending_team_decision"})


def validated_protocol(value):
    manifest = unseal(value)
    expected = protocol(manifest["development"], manifest["holdout"], manifest["selection"], manifest["acceptance"])
    if expected != value:
        raise HoldoutValidationError("Protocol/settings/prompt changed; preregister a new evaluation")
    return manifest


def run_pair(client, value, split, output):
    manifest = validated_protocol(value)
    if split not in {"development", "holdout"}:
        raise HoldoutValidationError("Unknown evaluation split")
    criteria = manifest[split]
    pack, _ = validated_criteria(criteria)
    settings = manifest["settings"]
    if client.timeout != settings["timeout_seconds"] or client.max_retries != settings["sdk_max_retries"]:
        raise HoldoutValidationError("Client differs from preregistered settings")
    output = Path(output)
    # Check both resume contracts before the first potentially paid request.
    for version in VERSIONS:
        folder = output / version
        expected = {**batch_conditions(pack["manifest"]["inputs"], pack["manifest"]["taxonomy"],
                                       settings["model"], client, version, settings["temperature"]),
                    "max_attempts": settings["max_attempts"], "human_criteria_hash": criteria["hash"]}
        if (folder / "run-conditions.json").exists():
            if read(folder / "run-conditions.json") != expected:
                raise HoldoutValidationError("Run conditions mismatch before paired execution")
        elif folder.exists() and any(folder.iterdir()):
            raise HoldoutValidationError("Unversioned paired output directory")
    binding = {"protocol_hash": value["hash"], "split": split}
    if (output / "evaluation-binding.json").exists():
        if read(output / "evaluation-binding.json") != binding:
            raise HoldoutValidationError("Evaluation binding differs")
    else:
        if output.exists() and any(output.iterdir()):
            raise HoldoutValidationError("New evaluation directory required")
        write_new(output / "evaluation-binding.json", binding)
    summaries = []
    for version in VERSIONS:
        folder = output / version
        summary = run_batch(client, pack["manifest"]["inputs"], pack["manifest"]["taxonomy"],
                            settings["model"], folder, settings["max_attempts"], version,
                            settings["temperature"], criteria)
        summaries.append(summary)
    return summaries


def compare_pair(value, split, summaries):
    manifest = validated_protocol(value)
    if split not in {"development", "holdout"}:
        raise HoldoutValidationError("Unknown evaluation split")
    if len(summaries) != 2:
        raise HoldoutValidationError("Exactly one fresh v2 and one v4 run required")
    for path in summaries:
        binding = read(Path(path).parent.parent / "evaluation-binding.json")
        if binding != {"protocol_hash": value["hash"], "split": split}:
            raise HoldoutValidationError("Evaluation binding differs")
        conditions = read(path)["conditions"]
        if conditions.get("human_criteria_hash") != manifest[split]["hash"]:
            raise HoldoutValidationError("Preregistered human criteria required")
        for key, expected in manifest["settings"].items():
            if conditions.get("model_version" if key == "model" else key) != expected:
                raise HoldoutValidationError("Preregistered run settings differ")
    report = evaluate(manifest[split], summaries)
    runs = {r["conditions"]["prompt_version"]: r for r in report["runs"]}
    if set(runs) != set(VERSIONS):
        raise HoldoutValidationError("Exactly one v2 and one v4 required")
    old, new = [runs[v]["all"] for v in VERSIONS]
    acceptance = manifest["acceptance"]
    checks = {"complete_human_review": report["human_review"] == "review_complete",
              "no_model_failures": old["failed_channels"] == new["failed_channels"] == 0,
              "topic_missing_reduction": old["topics"]["missing"] - new["topics"]["missing"] >= acceptance["min_topic_missing_reduction"],
              "topic_excess": new["topics"]["excess"] - old["topics"]["excess"] <= acceptance["max_topic_excess_increase"],
              "parent_missing": new["parents"]["missing"] - old["parents"]["missing"] <= acceptance["max_parent_missing_increase"],
              "parent_excess": new["parents"]["excess"] - old["parents"]["excess"] <= acceptance["max_parent_excess_increase"]}
    costs = []
    for path in summaries:
        attempts = [read(p) for p in (Path(path).parent / "attempts").rglob("*.json")]
        ids = {r["channel_id"] for r in manifest[split]["criteria"]["pack"]["manifest"]["inputs"]}
        if any(a.get("channel_id") not in ids for a in attempts):
            raise HoldoutValidationError("Foreign channel attempt in paired run")
        usage = [a.get("response", {}).get("usage") for a in attempts]
        costs.append({"prompt_version": read(path)["conditions"]["prompt_version"],
                      "attempts": len(attempts), "usage_missing_attempts": sum(not isinstance(u, dict) for u in usage),
                      "missing_token_fields": {k: sum(not isinstance(u, dict) or u.get(k) is None for u in usage)
                                               for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                      "status_counts": dict(Counter(a["status"] for a in attempts)),
                      "tokens": {k: sum(u.get(k, 0) or 0 for u in usage if isinstance(u, dict))
                                 for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                      "billing": "Actual charges unverified; usage totals are not prices."})
    return {**report, "protocol_hash": value["hash"], "split": split, "checks": checks, "usage": costs,
            "decision": "eligible_for_team_review" if split == "holdout" and all(checks.values()) else "retain_v2",
            "holdout_reuse": "If prompts change after viewing this report, reclassify this sample as development and select a new holdout."}


def record_decision(value, development_summaries, holdout_summaries, choice, reviewer, reason,
                    prompts_changed=False):
    if choice not in {"retain_v2", "additional_experiment", "adopt_v4"} or not reviewer.strip() or not reason.strip():
        raise HoldoutValidationError("Team decision, reviewer and reason required")
    development = compare_pair(value, "development", development_summaries)
    holdout = compare_pair(value, "holdout", holdout_summaries)
    if choice == "adopt_v4" and (prompts_changed or holdout["decision"] != "eligible_for_team_review"
                                 or development["human_review"] != "review_complete"):
        raise HoldoutValidationError("Adoption requires completed human review and eligible untouched holdout")
    return seal({"protocol_hash": value["hash"], "choice": choice, "reviewer": reviewer, "reason": reason,
                 "development_report_hash": digest(development), "holdout_report_hash": digest(holdout),
                 "holdout_status": "reclassified_as_development_new_holdout_required" if prompts_changed else "held_out",
                 "default_change": "Requires a separate reviewed code change; this record does not change v2.",
                 "limitations": "Team-declared decision; reviewer identity is not authenticated. Taxonomy/BE seeds/deployment are separate."})


def publish_pack(output, inputs, taxonomy, cohorts=None):
    pack = prepare(inputs, taxonomy, cohorts)
    with new_artifacts(output) as staging:
        write_new(staging / "inputs.private.json", inputs)
        write_new(staging / "pack.private.json", pack)
        with (staging / "human-review.private.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(review_rows(pack))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    history = commands.add_parser("cohorts")
    for name in ("inputs", "first", "final", "index", "revisions", "v3-summary", "output"):
        history.add_argument("--" + name, type=Path, required=True)
    history.add_argument("--taxonomy", type=Path, default=Path("data/creator-subtopics.json"))
    select = commands.add_parser("select")
    for name in ("candidates", "development-inputs", "examples", "quotas", "output"):
        select.add_argument("--" + name, type=Path, required=True)
    select.add_argument("--seed", required=True)
    select.add_argument("--taxonomy", type=Path, default=Path("data/creator-subtopics.json"))
    register = commands.add_parser("register")
    for name in ("development", "holdout", "selection", "acceptance", "output"):
        register.add_argument("--" + name, type=Path, required=True)
    run = commands.add_parser("run")
    score = commands.add_parser("report")
    for command in (run, score):
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--split", choices=("development", "holdout"), required=True)
        command.add_argument("--output", type=Path, required=True)
    run.add_argument("--approved-paid-transfer", action="store_true", required=True,
                     help="Explicit approval for this run's private text transfer and paid calls")
    score.add_argument("--summary", type=Path, action="append", required=True)
    decision = commands.add_parser("decide")
    decision.add_argument("--protocol", type=Path, required=True)
    decision.add_argument("--development-summary", type=Path, action="append", required=True)
    decision.add_argument("--holdout-summary", type=Path, action="append", required=True)
    decision.add_argument("--choice", choices=("retain_v2", "additional_experiment", "adopt_v4"), required=True)
    decision.add_argument("--reviewer", required=True)
    decision.add_argument("--reason", required=True)
    decision.add_argument("--prompts-changed", action="store_true")
    decision.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "cohorts":
            inputs, taxonomy = read(args.inputs), read(args.taxonomy)
            result = historical_cohorts(inputs, taxonomy, args.first, args.final,
                                        args.index, args.revisions, args.v3_summary)
            with new_artifacts(args.output) as staging:
                write_new(staging / "historical.private.json", result)
                publish_pack(staging / "development", inputs, taxonomy, result["cohorts"])
        elif args.command == "select":
            selection = select_holdout(read(args.candidates), read(args.development_inputs), read(args.examples),
                                       read(args.quotas), args.seed)
            with new_artifacts(args.output) as staging:
                write_new(staging / "selection.private.json", selection)
                publish_pack(staging / "holdout", selection["manifest"]["inputs"], read(args.taxonomy))
        elif args.command == "register":
            write_new(args.output, protocol(read(args.development), read(args.holdout),
                                           read(args.selection), read(args.acceptance)))
        elif args.command == "run":
            from dotenv import load_dotenv
            from openai import OpenAI
            value = read(args.protocol)
            validated_protocol(value)  # Validate before loading secrets or creating client.
            load_dotenv()
            summaries = run_pair(OpenAI(timeout=60, max_retries=0), value, args.split, args.output)
            return 0 if all(s["passed_after_retries"] == s["total"] for s in summaries) else 1
        elif args.command == "report":
            write_new(args.output, compare_pair(read(args.protocol), args.split, args.summary))
        else:
            write_new(args.output, record_decision(read(args.protocol), args.development_summary,
                                                 args.holdout_summary, args.choice, args.reviewer,
                                                 args.reason, args.prompts_changed))
    except HoldoutValidationError as error:
        parser.exit(2, f"Holdout evaluation failed: {error}\n")
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Holdout evaluation failed ({type(error).__name__}): check input schemas, "
                       "frozen criteria and attempt history. Private error details are suppressed.\n")
    except OSError:
        parser.exit(2, "Holdout evaluation failed (OSError): check private input paths, "
                       "permissions and whether the output already exists.\n")
    except RuntimeError:
        parser.exit(2, "Holdout evaluation failed (RuntimeError): execution stopped; "
                       "check preserved private attempt records before resuming.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
