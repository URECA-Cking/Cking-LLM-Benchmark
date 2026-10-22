"""Offline comparison of preserved prompt experiments; no model calls."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from src.detail_tagging import RevisionStore, classify, digest, CODE_MAP_VERSIONS, GROUPED_VERSIONS
from src.detail_tagging_batch import read, write_new
from src.detail_tagging_human_eval import evaluate, validated_criteria, _load_run, _metrics


def reference_metrics(metrics):
    """Add precision/recall/F1 with explicit zero-denominator behavior."""
    for key in list(metrics):
        if key.startswith("human_"):
            metrics["reference_" + key[6:]] = metrics.pop(key)
    for axis in ("parents", "topics"):
        counts = metrics[axis]
        positive = counts.pop("human_positive_tags", counts.get("reference_positive_tags", 0))
        counts["reference_positive_tags"] = positive
        predicted = counts["predicted_reviewed_tags"]
        tp = positive - counts["missing"]
        counts.update(true_positive=tp, precision=tp / predicted if predicted else None,
                      recall=tp / positive if positive else None,
                      f1=2 * tp / (positive + predicted) if positive + predicted else None)
    return metrics


def audit_run(path, pack):
    """Check original responses and report error calls separately from channels.

    Caller must validate summary/history/revisions with _load_run first.
    Private details deliberately retain all defects, including those masked by the
    first validator error. Never publish this object directly in public logs.
    """
    summary = read(path)
    conditions = summary["conditions"]
    texts = {r["channel_id"]: r["text"] for r in pack["manifest"]["inputs"]}
    entries = [read(p) for p in (Path(path).parent / "attempts").rglob("*.json")]
    if len(entries) != sum(r["attempts"] for r in summary["rows"]):
        raise ValueError("Unexpected attempt files")
    store = RevisionStore(Path(path).parent / "revisions")
    calls, channels = Counter(), defaultdict(set)
    details, usage, models = [], [], Counter()
    for entry in entries:
        cid = entry["channel_id"]
        if cid not in texts or entry["conditions"] != conditions:
            raise ValueError("Foreign attempt or conditions")
        response = entry.get("response", {})
        models[response.get("model", "missing")] += 1
        usage.append(response.get("usage"))
        if response and response.get("model") != conditions["model_version"]:
            raise ValueError("Response model differs")
        if entry["status"] == "provider_error":
            continue
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        try:
            candidates = json.loads(message.get("content") or "null")
        except (ValueError, TypeError):
            candidates = None
        raw_candidates = candidates
        normalization_error = None
        if (conditions["prompt_version"] in CODE_MAP_VERSIONS
                and choice.get("finish_reason") == "stop" and not message.get("refusal")
                and message.get("content")):
            from src.detail_tagging_code_map import parse, normalize, normalize_grouped
            try:
                raw_candidates = parse(message.get("content") or "null")
                converter = normalize_grouped if conditions["prompt_version"] in GROUPED_VERSIONS else normalize
                candidates = converter(raw_candidates, pack["manifest"]["taxonomy"])
            except ValueError as error:
                normalization_error = error
                candidates = None
        if entry["status"] == "passed":
            record = store.read(entry["revision"])
            if (choice.get("finish_reason") != "stop" or message.get("refusal")
                    or candidates != record["candidates"]
                    or (conditions["prompt_version"] in CODE_MAP_VERSIONS
                        and raw_candidates != record.get("provider_candidates"))):
                raise ValueError("Raw response differs from successful revision")
        else:
            if normalization_error and str(normalization_error) != entry.get("error"):
                raise ValueError("Recorded normalization error differs from raw response")
            if choice.get("finish_reason") == "stop" and not message.get("refusal") and isinstance(candidates, dict):
                try:
                    classify(texts[cid], candidates, pack["manifest"]["taxonomy"])
                except ValueError as error:
                    if str(error) != entry.get("error"):
                        raise ValueError("Recorded validation error differs from raw response") from error
                else:
                    raise ValueError("Validation failure contains valid candidates")
            error = entry.get("error", "unknown")
            calls[error] += 1
            channels[error].add(cid)
            defects = []
            if isinstance(candidates, dict):
                for axis in ("parents", "topics"):
                    seen = set()
                    items = candidates.get(axis)
                    for item in items if isinstance(items, list) else []:
                        if not isinstance(item, dict):
                            continue
                        code = item.get("code")
                        if code in seen:
                            defects.append({"axis": axis, "code": code, "defect": "duplicate"})
                        seen.add(code)
                        for field in ("evidence", "core_evidence"):
                            quote = item.get(field, "")
                            if (not isinstance(quote, str) or (field == "evidence" and not quote.strip())
                                    or (quote and quote not in texts[cid])):
                                defects.append({"axis": axis, "code": code, "field": field,
                                                "defect": "empty_or_nonexact_quote", "quote": quote})
            details.append({"channel_id": cid, "attempt": entry["attempt"],
                            "first_validation_error": error, "defects": defects})
    tokens = Counter()
    for value in usage:
        if not isinstance(value, dict):
            continue
        keys = ("prompt_tokens", "completion_tokens", "total_tokens")
        if all(type(value.get(k)) is int for k in keys):
            if value["prompt_tokens"] + value["completion_tokens"] != value["total_tokens"]:
                raise ValueError("Usage tokens inconsistent")
        for key in keys:
            tokens[key] += value.get(key, 0) or 0
        tokens["cached_prompt_tokens"] += (value.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return {"calls": len(entries), "status_counts": dict(Counter(e["status"] for e in entries)),
            "errors": {k: {"calls": v, "channels": len(channels[k])} for k, v in calls.items()},
            "tokens": dict(tokens), "usage_missing_attempts": sum(not isinstance(u, dict) for u in usage),
            "missing_token_fields": {k: sum(not isinstance(u, dict) or u.get(k) is None for u in usage)
                                     for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "actual_response_models": dict(models), "billing": "not_verified",
            "validation_details": details,
            "final_failures": [{"channel_id": r["channel_id"], "attempts": r["attempts"],
                                "errors": [e.get("error", e.get("error_type")) for e in
                                           sorted(entries, key=lambda e: e["attempt"]) if e["channel_id"] == r["channel_id"]]}
                               for r in summary["rows"] if not r["passed"]]}


def compare_experiments(frozen, paths, origin="ai_proxy", *, schema_experiment=False):
    """Reuse existing strict evaluator; preserve original criteria and run artifacts."""
    report = evaluate(frozen, paths, origin, allow_schema_change=schema_experiment)
    if origin == "human_declared" and any(
            run["conditions"].get("human_criteria_hash") != frozen["hash"] for run in report["runs"]):
        raise ValueError("Experiment runs require bound human criteria")
    versions = [run["conditions"]["prompt_version"] for run in report["runs"]]
    if len(set(versions)) != len(versions):
        raise ValueError("Duplicate experiment version")
    pack, decisions = validated_criteria(frozen, origin)
    loaded = [_load_run(path, pack) for path in paths]
    ids = [r["channel_id"] for r in pack["manifest"]["inputs"]]
    for run, path in zip(report["runs"], paths):
        reference_metrics(run["all"])
        for metrics in run["cohorts"].values():
            reference_metrics(metrics)
        run["audit"] = audit_run(path, pack)
    paired = []
    # Pair-specific success sets prevent a third version changing a previous pair's denominator.
    for first, second in combinations(range(len(paths)), 2):
        common = [cid for cid in ids if all(loaded[i][2][cid] is not None for i in (first, second))]
        paired.append({"versions": [versions[i] for i in (first, second)],
                       "changed_factors": [key for key in ("prompt_hash", "output_schema_hash")
                                           if loaded[first][0]["conditions"][key] != loaded[second][0]["conditions"][key]],
                       "channels": len(common), "excluded_failed_channels": len(ids) - len(common),
                       "runs": [{"prompt_version": versions[i],
                                 "metrics": reference_metrics(_metrics(common, loaded[i][1], loaded[i][2], decisions))}
                                for i in (first, second)]})
    report.update(paired_success=paired, independent_holdout="not_performed",
                  team_boundary_agreement="not_performed", ablation_effect="not_established",
                  decision="retain_v2_pending_independent_human_validation",
                  comparison_hash=digest([frozen["hash"], [r["summary_sha256"] for r in report["runs"]]]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ai-proxy", action="store_true")
    parser.add_argument("--schema-experiment", action="store_true",
                        help="Explicitly compare schema+prompt variants, never claim prompt-only isolation")
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(Path("results").resolve()):
        parser.error("Private experiment reports must stay inside results/")
    report = compare_experiments(read(args.criteria), args.summary,
                                 "ai_proxy" if args.ai_proxy else "human_declared",
                                 schema_experiment=args.schema_experiment)
    write_new(args.output, report)
    # Public stdout contains no source text, channel IDs, quotes or file paths.
    print(json.dumps({"sample_size": report["sample_size"], "decision": report["decision"],
                      "versions": [r["conditions"]["prompt_version"] for r in report["runs"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
