"""Report blind LLM relevance separately from tag proxies and human scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.flow_eval import digest


def score(directory):
    """Validate all frozen judgments and summarize correlated synthetic cases."""
    load = lambda name: json.loads((directory / name).read_text())
    plan = load("auto_plan.json")
    pairs = load("evaluation_pairs.json")
    candidates = load("candidates.json")
    judged = load("auto_judgments.json")
    if digest(plan) != judged["contract_hash"]:
        raise ValueError("Judge contract changed")
    if (
        digest(pairs) != plan["pairs_hash"]
        or digest(candidates) != plan["candidate_hash"]
    ):
        raise ValueError("Frozen evidence changed")
    scores = judged["scores"]
    if set(scores) != set(pairs):
        raise ValueError("Incomplete or extra judgments")
    for key, record in scores.items():
        values = record["topic_scores"]
        if record["hash"] != pairs[key]["hash"] or len(values) != len(
            pairs[key]["topics"]
        ):
            raise ValueError("Judgment evidence changed")
        if any(type(s) is not int or s not in (-1, 0, 1) for s in values):
            raise ValueError("Invalid topic judgments")
    case_ids = [case["id"] for case in plan["cases"]]
    if len(set(case_ids)) != len(case_ids) or set(case_ids) != set(candidates):
        raise ValueError("Case contract changed")
    per_case, groups, expected = {}, {}, set()
    for case in plan["cases"]:
        qid, topics = case["id"], case["topics"]
        metrics = {}
        for method in ("parent_only", "selected_subtopic"):
            rows = candidates[qid][method]
            if len(rows) != 5 or len({r["id"] for r in rows}) != 5:
                raise ValueError("Invalid top-five list")
            keys = [qid + "::" + row["id"] for row in rows]
            expected.update(keys)
            if any(pairs[key]["topics"] != topics for key in keys):
                raise ValueError("Case topic order changed")
            values = [scores[key]["topic_scores"] for key in keys]
            per_topic = [sum(v[i] == 1 for v in values) / 5 for i in range(len(topics))]
            metrics[method] = {
                "p_at_5_any": sum(1 in v for v in values) / 5,
                "p_at_5_all": sum(all(s == 1 for s in v) for v in values) / 5,
                "unknown_at_5": sum(1 not in v and -1 in v for v in values) / 5,
                "short_bio_at_5": sum(pairs[key]["short_candidate_bio"] for key in keys)
                / 5,
                "interest_coverage": sum(p > 0 for p in per_topic) / len(topics),
                "per_topic_p_at_5": dict(zip(topics, per_topic)),
            }
        metrics["difference_any"] = (
            metrics["selected_subtopic"]["p_at_5_any"]
            - metrics["parent_only"]["p_at_5_any"]
        )
        per_case[qid] = {"group": case["group"], "topics": topics, **metrics}
    if expected != set(pairs):
        raise ValueError("Candidate union mismatch")
    for group in sorted({c["group"] for c in plan["cases"]}):
        group_rows = [r for r in per_case.values() if r["group"] == group]
        summary = {"n": len(group_rows)}
        for method in ("parent_only", "selected_subtopic"):
            summary[method] = {
                metric: sum(r[method][metric] for r in group_rows) / len(group_rows)
                for metric in (
                    "p_at_5_any",
                    "p_at_5_all",
                    "unknown_at_5",
                    "interest_coverage",
                    "short_bio_at_5",
                )
            }
        summary["difference_any"] = (
            summary["selected_subtopic"]["p_at_5_any"]
            - summary["parent_only"]["p_at_5_any"]
        )
        groups[group] = summary
    agreement, disagreements, matrix = 0, [], {}
    for key, human in plan["agreement_targets"].items():
        if key not in scores or len(scores[key]["topic_scores"]) != 1:
            raise ValueError("Invalid human overlap")
        raw = scores[key]["topic_scores"][0]
        automatic = None if raw == -1 else raw
        matrix_key = f"human={human},llm={automatic}"
        matrix[matrix_key] = matrix.get(matrix_key, 0) + 1
        if automatic == human:
            agreement += 1
        else:
            disagreements.append(
                {
                    "key": key,
                    "human": human,
                    "llm": automatic,
                    "reason": scores[key]["reason"],
                }
            )
    return {
        "contract_verified": True,
        "plan_hash": digest(plan),
        "judgment_hash": digest(judged),
        "groups": groups,
        "per_case": per_case,
        "human_agreement": {
            "n": len(plan["agreement_targets"]),
            "matched": agreement,
            "matrix": matrix,
        },
        "human_disagreements": disagreements,
        "usage": {
            "unique_judgments": len(scores),
            "input_tokens": sum(r["input_tokens"] for r in scores.values()),
            "output_tokens": sum(r["output_tokens"] for r in scores.values()),
            "actual_invoice": "unverified; unsuccessful requests/retries may add usage",
        },
        "limits": "LLM proxy quality; synthetic correlated inputs; no population superiority or automatic adoption",
    }


def main():
    """Print a complete validated report without overwriting source evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.experiment_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
