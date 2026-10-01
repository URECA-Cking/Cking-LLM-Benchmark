"""Validate direct-interest ratings and report descriptive per-topic quality."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

from src.flow_eval import digest
from src.human_check_cli import prepare


def score(directory):
    """Keep a single user's correlated interests descriptive."""
    load = lambda name: json.loads((directory / name).read_text())
    plan = load("selected_plan.json")
    candidates = load("candidates.json")
    pairs = load("evaluation_pairs.json")
    human = load("human_review.json")
    rows = prepare(
        Namespace(mode="recommendations", experiment_dir=directory, limit=len(pairs))
    )
    if human["source"] != "human" or human["contract_hash"] != digest(rows):
        raise ValueError("Review contract changed")
    if set(human["scores"]) != set(pairs):
        raise ValueError("Incomplete or extra human ratings")
    if digest(candidates) != plan["candidate_hash"]:
        raise ValueError("Candidate contract changed")
    if {k: r["hash"] for k, r in pairs.items()} != plan["pair_hashes"]:
        raise ValueError("Pair contract changed")
    for row in rows:
        answer = human["scores"][row["id"]]
        value = answer["score"]
        if answer["evidence_hash"] != digest(row):
            raise ValueError("Human evidence changed")
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("Invalid score")
        if answer["status"] != ("uncertain" if value is None else "rated"):
            raise ValueError("Invalid score status")
    cases = plan["cases"]
    if len({c["id"] for c in cases}) != len(cases) or {c["id"] for c in cases} != set(
        candidates
    ):
        raise ValueError("Case IDs mismatch")
    per_topic = {}
    union = set()
    for case in cases:
        metrics = {}
        for method in ("parent_only", "selected_subtopic"):
            selected = candidates[case["id"]][method]
            if len(selected) != 5 or len({row["id"] for row in selected}) != 5:
                raise ValueError("Invalid top-five list")
            keys = [case["id"] + "::" + row["id"] for row in selected]
            union.update(keys)
            values = [human["scores"][key]["score"] for key in keys]
            metrics[method] = {
                "p_at_5": sum(v == 1 for v in values) / 5,
                "unknown_at_5": sum(v is None for v in values) / 5,
            }
        metrics["difference"] = (
            metrics["selected_subtopic"]["p_at_5"] - metrics["parent_only"]["p_at_5"]
        )
        per_topic[case["topic"]] = metrics
    if union != set(pairs):
        raise ValueError("Candidate union mismatch")
    means = {
        method: sum(row[method]["p_at_5"] for row in per_topic.values())
        / len(per_topic)
        for method in ("parent_only", "selected_subtopic")
    }
    return {
        "scope": plan["scope"],
        "contract_verified": True,
        "plan_hash": digest(plan),
        "human_hash": digest(human),
        "rated_pairs": len(pairs),
        "answers": dict(Counter(str(row["score"]) for row in human["scores"].values())),
        "per_topic": per_topic,
        "means": means,
        "difference": means["selected_subtopic"] - means["parent_only"],
        "interpretation": "descriptive single-rater selected-topic diagnosis, not population superiority",
    }


def main():
    """Print descriptive results only after complete human review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.experiment_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
