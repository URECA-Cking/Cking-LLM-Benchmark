"""Validate frozen human ratings and reproduce prespecified comparisons."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np

from src.flow_eval import digest
from src.human_check_cli import prepare

METHOD_KEYS = {
    "M2": "M2",
    "M3": "M3",
    "M4": "M4",
    "H3": "H3_t0.45_d1.0",
    "H4": "H4_d1.0",
}


def score(directory):
    """Reject incomplete or changed evidence before computing paired intervals."""
    load = lambda name: json.loads((directory / name).read_text())
    plan = load("final_plan.json")
    candidates = load("candidates.json")
    pairs = load("evaluation_pairs.json")
    human = load("human_review.json")
    rows = prepare(
        Namespace(mode="recommendations", experiment_dir=directory, limit=len(pairs))
    )
    if human.get("source") != "human" or human["contract_hash"] != digest(rows):
        raise ValueError("Human review contract changed")
    if set(human["scores"]) != set(pairs):
        raise ValueError("Human ratings are incomplete or contain extra pairs")
    if plan["candidate_hash"] != digest(candidates):
        raise ValueError("Frozen candidates changed")
    if {k: r["hash"] for k, r in pairs.items()} != plan["pair_hashes"]:
        raise ValueError("Frozen pair hashes changed")
    for row in rows:
        answer = human["scores"][row["id"]]
        if answer["evidence_hash"] != digest(row):
            raise ValueError("Human rating evidence changed")
        value = answer["score"]
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("Invalid human score")
        if answer["status"] != ("uncertain" if value is None else "rated"):
            raise ValueError("Human score status mismatch")
    case_ids = [case["id"] for case in plan["cases"]]
    if len(set(case_ids)) != len(case_ids) or set(case_ids) != set(candidates):
        raise ValueError("Frozen case IDs mismatch")
    rule = plan["decision_rule"]
    if "per contrast 99%" not in rule["interval_rule"]:
        raise ValueError("Unsupported interval contract")
    result = {
        "contract_verified": True,
        "plan_hash": digest(plan),
        "human_hash": digest(human),
        "rated_pairs": len(rows),
        "answers": dict(Counter(str(v["score"]) for v in human["scores"].values())),
        "groups": {},
        "decision_rule": rule,
    }
    expected_pairs = set()
    for group in ("regular", "short"):
        cases = [c for c in plan["cases"] if c["group"] == group]
        if not cases:
            raise ValueError("Missing intro group")
        arrays, unknown = {}, {}
        for label, key in METHOD_KEYS.items():
            values, uncertain = [], []
            for case in cases:
                selected = candidates[case["id"]][key]
                if len(selected) != plan["top_k"] or len(
                    {r["id"] for r in selected}
                ) != len(selected):
                    raise ValueError("Invalid top-K candidate list")
                keys = [case["id"] + "::" + row["id"] for row in selected]
                expected_pairs.update(keys)
                ratings = [human["scores"][key]["score"] for key in keys]
                values.append(sum(v == 1 for v in ratings) / plan["top_k"])
                uncertain.append(sum(v is None for v in ratings) / plan["top_k"])
            arrays[label] = np.array(values)
            unknown[label] = float(np.mean(uncertain))
        rng = np.random.default_rng(rule["bootstrap_seed"])
        indices = rng.integers(
            0, len(cases), size=(rule["bootstrap_resamples"], len(cases))
        )
        contrasts = {}
        for contrast in rule["primary_contrasts"]:
            a, b = contrast.split("-")
            differences = arrays[a] - arrays[b]
            low, high = np.quantile(differences[indices].mean(axis=1), [0.005, 0.995])
            contrasts[contrast] = {
                "difference": float(differences.mean()),
                "interval_99": [float(low), float(high)],
            }
        result["groups"][group] = {
            "n": len(cases),
            "p_at_5": {m: float(v.mean()) for m, v in arrays.items()},
            "unknown_at_5": unknown,
            "contrasts": contrasts,
        }
    if expected_pairs != set(pairs):
        raise ValueError("Candidate union and rated pairs mismatch")
    return result


def main():
    """Save an aggregate without changing the frozen plan or human answers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.experiment_dir)
    if args.output.resolve() in {
        (args.experiment_dir / name).resolve()
        for name in (
            "final_plan.json",
            "candidates.json",
            "evaluation_pairs.json",
            "human_review.json",
        )
    }:
        raise ValueError("Output must not overwrite frozen inputs")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
