"""Freeze a small explored development set for blind human review."""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from src import flow_eval as flow
from src.flow_m234 import evidence
from src.judge import pair_text_hash

METHODS = ("M2", "M3", "M4", "H3_t0.45_d1.0", "H4_d1.0")


def select_cases(cases, excluded, seed=42):
    """Select ten inputs per intro group, without reusing reviewed seeds."""
    rng = random.Random(seed)
    selected = []
    for group in ("regular", "short"):
        available = sorted(
            (
                c
                for c in cases
                if c["route"] == "creators"
                and c["group"] == group
                and not set(c["seeds"]) & excluded
            ),
            key=lambda c: c["id"],
        )
        rng.shuffle(available)
        if len(available) < 10:
            raise ValueError("Not enough eligible development cases")
        selected.extend(available[:10])
    return selected


def execute(args):
    """Verify frozen source and save blind union plus private provenance."""
    data, vectors, categories, llm, params = flow.load_source(
        args.data_dir, args.source_dir
    )
    frozen = json.loads((args.flow_dir / "plan.json").read_text())
    if frozen["hash"] != flow.digest(frozen["plan"]) or frozen["plan"][
        "source"
    ] != flow.source_snapshot(data, vectors, categories, llm, params):
        raise ValueError("Frozen source changed")
    prior = json.loads((args.hierarchy_dir / "hierarchy_plan.json").read_text())
    candidates = json.loads((args.hierarchy_dir / "candidates.json").read_text())
    baseline = json.loads((args.baseline_dir / "evaluation_plan.json").read_text())
    if (
        prior["candidate_hash"] != flow.digest(candidates)
        or prior["baseline_hash"] != flow.digest(baseline)
        or baseline["base_hash"] != frozen["hash"]
    ):
        raise ValueError("Prior candidate or input contract changed")
    by_case = {c["id"]: c for c in frozen["plan"]["cases"]}
    excluded = set()
    for path in args.review_dir.glob("human_check_*.json"):
        review = json.loads(path.read_text())
        for key in review.get("scores", {}):
            if "::" in key:
                case = by_case.get(key.split("::")[0])
                if case:
                    excluded.update(case["seeds"])
            else:
                excluded.add(key)
    chosen = select_cases(list(by_case.values()), excluded)
    channels = {c.id: c for c in data.pool}
    output_candidates = {}
    pairs = {}
    provenance = {}
    for case in chosen:
        selected = {m: candidates[case["id"]][m] for m in METHODS}
        output_candidates[case["id"]] = selected
        query = evidence(case, data)
        for method, rows in selected.items():
            for row in rows:
                cid = row["id"]
                key = f"{case['id']}::{cid}"
                if cid in case["seeds"]:
                    raise ValueError("Seed appears in recommendations")
                pairs[key] = {
                    "route": "creators",
                    "input": query,
                    "candidate": channels[cid].text(),
                    "hash": pair_text_hash(query, channels[cid].text()),
                }
                provenance.setdefault(key, []).append(method)
    plan = {
        "version": "final-dev-v1",
        "status": "explored_development_not_final",
        "source_hash": frozen["hash"],
        "prior_hierarchy_hash": flow.digest(prior),
        "cases": chosen,
        "methods": METHODS,
        "top_k": 5,
        "sample_seed": 42,
        "pair_hashes": {key: v["hash"] for key, v in sorted(pairs.items())},
        "candidate_hash": flow.digest(output_candidates),
        "excluded_reviewed_seed_ids": sorted(excluded),
        "final_excluded_seed_ids": sorted(
            {seed for case in frozen["plan"]["cases"] for seed in case["seeds"]}
            | set(frozen["plan"]["dev_ids"])
            | excluded
            | {s for c in chosen for s in c["seeds"]}
        ),
    }
    summary = {
        "development_inputs": len(chosen),
        "review_pairs": len(pairs),
        "max_pairs_per_input": 25,
        "new_api_calls": 0,
        "final_sample_available": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("development_plan.json", plan),
        ("candidates.json", output_candidates),
        ("evaluation_pairs.json", pairs),
        ("provenance.json", provenance),
        ("preparation_summary.json", summary),
    ]:
        flow.write_json(args.output_dir / name, value)
    print(json.dumps(summary), flush=True)


def main():
    """Prepare with explicit external data paths; never call an API."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-dir",
        "source-dir",
        "flow-dir",
        "baseline-dir",
        "hierarchy-dir",
        "review-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=lambda v: Path(v).expanduser(), required=True)
    execute(p.parse_args())


if __name__ == "__main__":
    main()
