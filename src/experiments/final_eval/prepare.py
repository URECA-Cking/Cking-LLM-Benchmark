"""Freeze 100 unseen query inputs while retaining the explored candidate pool."""

from __future__ import annotations
import argparse
import json
import random
import numpy as np
from collections import Counter
from pathlib import Path
from src import flow_eval as flow
from src.flow_m234 import rankings
from src.experiments.final_eval.development import METHODS
from src.hierarchy_eval import hierarchical_rank
from src.subtopic_tags import load_taxonomy, assign_zero_topics, validate_llm_topics
from src.tagging import rank_all
from src.config import LLM_TAG_MAX
from src.judge import pair_text_hash


def excluded_seeds(base, review_paths):
    """Exclude every prior seed and all directly reviewed query/candidate IDs."""
    excluded = set(base["dev_ids"]) | {s for c in base["cases"] for s in c["seeds"]}
    for path in review_paths:
        scores = json.loads(path.read_text()).get("scores", {})
        for key in scores:
            if "::" in key:
                query, candidate = key.split("::")
                excluded.add(candidate)
                if query.startswith("creator-"):
                    excluded.add(query.removeprefix("creator-"))
            else:
                excluded.add(key)
    return excluded


def sample_inputs(pool, excluded, gold, codes, seed=20261001):
    """Balance 75 regular queries by available gold fields, plus 25 short queries."""
    rng = random.Random(seed)
    buckets = {code: [] for code in codes}
    short = []
    for c in sorted(pool, key=lambda c: c.id):
        if c.id in excluded:
            continue
        if c.is_short():
            short.append(c.id)
        elif c.id in gold:
            buckets[gold[c.id]].append(c.id)
    for rows in buckets.values():
        rng.shuffle(rows)
    rng.shuffle(short)
    regular = []
    while len(regular) < 75:
        before = len(regular)
        for code in codes:
            if buckets[code] and len(regular) < 75:
                regular.append(buckets[code].pop())
        if before == len(regular):
            raise ValueError("Not enough unseen regular inputs")
    if len(short) < 25:
        raise ValueError("Not enough unseen short inputs")
    return [
        {
            "id": "creator-" + cid,
            "route": "creators",
            "group": group,
            "seeds": [cid],
            "tags": [],
        }
        for group, ids in [("regular", regular), ("short", short[:25])]
        for cid in ids
    ]


def execute(args):
    """Bind source, cached tags and decision rules before any final human scores."""
    data, vectors, categories, llm, previous = flow.load_source(
        args.data_dir, args.source_dir
    )
    frozen = json.loads((args.flow_dir / "plan.json").read_text())
    base = frozen["plan"]
    if frozen["hash"] != flow.digest(base) or base["source"] != flow.source_snapshot(
        data, vectors, categories, llm, previous
    ):
        raise ValueError("Base source changed")
    baseline = json.loads((args.baseline_dir / "evaluation_plan.json").read_text())
    if baseline["base_hash"] != frozen["hash"]:
        raise ValueError("Baseline changed")
    params = baseline["params"]
    taxonomy = load_taxonomy(args.taxonomy, data.codes)
    tagplan = json.loads((args.subtopic_dir / "subtopic_plan.json").read_text())
    tagpack = json.loads((args.subtopic_dir / "subtopics_llm.json").read_text())
    if tagplan["taxonomy_hash"] != flow.digest(taxonomy) or tagpack[
        "contract_hash"
    ] != flow.digest(tagplan):
        raise ValueError("Tag contract changed")
    source = flow.digest(
        {
            "texts": [(c.id, c.text()) for c in data.pool],
            "llm_parents": {cid: sorted(t) for cid, t in llm.items()},
        }
    )
    if tagplan["source_hash"] != source:
        raise ValueError("Tag source changed")
    topics = {}
    for c in data.pool:
        if not llm[c.id]:
            topics[c.id] = frozenset()
            continue
        r = tagpack["tags"].get(c.id)
        if r is None or r["text_hash"] != flow.digest(c.text()):
            raise ValueError("Tag evidence changed")
        topics[c.id] = validate_llm_topics(r["topics"], c.text(), llm[c.id], taxonomy)
    with np.load(args.subtopic_dir / "topic_vectors.npz", allow_pickle=False) as cached:
        if (
            str(cached["taxonomy_hash"]) != flow.digest(taxonomy)
            or str(cached["identity"]) != "bge-m3@local"
            or list(cached["codes"]) != [t["code"] for t in taxonomy["subtopics"]]
        ):
            raise ValueError("Topic vectors changed")
        tv = cached["vectors"]
    zero = {
        c.id: r.assigned(params["tau"], LLM_TAG_MAX)
        for c, r in zip(data.pool, rank_all(vectors, categories, data.codes))
    }
    zero_topics = {
        c.id: assign_zero_topics(v, tv, taxonomy, zero[c.id], 0.45)
        for c, v in zip(data.pool, vectors)
    }
    reviews = list(args.review_dir.glob("human_check_*.json")) + [
        args.development_dir / "human_review.json"
    ]
    excluded = excluded_seeds(base, reviews)
    development = json.loads(
        (args.development_dir / "development_plan.json").read_text()
    )
    excluded.update(s for c in development["cases"] for s in c["seeds"])
    cases = sample_inputs(data.pool, excluded, data.gold, data.codes)
    ids = [c.id for c in data.pool]
    channels = {c.id: c for c in data.pool}
    candidates = {}
    pairs = {}
    provenance = {}
    for case in cases:
        rows = rankings(
            case, ids, vectors, categories, data.codes, zero, llm, params, 5
        )
        rows["H3_t0.45_d1.0"] = hierarchical_rank(
            case,
            ids,
            vectors,
            categories,
            data.codes,
            zero,
            zero_topics,
            taxonomy,
            params["bonus_m3"],
            0.5,
            1.0,
            5,
        )
        rows["H4_d1.0"] = hierarchical_rank(
            case,
            ids,
            vectors,
            categories,
            data.codes,
            llm,
            topics,
            taxonomy,
            params["bonus_m4"],
            0.5,
            1.0,
            5,
        )
        candidates[case["id"]] = rows
        query = channels[case["seeds"][0]].text()
        for method, selected in rows.items():
            for row in selected:
                cid = row["id"]
                key = case["id"] + "::" + cid
                pairs[key] = {
                    "route": "creators",
                    "input": query,
                    "candidate": channels[cid].text(),
                    "hash": pair_text_hash(query, channels[cid].text()),
                }
                provenance.setdefault(key, []).append(method)
    plan = {
        "version": "single-rater-unseen-query-v1",
        "scope": "same_pool_unseen_queries_not_new_creator_generalization",
        "cases": cases,
        "methods": METHODS,
        "top_k": 5,
        "sample_seed": 20261001,
        "params": params,
        "h3_threshold": 0.45,
        "coarse_scale": 0.5,
        "detail_scale": 1.0,
        "source_hash": frozen["hash"],
        "taxonomy_hash": flow.digest(taxonomy),
        "tags_hash": flow.digest(tagpack),
        "topic_vectors_hash": flow.digest(tv.tolist()),
        "candidate_hash": flow.digest(candidates),
        "pair_hashes": {k: r["hash"] for k, r in sorted(pairs.items())},
        "excluded_ids": sorted(excluded),
        "decision_rule": {
            "minimum_gain": 0.03,
            "primary_contrasts": ["M3-M2", "M4-M3", "H3-M3", "H4-M4", "H4-H3"],
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20261001,
            "interval_rule": "Bonferroni: per contrast 99% percentile paired bootstrap within each intro group; exploratory single-rater uncertainty",
            "uncertainty": "paired input bootstrap with simultaneous intervals across five prespecified contrasts",
            "unclear": "simpler_method_provisional_not_equivalent",
            "final_adoption": "offline provisional; cost ceilings and service validation not established",
        },
        "unknown_scores": "U counts as not confirmed appropriate; reported separately",
        "fallback": "none tested; report uncertainty and info limitations",
        "human_raters": 1,
        "short_definition": "existing Channel.is_short length rule; not semantic information sufficiency",
    }
    summary = {
        "inputs": len(cases),
        "regular": 75,
        "short": 25,
        "pairs": len(pairs),
        "excluded_ids": len(excluded),
        "new_api_calls": 0,
        "regular_gold_fields": dict(
            Counter(data.gold[c["seeds"][0]] for c in cases if c["group"] == "regular")
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("final_plan.json", plan),
        ("candidates.json", candidates),
        ("evaluation_pairs.json", pairs),
        ("provenance.json", provenance),
        ("preparation_summary.json", summary),
    ]:
        flow.write_json(args.output_dir / name, value)
    print(json.dumps(summary), flush=True)


def main():
    """Require explicit data and cache inputs; make no paid calls."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-dir",
        "source-dir",
        "flow-dir",
        "baseline-dir",
        "subtopic-dir",
        "taxonomy",
        "review-dir",
        "development-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=lambda v: Path(v).expanduser(), required=True)
    execute(p.parse_args())


if __name__ == "__main__":
    main()
