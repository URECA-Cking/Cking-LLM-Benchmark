"""Offline recommendation-flow experiment using frozen real-data artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np

from src import real_eval as real
from src.config import REAL_SEED, REAL_TOP_K


def digest(value) -> str:
    """Hash a canonical JSON value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_snapshot(data, vectors, category_vectors, tags, params) -> str:
    """Pin source text, labels, vector contents, tags, and source split."""
    return digest({
        "pool": [(c.id, c.name, c.bio) for c in data.pool],
        "gold": data.gold, "categories": data.categories, "params": params,
        "tags": {cid: sorted(value) for cid, value in tags.items()},
        "vectors": hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest(),
        "category_vectors": hashlib.sha256(np.ascontiguousarray(category_vectors).tobytes()).hexdigest(),
    })


def normalized(vector):
    """Normalize a vector, keeping an exactly cancelled vector at zero."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros_like(vector)


def make_plan(data, params, source: str, seed: int = REAL_SEED) -> dict:
    """Freeze onboarding inputs and synthetic favorite sets without dev seeds."""
    dev, test = params["dev_ids"], params["test_ids"]
    if set(dev) & set(test) or set(dev) | set(test) != set(data.gold):
        raise ValueError("Source dev/test split does not match gold labels")
    ids = {c.id for c in data.pool}
    cases = []
    for code in data.codes:
        cases.append({"id": f"tag-{code}", "route": "tags", "group": "single_tag", "tags": [code], "seeds": []})
    for left, right in itertools.combinations(data.codes, 2):
        cases.append({"id": f"tag-{left}-{right}", "route": "tags", "group": "two_tags", "tags": [left, right], "seeds": []})
    for group, query_ids in params["queries"].items():
        if set(query_ids) & set(dev) or not set(query_ids) <= ids:
            raise ValueError("Source query IDs include dev or unknown channels")
        for cid in query_ids:
            cases.append({"id": f"creator-{cid}", "route": "creators", "group": group, "tags": [], "seeds": [cid]})
    rng = random.Random(seed)
    groups = {code: sorted(cid for cid in test if data.gold[cid] == code) for code in data.codes}
    for code, members in groups.items():
        if len(members) >= 3:
            cases.append({"id": f"favorites-same-{code}", "route": "favorites", "group": "same_field", "tags": [], "seeds": sorted(rng.sample(members, 3))})
    available = [code for code in data.codes if groups[code]]
    for left, right in zip(available, available[1:] + available[:1]):
        if left != right:
            cases.append({"id": f"favorites-mixed-{left}-{right}", "route": "favorites", "group": "mixed_fields", "tags": [], "seeds": [rng.choice(groups[left]), rng.choice(groups[right])]})
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return {"version": 1, "source": source, "seed": seed, "top_k": REAL_TOP_K,
            "tag_source": "cached_llm_proxy_not_self_selected", "history_source": "synthetic_not_user_history",
            "dev_ids": dev, "test_ids": test, "queries": params["queries"],
            "bonus": params["bonus_m4"], "cases": cases,
            "pending": ["service_tag_contract", "popularity_definition", "human_judgments", "adoption_rule"]}


def rank_candidates(case, ids, vectors, category_vectors, codes, tags, bonus, k, seed):
    """Compare methods on the same pool and exclude every selected creator."""
    index = {cid: i for i, cid in enumerate(ids)}
    seeds = case["seeds"]
    if len(set(seeds)) != len(seeds) or any(cid not in index for cid in seeds):
        raise ValueError("Unknown or duplicate seed creator")
    scores = {}
    eligible = np.array([cid not in seeds for cid in ids])
    rng = random.Random(f"{seed}:{case['id']}")
    scores["random"] = np.array([rng.random() for _ in ids])
    if case["route"] == "tags":
        selected = case["tags"]
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(codes):
            raise ValueError("Unknown, duplicate, or empty selected tags")
        if seeds:
            raise ValueError("Tag-only case cannot contain seed creators")
        query = normalized(category_vectors[[codes.index(code) for code in selected]].mean(axis=0))
        cosine = vectors @ query
        overlap = np.array([len(set(selected) & tags[cid]) for cid in ids])
        # Random breaks ties without using relevance labels.
        scores["tag_overlap"] = np.where(overlap > 0, overlap + scores["random"] * 0.01, -np.inf)
        scores["tag_embedding"] = cosine
        scores["tag_filtered_embedding"] = np.where(overlap > 0, cosine, -np.inf)
    elif case["route"] in {"creators", "favorites"}:
        if not seeds or case["tags"]:
            raise ValueError("Creator case requires seeds and no selected tags")
        similarities = vectors[[index[cid] for cid in seeds]] @ vectors.T
        scores["max_similarity"] = similarities.max(axis=0)
        scores["mean_embedding"] = vectors @ normalized(vectors[[index[cid] for cid in seeds]].mean(axis=0))
        # Each seed contributes an equal rank vote; selected creators never vote as candidates.
        votes = np.zeros(len(ids))
        for row in similarities:
            order = sorted((i for i in range(len(ids)) if eligible[i]), key=lambda i: (-float(row[i]), ids[i]))
            for rank, i in enumerate(order, 1):
                votes[i] += 1.0 / (60 + rank)
        scores["rank_fusion"] = votes / len(seeds)
        selected_tags = set().union(*(tags[cid] for cid in seeds))
        overlap = np.array([bool(selected_tags & tags[cid]) for cid in ids])
        scores["max_tag_bonus"] = scores["max_similarity"] + bonus * overlap
    else:
        raise ValueError("Unknown route")
    result = {}
    for method, values in scores.items():
        valid = [i for i in range(len(ids)) if eligible[i] and np.isfinite(values[i])]
        order = sorted(valid, key=lambda i: (-float(values[i]), ids[i]))[:k]
        result[method] = [{"id": ids[i], "score": float(values[i])} for i in order]
    return result


def load_source(data_dir, source_dir):
    """Reuse the original cache freshness checks without calling providers."""
    previous = real.REAL_DIR
    try:
        real.REAL_DIR = source_dir
        data = real.load_real_data(data_dir)
        vectors, category_vectors = real.load_embeddings(data)
        tags = real._llm_tags(data)
        params = json.loads((source_dir / "params.json").read_text())
    finally:
        real.REAL_DIR = previous
    if not np.isfinite(vectors).all() or not np.isfinite(category_vectors).all():
        raise ValueError("Non-finite embedding values")
    if category_vectors.shape[0] != len(data.codes) or vectors.shape[0] != len(data.pool):
        raise ValueError("Embedding shape does not match source")
    return data, vectors, category_vectors, tags, params


def write_json(path, value):
    """Write a new artifact and refuse to overwrite any existing result."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def run(args):
    """Prepare frozen inputs or generate candidates and a blind human sheet."""
    data, vectors, category_vectors, tags, params = load_source(args.data_dir, args.source_dir)
    fingerprint = source_snapshot(data, vectors, category_vectors, tags, params)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if args.step == "prepare":
        plan = make_plan(data, params, fingerprint)
        write_json(plan_path, {"plan": plan, "hash": digest(plan)})
        print(f"[prepare] Frozen {len(plan['cases'])} inputs; no API calls")
        return
    saved = json.loads(plan_path.read_text())
    plan = saved["plan"]
    if saved["hash"] != digest(plan) or plan["source"] != fingerprint:
        raise ValueError("Frozen plan or source changed; prepare a new output directory")
    outputs = [args.output_dir / name for name in ("candidates.json", "provenance.json", "human_sheet.csv")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Existing results will not be overwritten; use a new output directory")
    ids = [c.id for c in data.pool]
    channels = {c.id: c for c in data.pool}
    candidates, provenance, rows = {}, {}, []
    for case in plan["cases"]:
        methods = rank_candidates(case, ids, vectors, category_vectors, data.codes, tags, plan["bonus"], plan["top_k"], plan["seed"])
        candidates[case["id"]] = methods
        union = {}
        for method, selected in methods.items():
            for candidate in selected:
                union.setdefault(candidate["id"], []).append(method)
        if case["route"] == "tags":
            evidence = "\n".join(f"{name}: {description}" for code, name, description in data.categories if code in case["tags"])
        else:
            evidence = "\n---\n".join(channels[cid].text() for cid in case["seeds"])
        for cid, origin in sorted(union.items()):
            key = f"{case['id']}::{cid}"
            provenance[key] = origin
            rows.append({"pair_id": key, "route": case["route"], "input": evidence,
                         "candidate": channels[cid].text(), "score": ""})
    random.Random(plan["seed"]).shuffle(rows)
    write_json(outputs[0], {"plan_hash": saved["hash"], "candidates": candidates})
    write_json(outputs[1], {"plan_hash": saved["hash"], "provenance": provenance})
    with outputs[2].open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "route", "input", "candidate", "score"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {route: sum(case["route"] == route for case in plan["cases"]) for route in ("tags", "creators", "favorites")}
    print(f"[candidates] inputs={counts}; blind human pairs={len(rows)}; no API calls")


def main():
    """Parse the offline experiment command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["prepare", "candidates"])
    parser.add_argument("--data-dir", type=lambda value: Path(value).expanduser(), required=True)
    parser.add_argument("--source-dir", type=lambda value: Path(value).expanduser(), required=True)
    parser.add_argument("--output-dir", type=lambda value: Path(value).expanduser(), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
