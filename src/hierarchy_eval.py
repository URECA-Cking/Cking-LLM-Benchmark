"""Prepare fixed hierarchical variants without new paid relevance judgments."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src import flow_eval as flow
from src.subtopic_tags import load_taxonomy, assign_zero_topics, validate_llm_topics
from src.tagging import rank_all
from src.config import LLM_TAG_MAX
from src.flow_m234 import rankings, evidence


def bonus_vector(
    query_parents,
    query_topics,
    candidate_parents,
    candidate_topics,
    taxonomy,
    base,
    coarse,
    detail,
):
    """Vectorize the same per-parent fallback as hierarchical_bonus."""
    parent_of = {t["code"]: t["parent"] for t in taxonomy["subtopics"]}
    result = np.zeros(len(candidate_parents))
    for parent in query_parents:
        q = {code for code in query_topics if parent_of[code] == parent}
        present = np.array([parent in parents for parents in candidate_parents])
        if not q:
            values = np.full(len(candidate_parents), base)
        else:
            subsets = [
                {code for code in topics if parent_of[code] == parent}
                for topics in candidate_topics
            ]
            values = np.array(
                [
                    (
                        base
                        if not subset
                        else base * (coarse + (detail if q & subset else 0.0))
                    )
                    for subset in subsets
                ]
            )
        result = np.maximum(result, np.where(present, values, 0.0))
    return result


def hierarchical_rank(
    case,
    ids,
    vectors,
    categories,
    codes,
    parents,
    topics,
    taxonomy,
    base,
    coarse,
    detail,
    k,
):
    """Keep max-of-adjusted-seed-scores and exclude all selected seeds."""
    index = {cid: i for i, cid in enumerate(ids)}
    if case["route"] == "tags":
        cosine = (
            vectors
            @ flow.normalized(
                categories[[codes.index(c) for c in case["tags"]]].mean(axis=0)
            )
        )[None, :]
        qparents = [set(case["tags"])]
        qtopics = [set()]
    else:
        cosine = vectors[[index[cid] for cid in case["seeds"]]] @ vectors.T
        qparents = [parents[cid] for cid in case["seeds"]]
        qtopics = [topics[cid] for cid in case["seeds"]]
    cp = [parents[cid] for cid in ids]
    ct = [topics[cid] for cid in ids]
    adjusted = [
        row + bonus_vector(qp, qt, cp, ct, taxonomy, base, coarse, detail)
        for row, qp, qt in zip(cosine, qparents, qtopics)
    ]
    scores = np.array(adjusted).max(axis=0)
    eligible = [i for i, cid in enumerate(ids) if cid not in case["seeds"]]
    order = sorted(eligible, key=lambda i: (-float(scores[i]), ids[i]))[:k]
    return [{"id": ids[i], "score": float(scores[i])} for i in order]


def execute(args):
    """Validate complete source caches and freeze variant candidate contents."""
    data, vectors, categories, llm, previous = flow.load_source(
        args.data_dir, args.source_dir
    )
    taxonomy = load_taxonomy(args.taxonomy, data.codes)
    frozen = json.loads((args.flow_dir / "plan.json").read_text())
    plan = frozen["plan"]
    if frozen["hash"] != flow.digest(plan) or plan["source"] != flow.source_snapshot(
        data, vectors, categories, llm, previous
    ):
        raise ValueError("Frozen base source changed")
    original = json.loads((args.baseline_dir / "evaluation_plan.json").read_text())
    if original["base_hash"] != frozen["hash"]:
        raise ValueError("Baseline input contract mismatch")
    baseline = json.loads((args.baseline_dir / "candidates.json").read_text())
    if (
        baseline["contract_hash"] != flow.digest(original)
        or flow.digest(baseline["candidates"]) != original["candidate_hash"]
    ):
        raise ValueError("Baseline candidate cache changed")
    sourceplan = json.loads((args.subtopic_dir / "subtopic_plan.json").read_text())
    if sourceplan["taxonomy_hash"] != flow.digest(taxonomy):
        raise ValueError("Taxonomy content changed")
    tagpack = json.loads((args.subtopic_dir / "subtopics_llm.json").read_text())
    if tagpack["contract_hash"] != flow.digest(sourceplan):
        raise ValueError("Subtopic tagging contract mismatch")
    sourcehash = flow.digest(
        {
            "texts": [(c.id, c.text()) for c in data.pool],
            "llm_parents": {cid: sorted(tags) for cid, tags in llm.items()},
        }
    )
    if sourceplan["source_hash"] != sourcehash:
        raise ValueError("Subtopic text/parent source changed")
    llm_topics = {}
    for c in data.pool:
        if not llm[c.id]:
            llm_topics[c.id] = frozenset()
            continue
        record = tagpack["tags"].get(c.id)
        if not record or record["text_hash"] != flow.digest(c.text()):
            raise ValueError("Missing or stale subtopic tag")
        llm_topics[c.id] = validate_llm_topics(
            record["topics"], c.text(), llm[c.id], taxonomy
        )
    with np.load(args.subtopic_dir / "topic_vectors.npz", allow_pickle=False) as saved:
        if str(saved["taxonomy_hash"]) != flow.digest(taxonomy) or list(
            saved["codes"]
        ) != [t["code"] for t in taxonomy["subtopics"]]:
            raise ValueError("Topic vectors changed")
        topic_vectors = saved["vectors"]
        if (
            str(saved["identity"]) != "bge-m3@local"
            or topic_vectors.shape != (len(taxonomy["subtopics"]), vectors.shape[1])
            or not np.isfinite(topic_vectors).all()
        ):
            raise ValueError("Topic embedding identity or shape mismatch")
    params = original["params"]
    ids = [c.id for c in data.pool]
    zero = {
        c.id: r.assigned(params["tau"], LLM_TAG_MAX)
        for c, r in zip(data.pool, rank_all(vectors, categories, data.codes))
    }
    zero_variants = {
        threshold: {
            c.id: assign_zero_topics(
                vector, topic_vectors, taxonomy, zero[c.id], threshold
            )
            for c, vector in zip(data.pool, vectors)
        }
        for threshold in sourceplan["zero_thresholds"]
    }
    chosen = {c["id"]: c for c in plan["cases"] if c["id"] in original["case_ids"]}
    if set(chosen) != set(original["case_ids"]):
        raise ValueError("Missing baseline evaluation inputs")
    candidates = {}
    for case in chosen.values():
        selected = rankings(
            case, ids, vectors, categories, data.codes, zero, llm, params, plan["top_k"]
        )
        if selected != baseline["candidates"][case["id"]]:
            raise ValueError("Baseline rankings differ from source recomputation")
        for weight in sourceplan["bonus_conditions"]:
            coarse, detail = weight["coarse_scale"], weight["detail_scale"]
            for threshold, topics in zero_variants.items():
                name = f"H3_t{threshold:.2f}_d{detail:.1f}"
                selected[name] = hierarchical_rank(
                    case,
                    ids,
                    vectors,
                    categories,
                    data.codes,
                    zero,
                    topics,
                    taxonomy,
                    params["bonus_m3"],
                    coarse,
                    detail,
                    plan["top_k"],
                )
            name = f"H4_d{detail:.1f}"
            selected[name] = hierarchical_rank(
                case,
                ids,
                vectors,
                categories,
                data.codes,
                llm,
                llm_topics,
                taxonomy,
                params["bonus_m4"],
                coarse,
                detail,
                plan["top_k"],
            )
        candidates[case["id"]] = selected
    old = json.loads((args.baseline_dir / "judgments.json").read_text())
    if old["contract_hash"] != flow.digest(original):
        raise ValueError("Old judgments contract mismatch")
    known = old["scores"]
    rows = {}
    reused = {}
    channels = {c.id: c for c in data.pool}
    from src.judge import pair_text_hash

    for case in chosen.values():
        query = evidence(case, data)
        for selected in candidates[case["id"]].values():
            for candidate in selected:
                cid = candidate["id"]
                key = f'{case["id"]}::{cid}'
                h = pair_text_hash(query, channels[cid].text())
                rows[key] = {
                    "route": case["route"],
                    "input": query,
                    "candidate": channels[cid].text(),
                    "hash": h,
                }
                if key in known:
                    if type(known[key]["score"]) is not int or known[key][
                        "score"
                    ] not in (0, 1):
                        raise ValueError("Invalid reused score")
                    if known[key]["hash"] != h:
                        raise ValueError("Reused judgment evidence changed")
                    reused[key] = known[key]
    contract = {
        "baseline_hash": flow.digest(original),
        "taxonomy_hash": flow.digest(taxonomy),
        "subtopic_records_hash": flow.digest(tagpack),
        "topic_vector_hash": flow.digest(topic_vectors.tolist()),
        "conditions": sourceplan["bonus_conditions"],
        "thresholds": sourceplan["zero_thresholds"],
        "candidate_hash": flow.digest(candidates),
        "pair_hashes": {key: r["hash"] for key, r in sorted(rows.items())},
        "evaluation_inputs": len(chosen),
        "status": "prepared_relevance_judgments_pending",
        "formats_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("hierarchy_plan.json", contract),
        ("candidates.json", candidates),
        ("evaluation_pairs.json", rows),
        ("reused_judgments.json", reused),
    ]:
        flow.write_json(args.output_dir / name, value)
    summary = {
        "inputs": len(chosen),
        "methods": len(next(iter(candidates.values()))),
        "total_pairs": len(rows),
        "reused_pairs": len(reused),
        "new_pairs": len(rows) - len(reused),
        "llm_subtopic_assigned": sum(bool(tags) for tags in llm_topics.values()),
        "llm_subtopic_unassigned": sum(not tags for tags in llm_topics.values()),
        "zero_assigned": {
            str(t): sum(bool(tags) for tags in topics.values())
            for t, topics in zero_variants.items()
        },
    }
    flow.write_json(args.output_dir / "preparation_summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    """Generate candidates locally; does not invoke any external provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-dir",
        "source-dir",
        "flow-dir",
        "baseline-dir",
        "subtopic-dir",
        "taxonomy",
        "output-dir",
    ):
        parser.add_argument(
            "--" + name, type=lambda v: Path(v).expanduser(), required=True
        )
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
