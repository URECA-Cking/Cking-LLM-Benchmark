"""Freeze a four-condition bonus ablation using validated prior judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import flow_eval as flow
from src.flow_m234 import rankings, evidence
from src.judge import pair_text_hash
from src.tagging import rank_all
from src.config import LLM_TAG_MAX
from src.hierarchy_eval import hierarchical_rank
from src.subtopic_tags import load_taxonomy, validate_llm_topics


def increased_rank(case, ids, vectors, categories, codes, zero, llm, params, k):
    """Change only the parent-tag bonus, preserving per-seed aggregation."""
    changed = {**params, "bonus_m4": params["bonus_m4"] * 1.5}
    return rankings(case, ids, vectors, categories, codes, zero, llm, changed, k)["M4"]


def validate_prior(plan, candidates, judgments, baseline):
    """Reject changed candidate or judging contracts before reusing scores."""
    if plan["baseline_hash"] != flow.digest(baseline) or plan[
        "candidate_hash"
    ] != flow.digest(candidates):
        raise ValueError("Prior candidate contract changed")
    contract = {
        "hierarchy_hash": flow.digest(plan),
        **{
            k: baseline[k]
            for k in ("model", "creator_prompt", "other_prompt", "temperature")
        },
    }
    if judgments["contract_hash"] != flow.digest(contract):
        raise ValueError("Prior judgment contract changed")
    for key, value in judgments["scores"].items():
        if (
            value["hash"] != plan["pair_hashes"].get(key)
            or type(value["score"]) is not int
            or value["score"] not in (0, 1)
        ):
            raise ValueError("Prior score is stale or invalid")


def recompute_h4(
    case, ids, vectors, categories, codes, llm, topics, taxonomy, params, k, old
):
    """Reject stale C/D artifacts by recomputing both hierarchical conditions."""
    result = {}
    for method, detail in (("H4_d0.5", 0.5), ("H4_d1.0", 1.0)):
        result[method] = hierarchical_rank(
            case,
            ids,
            vectors,
            categories,
            codes,
            llm,
            topics,
            taxonomy,
            params["bonus_m4"],
            0.5,
            detail,
            k,
        )
        if result[method] != old[method]:
            raise ValueError("Prior H4 ranking differs from source recomputation")
    return result


def execute(args):
    """Generate candidates locally and freeze the union for optional judging."""
    data, vectors, categories, llm, previous = flow.load_source(
        args.data_dir, args.source_dir
    )
    frozen = json.loads((args.flow_dir / "plan.json").read_text())
    base = frozen["plan"]
    if frozen["hash"] != flow.digest(base) or base["source"] != flow.source_snapshot(
        data, vectors, categories, llm, previous
    ):
        raise ValueError("Frozen source changed")
    baseline = json.loads((args.baseline_dir / "evaluation_plan.json").read_text())
    if baseline["base_hash"] != frozen["hash"]:
        raise ValueError("Input contract changed")
    prior_plan = json.loads((args.hierarchy_dir / "hierarchy_plan.json").read_text())
    prior_candidates = json.loads((args.hierarchy_dir / "candidates.json").read_text())
    prior_judgments = json.loads((args.hierarchy_dir / "judgments.json").read_text())
    validate_prior(prior_plan, prior_candidates, prior_judgments, baseline)
    cases = [c for c in base["cases"] if c["id"] in baseline["case_ids"]]
    if len(cases) != len(baseline["case_ids"]):
        raise ValueError("Missing cases")
    params = baseline["params"]
    ids = [c.id for c in data.pool]
    zero = {
        c.id: r.assigned(params["tau"], LLM_TAG_MAX)
        for c, r in zip(data.pool, rank_all(vectors, categories, data.codes))
    }
    taxonomy = load_taxonomy(args.taxonomy, data.codes)
    tag_plan = json.loads((args.subtopic_dir / "subtopic_plan.json").read_text())
    tagpack = json.loads((args.subtopic_dir / "subtopics_llm.json").read_text())
    if prior_plan["taxonomy_hash"] != flow.digest(taxonomy) or tag_plan[
        "taxonomy_hash"
    ] != flow.digest(taxonomy):
        raise ValueError("Taxonomy changed")
    if tagpack["contract_hash"] != flow.digest(tag_plan) or prior_plan[
        "subtopic_records_hash"
    ] != flow.digest(tagpack):
        raise ValueError("Subtopic cache contract changed")
    source_hash = flow.digest(
        {
            "texts": [(c.id, c.text()) for c in data.pool],
            "llm_parents": {cid: sorted(tags) for cid, tags in llm.items()},
        }
    )
    if tag_plan["source_hash"] != source_hash:
        raise ValueError("Subtopic source changed")
    topics = {}
    for c in data.pool:
        if not llm[c.id]:
            topics[c.id] = frozenset()
            continue
        record = tagpack["tags"].get(c.id)
        if record is None or record["text_hash"] != flow.digest(c.text()):
            raise ValueError("Missing or stale subtopic evidence")
        topics[c.id] = validate_llm_topics(
            record["topics"], c.text(), llm[c.id], taxonomy
        )
    candidates, pairs, reused = {}, {}, {}
    channels = {c.id: c for c in data.pool}
    for case in cases:
        old = prior_candidates[case["id"]]
        recomputed = rankings(
            case, ids, vectors, categories, data.codes, zero, llm, params, base["top_k"]
        )
        if any(recomputed[m] != old[m] for m in ("M2", "M3", "M4")):
            raise ValueError("Baseline ranking changed")
        h4 = recompute_h4(
            case,
            ids,
            vectors,
            categories,
            data.codes,
            llm,
            topics,
            taxonomy,
            params,
            base["top_k"],
            old,
        )
        selected = {
            "A_M4": old["M4"],
            "B_M4_x1.5": increased_rank(
                case,
                ids,
                vectors,
                categories,
                data.codes,
                zero,
                llm,
                params,
                base["top_k"],
            ),
            "C_H4_equal": h4["H4_d0.5"],
            "D_H4_x1.5": h4["H4_d1.0"],
        }
        candidates[case["id"]] = selected
        query = evidence(case, data)
        for rows in selected.values():
            for row in rows:
                cid = row["id"]
                key = f"{case['id']}::{cid}"
                h = pair_text_hash(query, channels[cid].text())
                pairs[key] = {
                    "route": case["route"],
                    "input": query,
                    "candidate": channels[cid].text(),
                    "hash": h,
                }
                if key in prior_judgments["scores"]:
                    if prior_judgments["scores"][key]["hash"] != h:
                        raise ValueError("Prior evidence changed")
                    reused[key] = {
                        **prior_judgments["scores"][key],
                        "original_source": prior_judgments["scores"][key].get("source"),
                        "source": "reused_bonus_ablation",
                    }
    plan = {
        "baseline_hash": flow.digest(baseline),
        "prior_hierarchy_hash": flow.digest(prior_plan),
        "prior_judgments_hash": flow.digest(prior_judgments),
        "recomputed_subtopics_hash": flow.digest(tagpack),
        "recomputed_taxonomy_hash": flow.digest(taxonomy),
        "candidate_hash": flow.digest(candidates),
        "pair_hashes": {key: row["hash"] for key, row in sorted(pairs.items())},
        "evaluation_inputs": len(cases),
        "conditions": {
            "A_M4": "parent bonus 1x",
            "B_M4_x1.5": "parent bonus 1.5x",
            "C_H4_equal": "coarse .5 + detail .5",
            "D_H4_x1.5": "coarse .5 + detail 1",
        },
        "primary_contrasts": ["B-A", "C-A", "D-B", "D-C"],
        "status": "exploratory_not_final",
        "formats_used": False,
    }
    summary = {
        "inputs": len(cases),
        "total_pairs": len(pairs),
        "reused_pairs": len(reused),
        "new_pairs": len(pairs) - len(reused),
    }
    coverage = {}
    for case in cases:
        desired = (
            set(case["tags"])
            if case["route"] == "tags"
            else {data.gold[cid] for cid in case["seeds"] if cid in data.gold}
        )
        if not desired:
            continue
        group = case["route"] + "/" + case["group"]
        for method, rows in candidates[case["id"]].items():
            found = set().union(*(llm[row["id"]] for row in rows))
            coverage.setdefault(group, {}).setdefault(method, []).append(
                int(desired <= found)
            )
    diagnostic = {
        group: {
            method: {
                "n": len(values),
                "exposed_inputs": sum(values),
                "all_parent_fields_exposed": sum(values) / len(values),
            }
            for method, values in methods.items()
        }
        for group, methods in coverage.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("hierarchy_plan.json", plan),
        ("candidates.json", candidates),
        ("evaluation_pairs.json", pairs),
        ("reused_judgments.json", reused),
        ("preparation_summary.json", summary),
        ("proxy_coverage.json", diagnostic),
    ]:
        flow.write_json(args.output_dir / name, value)
    print(json.dumps(summary), flush=True)


def main():
    """Prepare only; paid judging is a separate explicit action."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-dir",
        "source-dir",
        "flow-dir",
        "baseline-dir",
        "hierarchy-dir",
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
