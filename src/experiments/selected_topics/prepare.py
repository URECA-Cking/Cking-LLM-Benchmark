"""Prepare direct-topic preference checks with a fixed cosine and bonus budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import flow_eval as flow
from src.judge import pair_text_hash
from src.subtopic_tags import load_taxonomy, validate_llm_topics


def rank_selected(parent, topic, ids, vectors, category, parents, topics, bonus=0.2):
    """Compare parent-only and explicit-topic bonuses with the same maximum."""
    cosine = vectors @ flow.normalized(category)
    matched = np.array([parent in parents[cid] for cid in ids])
    detailed = np.array([topic in topics[cid] for cid in ids])
    scores = {
        "parent_only": cosine + bonus * matched,
        "selected_subtopic": cosine + (bonus / 2) * matched * (1 + detailed),
    }
    return {
        method: [
            {"id": ids[i], "score": float(values[i])}
            for i in sorted(range(len(ids)), key=lambda i: (-float(values[i]), ids[i]))[
                :5
            ]
        ]
        for method, values in scores.items()
    }


def execute(args):
    """Freeze all taxonomy cases for diagnosis without claiming real preferences."""
    data, vectors, categories, parents, previous = flow.load_source(
        args.data_dir, args.source_dir
    )
    taxonomy = load_taxonomy(args.taxonomy, data.codes)
    source = json.loads((args.subtopic_dir / "subtopic_plan.json").read_text())
    pack = json.loads((args.subtopic_dir / "subtopics_llm.json").read_text())
    if source["taxonomy_hash"] != flow.digest(taxonomy) or pack[
        "contract_hash"
    ] != flow.digest(source):
        raise ValueError("Taxonomy/tagging contract changed")
    source_hash = flow.digest(
        {
            "texts": [(c.id, c.text()) for c in data.pool],
            "llm_parents": {cid: sorted(tags) for cid, tags in parents.items()},
        }
    )
    if source["source_hash"] != source_hash:
        raise ValueError("Source text/parent tags changed")
    topics = {}
    for channel in data.pool:
        if not parents[channel.id]:
            topics[channel.id] = frozenset()
            continue
        record = pack["tags"].get(channel.id)
        if not record or record["text_hash"] != flow.digest(channel.text()):
            raise ValueError("Stale or missing candidate subtopic")
        topics[channel.id] = validate_llm_topics(
            record["topics"], channel.text(), parents[channel.id], taxonomy
        )
    ids = [c.id for c in data.pool]
    channels = {c.id: c for c in data.pool}
    names = {code: name for code, name, _ in data.categories}
    candidates, pairs, cases, coverage = {}, {}, [], {}
    selected = None
    if args.selection_file is not None:
        selected = json.loads(args.selection_file.read_text())
        requested = [row["topic"] for row in selected]
        known = {topic["code"] for topic in taxonomy["subtopics"]}
        if (
            not requested
            or len(set(requested)) != len(requested)
            or not set(requested) <= known
        ):
            raise ValueError("Invalid selected topic IDs")
    for topic in taxonomy["subtopics"]:
        if selected is not None and topic["code"] not in requested:
            continue
        code, parent = topic["code"], topic["parent"]
        case_id = "selected-" + code
        rows = rank_selected(
            parent,
            code,
            ids,
            vectors,
            categories[data.codes.index(parent)],
            parents,
            topics,
        )
        candidates[case_id] = rows
        query = f"선택 관심사: {names[parent]} > {topic['name']}\n세부 주제 정의: {topic['definition']}"
        cases.append({"id": case_id, "parent": parent, "topic": code})
        coverage[code] = {
            "candidate_tag_count": sum(code in topics[cid] for cid in ids),
            "same_top5_order": [r["id"] for r in rows["parent_only"]]
            == [r["id"] for r in rows["selected_subtopic"]],
            "top5_tag_matches": {
                m: sum(code in topics[r["id"]] for r in selected)
                for m, selected in rows.items()
            },
        }
        for selected_rows in rows.values():
            for row in selected_rows:
                cid = row["id"]
                pairs[case_id + "::" + cid] = {
                    "route": "tags",
                    "input": query,
                    "candidate": channels[cid].text(),
                    "hash": pair_text_hash(query, channels[cid].text()),
                }
    plan = {
        "scope": (
            "single_user_selected_preferences"
            if selected is not None
            else "taxonomy_synthetic_inputs_not_actual_user_preferences"
        ),
        "user_selections": selected,
        "cases": cases,
        "top_k": 5,
        "source_hash": flow.digest(
            flow.source_snapshot(data, vectors, categories, parents, previous)
        ),
        "taxonomy_hash": flow.digest(taxonomy),
        "tag_hash": flow.digest(pack),
        "candidate_hash": flow.digest(candidates),
        "pair_hashes": {k: row["hash"] for k, row in sorted(pairs.items())},
        "rule": "same parent-category cosine; parent-only bonus 0.2; selected bonus 0.1 parent + 0.1 topic when parent matches",
        "unknown_candidate_topics": "parent bonus 0.1, no extra topic bonus",
        "quality": "human relevance to selected topic required; tag-match counts are proxy diagnostics only",
        "new_api_calls": 0,
    }
    summary = {
        "taxonomy_cases": len(cases),
        "pairs": len(pairs),
        "new_api_calls": 0,
        "changed_top5_order": sum(not r["same_top5_order"] for r in coverage.values()),
        "fewer_than_five_tagged_candidates": sum(
            r["candidate_tag_count"] < 5 for r in coverage.values()
        ),
        "coverage": coverage,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("selected_plan.json", plan),
        ("candidates.json", candidates),
        ("evaluation_pairs.json", pairs),
        ("preparation_summary.json", summary),
    ]:
        flow.write_json(args.output_dir / name, value)
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage"}))


def main():
    """Require explicit local sources and a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--subtopic-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
