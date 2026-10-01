"""Verify parent-only bonus changes and rejection of stale prior evidence."""

from copy import deepcopy
import numpy as np
import pytest
from src.bonus_ablation import increased_rank, validate_prior
from src.flow_m234 import rankings
from src.flow_eval import digest


def test_bonus_changes_ranking_without_mutating_params():
    ids = ["seed", "similar", "tagged"]
    vectors = np.array([[1.0, 0.0], [0.95, 0.0], [0.70, 0.0]])
    tags = {"seed": {"X"}, "similar": set(), "tagged": {"X"}}
    case = {"route": "creators", "seeds": ["seed"], "tags": []}
    params = {"bonus_m3": 0.1, "bonus_m4": 0.2}
    args = (case, ids, vectors, np.eye(2), ["X", "Y"], tags, tags, params, 2)
    assert rankings(*args)["M4"][0]["id"] == "similar"
    result = increased_rank(*args)
    assert result[0]["id"] == "tagged"
    assert all(r["id"] != "seed" for r in result)
    assert params["bonus_m4"] == 0.2


def test_prior_contract_and_score_are_validated():
    baseline = {
        k: k for k in ("model", "creator_prompt", "other_prompt", "temperature")
    }
    candidates = {}
    plan = {
        "baseline_hash": digest(baseline),
        "candidate_hash": digest(candidates),
        "pair_hashes": {"q::c": "hash"},
    }
    contract = {"hierarchy_hash": digest(plan), **baseline}
    judgments = {
        "contract_hash": digest(contract),
        "scores": {"q::c": {"hash": "hash", "score": 1}},
    }
    validate_prior(plan, candidates, judgments, baseline)
    for field, value in [("hash", "changed"), ("score", True)]:
        changed = deepcopy(judgments)
        changed["scores"]["q::c"][field] = value
        with pytest.raises(ValueError):
            validate_prior(plan, candidates, changed, baseline)
    with pytest.raises(ValueError):
        validate_prior(plan, {"changed": True}, judgments, baseline)
    changed = deepcopy(judgments)
    changed["contract_hash"] = "changed"
    with pytest.raises(ValueError):
        validate_prior(plan, candidates, changed, baseline)


def test_h4_recomputation_rejects_changed_prior_candidates():
    from src.bonus_ablation import recompute_h4
    from src.hierarchy_eval import hierarchical_rank

    ids = ["seed", "candidate"]
    vectors = np.eye(2)
    tags = {cid: {"X"} for cid in ids}
    topics = {cid: {"X_A"} for cid in ids}
    taxonomy = {"subtopics": [{"code": "X_A", "parent": "X"}]}
    case = {"route": "creators", "seeds": ["seed"], "tags": []}
    params = {"bonus_m4": 0.2}
    old = {
        f"H4_d{d:.1f}": hierarchical_rank(
            case,
            ids,
            vectors,
            vectors,
            ["X", "Y"],
            tags,
            topics,
            taxonomy,
            0.2,
            0.5,
            d,
            1,
        )
        for d in (0.5, 1.0)
    }
    args = (case, ids, vectors, vectors, ["X", "Y"], tags, topics, taxonomy, params, 1)
    assert recompute_h4(*args, old) == old
    changed = deepcopy(old)
    changed["H4_d0.5"][0]["score"] += 0.1
    with pytest.raises(ValueError):
        recompute_h4(*args, changed)


def test_uniform_shift_requires_same_ids_and_all_candidate_scores():
    from src.bonus_ablation import bonus_shift_diagnostic

    candidates = {
        "uniform": {
            "A_M4": [{"id": "a", "score": 0.5}, {"id": "b", "score": 0.4}],
            "B_M4_x1.5": [{"id": "a", "score": 0.6}, {"id": "b", "score": 0.5}],
        },
        "partial": {
            "A_M4": [{"id": "a", "score": 0.5}, {"id": "b", "score": 0.4}],
            "B_M4_x1.5": [{"id": "a", "score": 0.6}, {"id": "b", "score": 0.4}],
        },
        "changed": {
            "A_M4": [{"id": "a", "score": 0.5}],
            "B_M4_x1.5": [{"id": "c", "score": 0.6}],
        },
        "empty": {"A_M4": [], "B_M4_x1.5": []},
    }
    result = bonus_shift_diagnostic(candidates, 0.2)
    assert result["inputs"] == 4
    assert result["same_rankings"] == 3
    assert result["uniform_top_k_shift_inputs"] == 1
    assert result["expected_shift"] == 0.1
