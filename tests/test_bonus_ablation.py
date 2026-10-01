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
