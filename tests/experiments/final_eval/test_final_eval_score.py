"""Check frozen evidence rejection and paired human scoring."""

import json
from argparse import Namespace

import pytest

from src.final_eval_score import METHOD_KEYS, score
from src.flow_eval import digest
from src.human_check_cli import prepare


def fixture_files(path):
    """Build a tiny known comparison with an uncertain rating."""
    candidates, pairs, cases = {}, {}, []
    for group in ("regular", "short"):
        q = group
        cases.append({"id": q, "group": group})
        candidates[q] = {key: [{"id": "a"}] for key in METHOD_KEYS.values()}
        candidates[q]["M2"] = [{"id": "b"}]
        for cid in ("a", "b"):
            pairs[q + "::" + cid] = {
                "route": "creators",
                "input": q,
                "candidate": cid,
                "hash": cid,
            }
    plan = {
        "cases": cases,
        "top_k": 1,
        "candidate_hash": digest(candidates),
        "pair_hashes": {k: v["hash"] for k, v in pairs.items()},
        "decision_rule": {
            "primary_contrasts": ["M3-M2"],
            "bootstrap_seed": 20261001,
            "bootstrap_resamples": 10000,
            "interval_rule": "per contrast 99%",
        },
    }
    for name, data in [
        ("final_plan.json", plan),
        ("candidates.json", candidates),
        ("evaluation_pairs.json", pairs),
    ]:
        (path / name).write_text(json.dumps(data))
    rows = prepare(Namespace(mode="recommendations", experiment_dir=path, limit=4))
    answers = {}
    for row in rows:
        value = 1 if row["id"].endswith("::a") else None
        answers[row["id"]] = {
            "score": value,
            "status": "rated" if value == 1 else "uncertain",
            "evidence_hash": digest(row),
        }
    human = {"source": "human", "contract_hash": digest(rows), "scores": answers}
    (path / "human_review.json").write_text(json.dumps(human))
    return human


def test_known_paired_difference_and_unknown(tmp_path):
    fixture_files(tmp_path)
    result = score(tmp_path)
    for group in result["groups"].values():
        assert group["p_at_5"]["M2"] == 0
        assert group["unknown_at_5"]["M2"] == 1
        assert group["contrasts"]["M3-M2"] == {"difference": 1, "interval_99": [1, 1]}


@pytest.mark.parametrize("mutation", ["missing", "evidence", "score", "candidate"])
def test_reject_changed_or_incomplete_contract(tmp_path, mutation):
    human = fixture_files(tmp_path)
    if mutation == "candidate":
        file = tmp_path / "candidates.json"
        data = json.loads(file.read_text())
        data["regular"]["M2"][0]["id"] = "a"
    else:
        file = tmp_path / "human_review.json"
        data = human
        key = next(iter(data["scores"]))
        if mutation == "missing":
            del data["scores"][key]
        elif mutation == "evidence":
            data["scores"][key]["evidence_hash"] = "changed"
        else:
            data["scores"][key]["score"] = 2
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        score(tmp_path)
