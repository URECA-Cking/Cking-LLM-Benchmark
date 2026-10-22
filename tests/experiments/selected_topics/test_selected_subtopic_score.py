"""Check direct-interest scoring and changed-evidence rejection."""

import json
from argparse import Namespace

import pytest

from src.flow_eval import digest
from src.human_check_cli import prepare
from src.selected_subtopic_score import score


def fixture_files(path):
    """Build five candidates per condition with a known 0.2 improvement."""
    candidates = {
        "q": {
            "parent_only": [{"id": str(i)} for i in range(5)],
            "selected_subtopic": [{"id": str(i)} for i in range(1, 6)],
        }
    }
    pairs = {
        "q::"
        + str(i): {
            "route": "tags",
            "input": "baking",
            "candidate": str(i),
            "hash": str(i),
        }
        for i in range(6)
    }
    plan = {
        "scope": "single_user_selected_preferences",
        "cases": [{"id": "q", "topic": "BAKING"}],
        "candidate_hash": digest(candidates),
        "pair_hashes": {k: r["hash"] for k, r in pairs.items()},
    }
    for name, data in [
        ("selected_plan.json", plan),
        ("candidates.json", candidates),
        ("evaluation_pairs.json", pairs),
    ]:
        (path / name).write_text(json.dumps(data))
    rows = prepare(Namespace(mode="recommendations", experiment_dir=path, limit=6))
    human = {
        "source": "human",
        "contract_hash": digest(rows),
        "scores": {
            row["id"]: {
                "score": None if row["id"] == "q::0" else 1,
                "status": "uncertain" if row["id"] == "q::0" else "rated",
                "evidence_hash": digest(row),
            }
            for row in rows
        },
    }
    (path / "human_review.json").write_text(json.dumps(human))
    return human


def test_known_difference_and_uncertainty(tmp_path):
    fixture_files(tmp_path)
    result = score(tmp_path)
    assert result["means"] == {"parent_only": 0.8, "selected_subtopic": 1.0}
    assert result["difference"] == pytest.approx(0.2)
    assert result["per_topic"]["BAKING"]["parent_only"]["unknown_at_5"] == 0.2


@pytest.mark.parametrize("mutation", ["missing", "evidence", "candidate"])
def test_reject_missing_or_changed_ratings(tmp_path, mutation):
    human = fixture_files(tmp_path)
    if mutation == "candidate":
        file = tmp_path / "candidates.json"
        data = json.loads(file.read_text())
        data["q"]["parent_only"][0]["id"] = "5"
    else:
        file = tmp_path / "human_review.json"
        data = human
        if mutation == "missing":
            del data["scores"]["q::0"]
        else:
            data["scores"]["q::0"]["evidence_hash"] = "changed"
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        score(tmp_path)
