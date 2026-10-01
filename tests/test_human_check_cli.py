"""Human review checkpoints persist without overwriting or inventing scores."""

import json
import pytest
from src.human_check_cli import run_sheet


def test_resume_keeps_explicit_human_score(tmp_path):
    rows = [
        {"id": "a", "title": "test", "display": "evidence"},
        {"id": "b", "title": "test", "display": "other"},
    ]
    path = tmp_path / "scores.json"
    answers = iter(["1", "reason", "q"])
    run_sheet(rows, path, read=lambda _: next(answers), emit=lambda _: None)
    assert json.loads(path.read_text())["scores"]["a"]["score"] == 1
    answers = iter(["u", "unclear"])
    run_sheet(rows, path, read=lambda _: next(answers), emit=lambda _: None)
    saved = json.loads(path.read_text())
    assert saved["source"] == "human"
    assert saved["scores"]["a"]["note"] == "reason"
    assert saved["scores"]["b"]["score"] is None
    with pytest.raises(ValueError):
        run_sheet([{"id": "a", "title": "test", "display": "changed"}], path)


def test_skip_and_quit_never_create_judgment(tmp_path):
    rows = [{"id": "a", "title": "test", "display": "evidence"}]
    path = tmp_path / "scores.json"
    run_sheet(rows, path, read=lambda _: "s", emit=lambda _: None)
    assert json.loads(path.read_text())["scores"] == {}


def test_mode_specific_data_path_validation():
    from src.human_check_cli import parse_args
    import pytest

    with pytest.raises(SystemExit) as error:
        parse_args(["tags"])
    assert error.value.code == 2
    assert parse_args(["recommendations"]).data_dir is None
    assert (
        parse_args(["tags", "--data-dir", "./external-data"]).data_dir.name
        == "external-data"
    )
