import csv

from src.data import load_creators
from src.judge_cli import format_prompt, load_rows, write_rows


def test_load_rows_sorts_by_query_then_candidate(tmp_path) -> None:
    path = tmp_path / "sheet.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "query_id", "candidate_id", "score"])
        writer.writerow(["q2::b", "q2", "b", ""])
        writer.writerow(["q1::b", "q1", "b", ""])
        writer.writerow(["q1::a", "q1", "a", "1"])

    rows = load_rows(path)

    assert [(r["query_id"], r["candidate_id"]) for r in rows] == [("q1", "a"), ("q1", "b"), ("q2", "b")]


def test_write_rows_round_trip_preserves_scores(tmp_path) -> None:
    path = tmp_path / "sheet.csv"
    rows = [
        {"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "2"},
        {"pair_id": "q1::b", "query_id": "q1", "candidate_id": "b", "score": ""},
    ]

    write_rows(rows, path)
    reloaded = load_rows(path)

    assert reloaded[0]["score"] == "2"
    assert reloaded[1]["score"] == ""


def test_format_prompt_shows_bios_and_progress() -> None:
    creators_by_id = {c.id: c for c in load_creators()}
    row = {"pair_id": "F06::X15", "query_id": "F06", "candidate_id": "X15", "score": ""}

    prompt = format_prompt(row, creators_by_id, index=3, total=556)

    assert "[3/556]" in prompt
    assert "달리는거북이" in prompt
    assert "워치로운동분석" in prompt
    assert "가민" in prompt  # 소개글 내용이 실제로 노출되는지
    assert "0=무관" in prompt


def test_format_prompt_falls_back_to_events_when_bio_empty() -> None:
    creators_by_id = {c.id: c for c in load_creators()}
    row = {"pair_id": "F06::X10", "query_id": "F06", "candidate_id": "X10", "score": ""}

    prompt = format_prompt(row, creators_by_id, index=1, total=1)

    assert "겨울 캠핑 장비" in prompt
