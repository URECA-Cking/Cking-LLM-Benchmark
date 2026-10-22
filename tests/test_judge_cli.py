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

    rows, _fieldnames = load_rows(path)

    assert [(r["query_id"], r["candidate_id"]) for r in rows] == [("q1", "a"), ("q1", "b"), ("q2", "b")]


def test_write_rows_round_trip_preserves_scores(tmp_path) -> None:
    path = tmp_path / "sheet.csv"
    fieldnames = ["pair_id", "query_id", "candidate_id", "score"]
    rows = [
        {"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "2"},
        {"pair_id": "q1::b", "query_id": "q1", "candidate_id": "b", "score": ""},
    ]

    write_rows(rows, fieldnames, path)
    reloaded, _fieldnames = load_rows(path)

    assert reloaded[0]["score"] == "2"
    assert reloaded[1]["score"] == ""


def test_write_rows_preserves_text_hash_column(tmp_path) -> None:
    """PR #5 재리뷰 회귀 테스트: judge_cli가 저장할 때 text_hash 열을 지우면 안 된다.

    이전에는 FIELDNAMES가 4열로 고정돼 있어, 사람이 CLI로 점수를 입력할 때마다
    저장하면서 text_hash 열이 통째로 사라졌다 — 그러면 다음 judge-sheet 재생성 때
    입력 텍스트 변경 여부를 확인할 방법이 없어져 점수 보존 검증 자체가 무력화됐다.
    """
    path = tmp_path / "sheet.csv"
    fieldnames = ["pair_id", "query_id", "candidate_id", "score", "text_hash"]
    rows = [{"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "", "text_hash": "abc123"}]

    write_rows(rows, fieldnames, path)
    reloaded, reloaded_fieldnames = load_rows(path)

    assert reloaded_fieldnames == fieldnames
    assert reloaded[0]["text_hash"] == "abc123"


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
