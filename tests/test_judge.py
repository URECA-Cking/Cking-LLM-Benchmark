import csv

from src.judge import JudgeRow, build_judge_pairs, load_existing_scores, shuffle_rows, write_judge_sheet


def test_build_judge_pairs_deduplicates_across_methods() -> None:
    top5 = {
        "q1": {
            "M1": [("a", 0.9), ("b", 0.5)],
            "M2": [("a", 0.8), ("c", 0.4)],
        },
        "q2": {
            "M1": [("d", 0.7)],
        },
    }

    rows, provenance = build_judge_pairs(top5)

    pair_ids = {row.pair_id for row in rows}
    assert pair_ids == {"q1::a", "q1::b", "q1::c", "q2::d"}
    assert provenance["q1::a"] == ["M1", "M2"]
    assert provenance["q1::b"] == ["M1"]
    assert all(row.score == "" for row in rows)


def test_shuffle_rows_is_deterministic_for_same_seed() -> None:
    top5 = {"q1": {"M1": [(f"c{i}", 1.0) for i in range(5)]}}
    rows, _ = build_judge_pairs(top5)

    first = shuffle_rows(rows, seed=1)
    second = shuffle_rows(rows, seed=1)
    different = shuffle_rows(rows, seed=2)

    assert [r.pair_id for r in first] == [r.pair_id for r in second]
    assert {r.pair_id for r in first} == {r.pair_id for r in rows}
    assert [r.pair_id for r in first] != [r.pair_id for r in different] or len(rows) < 2


def test_load_existing_scores_returns_empty_dict_when_file_missing(tmp_path) -> None:
    assert load_existing_scores(tmp_path / "nope.csv") == {}


def test_load_existing_scores_ignores_blank_scores(tmp_path) -> None:
    path = tmp_path / "sheet.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "query_id", "candidate_id", "score", "text_hash"])
        writer.writerow(["q1::a", "q1", "a", "2", "hash-a"])
        writer.writerow(["q1::b", "q1", "b", "", "hash-b"])

    assert load_existing_scores(path) == {"q1::a": ("2", "hash-a")}


def test_write_judge_sheet_preserves_existing_scores_for_same_pair_id_and_text_hash(tmp_path) -> None:
    """리뷰 P1 회귀 테스트: judge-sheet를 다시 만들어도 이미 채운 판정은 사라지면 안 된다."""
    path = tmp_path / "judge_sheet.csv"
    first_rows = [
        JudgeRow(pair_id="q1::a", query_id="q1", candidate_id="a", text_hash="hash-a"),
        JudgeRow(pair_id="q1::b", query_id="q1", candidate_id="b", text_hash="hash-b"),
    ]
    write_judge_sheet(first_rows, path)
    # 판정자가 CLI로 q1::a에 점수 2를 채웠다고 가정하고 파일을 직접 덮어써 시뮬레이션한다.
    rows_after_judging = list(csv.DictReader(path.open(encoding="utf-8")))
    rows_after_judging[0]["score"] = "2"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "query_id", "candidate_id", "score", "text_hash"])
        writer.writeheader()
        writer.writerows(rows_after_judging)

    # 후보가 바뀌어 q1::c가 새로 추가된 상황을 재현 (judge-sheet 재실행). 텍스트는 안 바뀌었다.
    second_rows = [
        JudgeRow(pair_id="q1::a", query_id="q1", candidate_id="a", text_hash="hash-a"),
        JudgeRow(pair_id="q1::b", query_id="q1", candidate_id="b", text_hash="hash-b"),
        JudgeRow(pair_id="q1::c", query_id="q1", candidate_id="c", text_hash="hash-c"),
    ]
    write_judge_sheet(second_rows, path)

    result = {r["pair_id"]: r["score"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert result["q1::a"] == "2"  # 기존 판정 보존
    assert result["q1::b"] == ""  # 원래도 비어 있던 건 그대로
    assert result["q1::c"] == ""  # 새로 생긴 쌍만 빈 칸


def test_write_judge_sheet_clears_score_when_text_hash_changes(tmp_path) -> None:
    """PR #5 재리뷰 회귀 테스트: pair_id가 같아도 텍스트(text_hash)가 바뀌면 점수를 비워야 한다.

    예: X01의 bio가 바뀌어도 id는 그대로 "X01"이라, pair_id만 보면 예전 텍스트로 매긴
    점수가 새 텍스트에도 그대로 적용된 것처럼 남을 수 있었다.
    """
    path = tmp_path / "judge_sheet.csv"
    write_judge_sheet([JudgeRow(pair_id="q1::a", query_id="q1", candidate_id="a", text_hash="hash-old")], path)
    rows_after_judging = list(csv.DictReader(path.open(encoding="utf-8")))
    rows_after_judging[0]["score"] = "2"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "query_id", "candidate_id", "score", "text_hash"])
        writer.writeheader()
        writer.writerows(rows_after_judging)

    # a의 소개글이 바뀌어 text_hash가 달라진 상황을 재현
    write_judge_sheet([JudgeRow(pair_id="q1::a", query_id="q1", candidate_id="a", text_hash="hash-new")], path)

    result = {r["pair_id"]: r["score"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert result["q1::a"] == ""  # 텍스트가 바뀌어 예전 점수를 버리고 재판정 대상이 됨


def test_write_judge_sheet_never_preserves_rows_without_text_hash(tmp_path) -> None:
    """text_hash를 계산하지 않은 행(빈 문자열)은 안전하게 항상 재판정 대상으로 둔다."""
    path = tmp_path / "judge_sheet.csv"
    write_judge_sheet([JudgeRow(pair_id="q1::a", query_id="q1", candidate_id="a")], path)
    rows_after_judging = list(csv.DictReader(path.open(encoding="utf-8")))
    rows_after_judging[0]["score"] = "1"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "query_id", "candidate_id", "score", "text_hash"])
        writer.writeheader()
        writer.writerows(rows_after_judging)

    write_judge_sheet([JudgeRow(pair_id="q1::a", query_id="q1", candidate_id="a")], path)

    result = {r["pair_id"]: r["score"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert result["q1::a"] == ""
