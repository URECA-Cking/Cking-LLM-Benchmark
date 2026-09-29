import csv
import json

import pytest

from src.config import E2_JUDGMENTS_JSON, ROOT_DIR
from src.data import load_creators, query_creators
from src.paired_diff import (
    EMBEDDINGS,
    M4_BGE_MINUS_TOP,
    M4_MINUS_M3,
    compute_all,
    export_artifact,
    format_row,
    load_artifact,
    paired_difference,
    per_query_relevance,
    write_artifact,
)

DOC = ROOT_DIR / "docs" / "embedding-method-selection.md"


def test_paired_difference_counts_wins_and_reports_an_interval_around_the_mean() -> None:
    queries = ["q1", "q2", "q3"]
    a, b = {"q1": 1.0, "q2": 0.0, "q3": 1.0}, {"q1": 0.0, "q2": 0.0, "q3": 1.0}

    result = paired_difference(a, b, queries)

    assert (result["wins"], result["losses"], result["ties"]) == (1, 0, 2)
    assert result["mean"] == pytest.approx(1 / 3)
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


def test_paired_difference_depends_on_the_explicit_query_order_only() -> None:
    queries = ["q1", "q2", "q3", "q4"]
    a = {"q1": 1.0, "q2": 0.2, "q3": 0.6, "q4": 0.0}
    b = {"q4": 0.5, "q3": 0.6, "q2": 0.0, "q1": 0.4}  # 딕셔너리 삽입 순서가 달라도 결과가 같아야 한다

    assert paired_difference(a, b, queries) == paired_difference(dict(reversed(list(a.items()))), b, queries)


def test_committed_artifact_matches_its_pinned_hash() -> None:
    artifact = load_artifact()  # 고정된 해시와 다르면 ValueError

    assert len(artifact["query_ids"]) == 30 and artifact["top_k"] == 5
    assert E2_JUDGMENTS_JSON.exists()


def test_committed_artifact_reproduces_the_documented_e2_means() -> None:
    rel = per_query_relevance(load_artifact())
    mean = lambda method: sum(rel[method].values()) / len(rel[method])

    assert round(mean("M4_bge-m3"), 3) == 0.487
    assert round(mean("M3_bge-m3"), 3) == 0.447
    assert round(mean("M2_bge-m3"), 3) == 0.393


# 결과 문서 표의 첫 칸 이름(임베딩 이름 또는 설정 이름)
DOC_LABEL_BY_EMBEDDING = {"text-embedding-3-small": "text-embedding-3-small", "bge-m3": "bge-m3", "kure-v1": "KURE-v1", "qwen3-embedding-0.6b": "Qwen3-Embedding-0.6B"}
DOC_LABEL_BY_SETTING = {"M4_qwen3-embedding-0.6b": "M4_qwen3"}


def _doc_rows_of_paired_section() -> dict[str, str]:
    """결과 문서 "방식 간 차이" 절의 표 행을 첫 칸 이름으로 찾을 수 있게 모은다(같은 이름이 다른 표에 있어도 이 절 안에서만 본다)."""
    text = DOC.read_text(encoding="utf-8").replace("**", "")
    section = text[text.index("### 방식 간 차이"):text.index("### Qwen3 쿼리 프롬프트 재측정")]
    rows: dict[str, str] = {}
    for line in section.split("\n"):
        if line.startswith("| ") and not line.startswith("| ---"):
            first = line.split("|")[1].strip()
            assert first not in rows or first == "임베딩", f"절 안에 첫 칸 이름이 중복됩니다: {first}"
            rows[first] = line
    return rows


def test_every_row_of_the_documented_paired_tables_matches_the_code_output() -> None:
    """결과 문서 "방식 간 차이"의 두 표가 코드 출력과 행별로 어긋나면 실패한다(문서 수치를 손으로 고쳐 생기는 불일치 방지)."""
    rel = per_query_relevance(load_artifact())
    queries = load_artifact()["query_ids"]
    rows = _doc_rows_of_paired_section()

    for (a, b), label in [
        *[((a, b), DOC_LABEL_BY_EMBEDDING[a.removeprefix("M4_")]) for a, b in M4_MINUS_M3],
        *[((a, b), DOC_LABEL_BY_SETTING.get(b, b)) for a, b in M4_BGE_MINUS_TOP],
    ]:
        cells = format_row(paired_difference(rel[a], rel[b], queries)).split(" | ")
        line = rows[label]
        assert all(cell in line for cell in cells), f"{a} - {b}: 문서 행 '{line}'이 코드 출력 {cells}와 다릅니다"


def test_export_builds_a_compact_artifact_from_results_and_round_trips(tmp_path) -> None:
    query_ids = [c.id for c in query_creators(load_creators())]
    (tmp_path / "judge_sheet.csv").write_text(
        "pair_id,query_id,candidate_id,score\n" + "".join(f"p{i},{q},C{i},{i % 3}\n" for i, q in enumerate(query_ids)), encoding="utf-8"
    )
    candidates = {"M2_x": {q: [[f"C{i}", 0.9]] * 7 for i, q in enumerate(query_ids)}}
    (tmp_path / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")

    artifact = export_artifact(tmp_path)
    digest = write_artifact(artifact, tmp_path / "e2.json")

    assert artifact["query_ids"] == query_ids and len(artifact["judgments"]) == len(query_ids)
    assert all(len(ids) == 5 for ids in artifact["top_ids_by_method"]["M2_x"].values())  # 상위 5개만 남긴다
    assert load_artifact(tmp_path / "e2.json", expected_sha256=digest) == artifact
    with pytest.raises(ValueError, match="고정된 내용과 다릅니다"):
        load_artifact(tmp_path / "e2.json", expected_sha256="0" * 64)


def test_export_refuses_a_judge_sheet_with_an_empty_score(tmp_path) -> None:
    (tmp_path / "judge_sheet.csv").write_text("pair_id,query_id,candidate_id,score\np1,F01,F02,\n", encoding="utf-8")
    (tmp_path / "candidates.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="비어 있는"):
        export_artifact(tmp_path)


def test_comparison_lists_cover_every_embedding() -> None:
    assert [a for a, _ in M4_MINUS_M3] == [f"M4_{key}" for key in EMBEDDINGS]
