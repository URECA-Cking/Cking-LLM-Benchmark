"""API를 호출하지 않고, 가짜 클라이언트로 파이프라인 배선(저장·순서·모양)만 검증한다."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace

import numpy as np

import src.pipeline as pipeline
from src.clients.openai_judge import JudgeResult
from src.clients.openai_tagger import TagResult
from src.data import Creator, load_categories, load_creators


class _FakeEmbeddingClient:
    """텍스트 길이만으로 결정적인 가짜 벡터를 만든다. 실제 API를 호출하지 않는다."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.name = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            vectors[i] = rng.rand(self.dim)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


def test_cmd_embed_writes_vectors_for_all_creators_and_categories(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_embedding_client", lambda key: _FakeEmbeddingClient())

    pipeline.cmd_embed(argparse.Namespace(force=False))

    e4 = json.loads((tmp_path / "e4_embedding.json").read_text(encoding="utf-8"))
    for key in pipeline.EMBEDDING_MODEL_KEYS:
        assert e4[key]["dim"] == 8
        assert e4[key]["total_input_tokens"] is None  # 가짜 클라이언트는 토큰 수를 남기지 않는다

    creators = load_creators()
    categories = load_categories()
    for key in pipeline.EMBEDDING_MODEL_KEYS:
        creator_ids, creator_vectors = pipeline._load_vectors(tmp_path / f"creators_{key}.npz")
        category_ids, category_vectors = pipeline._load_vectors(tmp_path / f"categories_{key}.npz")

        assert creator_ids == [c.id for c in creators]
        assert creator_vectors.shape == (100, 8)
        assert category_ids == [c.code for c in categories]
        assert category_vectors.shape == (10, 8)
        # 정규화됐으므로 각 행의 길이는 1이어야 한다.
        assert np.allclose(np.linalg.norm(creator_vectors, axis=1), 1.0, atol=1e-5)


def test_cmd_embed_skips_api_call_when_cache_exists(tmp_path, monkeypatch) -> None:
    """리뷰 P2 회귀 테스트: 캐시된 벡터가 있으면 재실행해도 embed API를 다시 부르지 않는다."""
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    call_count = {"n": 0}

    def counting_client(key: str) -> _FakeEmbeddingClient:
        call_count["n"] += 1
        return _FakeEmbeddingClient()

    monkeypatch.setattr(pipeline, "_embedding_client", counting_client)

    pipeline.cmd_embed(argparse.Namespace(force=False))
    first_call_count = call_count["n"]
    pipeline.cmd_embed(argparse.Namespace(force=False))  # 재실행

    assert call_count["n"] == first_call_count  # 캐시가 있어 클라이언트를 다시 만들지 않음

    pipeline.cmd_embed(argparse.Namespace(force=True))  # --force는 다시 계산
    assert call_count["n"] > first_call_count


def test_cmd_embed_recomputes_when_input_text_changes(tmp_path, monkeypatch) -> None:
    """PR #5 리뷰 P1 회귀 테스트: 캐시 파일이 있어도 data/creators.csv가 바뀌면 다시 계산해야 한다.

    파일 존재 여부만 보고 건너뛰면, 이전 실행에서 남은 results/cache가 새 입력(예: X01/X02
    bio 수정)을 반영하지 않은 채 재사용될 수 있었다.
    """
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_embedding_client", lambda key: _FakeEmbeddingClient())

    original_creators = load_creators()
    monkeypatch.setattr(pipeline, "load_creators", lambda: original_creators)
    pipeline.cmd_embed(argparse.Namespace(force=False))
    before_ids, before_vectors = pipeline._load_vectors(tmp_path / f"creators_{pipeline.EMBEDDING_MODEL_KEYS[0]}.npz")

    changed_creators = [replace(original_creators[0], bio=original_creators[0].bio + " 경품은 닌텐도 스위치입니다")] + list(original_creators[1:])
    monkeypatch.setattr(pipeline, "load_creators", lambda: changed_creators)
    pipeline.cmd_embed(argparse.Namespace(force=False))  # 캐시 파일은 그대로 있지만 입력이 바뀜
    after_ids, after_vectors = pipeline._load_vectors(tmp_path / f"creators_{pipeline.EMBEDDING_MODEL_KEYS[0]}.npz")

    assert before_ids == after_ids
    assert not np.allclose(before_vectors[0], after_vectors[0])  # 바뀐 크리에이터만 벡터가 달라짐
    assert np.allclose(before_vectors[1], after_vectors[1])  # 나머지는 그대로


def test_cmd_tag_llm_recomputes_when_input_text_changes(tmp_path, monkeypatch) -> None:
    """PR #5 리뷰 P1 회귀 테스트: run 파일이 있어도 입력 텍스트가 바뀌면 다시 태깅해야 한다."""
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "OPENAI_LLM_MODEL_CANDIDATES", {"fake-model": {}})
    call_count = {"n": 0}

    class _CountingTagger:
        def __init__(self, model: str, category_codes: list[str]) -> None:
            pass

        def tag(self, input_text: str) -> TagResult:
            call_count["n"] += 1
            return TagResult(tags=(), input_tokens=1, output_tokens=1)

    monkeypatch.setattr(pipeline, "OpenAITagger", _CountingTagger)

    original_creators = load_creators()
    monkeypatch.setattr(pipeline, "load_creators", lambda: original_creators)
    pipeline.cmd_tag_llm(argparse.Namespace(force=False))
    first_calls = call_count["n"]

    changed_creators = [replace(original_creators[0], bio=original_creators[0].bio + " 변경됨")] + list(original_creators[1:])
    monkeypatch.setattr(pipeline, "load_creators", lambda: changed_creators)
    pipeline.cmd_tag_llm(argparse.Namespace(force=False))  # run 파일은 있지만 입력이 바뀜

    assert call_count["n"] == first_calls * 2  # 캐시를 못 쓰고 전부 다시 태깅함


def test_cmd_tag_llm_skips_completed_runs_when_cache_exists(tmp_path, monkeypatch) -> None:
    """리뷰 P2 회귀 테스트: run 파일이 이미 있으면 그 run은 다시 태깅하지 않는다."""
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "OPENAI_LLM_MODEL_CANDIDATES", {"fake-model": {}})
    call_count = {"n": 0}

    class _CountingTagger:
        def __init__(self, model: str, category_codes: list[str]) -> None:
            pass

        def tag(self, input_text: str) -> TagResult:
            call_count["n"] += 1
            return TagResult(tags=(), input_tokens=1, output_tokens=1)

    monkeypatch.setattr(pipeline, "OpenAITagger", _CountingTagger)

    pipeline.cmd_tag_llm(argparse.Namespace(force=False))
    first_calls = call_count["n"]
    assert first_calls == 100 * pipeline.LLM_CONSISTENCY_RUNS  # 크리에이터 100명 × run 수

    pipeline.cmd_tag_llm(argparse.Namespace(force=False))  # 재실행
    assert call_count["n"] == first_calls  # run 파일이 있어 다시 호출하지 않음

    pipeline.cmd_tag_llm(argparse.Namespace(force=True))
    assert call_count["n"] == first_calls * 2  # --force는 다시 계산


class _FakeJudge:
    """소개글 길이 합으로 결정적인 점수를 주는 가짜 판정기다. 실제 API를 호출하지 않는다."""

    def __init__(self, model: str) -> None:
        self.model = model

    def judge(self, query_text: str, candidate_text: str) -> JudgeResult:
        score = (len(query_text) + len(candidate_text)) % 3
        return JudgeResult(score=score, input_tokens=10, output_tokens=1)


def _write_sheet(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "query_id", "candidate_id", "score"])
        writer.writeheader()
        writer.writerows(rows)


def test_cmd_auto_judge_fills_only_empty_scores_and_writes_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "OpenAIJudge", _FakeJudge)
    sheet_path = tmp_path / "judge_sheet.csv"
    _write_sheet(
        sheet_path,
        [
            {"pair_id": "F01::F02", "query_id": "F01", "candidate_id": "F02", "score": ""},
            {"pair_id": "F01::K01", "query_id": "F01", "candidate_id": "K01", "score": "2"},  # 이미 채워짐, 건드리지 않음
        ],
    )

    pipeline.cmd_auto_judge(None)

    rows = list(csv.DictReader(sheet_path.open(encoding="utf-8")))
    assert rows[0]["score"] != ""  # 비어 있던 것만 채워짐
    assert rows[1]["score"] == "2"  # 기존 값 보존

    report = json.loads((tmp_path / "auto_judge_report.json").read_text(encoding="utf-8"))
    assert report["judged_count"] == 1  # 이미 채워진 행은 다시 호출하지 않음
    assert report["estimated_cost_usd"] > 0


def test_write_judge_sheet_rows_preserves_existing_scores(tmp_path) -> None:
    """리뷰 P1 회귀 테스트: spot-check 재실행도 이미 채운 판정을 잃으면 안 된다."""
    path = tmp_path / "spot_check.csv"
    _write_sheet(
        path,
        [
            {"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "1"},
            {"pair_id": "q1::b", "query_id": "q1", "candidate_id": "b", "score": ""},
        ],
    )

    pipeline.write_judge_sheet_rows(
        [
            {"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": ""},
            {"pair_id": "q1::b", "query_id": "q1", "candidate_id": "b", "score": ""},
            {"pair_id": "q1::c", "query_id": "q1", "candidate_id": "c", "score": ""},
        ],
        path,
    )

    result = {r["pair_id"]: r["score"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert result["q1::a"] == "1"  # 기존 판정 보존
    assert result["q1::b"] == ""
    assert result["q1::c"] == ""


def test_cmd_spot_check_samples_when_pool_exceeds_sample_size(tmp_path, monkeypatch) -> None:
    """리뷰 P1 회귀 테스트: 141쌍처럼 표본 크기를 넘으면 SPOT_CHECK_SAMPLE_SIZE로 무작위로 잘라낸다."""
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "SPOT_CHECK_SAMPLE_SIZE", 5)

    fake_queries = [
        Creator(
            id=f"q{i}", name=f"q{i}", bio="", events=(), subtopic="", gold=(), declared=(),
            written_by="test", note="", split="test", is_query=True,
        )
        for i in range(3)
    ]
    monkeypatch.setattr(pipeline, "load_creators", lambda: fake_queries)

    candidates: dict[str, dict[str, list[list]]] = {
        pipeline.SPOT_CHECK_TARGET: {},
        pipeline.SPOT_CHECK_BASELINES[0]: {},
        pipeline.SPOT_CHECK_BASELINES[1]: {},
    }
    for i, q in enumerate(fake_queries):
        # target과 baseline이 완전히 다른 후보를 골라, 쿼리당 여러 불일치 쌍이 나오게 한다.
        candidates[pipeline.SPOT_CHECK_TARGET][q.id] = [[f"a{i}{j}", 1.0] for j in range(5)]
        for baseline in pipeline.SPOT_CHECK_BASELINES:
            candidates[baseline][q.id] = [[f"b{i}{j}", 1.0] for j in range(5)]
    (tmp_path / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")

    pipeline.cmd_spot_check(None)

    rows = list(csv.DictReader((tmp_path / "spot_check.csv").open(encoding="utf-8")))
    assert len(rows) == 5  # 표본 크기로 잘림 (전체는 3쿼리 * 10쌍 = 30쌍)

    # 시드가 고정돼 있으므로 같은 입력이면 같은 표본이 나와야 한다(재현성).
    pipeline.cmd_spot_check(None)
    rows_again = list(csv.DictReader((tmp_path / "spot_check.csv").open(encoding="utf-8")))
    assert {r["pair_id"] for r in rows_again} == {r["pair_id"] for r in rows}


def test_save_and_load_vectors_round_trip(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    ids = ["a", "b", "c"]
    vectors = np.random.rand(3, 4).astype(np.float32)

    pipeline._save_vectors(path, ids, vectors)
    loaded_ids, loaded_vectors = pipeline._load_vectors(path)

    assert loaded_ids == ids
    assert np.allclose(loaded_vectors, vectors)
