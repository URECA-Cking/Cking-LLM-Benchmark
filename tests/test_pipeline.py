"""API를 호출하지 않고, 가짜 클라이언트로 파이프라인 배선(저장·순서·모양)만 검증한다."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

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


def test_cmd_embed_does_not_trust_a_half_updated_cache_after_failed_force(tmp_path, monkeypatch) -> None:
    """--force 중 두 번째 벡터 파일 저장이 실패해도, 섞인 벡터(새 크리에이터 + 이전 카테고리)를 캐시로 쓰면 안 된다."""
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "EMBEDDING_MODEL_KEYS", ("bge-m3",))
    dim = {"value": 8}
    call_count = {"n": 0}

    def client_factory(key: str) -> _FakeEmbeddingClient:
        call_count["n"] += 1
        return _FakeEmbeddingClient(dim["value"])

    monkeypatch.setattr(pipeline, "_embedding_client", client_factory)
    pipeline.cmd_embed(argparse.Namespace(force=False))

    real_save, saved = pipeline._save_vectors, []

    def save_then_fail_on_second(path, ids, vectors):
        saved.append(path.name)
        if len(saved) == 2:  # 크리에이터 파일은 새 벡터로 바뀌었고 카테고리 파일 저장에서 실패
            raise OSError("disk full")
        real_save(path, ids, vectors)

    dim["value"] = 4  # 다시 계산하면 이전과 다른 벡터가 나오도록
    monkeypatch.setattr(pipeline, "_save_vectors", save_then_fail_on_second)
    try:
        pipeline.cmd_embed(argparse.Namespace(force=True))
    except OSError:
        pass
    else:
        raise AssertionError("두 번째 저장 실패가 전파되어야 한다")
    monkeypatch.setattr(pipeline, "_save_vectors", real_save)

    calls_before_retry = call_count["n"]
    pipeline.cmd_embed(argparse.Namespace(force=False))

    assert call_count["n"] == calls_before_retry + 1  # 섞인 캐시를 재사용하지 않고 다시 계산함
    _, creator_vectors = pipeline._load_vectors(tmp_path / "creators_bge-m3.npz")
    _, category_vectors = pipeline._load_vectors(tmp_path / "categories_bge-m3.npz")
    assert creator_vectors.shape[1] == category_vectors.shape[1] == 4


def test_cmd_tag_llm_does_not_trust_a_run_whose_token_usage_was_not_saved_after_failed_force(tmp_path, monkeypatch) -> None:
    """--force 중 run 파일은 새로 쓰였는데 토큰 사용량 저장이 실패하면, 이전 사용량과 섞인 run을 캐시로 쓰면 안 된다."""
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "OPENAI_LLM_MODEL_CANDIDATES", {"fake-model": {}})
    usage = {"tokens": 2}
    call_count = {"n": 0}

    class _CountingTagger:
        def __init__(self, model: str, category_codes: list[str]) -> None:
            pass

        def tag(self, input_text: str) -> TagResult:
            call_count["n"] += 1
            return TagResult(tags=(), input_tokens=usage["tokens"], output_tokens=0)

    monkeypatch.setattr(pipeline, "OpenAITagger", _CountingTagger)
    pipeline.cmd_tag_llm(argparse.Namespace(force=False))

    real_write_text = Path.write_text

    def fail_on_usage(self, *args, **kwargs):
        if self.name.endswith(".token_usage.json"):
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    usage["tokens"] = 9
    monkeypatch.setattr(Path, "write_text", fail_on_usage)
    try:
        pipeline.cmd_tag_llm(argparse.Namespace(force=True))
    except OSError:
        pass
    else:
        raise AssertionError("사용량 저장 실패가 전파되어야 한다")
    monkeypatch.setattr(Path, "write_text", real_write_text)

    calls_before_retry = call_count["n"]
    pipeline.cmd_tag_llm(argparse.Namespace(force=False))

    assert call_count["n"] == calls_before_retry + 100  # 실패한 run0만 다시 태깅함 (run1은 이전 캐시)
    run_usage = json.loads((tmp_path / "llm_tags_fake-model_run0.token_usage.json").read_text(encoding="utf-8"))
    assert run_usage["input_tokens"] == 100 * 9  # 새 run과 맞는 사용량


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


def test_cmd_embed_recomputes_when_model_identity_changes(tmp_path, monkeypatch) -> None:
    """PR #5 재리뷰 회귀 테스트: 텍스트가 그대로여도 키가 가리키는 실제 모델·차원이 바뀌면 다시 계산해야 한다.

    캐시 파일명은 "creators_bge-m3.npz"처럼 키 이름뿐이라, config.py에서 그 키가 가리키는
    실제 HF 모델을 바꿔도 텍스트가 안 바뀌었으면 예전 벡터를 계속 쓸 수 있었다.
    """
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    call_count = {"n": 0}

    def counting_client(key: str) -> _FakeEmbeddingClient:
        call_count["n"] += 1
        return _FakeEmbeddingClient()

    monkeypatch.setattr(pipeline, "_embedding_client", counting_client)
    monkeypatch.setattr(pipeline, "_embedding_model_identity", lambda key: "model-v1")

    pipeline.cmd_embed(argparse.Namespace(force=False))
    first_call_count = call_count["n"]

    monkeypatch.setattr(pipeline, "_embedding_model_identity", lambda key: "model-v2")  # 같은 키, 다른 실제 모델
    pipeline.cmd_embed(argparse.Namespace(force=False))

    assert call_count["n"] > first_call_count  # 모델 정체가 바뀌어 캐시를 못 쓰고 다시 계산함


def test_cmd_tag_llm_recomputes_when_system_prompt_changes(tmp_path, monkeypatch) -> None:
    """PR #5 재리뷰 회귀 테스트: LLM_TAG_MAX 등 요청 설정이 바뀌어도(=SYSTEM_PROMPT 변경) 다시 태깅해야 한다."""
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
    monkeypatch.setattr("src.clients.openai_tagger.SYSTEM_PROMPT", "프롬프트 v1")

    pipeline.cmd_tag_llm(argparse.Namespace(force=False))
    first_calls = call_count["n"]

    monkeypatch.setattr("src.clients.openai_tagger.SYSTEM_PROMPT", "프롬프트 v2 (예: LLM_TAG_MAX 변경)")
    pipeline.cmd_tag_llm(argparse.Namespace(force=False))

    assert call_count["n"] == first_calls * 2  # 프롬프트가 바뀌어 캐시를 못 씀


def test_cmd_tag_llm_preserves_token_usage_when_fully_cached(tmp_path, monkeypatch) -> None:
    """PR #5 재리뷰 회귀 테스트: 캐시 재사용만 있었던 실행은 누적 토큰 사용량을 0으로 덮으면 안 된다."""
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "OPENAI_LLM_MODEL_CANDIDATES", {"fake-model": {}})

    class _CountingTagger:
        def __init__(self, model: str, category_codes: list[str]) -> None:
            pass

        def tag(self, input_text: str) -> TagResult:
            return TagResult(tags=(), input_tokens=5, output_tokens=2)

    monkeypatch.setattr(pipeline, "OpenAITagger", _CountingTagger)

    pipeline.cmd_tag_llm(argparse.Namespace(force=False))
    first_usage = json.loads((tmp_path / "llm_model_selection.json").read_text(encoding="utf-8"))["token_usage"]
    assert first_usage["fake-model"]["input_tokens"] > 0

    pipeline.cmd_tag_llm(argparse.Namespace(force=False))  # 이번엔 전부 캐시 히트, 새 API 호출 없음
    second_usage = json.loads((tmp_path / "llm_model_selection.json").read_text(encoding="utf-8"))["token_usage"]

    assert second_usage == first_usage  # 0으로 덮이지 않고 이전 누적치를 유지


def test_cmd_tag_llm_sums_token_usage_when_only_some_runs_recomputed(tmp_path, monkeypatch) -> None:
    """PR #5 재리뷰 회귀 테스트: run 일부만 다시 계산해도 전체(양쪽 run 합) 토큰 사용량을 보고해야 한다.

    "새로 쓴 토큰만 기록"하는 이전 방식은 run 하나만 캐시에서 빠져 다시 계산되면, 캐시로
    읽은 다른 run의 과거 사용량이 통째로 빠져 기록이 실제보다 줄어들었다.
    """
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "OPENAI_LLM_MODEL_CANDIDATES", {"fake-model": {}})

    class _CountingTagger:
        def __init__(self, model: str, category_codes: list[str]) -> None:
            pass

        def tag(self, input_text: str) -> TagResult:
            return TagResult(tags=(), input_tokens=2, output_tokens=1)

    monkeypatch.setattr(pipeline, "OpenAITagger", _CountingTagger)

    pipeline.cmd_tag_llm(argparse.Namespace(force=False))  # run0·run1 둘 다 새로 계산
    full_usage = json.loads((tmp_path / "llm_model_selection.json").read_text(encoding="utf-8"))["token_usage"]["fake-model"]

    (tmp_path / "llm_tags_fake-model_run1.json").unlink()  # run1만 캐시에서 지워 다시 계산되게 함
    (tmp_path / "llm_tags_fake-model_run1.input_hash").unlink()
    pipeline.cmd_tag_llm(argparse.Namespace(force=False))
    partial_recompute_usage = json.loads((tmp_path / "llm_model_selection.json").read_text(encoding="utf-8"))["token_usage"]["fake-model"]

    assert partial_recompute_usage == full_usage  # run0(캐시)+run1(재계산) 합이 원래 전체와 같아야 함


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
        writer = csv.DictWriter(f, fieldnames=["pair_id", "query_id", "candidate_id", "score", "text_hash"])
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
    """리뷰 P1 회귀 테스트: spot-check 재실행도 이미 채운 판정을 잃으면 안 된다(텍스트가 안 바뀌었다면)."""
    path = tmp_path / "spot_check.csv"
    _write_sheet(
        path,
        [
            {"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "1", "text_hash": "hash-a"},
            {"pair_id": "q1::b", "query_id": "q1", "candidate_id": "b", "score": "", "text_hash": "hash-b"},
        ],
    )

    pipeline.write_judge_sheet_rows(
        [
            {"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "", "text_hash": "hash-a"},
            {"pair_id": "q1::b", "query_id": "q1", "candidate_id": "b", "score": "", "text_hash": "hash-b"},
            {"pair_id": "q1::c", "query_id": "q1", "candidate_id": "c", "score": "", "text_hash": "hash-c"},
        ],
        path,
    )

    result = {r["pair_id"]: r["score"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert result["q1::a"] == "1"  # 기존 판정 보존
    assert result["q1::b"] == ""
    assert result["q1::c"] == ""


def test_write_judge_sheet_rows_clears_score_when_text_hash_changes(tmp_path) -> None:
    """PR #5 재리뷰 회귀 테스트: pair_id가 같아도 text_hash가 다르면 예전 점수를 버려야 한다.

    예: X01의 bio가 바뀌어도 id는 그대로라, pair_id만 보면 예전 텍스트로 매긴 점수가
    새 텍스트에도 그대로 쓰인 것처럼 남을 수 있었다.
    """
    path = tmp_path / "spot_check.csv"
    _write_sheet(path, [{"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "2", "text_hash": "hash-old"}])

    pipeline.write_judge_sheet_rows(
        [{"pair_id": "q1::a", "query_id": "q1", "candidate_id": "a", "score": "", "text_hash": "hash-new"}],
        path,
    )

    result = {r["pair_id"]: r["score"] for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert result["q1::a"] == ""  # 텍스트가 바뀌어 재판정 대상이 됨


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

    def _fake_candidate(cid: str) -> Creator:
        return Creator(
            id=cid, name=cid, bio=f"bio of {cid}", events=(), subtopic="", gold=(), declared=(),
            written_by="test", note="", split="test", is_query=False,
        )

    candidates: dict[str, dict[str, list[list]]] = {
        pipeline.SPOT_CHECK_TARGET: {},
        pipeline.SPOT_CHECK_BASELINES[0]: {},
        pipeline.SPOT_CHECK_BASELINES[1]: {},
    }
    candidate_ids: set[str] = set()
    for i, q in enumerate(fake_queries):
        # target과 baseline이 완전히 다른 후보를 골라, 쿼리당 여러 불일치 쌍이 나오게 한다.
        a_ids = [f"a{i}{j}" for j in range(5)]
        b_ids = [f"b{i}{j}" for j in range(5)]
        candidates[pipeline.SPOT_CHECK_TARGET][q.id] = [[cid, 1.0] for cid in a_ids]
        for baseline in pipeline.SPOT_CHECK_BASELINES:
            candidates[baseline][q.id] = [[cid, 1.0] for cid in b_ids]
        candidate_ids.update(a_ids + b_ids)
    (tmp_path / "candidates.json").write_text(json.dumps(candidates), encoding="utf-8")

    all_fake_creators = fake_queries + [_fake_candidate(cid) for cid in candidate_ids]
    monkeypatch.setattr(pipeline, "load_creators", lambda: all_fake_creators)

    pipeline.cmd_spot_check(None)

    rows = list(csv.DictReader((tmp_path / "spot_check.csv").open(encoding="utf-8")))
    assert len(rows) == 5  # 표본 크기로 잘림 (전체는 3쿼리 * 10쌍 = 30쌍)

    # 시드가 고정돼 있으므로 같은 입력이면 같은 표본이 나와야 한다(재현성).
    pipeline.cmd_spot_check(None)
    rows_again = list(csv.DictReader((tmp_path / "spot_check.csv").open(encoding="utf-8")))
    assert {r["pair_id"] for r in rows_again} == {r["pair_id"] for r in rows}


def _minimal_creator(cid: str, is_query: bool) -> Creator:
    return Creator(
        id=cid, name=cid, bio=f"bio of {cid}", events=(), subtopic="", gold=(), declared=(),
        written_by="test", note="", split="test", is_query=is_query,
    )


def test_query_prompted_vectors_substitutes_only_query_rows(monkeypatch) -> None:
    """이슈 #8 회귀 테스트: query_prompt_name이 등록된 모델은 평가 쿼리 행만 바꿔치기해야 한다.

    Qwen3-Embedding처럼 모델에 쿼리용 instruct 프롬프트가 있는데 이를 안 쓰면(기존 버그),
    공식 권장 사용법과 다른 조건으로 측정하게 된다(리뷰로 발견).
    """
    ids = ["q1", "c1", "q2"]
    creators = [_minimal_creator("q1", True), _minimal_creator("c1", False), _minimal_creator("q2", True)]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    class _FakeQueryClient:
        def __init__(self, key: str) -> None:
            pass

        def embed(self, texts: list[str], prompt_name: str | None = None) -> np.ndarray:
            assert prompt_name == "fake-query-prompt"
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(pipeline, "LocalEmbeddingClient", _FakeQueryClient)
    monkeypatch.setattr(pipeline, "LOCAL_EMBEDDING_MODELS", {"fake-key": {"query_prompt_name": "fake-query-prompt"}})

    result = pipeline._query_prompted_vectors("fake-key", ids, vectors, creators)

    assert np.allclose(result[0], [0.0, 0.0])  # q1: 쿼리 프롬프트로 바뀜
    assert np.allclose(result[1], [0.0, 1.0])  # c1: 후보 전용, 원래 임베딩 그대로
    assert np.allclose(result[2], [0.0, 0.0])  # q2: 쿼리 프롬프트로 바뀜


def test_query_prompted_vectors_noop_when_model_has_no_query_prompt(monkeypatch) -> None:
    """query_prompt_name이 없는 모델(bge-m3, KURE-v1)은 원본 벡터를 그대로 반환해야 한다."""
    ids = ["q1"]
    creators = [_minimal_creator("q1", True)]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(pipeline, "LOCAL_EMBEDDING_MODELS", {"fake-key": {}})

    result = pipeline._query_prompted_vectors("fake-key", ids, vectors, creators)

    assert result is vectors


class _FakeReranker:
    """(쿼리, 후보) 텍스트를 그대로 점수로 쓴다 — 후보 텍스트 끝 숫자가 클수록 높은 점수."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(candidate_text[-1]) for _, candidate_text in pairs]


def test_rerank_bge_m2_candidates_reorders_pool_by_reranker_score() -> None:
    ids = ["q", "a1", "a2", "a3"]
    # 코사인 기준으로는 a1 > a2 > a3 순이지만, 가짜 리랭커는 텍스트 끝자리로 점수를 매겨 뒤집는다.
    # 대각선은 -inf로 둬 top_n이 자기 자신을 제외하게 한다(실제 cosine_matrix와 동일한 관례).
    cosine = np.array(
        [
            [-np.inf, 0.9, 0.8, 0.7],
            [0.9, -np.inf, 0.5, 0.5],
            [0.8, 0.5, -np.inf, 0.5],
            [0.7, 0.5, 0.5, -np.inf],
        ]
    )
    text_by_id = {"q": "query text 0", "a1": "candidate text 1", "a2": "candidate text 2", "a3": "candidate text 9"}

    result = pipeline._rerank_bge_m2_candidates(ids, cosine, text_by_id, _FakeReranker(), pool_size=3)

    ranked_ids = [cid for cid, _ in result["q"]]
    assert ranked_ids == ["a3", "a2", "a1"]  # 코사인 순위가 아니라 리랭커 점수 순
    assert result["q"][0][1] == 9.0


def test_save_and_load_vectors_round_trip(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    ids = ["a", "b", "c"]
    vectors = np.random.rand(3, 4).astype(np.float32)

    pipeline._save_vectors(path, ids, vectors)
    loaded_ids, loaded_vectors = pipeline._load_vectors(path)

    assert loaded_ids == ids
    assert np.allclose(loaded_vectors, vectors)


def test_query_prompted_vectors_target_ids_overrides_default_query_rows(monkeypatch) -> None:
    """select-params는 dev 행에 프롬프트를 적용하므로 target_ids로 대상 행을 바꿀 수 있어야 한다."""
    ids = ["q1", "c1"]
    creators = [_minimal_creator("q1", True), _minimal_creator("c1", False)]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakeClient:
        def __init__(self, _key): ...
        def embed(self, texts, prompt_name=None):
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(pipeline, "LocalEmbeddingClient", FakeClient)
    monkeypatch.setattr(pipeline, "LOCAL_EMBEDDING_MODELS", {"fake-key": {"query_prompt_name": "p"}})

    result = pipeline._query_prompted_vectors("fake-key", ids, vectors, creators, target_ids={"c1"})

    assert np.allclose(result[0], [1.0, 0.0]) and np.allclose(result[1], [0.0, 0.0])


def test_model_spot_check_args_default_and_override(tmp_path, monkeypatch) -> None:
    import argparse

    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)

    default = pipeline._model_spot_check_args(argparse.Namespace())
    custom = pipeline._model_spot_check_args(argparse.Namespace(target="A", baseline=["B", "C"], out="spot_check_x.csv"))

    assert default == (pipeline.MODEL_SPOT_CHECK_TARGET, pipeline.MODEL_SPOT_CHECK_BASELINES, tmp_path / "spot_check_models.csv")
    assert custom == ("A", ["B", "C"], tmp_path / "spot_check_x.csv")


def test_model_spot_check_args_rejects_path_outside_results(tmp_path, monkeypatch) -> None:
    import argparse

    import pytest

    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)

    for bad in ("../README.md", "sub/x.csv", "/tmp/x.csv", "notes.txt", "judge_sheet.csv", "spot_check.csv", "candidates.csv", "spot_check_x.txt"):
        with pytest.raises(ValueError):
            pipeline._model_spot_check_args(argparse.Namespace(out=bad))


def test_cmd_spot_check_models_prints_compare_flags_for_custom_comparison(tmp_path, monkeypatch, capsys) -> None:
    """사용자 지정 비교의 안내 명령에는 --target/--baseline/--out이 모두 남아야 한다."""
    import argparse

    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_build_spot_check_rows", lambda *a, **k: (0, 0))

    pipeline.cmd_spot_check_models(argparse.Namespace(target="A", baseline=["B"], out="spot_check_a_b.csv"))

    assert "--target A --baseline B --out spot_check_a_b.csv" in capsys.readouterr().out


def test_model_spot_check_args_requires_separate_file_for_non_default_comparison(tmp_path, monkeypatch) -> None:
    """다른 비교를 기본 파일(spot_check_models.csv)에 쓰면 기존 사람 판정이 사라지므로 별도 --out이 필요하다."""
    import argparse

    import pytest

    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)

    with pytest.raises(ValueError):
        pipeline._model_spot_check_args(argparse.Namespace(target="A", baseline=["B"]))
    with pytest.raises(ValueError):
        pipeline._model_spot_check_args(argparse.Namespace(target="A", baseline=["B"], out="spot_check_models.csv"))
    with pytest.raises(ValueError):
        pipeline._model_spot_check_args(argparse.Namespace(baseline=["B"]))


def test_method_cutoff_maps_method_id_including_m5() -> None:
    params = {"cutoff_r1": 0.5, "per_embedding": {"bge-m3": {"cutoff_m2": 0.51, "cutoff_r2": 0.6, "cutoff_m4": 0.7, "cutoff_m5": 0.2}}}

    assert pipeline._method_cutoff("R1", params) == 0.5
    assert pipeline._method_cutoff("M2_bge-m3", params) == 0.51
    assert pipeline._method_cutoff("R2_bge-m3", params) == 0.6
    assert pipeline._method_cutoff("M5_bge-m3", params) == 0.2
