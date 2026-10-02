from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.clients.openai_tagger import TagResult
from src.recommendation.cache import JsonModelCache
from src.recommendation.models import CreatorProfile
from src.recommendation.service import RecommendationConfig, SimilarCreatorRecommender


class FakeEmbeddingClient:
    name = "fake"
    dim = 2

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class FakeTaggingClient:
    def __init__(self, tags: dict[str, tuple[str, ...]], failure: Exception | None = None) -> None:
        self.tags = tags
        self.failure = failure
        self.calls: list[str] = []

    def tag(self, input_text: str) -> TagResult:
        self.calls.append(input_text)
        if self.failure is not None:
            raise self.failure
        return TagResult(self.tags[input_text], 0, 0)


def config(**changes) -> RecommendationConfig:
    base = RecommendationConfig(
        short_introduction_chars=5,
        top_n=3,
        m4_bonus=0.2,
        embedding_model_version="embed-v1",
        tag_model_version="tag-v1",
        tag_prompt_version="prompt-v1",
        tag_prompt="분류 프롬프트 v1",
        taxonomy_version="taxonomy-v1",
        allowed_tags=frozenset({"FITNESS", "FOOD"}),
    )
    return replace(base, **changes)


def build_service(tmp_path, embedder, tagger, settings=None) -> SimilarCreatorRecommender:
    return SimilarCreatorRecommender(
        embedder,
        tagger,
        JsonModelCache(tmp_path / "model-cache.json"),
        settings or config(),
    )


def test_regular_introduction_uses_m4_and_stable_creator_id_tie_break(tmp_path) -> None:
    seed = CreatorProfile(10, "홈트 운동")
    candidates = [
        CreatorProfile(30, "근력 운동"),
        CreatorProfile(20, "요리 채널"),
        CreatorProfile(20, "요리 채널"),
        CreatorProfile(10, "무시할 자기 자신"),
    ]
    embedder = FakeEmbeddingClient({"홈트 운동": [1, 0], "근력 운동": [1, 0], "요리 채널": [1, 0]})
    tagger = FakeTaggingClient({"홈트 운동": ("FITNESS",), "근력 운동": ("FITNESS",), "요리 채널": ("FOOD",)})
    service = build_service(tmp_path, embedder, tagger)

    result = service.recommend(seed, candidates)

    assert [item.similar_creator_id for item in result.candidates] == [30, 20]
    assert [item.score for item in result.candidates] == [1.2, 1.0]
    assert [item.rank for item in result.candidates] == [1, 2]
    assert {item.method for item in result.candidates} == {"M4"}
    assert all(item.model_version == "embed-v1+tag-v1@prompt-v1" for item in result.candidates)
    assert len({item.input_hash for item in result.candidates}) == 1
    assert result.to_backend_payload()["candidates"][0] == result.candidates[0].to_dict()


def test_same_score_is_sorted_by_creator_id(tmp_path) -> None:
    seed = CreatorProfile(10, "긴 소개글")
    candidates = [CreatorProfile(30, "후보 삼십"), CreatorProfile(20, "후보 이십")]
    embedder = FakeEmbeddingClient({profile.text: [1, 0] for profile in [seed, *candidates]})
    tagger = FakeTaggingClient({profile.text: ("FITNESS",) for profile in [seed, *candidates]})

    result = build_service(tmp_path, embedder, tagger).recommend(seed, candidates)

    assert [item.similar_creator_id for item in result.candidates] == [20, 30]


def test_short_introduction_uses_m2_without_tagger(tmp_path) -> None:
    seed = CreatorProfile(10, "짧음")
    candidate = CreatorProfile(20, "충분히 긴 후보 소개")
    embedder = FakeEmbeddingClient({seed.text: [1, 0], candidate.text: [1, 0]})
    tagger = FakeTaggingClient({})

    result = build_service(tmp_path, embedder, tagger).recommend(seed, [candidate])

    assert result.candidates[0].method == "M2"
    assert result.candidates[0].model_version == "embed-v1"
    assert tagger.calls == []


def test_unclassified_regular_seed_falls_back_to_m2(tmp_path) -> None:
    seed = CreatorProfile(10, "분류 근거 없는 긴 소개")
    candidate = CreatorProfile(20, "충분히 긴 후보 소개")
    embedder = FakeEmbeddingClient({seed.text: [1, 0], candidate.text: [1, 0]})
    tagger = FakeTaggingClient({seed.text: ()})

    result = build_service(tmp_path, embedder, tagger).recommend(seed, [candidate])

    assert result.candidates[0].method == "M2"
    assert tagger.calls == [seed.text]


def test_empty_seed_and_empty_candidates_do_not_call_models(tmp_path) -> None:
    embedder = FakeEmbeddingClient({})
    tagger = FakeTaggingClient({})
    service = build_service(tmp_path, embedder, tagger)

    empty_seed = service.recommend(CreatorProfile(10, "  "), [CreatorProfile(20, "후보 소개")])
    no_valid_candidate = service.recommend(CreatorProfile(10, "충분히 긴 소개"), [CreatorProfile(20, "  ")])

    assert empty_seed.candidates == ()
    assert no_valid_candidate.candidates == ()
    assert embedder.calls == []
    assert tagger.calls == []


def test_cache_reuses_calls_and_invalidates_model_and_prompt(tmp_path) -> None:
    seed = CreatorProfile(10, "홈트 운동 소개")
    candidate = CreatorProfile(20, "근력 운동 소개")
    vectors = {seed.text: [1, 0], candidate.text: [1, 0]}
    tags = {seed.text: ("FITNESS",), candidate.text: ("FITNESS",)}

    first_embedder = FakeEmbeddingClient(vectors)
    first_tagger = FakeTaggingClient(tags)
    first_result = build_service(tmp_path, first_embedder, first_tagger).recommend(seed, [candidate])
    assert len(first_embedder.calls) == 1 and len(first_tagger.calls) == 2

    reused_embedder = FakeEmbeddingClient(vectors)
    reused_tagger = FakeTaggingClient(tags)
    build_service(tmp_path, reused_embedder, reused_tagger).recommend(seed, [candidate])
    assert reused_embedder.calls == [] and reused_tagger.calls == []

    prompt_tagger = FakeTaggingClient(tags)
    prompt_result = build_service(
        tmp_path,
        FakeEmbeddingClient(vectors),
        prompt_tagger,
        config(tag_prompt="분류 프롬프트 v2"),
    ).recommend(seed, [candidate])
    assert len(prompt_tagger.calls) == 2
    assert prompt_result.candidates[0].input_hash != first_result.candidates[0].input_hash

    model_embedder = FakeEmbeddingClient(vectors)
    build_service(tmp_path, model_embedder, FakeTaggingClient(tags), config(embedding_model_version="embed-v2")).recommend(seed, [candidate])
    assert len(model_embedder.calls) == 1


def test_input_change_invalidates_only_changed_model_results(tmp_path) -> None:
    seed = CreatorProfile(10, "홈트 운동 소개")
    old_candidate = CreatorProfile(20, "근력 운동 소개")
    new_candidate = CreatorProfile(20, "필라테스 운동 소개")
    vectors = {seed.text: [1, 0], old_candidate.text: [1, 0], new_candidate.text: [0.9, 0.1]}
    tags = {seed.text: ("FITNESS",), old_candidate.text: ("FITNESS",), new_candidate.text: ("FITNESS",)}
    first = build_service(tmp_path, FakeEmbeddingClient(vectors), FakeTaggingClient(tags))
    first.recommend(seed, [old_candidate])

    embedder = FakeEmbeddingClient(vectors)
    tagger = FakeTaggingClient(tags)
    build_service(tmp_path, embedder, tagger).recommend(seed, [new_candidate])

    assert embedder.calls == [[new_candidate.text]]
    assert tagger.calls == [new_candidate.text]


def test_embeddings_are_requested_in_configured_batches(tmp_path) -> None:
    seed = CreatorProfile(10, "짧음")
    candidates = [CreatorProfile(creator_id, f"후보 {creator_id}") for creator_id in range(20, 25)]
    vectors = {profile.text: [1, 0] for profile in [seed, *candidates]}
    embedder = FakeEmbeddingClient(vectors)

    build_service(
        tmp_path,
        embedder,
        FakeTaggingClient({}),
        config(embedding_batch_size=2),
    ).recommend(seed, candidates)

    assert [len(batch) for batch in embedder.calls] == [2, 2, 2]


def test_tagger_failure_aborts_without_partial_result(tmp_path) -> None:
    seed = CreatorProfile(10, "홈트 운동 소개")
    candidate = CreatorProfile(20, "근력 운동 소개")
    service = build_service(
        tmp_path,
        FakeEmbeddingClient({seed.text: [1, 0], candidate.text: [1, 0]}),
        FakeTaggingClient({}, failure=RuntimeError("API unavailable")),
    )

    with pytest.raises(RuntimeError, match="API unavailable"):
        service.recommend(seed, [candidate])


def test_conflicting_duplicate_creator_is_rejected(tmp_path) -> None:
    service = build_service(tmp_path, FakeEmbeddingClient({}), FakeTaggingClient({}))

    with pytest.raises(ValueError, match="같은 creatorId"):
        service.recommend(
            CreatorProfile(10, "충분히 긴 소개"),
            [CreatorProfile(20, "첫 소개"), CreatorProfile(20, "다른 소개")],
        )
