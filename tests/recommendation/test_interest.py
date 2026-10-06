from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.recommendation.cache import JsonModelCache, text_hash
from src.recommendation.interest import (
    M2_METHOD, M3_METHOD, InterestRecommendationConfig, InterestRecommender,
)
from src.recommendation.manifest import CreatorManifest, build_manifest
from src.recommendation.models import CreatorProfile
from src.recommendation.taxonomy import DEFAULT_CATEGORIES_CSV, load_service_categories


def categories():
    return load_service_categories(DEFAULT_CATEGORIES_CSV)


class FakeEmbedding:
    name = "BAAI/bge-m3"
    dim = 17

    def __init__(self, vectors=None):
        self.calls = []
        self.vectors = dict(zip((row.description for row in categories()), np.eye(17)))
        self.vectors.update(vectors or {})

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts])


def vector(*values):
    return np.array([*values, *([0] * (17 - len(values)))], dtype=float)


def recommender(tmp_path, profiles=None, vectors=None, config=None, client=None):
    manifest = build_manifest(profiles if profiles is not None else [
        CreatorProfile(3, "third"), CreatorProfile(1, "first"), CreatorProfile(2, "second"),
        CreatorProfile(4, "  "),
    ])
    client = client or FakeEmbedding(vectors or {
        "first": vector(.8, .6), "second": vector(.75, .6614378278),
        "third": vector(.75, .6614378278),
    })
    return InterestRecommender(manifest, categories(), client, JsonModelCache(tmp_path / "cache.json"), config)


def test_all_17_independent_queries_default_top20_and_metadata(tmp_path):
    profiles = [CreatorProfile(index, "same") for index in range(25, 0, -1)]
    client = FakeEmbedding({"same": vector(1)})
    service = recommender(tmp_path, profiles, client=client)
    for category in categories():
        result = service.recommend(category.code)
        assert result.method == M3_METHOD
        assert len(result.candidates) == 20
        assert [cid for cid, _ in result.candidates] == list(range(1, 21))
        payload = result.to_backend_payload()
        assert payload["taxonomyVersion"] == "v0.2"
        assert payload["interestCode"] == category.code
        assert [candidate["rank"] for candidate in payload["candidates"]] == list(range(1, 21))
        for candidate in payload["candidates"]:
            for key in ("method", "modelVersion", "inputHash", "taxonomyVersion", "taxonomyHash", "interestCode"):
                assert candidate[key] == payload[key]
    assert len(client.calls) == 1
    assert len(client.calls[0]) == 18  # 동일한 소개는 한 번만 임베딩한다.


def test_bonus_only_for_thresholded_top_tags_and_m2_has_no_bonus(tmp_path):
    service = recommender(tmp_path, config=InterestRecommendationConfig(
        zero_shot_tau=.77, m3_bonus=.1, top_n=20,
    ))
    code = categories()[0].code
    m3, m2 = service.recommend(code), service.recommend(code, M2_METHOD)
    assert m3.candidates == ((1, .9), (2, .75), (3, .75))
    assert m2.candidates == ((1, .8), (2, .75), (3, .75))
    assert m3.input_hash != m2.input_hash
    assert all(cid != 4 for cid, _ in m3.candidates)


def test_zero_shot_tag_cap_and_ties_use_taxonomy_order(tmp_path):
    service = recommender(tmp_path, [CreatorProfile(1, "tie")], {"tie": vector(1, 1, 1, 1)},
                          InterestRecommendationConfig(zero_shot_tau=.5, zero_shot_max_tags=3))
    assert [service.recommend(row.code).candidates[0][1] for row in categories()[:4]] == [.6, .6, .6, .5]


def test_round_before_sort_breaks_score_ties_by_id(tmp_path):
    service = recommender(tmp_path, [CreatorProfile(2, "high"), CreatorProfile(1, "low")],
                          {"high": vector(.800004, .6), "low": vector(.8, .6)},
                          InterestRecommendationConfig(score_decimals=4))
    assert service.recommend(categories()[0].code).candidates == ((1, .9), (2, .9))


def test_m3_can_promote_tagged_candidate_over_m2_neighbor(tmp_path):
    service = recommender(tmp_path, [CreatorProfile(1, "broad"), CreatorProfile(2, "focused")],
                          {"broad": vector(.45, .51, .51, .51), "focused": vector(.44, .9)})
    code = categories()[0].code
    assert service.recommend(code, M2_METHOD).candidates[0][0] == 1
    assert service.recommend(code, M3_METHOD).candidates[0][0] == 2


def test_parameter_changes_reuse_vectors_but_recompute_scores(tmp_path):
    first = recommender(tmp_path)
    code = categories()[0].code
    initial = first.recommend(code)
    second = recommender(tmp_path, config=replace(first.config, m3_bonus=.2))
    changed = second.recommend(code)
    assert second.embedding_client.calls == []
    assert changed.candidates[0][1] == 1.0
    assert changed.input_hash != initial.input_hash


def test_empty_manifest_and_blank_introductions_keep_generation_metadata_without_calls(tmp_path):
    client = FakeEmbedding()
    for profiles in ([], [CreatorProfile(1, " \r\n ")]):
        service = recommender(tmp_path, profiles, client=client)
        payload = service.recommend(categories()[0].code).to_backend_payload()
        assert payload["candidates"] == []
        assert len(payload["inputHash"]) == 64
        assert payload["method"] == M3_METHOD
    assert client.calls == []


def test_cached_rerun_has_identical_payload_and_skips_model(tmp_path):
    first = recommender(tmp_path)
    expected = [first.recommend(row.code, method).to_backend_payload()
                for row in categories() for method in (M2_METHOD, M3_METHOD)]
    client = FakeEmbedding()
    second = recommender(tmp_path, client=client)
    assert [second.recommend(row.code, method).to_backend_payload()
            for row in categories() for method in (M2_METHOD, M3_METHOD)] == expected
    assert client.calls == []


@pytest.mark.parametrize("bad", [None, [0] * 17, [float("nan")] * 17, [float("inf")] * 17, [1, 2]])
def test_corrupt_cached_vector_is_regenerated(tmp_path, bad):
    service = recommender(tmp_path)
    service.cache.put_embedding(text_hash("first"), service.config.embedding_model_version, bad)
    assert service.recommend(categories()[0].code).candidates[0] == (1, .9)
    assert "first" in service.embedding_client.calls[0]


@pytest.mark.parametrize("bad", [vector(0), vector(float("nan")), vector(float("inf")), [1, 2]])
def test_invalid_model_vector_is_rejected_without_caching_batch(tmp_path, bad):
    client = FakeEmbedding({"bad": bad})
    service = recommender(tmp_path, [CreatorProfile(1, "bad")], client=client)
    with pytest.raises(ValueError):
        service.recommend(categories()[0].code)
    assert service.cache.get_embedding(text_hash("bad"), service.config.embedding_model_version) is None


@pytest.mark.parametrize("change", [
    {"top_n": 10}, {"zero_shot_tau": .5}, {"zero_shot_max_tags": 2}, {"m3_bonus": .2},
    {"score_decimals": 6}, {"embedding_model_version": "BAAI/bge-m3@different"},
])
def test_every_generation_setting_changes_input_hash(tmp_path, change):
    baseline = recommender(tmp_path)
    changed = recommender(tmp_path, config=replace(baseline.config, **change))
    assert changed.input_hash(categories()[0].code) != baseline.input_hash(categories()[0].code)


def test_manifest_and_interest_change_hash_but_batch_size_does_not(tmp_path):
    baseline = recommender(tmp_path)
    code = categories()[0].code
    changed = recommender(tmp_path, [CreatorProfile(1, "changed")])
    assert changed.input_hash(code) != baseline.input_hash(code)
    assert baseline.input_hash(code) != baseline.input_hash(categories()[1].code)
    batching = recommender(tmp_path, config=replace(baseline.config, embedding_batch_size=1))
    assert batching.input_hash(code) == baseline.input_hash(code)


@pytest.mark.parametrize("change", [
    {"top_n": True}, {"top_n": 0}, {"top_n": 101}, {"score_decimals": 1.5}, {"score_decimals": -1},
    {"score_decimals": 16}, {"embedding_batch_size": False}, {"zero_shot_max_tags": 0},
    {"zero_shot_max_tags": 18}, {"zero_shot_tau": float("nan")}, {"zero_shot_tau": 1.1},
    {"zero_shot_tau": -.0 - 1.1}, {"zero_shot_tau": True}, {"m3_bonus": -.1},
    {"m3_bonus": float("inf")}, {"m3_bonus": "0.1"}, {"embedding_model_version": ""},
    {"embedding_model_version": "another-model"}, {"taxonomy_version": "v0.1"}, {"taxonomy_hash": "a" * 64},
])
def test_invalid_config_rejected_before_execution(change):
    with pytest.raises(ValueError):
        InterestRecommendationConfig(**change)


def test_wrong_taxonomy_duplicate_creator_and_tampered_manifest_rejected(tmp_path):
    manifest = build_manifest([CreatorProfile(1, "first")])
    rows = categories()
    rows[0] = replace(rows[0], description="changed")
    with pytest.raises(ValueError, match="17개 정본"):
        InterestRecommender(manifest, rows, FakeEmbedding(), JsonModelCache(tmp_path / "c.json"))
    with pytest.raises(ValueError, match="중복"):
        build_manifest([CreatorProfile(1, "first"), CreatorProfile(1, "first")])
    with pytest.raises(ValueError, match="해시"):
        InterestRecommender(CreatorManifest(manifest.creators, "a" * 64), categories(),
                            FakeEmbedding(), JsonModelCache(tmp_path / "c.json"))
    service = recommender(tmp_path)
    with pytest.raises(ValueError, match="interestCode"):
        service.recommend("UNKNOWN")
    with pytest.raises(ValueError, match="방식"):
        service.recommend(categories()[0].code, "M4")

@pytest.mark.parametrize("method", [M2_METHOD, M3_METHOD])
def test_shared_cache_order_and_disk_reload_preserve_exact_payload(tmp_path, method):
    import hashlib
    from src.recommendation.service import RecommendationConfig, SimilarCreatorRecommender

    class DenseEmbedding:
        name = "BAAI/bge-m3"
        dim = 1024

        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            rows = []
            for text in texts:
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
                row = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
                rows.append(row / np.linalg.norm(row))
            return np.asarray(rows)

    profiles = [CreatorProfile(i, f"creator introduction {i}") for i in range(1, 31)]
    manifest = build_manifest(profiles)
    settings = InterestRecommendationConfig(zero_shot_tau=.02)
    similar_settings = RecommendationConfig(embedding_model_version=settings.embedding_model_version, short_introduction_chars=100)

    def payloads(cache, client):
        service = InterestRecommender(manifest, categories(), client, cache, settings)
        return [service.recommend(row.code, method).to_backend_payload() for row in categories()]

    direct = payloads(JsonModelCache(tmp_path / "direct.json"), DenseEmbedding())
    path = tmp_path / "shared.json"
    cache, client = JsonModelCache(path), DenseEmbedding()
    similar = SimilarCreatorRecommender(client, None, cache, similar_settings)
    first_similar = similar.recommend(profiles[0], profiles).to_backend_payload()
    assert payloads(cache, client) == direct
    reloaded, cached_client = JsonModelCache(path), DenseEmbedding()
    assert payloads(reloaded, cached_client) == direct
    assert SimilarCreatorRecommender(cached_client, None, reloaded, similar_settings).recommend(
        profiles[0], profiles,
    ).to_backend_payload() == first_similar
    assert cached_client.calls == []
    # 관심 분야가 먼저 캐시를 채운 반대 순서도 유사 추천 payload를 보존한다.
    reverse_cache, reverse_client = JsonModelCache(tmp_path / "reverse.json"), DenseEmbedding()
    assert payloads(reverse_cache, reverse_client) == direct
    assert SimilarCreatorRecommender(reverse_client, None, reverse_cache, similar_settings).recommend(
        profiles[0], profiles,
    ).to_backend_payload() == first_similar
