import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import pipeline
from src.api_parity import compare_embeddings
from src.clients import OpenAIEmbeddingClient
from src.clients.base import normalize_rows


def _random_unit(rows: int, dim: int, seed: int) -> np.ndarray:
    return normalize_rows(np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32))


def _compare(local_c, api_c, local_k, api_k):
    return compare_embeddings(local_c, api_c, local_k, api_k, k=5, max_abs_diff_limit=0.01, min_top_k_overlap=0.95)


def test_identical_vectors_are_equivalent() -> None:
    creators, categories = _random_unit(20, 16, 1), _random_unit(4, 16, 2)

    report = _compare(creators, creators.copy(), categories, categories.copy())

    assert report["equivalent"] is True
    assert report["creator_similarity_max_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    assert report["top5_overlap_mean"] == 1.0 and report["top1_agreement"] == 1.0
    assert report["self_cosine_min"] == pytest.approx(1.0, abs=1e-6)


def test_tiny_noise_stays_equivalent_but_large_noise_does_not() -> None:
    creators, categories = _random_unit(30, 32, 3), _random_unit(5, 32, 4)
    rng = np.random.default_rng(5)

    tiny = normalize_rows(creators + rng.normal(scale=1e-4, size=creators.shape).astype(np.float32))
    assert _compare(creators, tiny, categories, categories)["equivalent"] is True

    large = normalize_rows(creators + rng.normal(scale=0.5, size=creators.shape).astype(np.float32))
    report = _compare(creators, large, categories, categories)
    assert report["equivalent"] is False
    assert report["creator_similarity_max_abs_diff"] > 0.01


def test_category_similarity_drift_alone_breaks_equivalence() -> None:
    creators, categories = _random_unit(20, 16, 6), _random_unit(4, 16, 7)
    shifted = normalize_rows(categories + 0.3)  # 크리에이터 벡터는 그대로, 카테고리만 달라짐 → zero-shot 태깅이 흔들린다

    report = _compare(creators, creators.copy(), categories, shifted)

    assert report["top5_overlap_mean"] == 1.0
    assert report["category_similarity_max_abs_diff"] > 0.01
    assert report["equivalent"] is False


def test_client_sends_configured_model_name() -> None:
    calls = []

    def create(model, input):
        calls.append(model)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=7), data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])

    fake = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    client = OpenAIEmbeddingClient(client=fake, model="BAAI/bge-m3", dim=2)

    vectors = client.embed(["a", "b"])

    assert calls == ["BAAI/bge-m3"] and client.name == "BAAI/bge-m3" and client.last_input_tokens == 7
    assert np.allclose(vectors, [[0.6, 0.8], [0.6, 0.8]])  # 정규화됨


def _write_local_cache(cache_dir, creators, categories, creator_vectors, category_vectors) -> None:
    pipeline._save_vectors(cache_dir / "creators_bge-m3.npz", [c.id for c in creators], creator_vectors)
    pipeline._save_vectors(cache_dir / "categories_bge-m3.npz", [c.code for c in categories], category_vectors)
    input_hash = pipeline._content_hash(
        pipeline._embedding_model_identity("bge-m3"),
        *[c.input_text() for c in creators],
        *[c.description for c in categories],
    )
    (cache_dir / "embed_bge-m3.input_hash").write_text(input_hash, encoding="utf-8")


class _FakeApiClient:
    """로컬 캐시와 같은 벡터를 돌려주는 API 대역이다. 호출 순서대로 크리에이터 → 카테고리 벡터를 준다."""

    name = "BAAI/bge-m3"

    def __init__(self, *outputs: np.ndarray) -> None:
        self._outputs = list(outputs)
        self.last_input_tokens = None

    def embed(self, texts):
        self.last_input_tokens = 100
        return self._outputs.pop(0)


def test_cmd_api_parity_uses_cache_and_writes_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    creators, categories = pipeline.load_creators(), pipeline.load_categories()
    creator_vectors, category_vectors = _random_unit(len(creators), 8, 8), _random_unit(len(categories), 8, 9)
    _write_local_cache(tmp_path, creators, categories, creator_vectors, category_vectors)

    pipeline.cmd_api_parity(SimpleNamespace(), api_client=_FakeApiClient(creator_vectors, category_vectors))

    saved = json.loads((tmp_path / "api_parity.json").read_text(encoding="utf-8"))
    assert saved["equivalent"] is True and saved["creator_count"] == len(creators)
    assert saved["input_tokens"] == 200 and saved["cost_usd"] == pytest.approx(200 / 1_000_000 * 0.01)


def test_cmd_api_parity_stops_when_local_cache_is_stale(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    creators, categories = pipeline.load_creators(), pipeline.load_categories()
    _write_local_cache(tmp_path, creators, categories, _random_unit(len(creators), 8, 10), _random_unit(len(categories), 8, 11))
    (tmp_path / "embed_bge-m3.input_hash").write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="embed"):
        pipeline.cmd_api_parity(SimpleNamespace(), api_client=_FakeApiClient())

    assert not (tmp_path / "api_parity.json").exists()  # API를 부르기 전에 멈춘다


def test_api_client_requires_deepinfra_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPINFRA_API_KEY"):
        pipeline._api_bge_m3_client()
