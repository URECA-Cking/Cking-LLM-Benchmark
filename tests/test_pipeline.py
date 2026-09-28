"""API를 호출하지 않고, 가짜 클라이언트로 파이프라인 배선(저장·순서·모양)만 검증한다."""

from __future__ import annotations

import json

import numpy as np

import src.pipeline as pipeline
from src.data import load_categories, load_creators


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

    pipeline.cmd_embed(None)

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


def test_save_and_load_vectors_round_trip(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    ids = ["a", "b", "c"]
    vectors = np.random.rand(3, 4).astype(np.float32)

    pipeline._save_vectors(path, ids, vectors)
    loaded_ids, loaded_vectors = pipeline._load_vectors(path)

    assert loaded_ids == ids
    assert np.allclose(loaded_vectors, vectors)
