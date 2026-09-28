"""sentence-transformers로 로컬 실행하는 임베딩 클라이언트다. 접두어 없이 dense 벡터만 사용한다.

bge-m3, KURE-v1(bge-m3의 한국어 파인튜닝), Qwen3-Embedding 모두 같은 방식(dense 벡터,
query/passage 접두어 불필요)으로 로드·인코딩되어 하나의 클래스를 모델명·차원만 바꿔 공유한다.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.clients.base import normalize_rows
from src.config import LOCAL_EMBEDDING_MODELS


class LocalEmbeddingClient:
    """sentence-transformers 모델을 로컬에서 실행한다."""

    def __init__(self, key: str, model: SentenceTransformer | None = None) -> None:
        spec = LOCAL_EMBEDDING_MODELS[key]
        self.name = key
        self.dim = spec["dim"]
        self._model = model or SentenceTransformer(spec["model_name"])
        self.last_input_tokens: int | None = None  # 로컬 실행이라 과금 토큰 개념이 없다

    def embed(self, texts: list[str]) -> np.ndarray:
        """texts를 정규화된 dense 벡터로 바꾼다."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return normalize_rows(np.asarray(vectors, dtype=np.float32))
