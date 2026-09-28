"""로컬 bge-m3 임베딩 클라이언트다. 접두어 없이 dense 벡터만 사용한다."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.clients.base import normalize_rows
from src.config import BGE_EMBEDDING_DIM, BGE_MODEL_NAME


class BgeEmbeddingClient:
    """sentence-transformers로 bge-m3를 로컬에서 실행한다."""

    name = "bge-m3"
    dim = BGE_EMBEDDING_DIM

    def __init__(self, model: SentenceTransformer | None = None) -> None:
        self._model = model or SentenceTransformer(BGE_MODEL_NAME)
        self.last_input_tokens: int | None = None  # 로컬 실행이라 과금 토큰 개념이 없다

    def embed(self, texts: list[str]) -> np.ndarray:
        """texts를 정규화된 dense 벡터로 바꾼다. bge-m3는 query/passage 접두어가 필요 없다."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return normalize_rows(np.asarray(vectors, dtype=np.float32))
