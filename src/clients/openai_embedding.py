"""OpenAI 호환 임베딩 API 호출 클라이언트다. 기본은 OpenAI text-embedding-3-small이다."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from src.clients.base import normalize_rows
from src.config import OPENAI_EMBEDDING_DIM, OPENAI_EMBEDDING_MODEL


class OpenAIEmbeddingClient:
    """OpenAI 호환 임베딩 API를 호출해 정규화된 벡터를 만든다. model·dim을 바꾸면 다른 호환 업체에도 쓴다."""

    def __init__(
        self, client: OpenAI | None = None, model: str = OPENAI_EMBEDDING_MODEL, dim: int = OPENAI_EMBEDDING_DIM
    ) -> None:
        self._client = client or OpenAI()
        self.name = model
        self.dim = dim
        self.last_input_tokens: int | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        """빈 문자열 없이 texts를 한 번의 API 호출로 임베딩한다. 사용 토큰 수를 last_input_tokens에 남긴다."""
        if not texts:
            self.last_input_tokens = 0
            return np.empty((0, self.dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self.name, input=texts)
        self.last_input_tokens = response.usage.total_tokens
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return normalize_rows(vectors)
