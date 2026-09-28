"""임베딩 클라이언트가 공통으로 따르는 인터페이스를 정의한다."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingClient(Protocol):
    """텍스트 목록을 길이 1로 정규화된 벡터 배열로 바꾸는 클라이언트다."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """texts와 같은 순서의 (len(texts), dim) 정규화 벡터 배열을 반환한다."""
        ...


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """각 행 벡터의 길이를 1로 만든다. 정규화된 벡터끼리는 내적이 곧 코사인 유사도다."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
