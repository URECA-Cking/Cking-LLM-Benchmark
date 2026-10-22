"""관심 분야 생성에 필요한 BGE-M3 클라이언트를 지연 생성한다."""

from __future__ import annotations

import numpy as np

from src.clients.base import EmbeddingClient


class LazyLocalBgeM3:
    name = "BAAI/bge-m3"
    dim = 1024

    def __init__(self) -> None:
        self._delegate = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._delegate is None:
            from src.clients.local_embedding import LocalEmbeddingClient

            self._delegate = LocalEmbeddingClient("bge-m3")
        return self._delegate.embed(texts)


def create_interest_embedding_client(provider: str) -> EmbeddingClient:
    if provider == "local":
        return LazyLocalBgeM3()
    if provider == "deepinfra":
        from src.recommendation.runtime import LazyDeepInfraEmbeddingClient

        return LazyDeepInfraEmbeddingClient()
    raise ValueError(f"지원하지 않는 BGE-M3 제공자입니다: {provider}")
