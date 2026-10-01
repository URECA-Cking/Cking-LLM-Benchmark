"""sentence-transformers로 로컬 실행하는 임베딩 클라이언트다. 기본은 접두어 없이 dense 벡터만 사용한다.

bge-m3, KURE-v1(bge-m3의 한국어 파인튜닝)은 query/passage 접두어가 필요 없어 그대로 쓰면
된다. Qwen3-Embedding은 모델에 쿼리용 instruct 프롬프트가 등록돼 있고(`model.prompts`),
공식 사용법은 검색 질의 쪽에 이를 적용하길 권장한다(리뷰로 발견) — `embed(..., prompt_name=...)`로
호출부에서 선택적으로 적용한다. `LOCAL_EMBEDDING_MODELS[key]["query_prompt_name"]`에 등록된
모델만 해당하며, 평가 쿼리 역할일 때만 쓴다(이슈 #8, `docs/history/model-selection/index.md`의
Qwen3 관련 한계 참고).
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

    def embed(self, texts: list[str], prompt_name: str | None = None, prompt: str | None = None) -> np.ndarray:
        """texts를 정규화된 dense 벡터로 바꾼다. prompt_name은 모델에 등록된 프롬프트를, prompt는 직접 쓴 접두 문구를 붙인다."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, prompt_name=prompt_name, prompt=prompt, normalize_embeddings=True, convert_to_numpy=True)
        return normalize_rows(np.asarray(vectors, dtype=np.float32))
