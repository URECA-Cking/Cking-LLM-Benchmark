"""sentence-transformers로 로컬 실행하는 임베딩 클라이언트다. 접두어 없이 dense 벡터만 사용한다.

bge-m3, KURE-v1(bge-m3의 한국어 파인튜닝)은 query/passage 접두어가 필요 없어 이 클래스로
충분하다. Qwen3-Embedding은 모델에 쿼리용 instruct 프롬프트(`prompt_name="query"`)가
등록돼 있고 공식 사용법은 검색 질의 쪽에 이를 적용하길 권장하지만, 이 클래스는 세 모델 모두
접두어 없이 동일하게 인코딩한다 — M1~M4가 공유하는 대칭 코사인 행렬 구조(크리에이터 한 명을
같은 임베딩으로 쿼리·후보 양쪽에 다 쓰는 방식) 때문에 비대칭 프롬프트를 끼워 넣으려면 이
클래스 밖에서 별도 처리가 필요하다(리뷰로 발견, `docs/embedding-method-selection.md`의
Qwen3 관련 한계 참고). 현재 Qwen3 결과는 이 프롬프트 없이 측정한 값이다.
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
