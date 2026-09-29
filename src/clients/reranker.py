"""bge-reranker-v2-m3 cross-encoder로 (쿼리, 후보) 텍스트 쌍의 관련도를 직접 채점한다.

임베딩(bi-encoder)과 달리 벡터를 미리 만들어 코사인으로 비교하지 않고, 두 텍스트를
같이 모델에 넣어 관련도 점수 하나를 뽑는다 — 그만큼 정확하지만 쌍마다 추론이 필요해
느리므로, M2가 이미 추린 소수의 후보만 재정렬하는 용도로 쓴다.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL_NAME


class RerankerClient:
    """sentence-transformers CrossEncoder로 bge-reranker-v2-m3를 로컬에서 실행한다."""

    def __init__(self, model: CrossEncoder | None = None) -> None:
        self._model = model or CrossEncoder(RERANKER_MODEL_NAME)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """pairs와 같은 순서로 관련도 점수를 반환한다. 순서 자체가 유사도 랭킹은 아니다."""
        if not pairs:
            return []
        scores = self._model.predict(list(pairs))
        return [float(s) for s in scores]
