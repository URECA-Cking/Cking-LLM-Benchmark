"""서비스 추천 생성의 입력과 BE 전달 결과 모델을 정의한다."""

from __future__ import annotations

from dataclasses import dataclass


MAX_BACKEND_CANDIDATES = 100


@dataclass(frozen=True)
class CreatorProfile:
    """추천 생성에 필요한 Cking 크리에이터 최소 입력이다."""

    creator_id: int
    introduction: str

    def __post_init__(self) -> None:
        if isinstance(self.creator_id, bool) or not isinstance(self.creator_id, int) or self.creator_id <= 0:
            raise ValueError("creator_id는 양의 정수여야 합니다.")
        if not isinstance(self.introduction, str):
            raise TypeError("introduction은 문자열이어야 합니다.")

    @property
    def text(self) -> str:
        """모델 입력과 해시에 함께 쓰는 정규화된 소개다."""
        return self.introduction.replace("\r\n", "\n").replace("\r", "\n").strip()


@dataclass(frozen=True)
class RecommendationCandidate:
    """Cking-BE가 저장하는 유사 추천 후보 한 건이다."""

    creator_id: int
    similar_creator_id: int
    score: float
    rank: int
    method: str
    model_version: str
    input_hash: str

    def to_dict(self) -> dict[str, int | float | str]:
        """BE Issue #393의 camelCase 전달 계약으로 변환한다."""
        return {
            "creatorId": self.creator_id,
            "similarCreatorId": self.similar_creator_id,
            "score": self.score,
            "rank": self.rank,
            "method": self.method,
            "modelVersion": self.model_version,
            "inputHash": self.input_hash,
        }


@dataclass(frozen=True)
class RecommendationResult:
    """한 seed 크리에이터에서 생성된 완결된 후보 묶음이다."""

    creator_id: int
    method: str
    model_version: str
    input_hash: str
    candidates: tuple[RecommendationCandidate, ...]

    def to_backend_payload(self) -> dict[str, object]:
        """부분 상태 없이 한 번에 적재할 수 있는 BE 전달 payload를 만든다."""
        if len(self.candidates) > MAX_BACKEND_CANDIDATES:
            raise ValueError("BE 적재 후보는 0~100건이어야 합니다.")
        if any(
            candidate.creator_id != self.creator_id
            or candidate.method != self.method
            or candidate.model_version != self.model_version
            or candidate.input_hash != self.input_hash
            for candidate in self.candidates
        ):
            raise ValueError("후보 메타데이터는 생성 세대의 최상위 메타데이터와 같아야 합니다.")
        return {
            "creatorId": self.creator_id,
            "method": self.method,
            "modelVersion": self.model_version,
            "inputHash": self.input_hash,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
