"""저장된 후보 순위만 사용하는 Python·Java 공용 개인화 정책 참조 구현."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from typing import Any


HYBRID_POLICY_VERSION = "HYBRID_PERSONALIZED_V1"
INTEREST_POLICY_VERSION = "INTEREST_PERSONALIZED_V1"
FOLLOW_POLICY_VERSION = "FOLLOW_PERSONALIZED_V2"
RRF_K = 60
SCORE_SCALE = 8
INTEREST_WEIGHT = Decimal("0.5")
FOLLOW_WEIGHT = Decimal("0.5")
_QUANTUM = Decimal("0.00000001")
_ZERO = Decimal("0.00000000")


def policy_settings() -> dict[str, object]:
    """V1 고정 설정이다. 설정 변경은 별도 정책 버전과 fixture가 필요하다."""
    return {
        "rrfK": RRF_K,
        "scoreScale": SCORE_SCALE,
        "roundingMode": "HALF_UP",
        "interestWeight": str(INTEREST_WEIGHT),
        "followWeight": str(FOLLOW_WEIGHT),
    }


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}은 JSON 객체여야 합니다.")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name}은 JSON 배열이어야 합니다.")
    return value


def _positive_integer(value: Any, name: str, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name}은 1~{maximum}의 정수여야 합니다.")
    return value


def _interest_code(value: Any) -> str:
    # ASCII 코드의 사전순은 Python과 Java에서 동일하다. 표시 순서가 아니다.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
        raise ValueError("interestCode는 대문자 ASCII 분야 코드여야 합니다.")
    return value


def _sources(value: Any, key: str) -> dict[str | int, tuple[tuple[int, int], ...]]:
    sources = {}
    field = "interestSources" if key == "interestCode" else "followSources"
    for raw_source in _array(value, field):
        source = _object(raw_source, "source")
        identity = (_interest_code(source[key]) if key == "interestCode"
                    else _positive_integer(source[key], key))
        if identity in sources:
            raise ValueError(f"source 식별자가 중복되었습니다: {identity}")
        generation = source["activeGeneration"]
        if generation is None:
            sources[identity] = ()
            continue
        generation = _object(generation, "activeGeneration")
        if not isinstance(generation["generationId"], str) or not generation["generationId"].strip():
            raise ValueError("generationId는 비어 있지 않은 문자열이어야 합니다.")
        candidates = []
        creator_ids = set()
        ranks = set()
        for raw_candidate in _array(generation["candidates"], "candidates"):
            candidate = _object(raw_candidate, "candidate")
            creator_id = _positive_integer(candidate["creatorId"], "creatorId")
            rank = _positive_integer(candidate["rank"], "rank", 2**31 - 1)
            if creator_id in creator_ids or rank in ranks:
                raise ValueError("한 source 안에서 creatorId와 rank는 각각 고유해야 합니다.")
            creator_ids.add(creator_id)
            ranks.add(rank)
            candidates.append((creator_id, rank))
        sources[identity] = tuple(candidates)
    return sources


def _group_scores(
    selected: set[str] | set[int],
    sources: Mapping[str | int, tuple[tuple[int, int], ...]],
    excluded: set[int],
    spaces: set[int],
) -> tuple[dict[int, Decimal], dict[int, set[str | int]], int]:
    totals: dict[int, Decimal] = {}
    provenance: dict[int, set[str | int]] = {}
    valid_count = 0
    for identity in sorted(selected):
        remaining = [(creator_id, rank) for creator_id, rank in sources.get(identity, ())
                     if creator_id not in excluded and creator_id in spaces]
        # 모든 제외 규칙 적용 후 후보가 남은 source만 그룹 평균의 분모에 넣는다.
        if not remaining:
            continue
        valid_count += 1
        for creator_id, rank in remaining:
            # 제외 후 새 순위를 붙이지 않는다. 활성 세대에 저장된 원래 rank다.
            contribution = (Decimal(1) / Decimal(RRF_K + rank)).quantize(
                _QUANTUM, rounding=ROUND_HALF_UP,
            )
            totals[creator_id] = totals.get(creator_id, _ZERO) + contribution
            provenance.setdefault(creator_id, set()).add(identity)
    averages = {
        creator_id: (total / Decimal(valid_count)).quantize(_QUANTUM, rounding=ROUND_HALF_UP)
        for creator_id, total in totals.items()
    }
    return averages, provenance, valid_count


def recommend_personalized(request: Mapping[str, Any]) -> dict[str, object]:
    """공용 fixture의 input 객체를 받아 정렬된 전체 후보와 문자열 점수를 반환한다.

    누락된 필수 필드는 KeyError, 잘못된 타입·중복 source/후보 순위는 ValueError다.
    누락된 source·Space는 무효로 취급한다. 모델, DB, 네트워크를 호출하지 않는다.
    """
    request = _object(request, "input")
    member_id = request["memberCreatorId"]
    if member_id is not None:
        member_id = _positive_integer(member_id, "memberCreatorId")
    interests = {_interest_code(code) for code in _array(request["selectedInterestCodes"], "selectedInterestCodes")}
    followed = {_positive_integer(creator_id, "followedCreatorIds")
                for creator_id in _array(request["followedCreatorIds"], "followedCreatorIds")}
    excluded = followed | ({member_id} if member_id is not None else set())
    spaces = set()
    seen_spaces = set()
    for raw_space in _array(request["creatorSpaces"], "creatorSpaces"):
        space = _object(raw_space, "creatorSpace")
        creator_id = _positive_integer(space["creatorId"], "creatorId")
        if creator_id in seen_spaces:
            raise ValueError("creatorSpaces의 creatorId가 중복되었습니다.")
        seen_spaces.add(creator_id)
        if not isinstance(space["hasCreatorSpace"], bool):
            raise ValueError("hasCreatorSpace는 boolean이어야 합니다.")
        if space["hasCreatorSpace"]:
            spaces.add(creator_id)
    interest_sources = _sources(request["interestSources"], "interestCode")
    follow_sources = _sources(request["followSources"], "seedCreatorId")

    # 호출자의 Decimal 정밀도·반올림·trap 설정에 결과가 의존하지 않게 격리한다.
    # Java long ID / int rank와 8자리 점수 연산에 충분한 중간 정밀도를 확보한다.
    with localcontext(Context(prec=80, rounding=ROUND_HALF_UP)):
        interest_scores, interest_provenance, interest_count = _group_scores(
            interests, interest_sources, excluded, spaces,
        )
        follow_scores, follow_provenance, follow_count = _group_scores(
            followed, follow_sources, excluded, spaces,
        )
        if interest_count and follow_count:
            version = HYBRID_POLICY_VERSION
        elif interest_count:
            version = INTEREST_POLICY_VERSION
        else:
            # 둘 다 무효일 때도 평균 방식의 팔로우 V2 정책 이름과 빈 목록을 반환한다.
            version = FOLLOW_POLICY_VERSION
        scores = {}
        for creator_id in interest_scores.keys() | follow_scores.keys():
            if interest_count and follow_count:
                score = (interest_scores.get(creator_id, _ZERO) * INTEREST_WEIGHT
                         + follow_scores.get(creator_id, _ZERO) * FOLLOW_WEIGHT)
                score = score.quantize(_QUANTUM, rounding=ROUND_HALF_UP)
            else:
                score = interest_scores.get(creator_id, follow_scores.get(creator_id, _ZERO))
            scores[creator_id] = score
        return {
            "policyVersion": version,
            "validSourceCounts": {"interest": interest_count, "follow": follow_count},
            "items": [
                {
                    "creatorId": creator_id,
                    "aggregateScore": format(scores[creator_id], ".8f"),
                    "interestCodes": sorted(interest_provenance.get(creator_id, set())),
                    "seedCreatorIds": sorted(follow_provenance.get(creator_id, set())),
                }
                for creator_id in sorted(scores, key=lambda key: (-scores[key], key))
            ],
        }
