"""E2 자동 판정(LLM) 결과 중, 결정을 가르는 지점만 사람이 다시 보게 골라내는 도구다.

전체 556쌍을 다 보는 대신, 비교 대상 방식(예: M4)이 기준 방식(예: M3)과 다르게 고른
후보만 추려 사람이 직접 소규모로 재판정하고, 자동 판정과의 일치율을 확인한다.
"""

from __future__ import annotations

from typing import Iterable


def select_disagreement_pairs(
    candidates: dict[str, dict[str, list[list]]],
    query_ids: Iterable[str],
    target_method: str,
    baseline_methods: list[str],
    k: int = 5,
) -> list[tuple[str, str]]:
    """target_method의 top-k에는 있지만 baseline_methods 어디의 top-k에도 없는 (query, candidate) 쌍을 고른다.

    이 쌍들이 target_method의 점수를 baseline보다 높게 만드는 실제 원인이다.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for query_id in query_ids:
        target_top = {item[0] for item in candidates[target_method][query_id][:k]}
        baseline_top: set[str] = set()
        for method in baseline_methods:
            baseline_top |= {item[0] for item in candidates[method][query_id][:k]}
        for candidate_id in target_top - baseline_top:
            pair = (query_id, candidate_id)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def select_pairwise_symmetric_disagreement(
    candidates: dict[str, dict[str, list[list]]],
    query_ids: Iterable[str],
    method_a: str,
    method_b: str,
    k: int = 5,
) -> list[tuple[str, str]]:
    """method_a·method_b의 top-k가 서로 다른 (query, candidate) 쌍을 **양쪽** 차집합으로 모은다.

    `select_disagreement_pairs`(target − baseline 합집합)는 "M4=A, M3=B, R2=A"처럼 다른 baseline이
    같은 후보를 가지고 있으면 실제 차이를 놓친다(리뷰 P2). 이 함수는 두 방식을 정확히 한 쌍씩 비교해
    한쪽에만 있는 후보(대칭차집합, A만 또는 B만)를 전부 잡아낸다 — A가 B에서 뺀 것과 더한 것 둘 다.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for query_id in query_ids:
        top_a = {item[0] for item in candidates[method_a][query_id][:k]}
        top_b = {item[0] for item in candidates[method_b][query_id][:k]}
        for candidate_id in top_a ^ top_b:
            pair = (query_id, candidate_id)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def agreement_stats(human: dict[tuple[str, str], int], auto: dict[tuple[str, str], int]) -> dict[str, float]:
    """같은 쌍에 대한 사람 점수와 자동 점수를 비교해 일치율을 계산한다."""
    keys = [k for k in human if k in auto]
    if not keys:
        return {"count": 0, "exact_match_rate": 0.0, "within_1_rate": 0.0, "mean_abs_diff": 0.0}
    exact = sum(1 for k in keys if human[k] == auto[k])
    within_1 = sum(1 for k in keys if abs(human[k] - auto[k]) <= 1)
    diffs = [abs(human[k] - auto[k]) for k in keys]
    return {
        "count": len(keys),
        "exact_match_rate": exact / len(keys),
        "within_1_rate": within_1 / len(keys),
        "mean_abs_diff": sum(diffs) / len(diffs),
    }
