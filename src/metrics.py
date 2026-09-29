"""판정 결과(E2)와 대표 사례 통과 여부(E3)를 계산하는 순수 함수 모음이다.

nDCG는 풀링(pooling) 방식의 관례를 따른다. 쿼리 하나의 "이상적인 순서"는 그 쿼리에서
판정된 모든 후보의 점수를 내림차순으로 정렬한 것이며, 풀에 없던(=한 번도 뽑히지 않은)
후보는 이상적 순서 계산에도 포함하지 않는다.
"""

from __future__ import annotations

import math
import random


def mean_relevance_at_k(ranked_ids: list[str], query_id: str, judged: dict[tuple[str, str], int]) -> float:
    """상위 k개 후보의 판정 점수(0~2) 평균이다. 풀에 없어 판정되지 않은 후보는 0으로 본다."""
    if not ranked_ids:
        return 0.0
    scores = [judged.get((query_id, cid), 0) for cid in ranked_ids]
    return sum(scores) / len(scores)


def irrelevant_rate_at_k(ranked_ids: list[str], query_id: str, judged: dict[tuple[str, str], int]) -> float:
    """상위 k개 중 점수가 0(무관)인 후보의 비율이다."""
    if not ranked_ids:
        return 0.0
    zero_count = sum(1 for cid in ranked_ids if judged.get((query_id, cid), 0) == 0)
    return zero_count / len(ranked_ids)


def _dcg(scores: list[int]) -> float:
    """0-index 기준 표준 DCG. 등급이 0/1/2뿐이라 log 밑은 2를 그대로 쓴다."""
    return sum(score / math.log2(rank + 2) for rank, score in enumerate(scores))


def ndcg_at_k(ranked_ids: list[str], query_id: str, judged_pool_for_query: list[int], judged: dict[tuple[str, str], int], k: int) -> float:
    """쿼리 하나의 nDCG@k. judged_pool_for_query는 그 쿼리에서 판정된 모든 점수 목록이다."""
    if not judged_pool_for_query:
        return 0.0
    actual = _dcg([judged.get((query_id, cid), 0) for cid in ranked_ids[:k]])
    ideal = _dcg(sorted(judged_pool_for_query, reverse=True)[:k])
    return 0.0 if ideal == 0 else actual / ideal


def paired_win_counts(per_query_a: dict[str, float], per_query_b: dict[str, float]) -> tuple[int, int, int]:
    """쿼리별 지표(예: mean_relevance_at_k)를 짝지어 A가 이긴 수·B가 이긴 수·동점 수를 센다."""
    wins_a = wins_b = ties = 0
    for query_id in per_query_a:
        a, b = per_query_a[query_id], per_query_b[query_id]
        if a > b:
            wins_a += 1
        elif a < b:
            wins_b += 1
        else:
            ties += 1
    return wins_a, wins_b, ties


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 20260928, alpha: float = 0.05) -> tuple[float, float]:
    """값 목록의 평균에 대한 percentile 부트스트랩 신뢰구간을 계산한다."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((alpha / 2) * n_boot)
    upper_idx = int((1 - alpha / 2) * n_boot) - 1
    return means[max(lower_idx, 0)], means[min(upper_idx, n_boot - 1)]


def check_case(top_ids: list[str], must_include: set[str] | None = None, must_exclude: set[str] | None = None) -> bool:
    """대표 사례(E3) 하나가 통과했는지 본다. must_include는 하나라도 있으면 통과, must_exclude는 하나도 없어야 통과."""
    present = set(top_ids)
    if must_include is not None and not (present & must_include):
        return False
    if must_exclude is not None and (present & must_exclude):
        return False
    return True


def cutoff_effect(
    top_lists: dict[str, list[list]],
    cutoff: float,
    gold_by_id: dict[str, frozenset[str]],
    query_ids: list[str],
    k: int = 5,
) -> dict[str, float | None]:
    """쿼리들의 상위 k 후보에 컷오프(점수 >= cutoff만 남김)를 적용했을 때의 효과를 잰다.

    무관 = 정답 분야를 하나도 공유하지 않는 후보. 무관 쌍 제거율·관련 쌍 보존율은 해당 쌍이 하나도
    없으면 None이다. 빈 결과 비율은 컷오프 뒤 후보가 하나도 안 남은 쿼리의 비율이다.
    """
    unrelated = removed_unrelated = related = kept_related = empty = 0
    for query_id in query_ids:
        gold = gold_by_id[query_id]
        kept = 0
        for candidate_id, score in top_lists[query_id][:k]:
            keep = score >= cutoff
            kept += keep
            if gold_by_id[candidate_id] & gold:
                related += 1
                kept_related += keep
            else:
                unrelated += 1
                removed_unrelated += not keep
        empty += kept == 0
    return {
        "unrelated_removed_rate": removed_unrelated / unrelated if unrelated else None,
        "related_kept_rate": kept_related / related if related else None,
        "empty_result_rate": empty / len(query_ids) if query_ids else 0.0,
    }
