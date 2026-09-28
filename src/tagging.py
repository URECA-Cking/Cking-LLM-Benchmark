"""zero-shot 태깅(임베딩 기반)과 LLM 태깅 결과를 다루는 순수 로직이다.

zero-shot 태깅은 카테고리 이름이 아니라 설명문 벡터와 크리에이터 벡터의 코사인 유사도로
점수를 매긴다. LLM 태깅은 순위가 없는 태그 집합만 돌려주므로, 두 방식의 평가 지표를
다르게 정의한다 (zero-shot: Top-1·Top-3, LLM: 태그 집합 적중률·일관성).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankedTags:
    """한 크리에이터에 대해 점수 내림차순으로 정렬된 카테고리 후보다."""

    codes_by_rank: tuple[str, ...]
    scores_by_rank: tuple[float, ...]

    def top(self, k: int) -> tuple[str, ...]:
        """상위 k개 카테고리 코드를 반환한다."""
        return self.codes_by_rank[:k]

    def assigned(self, tau: float, max_tags: int) -> frozenset[str]:
        """점수 상위 max_tags개 중 tau 이상인 카테고리만 태그로 확정한다."""
        top_ranked = zip(self.codes_by_rank[:max_tags], self.scores_by_rank[:max_tags])
        return frozenset(code for code, score in top_ranked if score >= tau)


def rank_categories(creator_vector: np.ndarray, category_vectors: np.ndarray, category_codes: list[str]) -> RankedTags:
    """정규화된 크리에이터 벡터 하나를 모든 카테고리 벡터와 비교해 점수 내림차순으로 정렬한다."""
    scores = category_vectors @ creator_vector
    order = np.argsort(-scores)
    return RankedTags(
        codes_by_rank=tuple(category_codes[i] for i in order),
        scores_by_rank=tuple(float(scores[i]) for i in order),
    )


def rank_all(creator_vectors: np.ndarray, category_vectors: np.ndarray, category_codes: list[str]) -> list[RankedTags]:
    """크리에이터 벡터 여러 개를 한 번에 순위화한다. 행 순서를 그대로 유지한다."""
    return [rank_categories(vec, category_vectors, category_codes) for vec in creator_vectors]


def f1_against_gold(predicted: frozenset[str], gold: frozenset[str]) -> float:
    """예측 태그 집합과 정답 분야 집합의 F1 점수를 계산한다. 둘 다 비면 1.0으로 본다."""
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def select_tau(
    ranked_by_creator: dict[str, RankedTags],
    gold_by_creator: dict[str, frozenset[str]],
    max_tags: int,
    candidate_taus: list[float] | None = None,
) -> float:
    """dev 크리에이터의 평균 F1을 최대화하는 tau를 그리드서치로 고른다.

    후보 tau는 관측된 점수값 자체에서 가져온다 (임의의 고정 격자 대신, 실제로 임계선이
    될 수 있는 지점만 시험한다).
    """
    if candidate_taus is None:
        all_scores = sorted({score for ranked in ranked_by_creator.values() for score in ranked.scores_by_rank})
        candidate_taus = all_scores
    if not candidate_taus:
        raise ValueError("tau 후보가 없습니다. ranked_by_creator가 비어 있는지 확인하세요.")

    best_tau, best_f1 = candidate_taus[0], -1.0
    for tau in candidate_taus:
        f1_scores = [
            f1_against_gold(ranked.assigned(tau, max_tags), gold_by_creator[creator_id])
            for creator_id, ranked in ranked_by_creator.items()
        ]
        mean_f1 = sum(f1_scores) / len(f1_scores)
        if mean_f1 > best_f1:
            best_tau, best_f1 = tau, mean_f1
    return best_tau


def top1_accuracy(ranked_by_creator: dict[str, RankedTags], gold_by_creator: dict[str, frozenset[str]]) -> float:
    """1등 카테고리가 정답 분야 중 하나인 크리에이터의 비율이다."""
    hits = sum(1 for cid, ranked in ranked_by_creator.items() if ranked.top(1)[0] in gold_by_creator[cid])
    return hits / len(ranked_by_creator)


def top3_inclusion_rate(ranked_by_creator: dict[str, RankedTags], gold_by_creator: dict[str, frozenset[str]]) -> float:
    """정답 분야 중 하나 이상이 상위 3개 안에 포함된 크리에이터의 비율이다."""
    hits = sum(1 for cid, ranked in ranked_by_creator.items() if set(ranked.top(3)) & gold_by_creator[cid])
    return hits / len(ranked_by_creator)


def confusion_pairs(ranked_by_creator: dict[str, RankedTags], gold_by_creator: dict[str, frozenset[str]]) -> dict[tuple[str, str], int]:
    """1등이 정답과 다를 때 (정답, 예측1등) 쌍이 몇 번 나오는지 센다."""
    counts: dict[tuple[str, str], int] = {}
    for cid, ranked in ranked_by_creator.items():
        predicted = ranked.top(1)[0]
        gold = gold_by_creator[cid]
        if predicted not in gold:
            for g in gold:
                counts[(g, predicted)] = counts.get((g, predicted), 0) + 1
    return counts


def llm_hit_rate(tags_by_creator: dict[str, frozenset[str]], gold_by_creator: dict[str, frozenset[str]]) -> float:
    """LLM이 반환한 태그 집합이 정답 분야와 하나라도 겹치는 크리에이터의 비율이다."""
    hits = sum(1 for cid, tags in tags_by_creator.items() if tags & gold_by_creator[cid])
    return hits / len(tags_by_creator)


def llm_mean_f1(tags_by_creator: dict[str, frozenset[str]], gold_by_creator: dict[str, frozenset[str]]) -> float:
    """LLM 태그 집합과 정답 분야의 평균 F1이다."""
    scores = [f1_against_gold(tags, gold_by_creator[cid]) for cid, tags in tags_by_creator.items()]
    return sum(scores) / len(scores)


def unclassified_rate(tag_sets: list[frozenset[str]]) -> float:
    """태그가 하나도 없는(UNCLASSIFIED) 크리에이터의 비율이다."""
    return sum(1 for tags in tag_sets if not tags) / len(tag_sets)


def pick_llm_model(dev_metrics: dict[str, dict[str, float]]) -> str:
    """dev 지표(평균 F1 우선, 동률이면 hit_rate)가 가장 좋은 LLM 모델명을 고른다."""
    return max(dev_metrics, key=lambda model: (dev_metrics[model]["f1"], dev_metrics[model]["hit_rate"]))


def consistency_rate(run1: dict[str, frozenset[str]], run2: dict[str, frozenset[str]]) -> float:
    """같은 입력을 두 번 태깅했을 때 Jaccard 유사도의 평균이다. 1.0이면 완전히 일치."""
    ids = list(run1)
    scores = []
    for cid in ids:
        a, b = run1[cid], run2[cid]
        if not a and not b:
            scores.append(1.0)
            continue
        scores.append(len(a & b) / len(a | b))
    return sum(scores) / len(scores)
