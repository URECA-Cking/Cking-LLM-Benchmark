"""크리에이터 간 유사도 행렬을 방식(M1~M4, R1, R2)별로 만든다.

행렬은 항상 크리에이터 순서를 고정한 정사각 행렬이며, 자기 자신과의 유사도는
`-inf`로 채워 상위 후보에서 자연히 제외되게 한다.
"""

from __future__ import annotations

import numpy as np


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """두 태그 집합의 Jaccard 유사도다. 둘 다 비어 있으면 0으로 본다 (겹치는 근거가 없음)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_matrix(tag_sets: list[frozenset[str]]) -> np.ndarray:
    """태그 집합 목록으로 Jaccard 유사도 정사각 행렬을 만든다 (M1, R1)."""
    n = len(tag_sets)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = jaccard(tag_sets[i], tag_sets[j])
    _mask_diagonal(matrix)
    return matrix


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """정규화된 벡터 배열로 코사인 유사도 행렬을 만든다 (M2). 정규화 벡터의 내적 = 코사인."""
    matrix = vectors @ vectors.T
    _mask_diagonal(matrix)
    return matrix


def cosine_with_tag_bonus(cosine: np.ndarray, tag_sets: list[frozenset[str]], bonus: float) -> np.ndarray:
    """코사인 유사도에 태그 공유 여부(1개 이상 겹치면 +bonus)를 더한다 (M3, M4, R2)."""
    n = len(tag_sets)
    overlap = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j and (tag_sets[i] & tag_sets[j]):
                overlap[i, j] = 1.0
    combined = cosine.copy()
    finite = np.isfinite(combined)
    combined[finite] = combined[finite] + bonus * overlap[finite]
    return combined


def _mask_diagonal(matrix: np.ndarray) -> None:
    """정사각 행렬의 대각선을 -inf로 채워 자기 자신이 후보에 나오지 않게 한다."""
    np.fill_diagonal(matrix, -np.inf)


def precision_at_k_by_gold(
    matrix: np.ndarray,
    ids: list[str],
    gold_by_id: dict[str, frozenset[str]],
    query_ids: list[str],
    k: int,
) -> float:
    """쿼리 크리에이터의 상위 k명 중, 정답 분야(gold)를 하나라도 공유하는 비율의 평균이다.

    bonus·tau 같은 파라미터를 고를 때 쓰는 dev 전용 대리 지표다. 태깅 결과 자체가 아니라
    독립적인 gold 라벨을 기준으로 삼아, "태그를 더하면 실제로 같은 분야 크리에이터가
    상위에 더 오는가"를 순환 논리 없이 확인한다.
    """
    scores = []
    for query_id in query_ids:
        gold = gold_by_id[query_id]
        if not gold:
            continue
        row_index = ids.index(query_id)
        top = top_n(matrix, ids, row_index, k)
        relevant = sum(1 for candidate_id, _ in top if gold_by_id[candidate_id] & gold)
        scores.append(relevant / k)
    return sum(scores) / len(scores) if scores else 0.0


def select_bonus(
    cosine: np.ndarray,
    ids: list[str],
    tag_sets: list[frozenset[str]],
    gold_by_id: dict[str, frozenset[str]],
    dev_ids: list[str],
    candidate_bonuses: list[float],
    k: int = 5,
) -> float:
    """dev 쿼리들의 precision_at_k_by_gold를 최대화하는 bonus를 그리드서치로 고른다."""
    best_bonus, best_score = candidate_bonuses[0], -1.0
    for bonus in candidate_bonuses:
        combined = cosine_with_tag_bonus(cosine, tag_sets, bonus)
        score = precision_at_k_by_gold(combined, ids, gold_by_id, dev_ids, k)
        if score > best_score:
            best_bonus, best_score = bonus, score
    return best_bonus


def top_n(matrix: np.ndarray, ids: list[str], row_index: int, n: int) -> list[tuple[str, float]]:
    """row_index 크리에이터 기준 상위 n명을 (id, score) 내림차순으로 반환한다.

    -inf(자기 자신) 또는 동점은 id 오름차순으로 안정적인 순서를 만든다.
    """
    row = matrix[row_index]
    order = sorted(range(len(ids)), key=lambda j: (-row[j], ids[j]))
    result = []
    for j in order:
        if not np.isfinite(row[j]):
            continue
        result.append((ids[j], float(row[j])))
        if len(result) == n:
            break
    return result
