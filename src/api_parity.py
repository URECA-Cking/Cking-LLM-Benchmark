"""같은 bge-m3를 로컬과 API로 돌린 벡터가 추천 결과를 바꾸지 않을 만큼 같은지 비교한다.

서버 정밀도(fp16 등) 차이로 벡터가 미세하게 다르면 로컬에서 고른 tau·bonus·컷오프가 어긋날 수 있어,
추천에 실제로 쓰는 값(크리에이터끼리·크리에이터↔카테고리 유사도, 상위 이웃 순위)으로 차이를 잰다.
유사도가 거의 같아도 tau·컷오프는 "이상이면 통과"라는 경계 판정이라, 경계에 놓인 크리에이터는 아주 작은 차이로도
결과가 바뀐다. 그래서 같은 파라미터로 태그 집합과 컷오프 뒤 상위 후보를 직접 비교하는 단계(compare_decisions)를 따로 둔다.
"""

from __future__ import annotations

import numpy as np

from src.similarity import cosine_matrix, cosine_with_tag_bonus, jaccard_matrix, top_n
from src.tagging import rank_all


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """두 유사도 행렬에서 자기 자신(-inf)을 뺀 칸들의 최대 절댓값 차이다."""
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite])))


def _top_k(matrix: np.ndarray, k: int) -> np.ndarray:
    """행마다 유사도가 높은 순서의 상위 k개 열 인덱스다."""
    return np.argsort(-matrix, axis=1, kind="stable")[:, :k]


def compare_embeddings(
    local_creators: np.ndarray,
    api_creators: np.ndarray,
    local_categories: np.ndarray,
    api_categories: np.ndarray,
    k: int,
    max_abs_diff_limit: float,
    min_top_k_overlap: float,
) -> dict:
    """같은 순서로 정렬된 로컬·API 정규화 벡터를 비교해 지표와 동등 여부를 돌려준다."""
    self_cosine = np.sum(local_creators * api_creators, axis=1)
    local_cc, api_cc = cosine_matrix(local_creators), cosine_matrix(api_creators)
    local_top, api_top = _top_k(local_cc, k), _top_k(api_cc, k)
    overlaps = np.array([len(set(l) & set(a)) / k for l, a in zip(local_top, api_top)])
    creator_diff = _max_abs_diff(local_cc, api_cc)
    category_diff = _max_abs_diff(local_creators @ local_categories.T, api_creators @ api_categories.T)
    report = {
        "creator_count": len(local_creators),
        "self_cosine_min": float(self_cosine.min()),
        "self_cosine_mean": float(self_cosine.mean()),
        "creator_similarity_max_abs_diff": creator_diff,
        "category_similarity_max_abs_diff": category_diff,
        f"top{k}_overlap_mean": float(overlaps.mean()),
        f"top{k}_overlap_min": float(overlaps.min()),
        "top1_agreement": float(np.mean(local_top[:, 0] == api_top[:, 0])),
    }
    report["similarity_equivalent"] = bool(
        max(creator_diff, category_diff) <= max_abs_diff_limit and overlaps.mean() >= min_top_k_overlap
    )
    return report


def compare_decisions(
    creator_ids: list[str],
    local_creators: np.ndarray,
    api_creators: np.ndarray,
    local_categories: np.ndarray,
    api_categories: np.ndarray,
    category_codes: list[str],
    llm_tag_sets: list[frozenset[str]],
    tau: float,
    max_tags: int,
    bonuses: dict[str, float],
    cutoffs: dict[str, float],
    k: int,
) -> dict:
    """같은 tau·bonus·컷오프로 로컬·API 벡터의 zero-shot 태그와 컷오프 뒤 상위 k 후보(순위 순서 포함)를 비교한다.

    태그는 점수가 tau 이상인지, 후보는 점수가 컷오프 이상인지로 정해져서 경계 근처에서는 미세한 차이로도 뒤집힌다.
    M1~M4(bge-m3 임베딩과 tau·bonus·컷오프를 쓰는 방식)만 본다. M5는 리랭커 점수가 따로 필요해 포함하지 않는다.
    """

    def tags(creators: np.ndarray, categories: np.ndarray) -> list[frozenset[str]]:
        return [r.assigned(tau, max_tags) for r in rank_all(creators, categories, category_codes)]

    def matrices(creators: np.ndarray, zero_shot: list[frozenset[str]]) -> dict[str, np.ndarray]:
        cosine = cosine_matrix(creators)
        return {
            "M1": jaccard_matrix(zero_shot),
            "M2": cosine,
            "M3": cosine_with_tag_bonus(cosine, zero_shot, bonuses["M3"]),
            "M4": cosine_with_tag_bonus(cosine, llm_tag_sets, bonuses["M4"]),
        }

    local_tags, api_tags = tags(local_creators, local_categories), tags(api_creators, api_categories)
    local_matrices, api_matrices = matrices(local_creators, local_tags), matrices(api_creators, api_tags)

    def kept(matrix: np.ndarray, row: int, cutoff: float) -> list[str]:
        """컷오프를 넘은 상위 k 후보를 순위 순서 그대로 돌려준다. 순서가 달라져도 다른 결과로 본다."""
        return [cid for cid, score in top_n(matrix, creator_ids, row, k) if score >= cutoff]

    def summarize(changed: list[str]) -> dict:
        return {"count": len(changed), "creators": changed}

    tag_mismatch = [cid for cid, a, b in zip(creator_ids, local_tags, api_tags) if a != b]
    candidate_mismatch = {
        method: [
            cid
            for row, cid in enumerate(creator_ids)
            if kept(local_matrices[method], row, cutoffs[method]) != kept(api_matrices[method], row, cutoffs[method])
        ]
        for method in local_matrices
    }
    return {
        "zero_shot_tag_mismatch": summarize(tag_mismatch),
        "candidates_after_cutoff_mismatch": {method: summarize(ids) for method, ids in candidate_mismatch.items()},
        "decisions_identical": not tag_mismatch and not any(candidate_mismatch.values()),
    }
