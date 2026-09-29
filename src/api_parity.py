"""같은 bge-m3를 로컬과 API로 돌린 벡터가 추천 결과를 바꾸지 않을 만큼 같은지 비교한다.

서버 정밀도(fp16 등) 차이로 벡터가 미세하게 다르면 로컬에서 고른 tau·bonus·컷오프가 어긋날 수 있어,
추천에 실제로 쓰는 값(크리에이터끼리·크리에이터↔카테고리 유사도, 상위 이웃 순위)으로 차이를 잰다.
"""

from __future__ import annotations

import numpy as np

from src.similarity import cosine_matrix


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
    report["equivalent"] = bool(
        max(creator_diff, category_diff) <= max_abs_diff_limit and overlaps.mean() >= min_top_k_overlap
    )
    return report
