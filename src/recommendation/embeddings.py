"""두 추천 생성기가 공유하는 원벡터 저장 및 정규화 계약."""
import numpy as np

EMBEDDING_CONTRACT_VERSION = "raw-float64-normalize-v2"


def validated_embeddings(values: object, rows: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != rows:
        raise ValueError("임베딩 행 수가 요청과 다릅니다.")
    if raw.shape[1] != dim:
        raise ValueError("임베딩 차원이 요청과 다릅니다.")
    if not np.isfinite(raw).all():
        raise ValueError("임베딩에 유한하지 않은 값이 있습니다.")
    # 원벡터를 보존하고 overflow 없이 float64에서 한 번 정규화한다.
    scale = np.max(np.abs(raw), axis=1)
    if np.any(scale <= 0):
        raise ValueError("임베딩 norm은 양수여야 합니다.")
    scaled = raw / scale[:, None]
    return raw, scaled / np.linalg.norm(scaled, axis=1)[:, None]
