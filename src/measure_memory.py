"""로컬 임베딩 모델·리랭커의 메모리 사용량을 잰다 (이슈 #13).

프로세스의 최대 상주 메모리(peak RSS)는 줄어들지 않는 값이라 한 프로세스에서 모델을 여러 개 재면 앞선 모델이
뒤 모델 값에 섞인다. 그래서 이 모듈은 모델 하나를 새 프로세스에서 재고 JSON 한 줄을 출력하며, `memory` 단계가
모델마다 하위 프로세스로 호출한다. 직접 실행: `python3 -m src.measure_memory <모델 키|reranker> [--device cpu|auto]`
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys

RERANKER_KEY = "reranker"
M5_KEY = "m5"  # bge-m3 임베딩과 리랭커를 한 프로세스에 함께 올린 조건(M5 서버 구성)
MEASURE_PAIRS = 100  # 리랭커 측정용으로 채점하는 (쿼리, 후보) 쌍 수(M5는 쿼리당 20쌍씩 채점한다)


def peak_rss_mb() -> float:
    """이 프로세스가 지금까지 쓴 최대 상주 메모리를 MB로 돌려준다. macOS는 바이트, Linux는 KB 단위로 준다."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def measure(name: str, device: str) -> dict:
    """모델 하나를 로드하고 크리에이터 100명(또는 리랭커 100쌍)을 처리하며 단계별 최대 메모리를 잰다."""
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from src.clients import LocalEmbeddingClient, RerankerClient
    from src.config import LOCAL_EMBEDDING_MODELS, RERANKER_MODEL_NAME
    from src.data import load_creators

    texts = [c.input_text() for c in load_creators()]
    baseline = peak_rss_mb()  # torch·sentence-transformers를 불러온 직후, 모델을 올리기 전
    kwargs = {} if device == "auto" else {"device": device}
    pairs = [(texts[i % len(texts)], texts[(i * 7 + 3) % len(texts)]) for i in range(MEASURE_PAIRS)]
    if name == RERANKER_KEY:
        model = CrossEncoder(RERANKER_MODEL_NAME, **kwargs)
        after_load = peak_rss_mb()
        RerankerClient(model=model).score(pairs)
        model_name = RERANKER_MODEL_NAME
    elif name == M5_KEY:
        # M5는 bge-m3 임베딩으로 후보를 뽑고 리랭커로 다시 채점하므로 둘을 한 프로세스에 함께 올린다
        embedder = SentenceTransformer(LOCAL_EMBEDDING_MODELS["bge-m3"]["model_name"], **kwargs)
        model = CrossEncoder(RERANKER_MODEL_NAME, **kwargs)
        after_load = peak_rss_mb()
        LocalEmbeddingClient("bge-m3", model=embedder).embed(texts)
        RerankerClient(model=model).score(pairs)
        model_name = f"{LOCAL_EMBEDDING_MODELS['bge-m3']['model_name']} + {RERANKER_MODEL_NAME}"
    else:
        model_name = LOCAL_EMBEDDING_MODELS[name]["model_name"]
        model = SentenceTransformer(model_name, **kwargs)
        after_load = peak_rss_mb()
        LocalEmbeddingClient(name, model=model).embed(texts)
    after_run = peak_rss_mb()
    return {
        "model": model_name,
        "requested_device": device,
        "actual_device": str(model.device) if hasattr(model, "device") else "unknown",
        "baseline_mb": round(baseline, 1),
        "peak_after_load_mb": round(after_load, 1),
        "peak_after_run_mb": round(after_run, 1),
        "load_delta_mb": round(after_load - baseline, 1),
        "run_delta_mb": round(after_run - after_load, 1),
        "items": MEASURE_PAIRS if name == RERANKER_KEY else len(texts),  # m5는 임베딩 100건 + 리랭커 100쌍을 모두 처리
        "platform": platform.platform(),
        "torch": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 모델 메모리 측정(새 프로세스에서 하나씩)")
    parser.add_argument("name", help="LOCAL_EMBEDDING_MODELS의 키 또는 reranker")
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto"], help="cpu는 서버 기준, auto는 이 장비의 가속기(MPS 등)")
    args = parser.parse_args()
    print(json.dumps(measure(args.name, args.device), ensure_ascii=False))


if __name__ == "__main__":
    main()
