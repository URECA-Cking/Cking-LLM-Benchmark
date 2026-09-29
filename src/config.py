"""실행 경로, 고정 모델·가격, 실험 파라미터를 한곳에서 관리한다."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

CATEGORIES_CSV = DATA_DIR / "categories.csv"
CREATORS_CSV = DATA_DIR / "creators.csv"
SPLIT_CSV = DATA_DIR / "split.csv"
# dev 규모 민감도 실험용 추가 크리에이터(이슈 #8). 기존 100명과 별개 파일이며 평가에는 쓰지 않는다.
CREATORS_LARGE_CSV = DATA_DIR / "creators_large.csv"

# 실험 착수 시 확정한 데이터. 값이 바뀌면 README와 함께 갱신한다.
CREATORS_CSV_SHA256 = "e4c14bf750a17ec33d430b1958e397c986238dcd3073a41be566cec32ea2e01a"
# 2026-09-28 (1차): 85개 행에서 누락됐던 declared 칸(빈 문자열)을 명시적으로 채운 서식 수정.
# 2026-09-28 (2차, 리뷰 P2): csv 모듈이 기본으로 쓰는 CRLF 줄바꿈을 LF로 정규화(.gitattributes 추가).
# Windows 클론에서 Git 자동 줄바꿈 변환과 겹쳐 무결성 검사가 실패하는 문제였다.
# 2026-09-28 (3차, 리뷰 P2): X01·X02의 bio에 경품 문구를 직접 포함시켰다. events는 bio가 있으면
# input_text()에 전달되지 않아, "경품 잡음에 강하다"던 E3 사례가 실제로는 잡음을 넣어본 적이 없었다.
# 세 수정 모두 X01·X02 bio 외 gold·declared 값은 바뀌지 않았다.
SPLIT_CSV_SHA256 = "cb95606e41c46dea323d74db75ee8b8edb4e136e8eb11e875850ab574d5545b6"
# creators_large.csv(2026-09-29, gpt-5.4-mini로 생성한 합성 300명, 카테고리당 30명)의 확정 hash.
# 값이 바뀌면 이 hash와 docs를 함께 갱신한다(None이면 검증하지 않음).
CREATORS_LARGE_CSV_SHA256: str | None = "152aeb4be4dce9cc65fbd6261ddc894965ee611a461c437b8354d5ce08869aac"

TAXONOMY_VERSION = "v0.1"

PRICING_CHECKED_AT = "2026-09-28"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
OPENAI_EMBEDDING_PRICE_PER_1M = 0.02  # USD / 1,000,000 input tokens

LOCAL_EMBEDDING_MODELS = {
    "bge-m3": {"model_name": "BAAI/bge-m3", "dim": 1024},
    # bge-m3를 한국어 데이터로 파인튜닝한 모델. 같은 베이스에서 한국어 특화가 도움이 되는지 비교 (2026-09-28 추가)
    "kure-v1": {"model_name": "nlpai-lab/KURE-v1", "dim": 1024},
    # 다국어(한국어 포함) 임베딩. 리뷰에서 제안된 비교 후보 (2026-09-28 추가)
    # query_prompt_name: 모델에 등록된 쿼리용 instruct 프롬프트 이름. 평가 쿼리 30명의
    # 임베딩만 이 프롬프트로 다시 인코딩해 M2~M4/R2에 쓴다(이슈 #8, 리뷰로 발견 — Qwen3
    # 공식 사용법은 검색 쿼리 쪽에 이 프롬프트를 쓰길 권장하는데 기존엔 적용하지 않았다).
    "qwen3-embedding-0.6b": {"model_name": "Qwen/Qwen3-Embedding-0.6B", "dim": 1024, "query_prompt_name": "query"},
}

# M5(bge-m3 M2 + 재정렬)용 cross-encoder 리랭커. 임베딩처럼 벡터를 미리 만들어두지 않고
# (쿼리, 후보) 쌍을 직접 채점하므로 M2가 이미 추린 후보군(TOP_N_STORED)에만 적용한다.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# M4 LLM 태깅 후보. dev 정확도로 하나를 골라 OPENAI_LLM_MODEL_SELECTED에 기록한다.
OPENAI_LLM_MODEL_CANDIDATES = {
    "gpt-5.4-nano-2026-03-17": {"input_price": 0.20, "output_price": 1.25, "supports_temperature_zero": True},
    "gpt-4.1-nano-2025-04-14": {"input_price": 0.10, "output_price": 0.40, "supports_temperature_zero": True},
}
# gpt-5-nano-2025-08-07 은 temperature=0을 거부해 제외 (2026-09-28 실측, 400 Unsupported value)

LLM_TEMPERATURE = 0
LLM_TAG_MAX = 3
LLM_CONSISTENCY_RUNS = 2

# E2 자동 판정(LLM-as-judge)용 모델. M4 태깅에 쓴 모델(gpt-5.4-nano)보다 강한 모델을 써서
# "같은 LLM이 자기 태깅을 스스로 좋게 채점하는" 순환 편향을 줄인다.
OPENAI_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
OPENAI_JUDGE_MODEL_PRICE = {"input_price": 0.75, "output_price": 4.50}

TOP_N_STORED = 20  # 크리에이터별 저장할 유사 후보 수
JUDGE_TOP_K = 5  # 판정 시트에 합집합으로 모을 상위 개수
JUDGE_SHUFFLE_SEED = 20260928
