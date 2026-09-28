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

# 실험 착수 시 확정한 데이터. 값이 바뀌면 README와 함께 갱신한다.
CREATORS_CSV_SHA256 = "e4c14bf750a17ec33d430b1958e397c986238dcd3073a41be566cec32ea2e01a"
# 2026-09-28 (1차): 85개 행에서 누락됐던 declared 칸(빈 문자열)을 명시적으로 채운 서식 수정.
# 2026-09-28 (2차, 리뷰 P2): csv 모듈이 기본으로 쓰는 CRLF 줄바꿈을 LF로 정규화(.gitattributes 추가).
# Windows 클론에서 Git 자동 줄바꿈 변환과 겹쳐 무결성 검사가 실패하는 문제였다.
# 2026-09-28 (3차, 리뷰 P2): X01·X02의 bio에 경품 문구를 직접 포함시켰다. events는 bio가 있으면
# input_text()에 전달되지 않아, "경품 잡음에 강하다"던 E3 사례가 실제로는 잡음을 넣어본 적이 없었다.
# 세 수정 모두 X01·X02 bio 외 gold·declared 값은 바뀌지 않았다.
SPLIT_CSV_SHA256 = "cb95606e41c46dea323d74db75ee8b8edb4e136e8eb11e875850ab574d5545b6"

TAXONOMY_VERSION = "v0.1"

PRICING_CHECKED_AT = "2026-09-28"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
OPENAI_EMBEDDING_PRICE_PER_1M = 0.02  # USD / 1,000,000 input tokens

BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_EMBEDDING_DIM = 1024

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

# 태깅·유사도 임계값은 dev로만 결정한다. 여기 값은 select 단계가 채운 뒤 고정한다.
ZERO_SHOT_TAU: dict[str, float | None] = {
    "text-embedding-3-small": None,
    "bge-m3": None,
}
SIMILARITY_BONUS: dict[str, float | None] = {
    "text-embedding-3-small": None,
    "bge-m3": None,
}

TOP_N_STORED = 20  # 크리에이터별 저장할 유사 후보 수
JUDGE_TOP_K = 5  # 판정 시트에 합집합으로 모을 상위 개수
JUDGE_SHUFFLE_SEED = 20260928

METHOD_IDS = ("M1", "M2", "M3", "M4", "R1", "R2")
