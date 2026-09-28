"""실행 경로, 고정 모델·가격, 실험 파라미터를 한곳에서 관리한다."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"

CATEGORIES_CSV = DATA_DIR / "categories.csv"
CREATORS_CSV = DATA_DIR / "creators.csv"
SPLIT_CSV = DATA_DIR / "split.csv"

# 실험 착수 시 확정한 데이터. 값이 바뀌면 README와 함께 갱신한다.
CREATORS_CSV_SHA256 = "70472395d9154f1c51a161d89514664d2cb03188a34a50a7216c99999ca8f5dd"
# 2026-09-28: 85개 행에서 누락됐던 declared 칸(빈 문자열)을 명시적으로 채운 서식 수정.
# gold·declared 값 자체는 바뀌지 않았다 (csv.DictReader가 None -> ""로 취급하던 것을 파일에 반영).
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
