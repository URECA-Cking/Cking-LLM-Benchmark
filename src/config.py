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
# 사용자 취향 요약문 쿼리(이슈 #12). 소개글 → 크리에이터가 아닌 취향 → 크리에이터 E2E 검증용이다.
TASTE_QUERIES_CSV = DATA_DIR / "taste_queries.csv"

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

# taste_queries.csv의 확정 hash. 생성 직후 고정한다(None이면 검증하지 않음).
TASTE_QUERIES_CSV_SHA256: str | None = "ed3c37d709dd12a4e0bae59018038dce8a0c2ea9e19afde934b3e2ab31df9afb"

# 방식 간 관련도 차이(쿼리별 짝 차이)의 입력. results/는 커밋하지 않으므로, 필요한 부분(쿼리 30명의 설정별 상위 5 후보와
# 판정 점수)만 이 파일로 고정한다. 값이 바뀌면(`paired-diff --export`로 다시 만들면) 이 hash와 docs를 함께 갱신한다.
E2_JUDGMENTS_JSON = DATA_DIR / "e2_judgments.json"
E2_JUDGMENTS_JSON_SHA256: str | None = "07c4274a356c823f39dae6a5af31fc7d37d2538ed4f0ec5227dc6a674de272ea"

TAXONOMY_VERSION = "v0.1"

PRICING_CHECKED_AT = "2026-09-28"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
OPENAI_EMBEDDING_PRICE_PER_1M = 0.02  # USD / 1,000,000 input tokens

# 로컬 bge-m3와 같은 가중치를 API로 제공하는 호스팅(DeepInfra, OpenAI 호환). 가격은 2026-09-29 공식 문서 기준
API_BGE_M3_BASE_URL = "https://api.deepinfra.com/v1/openai"
API_BGE_M3_MODEL = "BAAI/bge-m3"
API_BGE_M3_PRICE_PER_1M = 0.01  # USD / 1,000,000 input tokens
# 로컬↔API 유사도 동등성 판정 기준. 서버 정밀도(fp16 등) 차이로 유사도가 흔들려도 bonus 그리드 간격(0.1)보다
# 한 자릿수 작고 상위 5 이웃이 거의 그대로면 "유사도가 같다"고 본다. 이 기준은 tau·컷오프 같은 경계 판정의
# 결과까지 보장하지 않아, 파라미터 재사용 여부는 같은 파라미터로 태그·후보를 직접 비교해(compare_decisions) 따로 본다
PARITY_MAX_ABS_DIFF = 0.01
PARITY_MIN_TOP5_OVERLAP = 0.95

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

# Qwen3 쿼리 instruct 실험용(이슈 #12): 등록된 웹 검색용 프롬프트 대신 취향 → 크리에이터 과제에 맞춘 문구
TASTE_QWEN3_INSTRUCT = "Instruct: 사용자의 취향 설명을 보고 그 취향에 맞는 크리에이터 소개를 찾으세요\nQuery:"
TASTE_JUDGE_STYLES = ("sentence",)  # LLM 판정은 비용 때문에 문장형 쿼리만 한다(키워드형·서술형은 정답 분야 기준 지표만)
TASTE_TOP_K = 5

TOP_N_STORED = 20  # 크리에이터별 저장할 유사 후보 수
JUDGE_TOP_K = 5  # 판정 시트에 합집합으로 모을 상위 개수
JUDGE_SHUFFLE_SEED = 20260928

# 실제 YouTube 채널 데이터로 M2·M3·M4를 비교하는 실행 도구(src/real_eval.py, 이슈 #28)의 고정값.
# 소개글 원문·정답 라벨은 저장소에 두지 않고 외부 폴더에서 읽는다(환경변수 또는 --data-dir).
REAL_DATA_ENV = "REAL_DATA_DIR"
REAL_DIR = RESULTS_DIR / "real"  # 산출물 폴더(results/는 git 제외)
REAL_SEED = 20260930
REAL_DEV_FRACTION = 0.3  # 정답이 있는 채널 중 tau·bonus를 고르는 dev 비율(분야별 층화)
REAL_SHORT_BIO_CHARS = 15  # 소개글이 이보다 짧으면 "짧은 소개글" 쿼리 묶음
REAL_N_REGULAR_QUERIES = 300  # 일반 쿼리(소개글 15자 이상) 수
REAL_N_SHORT_QUERIES = 100  # 짧은 소개글 쿼리 수
REAL_TOP_K = 5  # 판정하는 상위 후보 수(정밀도@5)
REAL_STORE_K = 10  # 후보 파일에 저장하는 상위 후보 수
REAL_MIN_GAIN = 0.05  # 사전 기준: M4 - M3 정밀도@5가 이 값 이상이고 95% 구간의 하한이 0보다 크면 M4 채택
REAL_HUMAN_SAMPLE = 150  # 사람이 채점할 무작위 쌍 수
REAL_HUMAN_AGREE_MIN = 0.80  # 사람과 LLM 판정 일치율이 이 값 이상이어야 LLM 판정을 쓴다
REAL_CONCURRENCY = 8  # LLM 호출 동시 실행 수
REAL_BONUS_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
REAL_TAG_MIN_INTERVAL = 0.5  # 태깅 호출 시작 간격(초). 분당 120건(약 15만 토큰)으로 gpt-5.4-nano의 분당 토큰 한도(TPM 200,000)를 넘지 않게 한다
REAL_JUDGE_MIN_INTERVAL = 0.25  # 판정 호출 시작 간격(초). 분당 240건
REAL_RETRIES = 6  # 일시 오류(호출 한도 초과 등) 재시도 횟수. 대기는 2·4·8·16·32초로 늘어난다
REAL_TARGETED_EACH = 25  # 결정을 가르는 쌍(M4만·M3만·M2+M4·M2+M3 조합)에서 조합마다 사람이 채점할 개수
REAL_REJUDGE_MODEL = "gpt-5.5-2026-04-23"  # 두 방식이 다르게 뽑은 쌍을 다시 판정하는 더 큰 모델(temperature 0을 지원하지 않아 기본값으로 호출한다)
