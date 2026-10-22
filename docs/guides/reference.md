# 지표·데이터·문제 해결

[문서 안내](../README.md) · [현재 결론](../results/recommendation-decision.md)

## 6. 문제 해결

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'dotenv'` | 가상환경을 안 켰거나 `pip install -r requirements.txt`를 안 함 | `source .venv/bin/activate` 후 다시 설치 |
| `openai.OpenAIError: Missing credentials` | `.env` 파일이 없거나 `OPENAI_API_KEY`가 비어 있음 | `cp .env.example .env` 후 키 입력, 같은 터미널 세션에서 재실행 |
| `ValueError: 데이터 파일이 실험 착수 시 확정본과 다릅니다` | `data/creators.csv` 또는 `data/split.csv`를 수정함 | 의도한 수정이 아니면 `git checkout -- data/`로 되돌림. 의도한 수정이면 `src/config.py`의 hash를 갱신 |
| `tag-llm`에서 특정 모델이 400 에러 | 그 모델이 `temperature=0`을 지원하지 않음 (예: `gpt-5-nano`) | 이미 후보에서 제외되어 있음. 새 모델을 추가할 때는 `src/config.py`에서 먼저 단독 테스트 |
| `score-judgments`에서 `ValueError: 판정이 비어 있는 행이 있습니다` | `judge_sheet.csv`를 다 안 채움 | 5장의 방법 A/B/C 중 하나로 마저 채움 |
| `RuntimeError: ... 임베딩 캐시가 없거나, 저장이 끝나지 않았거나, 현재 데이터와 다릅니다` / `llm_tags_..._run0.json이 없거나, 저장이 끝나지 않았거나...` | 후속 단계(`select-params` 등)는 `embed`·`tag-llm`이 **끝까지 저장하고 남기는 입력 해시**가 맞을 때만 캐시를 읽습니다. `--force` 갱신이 도중에 실패했거나 데이터가 바뀌면 해시가 없어 막습니다 | 안내대로 `python3 -m src.pipeline embed`(또는 `tag-llm`)를 다시 실행 (API 벡터는 `api-select-params --force`) |
| 로컬 모델 다운로드가 느리거나 멈춤 | 첫 실행 때 bge-m3·kure-v1·qwen3-embedding-0.6b·bge-reranker-v2-m3 가중치(합쳐서 약 8GB)를 Hugging Face에서 내려받는 중 | 네트워크 확인 후 재실행 (다운로드는 이어받기됨) |

## 7. 데이터 설명

| 파일 | 내용 |
| --- | --- |
| `data/categories.csv` | 카테고리 10개 + zero-shot 태깅용 설명문 (taxonomy v0.1) |
| `data/creators.csv` | 가상 크리에이터 100명 (일반 85 + 어려운 사례 X01~X15) |
| `data/split.csv` | dev 30 / test 70, `query=Y`인 test 30명이 평가 쿼리 |
| `data/taste_queries.csv` | 사용자 취향 요약문 쿼리 90개(프로필 30 × 표현 3종, `python3 -m src.generate_taste --force`로 새로 생성, 평가에는 커밋본 사용) |
| `data/e2_judgments.json` | E2 판정 점수 681쌍과 쿼리 30명의 설정별 상위 5 후보(`paired-diff`의 입력, `paired-diff --export`로 생성, 해시 고정) |
| `data/creators_large.csv` | dev 규모 민감도 실험용 합성 300명(`python3 -m src.generate_large --force`로 새로 생성, 평가에는 쓰지 않음) |

`creators.csv` 컬럼

| 컬럼 | 의미 | 모델 입력 |
| --- | --- | --- |
| `bio` | 소개글 | ✅ |
| `events` | 이벤트 제목 (`/` 구분) | ✅ (소개 없을 때 대체) |
| `gold` | 정답 분야 (`\|` 구분). 주목적 하나, 대등할 때만 복수 | ❌ 평가용 |
| `declared` | 크리에이터가 입력한 분야. 비어 있으면 `gold`와 같음 | R1·R2만 |
| `subtopic`, `note` | 판정 참고용 | ❌ |
| `written_by` | `claude` / `user` (직접 작성 14%) | ❌ |

- `gold`·`declared`·`split`은 실험 전 확정본입니다. 수정하면 아래 hash를 갱신하고 사유를 남깁니다
- τ·bonus·설명문 조정은 **dev로만**, 성적표는 test로 매깁니다
- 임베딩 입력에 카테고리 이름을 넣지 않습니다 (zero-shot 태깅이 자기 분야를 되맞히는 순환 방지)

```
creators.csv sha256 e4c14bf750a17ec33d430b1958e397c986238dcd3073a41be566cec32ea2e01a
split.csv    sha256 cb95606e41c46dea323d74db75ee8b8edb4e136e8eb11e875850ab574d5545b6
taste_queries.csv sha256 ed3c37d709dd12a4e0bae59018038dce8a0c2ea9e19afde934b3e2ab31df9afb
creators_large.csv sha256 152aeb4be4dce9cc65fbd6261ddc894965ee611a461c437b8354d5ce08869aac
e2_judgments.json sha256 07c4274a356c823f39dae6a5af31fc7d37d2538ed4f0ec5227dc6a674de272ea
```

## 8. 비교 대상 방식

**임베딩 모델**

| 모델 | 실행 |
| --- | --- |
| OpenAI `text-embedding-3-small` | 외부 API |
| `BAAI/bge-m3` | 로컬 (sentence-transformers) |
| `nlpai-lab/KURE-v1` (bge-m3의 한국어 파인튜닝) | 로컬 (sentence-transformers) |
| `Qwen/Qwen3-Embedding-0.6B` | 로컬 (sentence-transformers) |

**리랭커** (임베딩이 아니라 (쿼리, 후보) 쌍을 직접 채점하는 cross-encoder, M5 전용)

| 모델 | 실행 |
| --- | --- |
| `BAAI/bge-reranker-v2-m3` | 로컬 (sentence-transformers CrossEncoder) |

**추천 방식** (크리에이터 간 유사도)

| ID | 방식 | 유사도 |
| --- | --- | --- |
| M1 | 태깅(zero-shot) 단독 | 카테고리 설명문과 비교해 태그 부여 → 공유 태그 Jaccard |
| M2 | 임베딩 단독 | 소개글 벡터 코사인 |
| M3 | 임베딩 + zero-shot 태그 보정 | 코사인 + (zero-shot 태그 공유 시 +bonus) |
| M4 | 임베딩 + LLM 태그 보정 | 코사인 + (OpenAI LLM 태그 공유 시 +bonus) |
| M5 | bge-m3 M2 후보 재정렬 (bge-m3 전용) | M2 코사인 상위 20명을 bge-reranker-v2-m3로 다시 채점·정렬 |
| R1 | (참고) 입력 분야 단독 | `declared` Jaccard |
| R2 | (참고) 임베딩 + 입력 분야 보정 | 코사인 + (`declared` 공유 시 +bonus) |

M1~M3·R2는 모델 4종 각각, M4의 LLM 태그는 1회 생성해 네 모델에 공통 사용, R1은 모델 무관, M5는 bge-m3 M2 후보를 재정렬(모델 무관하게 1회) → **22개 설정**. R1·R2는 사람 입력 분야를 쓰므로 실제보다 유리하게 나올 수 있어 **상한선 참고치**로만 해석합니다.

## 9. 평가 지표

| ID | 항목 | 방법 |
| --- | --- | --- |
| E1 | 태깅 정확도 | Top-1 정확도, Top-3 포함률, 혼동 쌍, UNCLASSIFIED 비율, LLM 2회 일관성 (자동 계산) |
| E2 | 유사 크리에이터 품질 | 쿼리 30명 × 설정별 상위 5명 합집합을 0/1/2점 판정(자동 또는 사람) → 평균 관련도@5, nDCG@5, 무관 비율@5, 쿼리별 짝비교 + 부트스트랩 신뢰구간 |
| E3 | 대표 사례 | ①동의어 ②경품 잡음 ③분야 넘기 ④소개 부족(한 줄·이벤트만) 통과 여부, ⑤컷오프 효과(무관 쌍 제거율·관련 쌍 보존율·빈 결과 비율). ④·⑤의 컷오프는 `select-params`가 dev로 방식별로 고르며 M5는 리랭커 점수로 따로 고름 |
| E4 | 운영 | 임베딩 소요 시간, 벡터 차원·크기, 비용 |

## 10. 코드 규칙

- 모델명·경로·가격·고정 파라미터는 `src/config.py` 한곳에 둡니다. `.env`에는 API 키만 둡니다
- 실행은 `python3 -m src.<모듈>` 형태로 합니다
- 테스트는 `tests/`에 pytest로 작성합니다. 순수 로직은 반드시 테스트를 붙이고, 실제 API를 부르는 클라이언트는 가짜 클라이언트로 배선만 검증합니다
- 모델·방식 채택 근거는 `docs/<주제>-selection.md`로 분리해 기록합니다
- 모델 호출 결과·raw·summary는 `results/`에 저장하고 커밋하지 않습니다. 다만 커밋된 통계 결과를 재현하는 데 꼭 필요한 최소 입력(`data/e2_judgments.json`)만 해시를 고정해 `data/`에 둡니다
