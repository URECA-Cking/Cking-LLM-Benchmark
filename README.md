# Cking LLM Benchmark

Ticle(Cking) 2차 MVP **AI 크리에이터 추천**의 임베딩 모델과 추천 방식을 같은 데이터로 비교하는 Python 3.11+ 오프라인 벤치마크입니다. 앱·DB·배치와 무관하며, 결과는 **경향 확인용**이라 확정 성능 수치로 발표하지 않습니다. 모든 메서드는 역할을 설명하는 한글 docstring을 포함합니다.

## 준비와 실행

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# OPENAI_API_KEY 설정
```

`bge-m3`는 첫 실행 때 약 2.3GB를 내려받습니다.

## 파이프라인 실행

순서대로 실행합니다. `embed`와 `tag-llm`은 OpenAI API를 실제로 호출하므로(비용은 합쳐서 1달러 미만 수준), 실행 전 `.env`의 `OPENAI_API_KEY`와 예상 비용을 확인하세요.

```bash
python3 -m src.pipeline embed             # 두 임베딩 모델로 카테고리·크리에이터 100명 인코딩 (API 호출)
python3 -m src.pipeline tag-llm           # LLM 후보 모델을 전원에게 2회씩 태깅, dev 정확도로 하나 선택 (API 호출)
python3 -m src.pipeline select-params     # zero-shot tau와 M3·M4·R2 bonus를 dev로 결정 (API 호출 없음)
python3 -m src.pipeline candidates        # 11개 설정 × 크리에이터 100명의 상위 20명 계산 (API 호출 없음)
python3 -m src.pipeline judge-sheet       # 쿼리 30명의 상위 5명을 합집합으로 모은 블라인드 판정 시트 생성
python3 -m src.pipeline report            # 사람 판정 없이 계산되는 E1(태깅 정확도)·E3(대표 사례) 출력
# results/judge_sheet.csv의 score 열(0/1/2)을 직접 채운 뒤:
python3 -m src.pipeline score-judgments   # E2(관련도·nDCG·무관 비율·짝비교) 계산
```

각 단계의 산출물은 `results/`(git 제외)에 쌓입니다. `results/cache/`에는 임베딩 벡터와 LLM 태깅 원본이 저장되어, 같은 단계를 다시 돌려도 API를 재호출하지 않습니다.

## 코드 규칙

- 모델명·경로·가격·고정 파라미터는 `src/config.py` 한곳에 둡니다. `.env`에는 API 키만 둡니다
- 실행은 `python3 -m src.<모듈>` 형태로 합니다
- 테스트는 `tests/`에 pytest로 작성합니다
- 모델·방식 채택 근거는 `docs/<주제>-selection.md`로 분리해 기록합니다
- 모델 호출 결과·raw·summary는 `results/`에 저장하고 커밋하지 않습니다

## 비교 대상

**임베딩 모델**

| 모델 | 실행 |
| --- | --- |
| OpenAI `text-embedding-3-small` | 외부 API |
| `BAAI/bge-m3` | 로컬 (sentence-transformers) |

**추천 방식** (크리에이터 간 유사도)

| ID | 방식 | 유사도 |
| --- | --- | --- |
| M1 | 태깅(zero-shot) 단독 | 카테고리 설명문과 비교해 태그 부여 → 공유 태그 Jaccard |
| M2 | 임베딩 단독 | 소개글 벡터 코사인 |
| M3 | 임베딩 + zero-shot 태그 보정 | 코사인 + (zero-shot 태그 공유 시 +bonus) |
| M4 | 임베딩 + LLM 태그 보정 | 코사인 + (OpenAI LLM 태그 공유 시 +bonus) |
| R1 | (참고) 입력 분야 단독 | `declared` Jaccard |
| R2 | (참고) 임베딩 + 입력 분야 보정 | 코사인 + (`declared` 공유 시 +bonus) |

M1~M3·R2는 모델 2종 각각, M4의 LLM 태그는 1회 생성해 두 모델에 공통 사용, R1은 모델 무관 → **11개 설정**. R1·R2는 사람 입력 분야를 쓰므로 실제보다 유리하게 나올 수 있어 **상한선 참고치**로만 해석합니다.

## 평가

| ID | 항목 | 방법 |
| --- | --- | --- |
| E1 | 태깅 정확도 | Top-1 정확도, Top-3 포함률, 혼동 쌍, UNCLASSIFIED 비율, LLM 2회 일관성 (자동 계산) |
| E2 | 유사 크리에이터 품질 | 쿼리 30명 × 설정별 상위 5명 합집합을 블라인드 0/1/2점 판정 → 평균 관련도@5, nDCG@5, 무관 비율@5, 빈 결과 비율, 쿼리별 짝비교 + 부트스트랩 신뢰구간 |
| E3 | 대표 사례 | 동의어 / 경품 잡음 / 분야 넘기 / 정보 부족 / 분야 오입력 통과 여부 |
| E4 | 운영 | 임베딩 소요 시간, 벡터 차원·크기, 비용, 메모리 |

## 데이터

| 파일 | 내용 |
| --- | --- |
| `data/categories.csv` | 카테고리 10개 + zero-shot 태깅용 설명문 (taxonomy v0.1) |
| `data/creators.csv` | 가상 크리에이터 100명 (일반 85 + 어려운 사례 X01~X15) |
| `data/split.csv` | dev 30 / test 70, `query=Y`인 test 30명이 평가 쿼리 |

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
creators.csv sha256 70472395d9154f1c51a161d89514664d2cb03188a34a50a7216c99999ca8f5dd
split.csv    sha256 cb95606e41c46dea323d74db75ee8b8edb4e136e8eb11e875850ab574d5545b6
```
