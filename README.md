# Cking LLM Benchmark

Ticle(Cking) 2차 MVP **AI 크리에이터 추천**의 임베딩 모델과 추천 방식을 같은 데이터로 비교하는 Python 오프라인 벤치마크입니다. 앱·DB·배치와 무관하며, 결과는 **경향 확인용**이라 확정 성능 수치로 발표하지 않습니다. 모든 메서드는 역할을 설명하는 한글 docstring을 포함합니다.

## 준비와 실행

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# OPENAI_API_KEY 설정
```

`bge-m3`는 첫 실행 때 약 2.3GB를 내려받습니다. 실행 명령은 파이프라인 구현(#1) 후 이 절에 추가합니다.

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
creators.csv sha256 7caea476c6f813efafb8a9d3d7e4d616460a22dfddf9a06813db86b05ed06493
split.csv    sha256 cb95606e41c46dea323d74db75ee8b8edb4e136e8eb11e875850ab574d5545b6
```
