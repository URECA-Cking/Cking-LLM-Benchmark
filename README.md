# Cking LLM Benchmark

Ticle(Cking) 2차 MVP **AI 크리에이터 추천**의 임베딩 모델과 추천 방식을 같은 데이터로 비교하는 Python 3.11+ 오프라인 벤치마크입니다. 앱·DB·배치와 무관한 독립 실험 저장소이며, 결과는 **경향 확인용**이라 확정 성능 수치로 발표하지 않습니다. 함수는 역할을 설명하는 docstring을 포함하며 기존 한글과 새 코드의 영어 설명을 허용합니다.

## 현재 잠정 선택: `bge-m3` + M4

최종 사람 평가에서 **일반 소개는 M4, 짧은 소개는 M2를 잠정 선택**했습니다.
일반 소개75개에서 M4의 P@5는0.827이며 M3 대비+0.072입니다. H4는0.835지만 M4 대비 추가 개선+0.008이 작고 불확실해 세부 태그를 기본 채택하지 않습니다.
짧은 소개25개에서는 차이가 불명확해 단순한 M2를 잠정 선택합니다.

[최종 결과·선택 근거](docs/final-recommendation-eval.md#14-최종-사람-평가-결과와-선택)를 먼저 읽으세요.
이번 평가는 같은4,743개 후보 풀의 미평가 입력100개·중복 제거801쌍을 사용자1명이 평가한 오프라인 비교입니다.
실제 선택 태그·즐겨찾기·신규 채널 일반화·서비스 전환 성능은 검증하지 않았습니다.

서비스 적용의 기본 후보는 M4입니다. 짧거나 정보가 부족한 소개의 처리, API 임베딩 정합성·비용은 적용 단계에서 검증해야 합니다.
최종 비교는 로컬 bge-m3 캐시로 수행했습니다. 기존 API 비교 근거는 [임베딩 모델 선정 기록](docs/embedding-method-selection.md)에 있으며 이번 결과와 구분합니다.
M4는 상위 태그 분류 LLM을 오프라인에서 사용하고 추천 조회 중 LLM을 호출하지 않습니다. 평가 전용 판정 LLM은 서비스 구성에 포함하지 않습니다.

직접 선택 세부 태그는 [별도 비교 진단](docs/selected-subtopic-eval.md)을 준비했습니다. 7개 실제 선택·53쌍의 사람 평가에서0.714→0.800이었으며 개선은 베이킹1개에 집중됐습니다.

[자동 조합 평가](docs/selected-subtopic-auto-eval.md)는170개 입력·1,262쌍을 완료했습니다. LLM 품질 평균은 개선됐지만 사람 판정과23/53쌍만 일치해 단독 채택 근거로 쓰지 않습니다.

## 목차

1. [사전 준비물](#1-사전-준비물)
2. [설치](#2-설치)
3. [전체 흐름 한눈에 보기](#3-전체-흐름-한눈에-보기)
4. [단계별 실행](#4-단계별-실행)
5. [판정하기 (E2)](#5-판정하기-e2)
6. [문제 해결](#6-문제-해결)
7. [데이터 설명](#7-데이터-설명)
8. [비교 대상 방식](#8-비교-대상-방식)
9. [평가 지표](#9-평가-지표)
10. [코드 규칙](#10-코드-규칙)
11. [실제 크리에이터 데이터로 비교](#11-실제-크리에이터-데이터로-비교)

---

## 1. 사전 준비물

| 항목 | 확인 방법 | 없으면 |
| --- | --- | --- |
| Python 3.11 이상 | `python3 --version` | `brew install python@3.12` (macOS) |
| OpenAI API 키 (결제 등록 완료) | — | [platform.openai.com](https://platform.openai.com/api-keys)에서 발급, Billing에서 결제수단 등록 |
| 디스크 여유 공간 약 9GB | — | `bge-m3`·`KURE-v1`·`Qwen3-Embedding-0.6B`·`bge-reranker-v2-m3` 모델을 로컬에 내려받는 데 필요 |
| (선택) 인터넷 | — | 임베딩·태깅·판정은 실제 OpenAI API를 호출합니다 |

> ⚠️ **비용 안내**: 이 저장소의 기본 파이프라인(embed ~ auto-judge)을 한 번 실행하면 **API 비용이 약 $0.19 (260원 안팎, 약 16~17분)** 발생할 것으로 예상합니다(실제 청구액은 미확인이며, 이 프로젝트에서 여러 차례 재판정하며 쌓인 누적 비용은 약 $0.22입니다). 선택 실험(`dev-sensitivity`, `taste-eval`·`taste-judge`)까지 모두 돌리면 약 $0.8까지 늘고 시간도 수십 분 더 걸립니다. 상세 내역은 [비용·소요 시간](docs/embedding-method-selection.md#비용소요-시간)을 참고하세요. 큰돈은 아니지만 **본인 API 키에서 실제로 빠져나가는 돈**이니, `.env`에 다른 사람 키를 쓰지 말고 본인 키로 실행하세요.

## 2. 설치

```bash
git clone https://github.com/URECA-Cking/Cking-LLM-Benchmark.git
cd Cking-LLM-Benchmark

python3 -m venv .venv
source .venv/bin/activate        # Windows는 .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

`.env` 파일을 열어 `OPENAI_API_KEY=` 뒤에 본인 키를 붙여넣습니다.

```
OPENAI_API_KEY=sk-...본인_키...
```

설치가 끝났는지 아래 명령으로 확인합니다.

```bash
python3 -m pytest -q
```

```
.................................................................  [100%]
N passed in ...s  # 개수·시간은 변경에 따라 달라집니다
```

**모든 테스트가 통과하면 준비 완료입니다. 테스트 개수는 변경에 따라 달라집니다.** 이 테스트들은 API를 호출하지 않는 순수 로직 검증이라 비용이 들지 않습니다.

> 💡 이후 모든 명령은 `source .venv/bin/activate`로 가상환경을 켠 상태에서 실행한다고 가정합니다. 터미널을 새로 열었다면 저장소 폴더에서 이 명령을 다시 실행하세요.

## 3. 전체 흐름 한눈에 보기

```
① embed          카테고리·크리에이터 100명을 임베딩 모델 4종으로 인코딩
② tag-llm        LLM 태깅 후보 2개를 dev로 비교해 하나 선택
③ select-params  zero-shot 태깅 기준값(τ)과 보정 가중치(bonus)를 dev로 결정
④ candidates     22개 설정 각각의 유사 크리에이터 후보 계산
⑤ judge-sheet    쿼리 30명의 상위 5명을 합집합해 "무엇을 판정할지" 목록 생성
⑥ (판정)          auto-judge(LLM 자동) 또는 judge_cli(사람 직접)로 관련도 0/1/2 채점  ← 5장 참고
⑦ score-judgments 판정 결과로 관련도·nDCG 계산 (E2)
⑧ report         판정 없이 자동 계산되는 태깅 정확도(E1)·대표 사례(E3 ①~⑤, 컷오프 효과) 출력
```

①~⑤, ⑦, ⑧은 순서대로 한 번씩만 실행하면 됩니다. ⑥은 상황에 맞는 방법을 5장에서 고르세요.

## 4. 단계별 실행

각 명령은 `results/`(git에 안 올라감) 아래에 결과를 저장합니다. **`embed`·`tag-llm`은 저장이 끝난 결과(입력 해시가 남아 있고 현재 데이터와 맞는 것)가 이미 있으면 재실행해도 API를 다시 부르지 않고 건너뜁니다** — 캐시를 무시하고 새로 계산하려면 `--force`를 붙입니다. 나머지 단계(`select-params`~`report`)는 API를 호출하지 않는 순수 계산이라 매번 다시 실행해도 비용이 들지 않으며, 위 두 단계가 **끝까지 저장한 캐시만** 읽습니다. `--force` 갱신이 도중에 실패했다면 후속 단계가 섞인 캐시를 쓰지 않고 다시 실행하라고 안내합니다.

### ① embed — 임베딩 생성 (API 호출, 약 $0.0001, 5초)

```bash
python3 -m src.pipeline embed
```

```
[embed] text-embedding-3-small: 크리에이터 100명, 카테고리 10개, dim=1536, 3.1초
[embed] bge-m3: 크리에이터 100명, 카테고리 10개, dim=1024, 1.7초
[embed] kure-v1: 크리에이터 100명, 카테고리 10개, dim=1024, 1.2초
[embed] qwen3-embedding-0.6b: 크리에이터 100명, 카테고리 10개, dim=1024, 2.4초
```

`bge-m3`·`kure-v1`·`qwen3-embedding-0.6b`를 **처음 실행할 때만** 모델 가중치(로컬 3종 합쳐 약 6GB)를 자동으로 내려받습니다. 몇 분 걸릴 수 있습니다. 이미 계산된 벡터가 있으면 `[embed] {모델}: 캐시된 벡터 사용 (재계산하려면 --force)`만 찍고 끝납니다.

### ② tag-llm — LLM 태깅 모델 선택 (API 호출, 약 $0.02, 5~6분)

```bash
python3 -m src.pipeline tag-llm
```

크리에이터 100명을 **후보 LLM 2종 × 2회씩(총 400번)** 태깅하고, dev 30명 정확도로 더 나은 모델을 자동으로 고릅니다. 2회 태깅하는 이유는 같은 입력에 매번 같은 태그가 나오는지(일관성)를 확인하기 위해서입니다. run 파일이 이미 있으면 그 run은 다시 태깅하지 않고 넘어갑니다(`--force`로 재계산).

```
[tag-llm] gpt-5.4-nano-2026-03-17 run 1/2 완료
[tag-llm] gpt-5.4-nano-2026-03-17 run 2/2 완료
[tag-llm] gpt-4.1-nano-2025-04-14 run 1/2 완료
[tag-llm] gpt-4.1-nano-2025-04-14 run 2/2 완료
[tag-llm] dev 지표로 선택된 모델: gpt-5.4-nano-2026-03-17
```

### ③ select-params — 기준값 결정 (API 호출 없음, 즉시)

```bash
python3 -m src.pipeline select-params
```

```
[select-params] text-embedding-3-small: tau=0.2706 bonus_m3=0.2 bonus_m4=0.3 bonus_r2=0.3
   컷오프 m1=1.000 m2=0.424 m3=0.624 m4=0.415 r2=0.415
[select-params] bge-m3: tau=0.4801 bonus_m3=0.1 bonus_m4=0.2 bonus_r2=0.2
   컷오프 m1=0.500 m2=0.506 m3=0.597 m4=0.577 r2=0.577 m5=0.002
[select-params] kure-v1: tau=0.4745 bonus_m3=0.1 bonus_m4=0.1 bonus_r2=0.1
   컷오프 m1=0.500 m2=0.503 m3=0.529 m4=0.529 r2=0.529
[select-params] qwen3-embedding-0.6b: tau=0.3587 bonus_m3=0.2 bonus_m4=0.2 bonus_r2=0.2
   컷오프 m1=0.500 m2=0.377 m3=0.487 m4=0.388 r2=0.388
```

### 선택: memory — 로컬 모델 메모리 측정 (API 호출 없음, 수 분)

```bash
python3 -m src.pipeline memory
```

로컬 임베딩 모델 3종, 리랭커, M5 구성(bge-m3 + 리랭커를 함께 로드)의 최대 메모리를 새 프로세스에서 하나씩 재서 `results/e4_embedding.json`의 `memory`에 기록합니다. 기본은 CPU 기준(서버 메모리 계획용)이고 `--device auto`는 이 장비의 가속기 기준입니다. ①을 먼저 실행하지 않아도 되지만 모델이 내려받아져 있어야 합니다.

### 선택: api-parity — 로컬 bge-m3 ↔ API bge-m3 비교 (API 호출 1센트 미만, 수 초)

```bash
python3 -m src.pipeline api-parity
```

서버에 모델을 올리지 않고 같은 bge-m3를 API(DeepInfra)로 써도 되는지 확인합니다. ①이 만든 로컬 벡터 캐시와 API 벡터를 두 단계로 비교해 `results/api_parity.json`에 기록합니다.

1. **유사도 비교:** 크리에이터끼리·크리에이터↔카테고리 유사도의 최대 차이와 상위 5 이웃 겹침. 차이가 0.01 이하이고 겹침 평균이 0.95 이상이면 `similarity_equivalent`입니다(기준은 `src/config.py`의 `PARITY_*`).
2. **같은 파라미터로 결과 비교:** `select-params`가 고른 tau·bonus·컷오프를 그대로 써서 zero-shot 태그 집합과 컷오프 뒤 상위 5 후보(M1~M4)가 순위 순서까지 로컬과 API에서 같은지 봅니다. 전부 같아야 `parameters_reusable`입니다.

tau·컷오프는 "이상이면 통과"라는 경계 판정이라, 유사도가 거의 같아도 경계에 놓인 크리에이터는 결과가 달라질 수 있습니다. 그래서 1번만 통과하면 "유사도는 같지만 일부 결과가 바뀔 수 있음"으로 안내합니다. `.env`에 `DEEPINFRA_API_KEY`가 필요하고 ①~③(`embed`, `tag-llm`, `select-params`)을 먼저 실행해야 합니다.

### 선택: api-select-params — API bge-m3 벡터로 tau·bonus·컷오프 다시 고르기 (API 호출 1센트 미만, 수 초)

```bash
python3 -m src.pipeline api-select-params
```

서버에서 로컬 bge-m3 대신 API를 쓰기로 했다면, 로컬 벡터로 고른 tau·bonus·컷오프 대신 API 벡터로 다시 고른 값을 씁니다. API 벡터를 받아 캐시(`results/cache/*bge-m3-api*`)한 뒤 `select-params`와 같은 방식으로 dev 30명만으로 고르고, 로컬 선택값과 비교해 `results/api_selected_params.json`에 남깁니다. API 벡터는 호출마다 조금씩 달라서 **처음 받은 벡터를 캐시해 두고 이후 실행은 그 벡터를 씁니다**(`--force`로 다시 받음). 캐시는 입력 텍스트와 모델명이 같으면 재사용하므로, **제공 업체의 서빙 방식이 바뀌었거나 모델을 바꿨을 때는 `python3 -m src.pipeline api-select-params --force`로 다시 받아야 합니다.** 벡터 파일이 하나라도 없거나 순서가 다르면 자동으로 다시 받습니다. `results/selected_params.json`은 바꾸지 않습니다. `.env`에 `DEEPINFRA_API_KEY`가 필요하고 ①~③(`embed`, `tag-llm`, `select-params`)을 먼저 실행해야 합니다.

### 선택: paired-diff — 방식 간 관련도 차이 (API 호출 없음, 즉시)

```bash
python3 -m src.pipeline paired-diff
```

같은 쿼리 30명에 대한 두 설정의 관련도@5 차이를 쿼리별로 구하고, 쿼리를 다시 뽑는(부트스트랩 2,000회) 방식으로 차이의 평균에 대한 95% 신뢰구간을 계산합니다(M4 − M3 임베딩별, `M4_bge-m3` − 상위권 설정). 구간이 0을 포함하면 차이를 확인하지 못한 것입니다. 입력은 커밋된 `data/e2_judgments.json`(해시 고정)이라 ①~⑦을 실행하지 않아도 됩니다. 이 파일을 판정 결과(`results/candidates.json`, `results/judge_sheet.csv`)에서 다시 만들려면 `python3 -m src.pipeline paired-diff --export`를 실행하고, 내용이 바뀌면 `src/config.py`의 `E2_JUDGMENTS_JSON_SHA256`과 결과 문서를 함께 갱신합니다. 결과는 `results/paired_diff.json`에 저장됩니다.

### 선택: 취향 쿼리 검증 — `taste-eval` → `taste-judge`

```bash
python3 -m src.pipeline taste-eval
python3 -m src.pipeline taste-judge
```

서비스가 사용자 취향 요약문을 쿼리로 크리에이터를 추천할 때를 가정해 커밋된 `data/taste_queries.csv`(90개, 키워드형·문장형·서술형, hash 고정)로 M2~M5를 검증합니다. `taste-eval`은 정답 분야 기준 P@5를 만들고(태깅 1센트 안팎, 나머지 로컬), `taste-judge`는 문장형 쿼리의 후보 쌍을 LLM으로 판정합니다(약 $0.12, 판정 모델·프롬프트가 바뀌면 다시 판정). ①~③을 먼저 실행해야 하며 결과는 `results/taste_*.json`에 저장됩니다. 해석은 `docs/embedding-method-selection.md`의 "사용자 취향 쿼리 검증" 절을 참고하세요.

쿼리를 **새로 만들 때만** `python3 -m src.generate_taste --force`를 실행합니다(gpt-5.4-mini 30회, 약 $0.03). `--force`는 확정 CSV 덮어쓰기를 허용할 뿐이고, 프로필별 생성 캐시(`results/cache/taste_gen/`)는 모델과 프롬프트(길이 요구 포함)가 그대로일 때만 재사용합니다 — 프롬프트를 바꾸면 자동으로 새로 만들고, 같은 프롬프트로 새 표본을 뽑으려면 그 캐시 폴더를 지우세요. 확정본을 덮어쓰므로 끝난 뒤 `src/config.py`의 `TASTE_QUERIES_CSV_SHA256`을 다시 고정해야 하고, 그렇지 않으면 이후 단계가 hash 불일치로 멈춥니다. 확정본을 평가하는 데는 이 단계가 필요 없습니다.

### 선택: dev-sensitivity — dev 규모 민감도 (API 호출 약 1센트, 수 분)

```bash
python3 -m src.pipeline dev-sensitivity
```

기존 dev 30명 + 합성 300명에서 크기별(30/60/120/240) 부분표본으로 tau·bonus를 골라 보고 얼마나 흔들리는지 `results/dev_sensitivity.json`에 저장합니다. ①~③을 먼저 실행해야 합니다. 결과 해석은 `docs/embedding-method-selection.md`의 "dev 규모 민감도" 절을 참고하세요.

### ④ candidates — 유사 크리에이터 후보 계산 (API 호출 없음, 몇 초)

```bash
python3 -m src.pipeline candidates
```

```
[candidates] 22개 설정 저장 완료
```

### ⑤ judge-sheet — 판정할 목록 만들기 (API 호출 없음, 즉시)

```bash
python3 -m src.pipeline judge-sheet
```

```
[judge-sheet] 681쌍. results/judge_sheet.csv의 score 열(0/1/2)을 채운 뒤 score-judgments를 실행하세요.
```

이 시점의 `results/judge_sheet.csv`는 `score` 열이 전부 빈칸입니다. **다음 5장에서 이 빈칸을 채우는 방법을 고릅니다.**

### ⑦ score-judgments — 관련도 계산 (API 호출 없음, 즉시)

5장에서 `score` 열을 다 채운 뒤 실행합니다.

```bash
python3 -m src.pipeline score-judgments
```

```
=== E2. 방식별 평균 관련도@5 / nDCG@5 / 무관 비율@5 (test 쿼리 30명) ===
   M1_text-embedding-3-small: 관련도=0.147 [0.060, 0.240]  nDCG=0.143  무관비율=0.867
   ...
   M4_bge-m3: 관련도=0.487 [0.333, 0.640]  nDCG=0.541  무관비율=0.587
   ...
```

### ⑧ report — 태깅 정확도·대표 사례 (API 호출 없음, 즉시)

```bash
python3 -m src.pipeline report
```

```
=== E1. zero-shot 태깅 정확도 (test) ===
-- text-embedding-3-small (tau=0.2706)
   Top-1 정확도: 0.729
   Top-3 포함률: 0.886
   ...
-- bge-m3 (tau=0.4801)
   Top-1 정확도: 0.871
   Top-3 포함률: 1.000
   ...

=== E3. 대표 사례 (top-5, 설정 전체) ===
-- M4_bge-m3
   [PASS] ① 동의어 F01<->F02 -> top5=[...]
   ...
-- M2_kure-v1
   [PASS] ① 동의어 F01<->F02 -> top5=[...]
   ...
```

## 5. 판정하기 (E2)

`judge-sheet` 직후 `results/judge_sheet.csv`는 681쌍인데 `score` 열이 비어 있습니다. 이 빈칸을 채우는 세 가지 방법이 있습니다. **상황에 맞게 하나만 골라도 되고, 순서대로 다 해도 됩니다.**

### 방법 A. 자동 판정 (추천 — 대부분 이 방법으로 충분)

```bash
python3 -m src.pipeline auto-judge
```

LLM(`gpt-5.4-mini`, 태깅에 쓴 모델보다 강한 모델)이 681쌍을 대신 채점합니다. **비용 약 $0.17, 소요 약 9~10분.** 중간에 멈춰도 이미 채운 건 저장돼 있어서 다시 실행하면 **비어 있는 것만** 이어서 채웁니다.

```
[auto-judge] 전체 681쌍 중 0쌍 완료, 681쌍 자동 판정 시작 (모델: gpt-5.4-mini-2026-03-17)
[auto-judge] 50/681 완료
...
[auto-judge] 완료. 비용 약 $0.17. 일부를 src.judge_cli로 직접 재판정해 일치율을 확인하는 것을 권장합니다.
```

### 방법 B. 자동 판정 신뢰도 확인 (선택, 권장 — 10~15분)

방법 A 다음에 하면 좋습니다. M4가 M3·R2 각각과 **top-5를 다르게 고른 쌍**(양쪽 차집합)을 모으면 보통 100쌍을 넘습니다. 전부 보기엔 부담이 커서, 고정 시드로 무작위 **30쌍만** 추려 사람이 직접 봐서 자동 판정을 신뢰할 수 있는지 확인합니다.

```bash
python3 -m src.pipeline spot-check
```

```
[spot-check] M4_bge-m3를 ['M3_bge-m3', 'R2_bge-m3']와 각각 양쪽 차집합으로 비교해 다른 30쌍 (전체 139쌍 중 무작위 표본)을 results/spot_check.csv에 저장했습니다.
python3 -m src.judge_cli --file results/spot_check.csv 로 채운 뒤 python3 -m src.pipeline spot-check-report 를 실행하세요.
```

안내대로 대화형 CLI로 30쌍을 직접 채점합니다 (아래 "대화형 판정 CLI 사용법" 참고).

```bash
python3 -m src.judge_cli --file results/spot_check.csv
```

다 채운 뒤 일치율을 확인합니다.

```bash
python3 -m src.pipeline spot-check-report
```

```
=== spot-check 일치율 (M4_bge-m3 vs ['M3_bge-m3', 'R2_bge-m3']만 고른 후보 30쌍) ===
   완전 일치율: 0.77
   ±1 이내 일치율: 1.00
   평균 절대 오차: 0.23
   불일치: K07::T07  사람=1  자동=0
   ...
```

`±1 이내 일치율`이 1.0에 가까우면(즉 사람과 자동 판정이 2점 이상 차이나는 극단적 불일치가 없으면) 방법 A의 결과를 신뢰할 근거가 됩니다.

### 방법 C. 681쌍 전부 사람이 직접 판정 (가장 정확, 4~5시간)

빠르게 확인하고 싶다면 방법 A만으로 충분합니다. 하지만 **사람 판정만으로 결과를 내고 싶다면** 이 방법을 씁니다.

```bash
python3 -m src.judge_cli
```

### 대화형 판정 CLI 사용법 (`src.judge_cli`)

`judge_cli`는 쌍 하나씩 소개글을 보여주고 점수를 입력받습니다. **`judge_sheet.csv`와 `spot_check.csv` 둘 다 이 도구로 채웁니다** — 어떤 파일을 채울지는 `--file` 옵션으로 정합니다(생략하면 `results/judge_sheet.csv`).

```
전체 681쌍 중 0쌍 완료, 681쌍 남음

[1/681] 쿼리 B02 데일리메이크업쌤
  소개: 출근 전 10분이면 끝나는 데일리 메이크업을 알려드려요. ...
  후보 B01 피부과가고싶은날
  소개: 민감성 피부 스킨케어 루틴이랑 성분 분석해요. ...
점수 (0=무관 1=관련 2=매우관련, s=건너뛰기, q=저장 후 종료): 
```

| 입력 | 동작 |
| --- | --- |
| `0` | 무관 — 분야도 다르고 내용도 안 겹침 |
| `1` | 관련 있음 — 큰 분야는 같은데 결이 다름, 또는 분야는 달라도 내용이 실제로 겹침 |
| `2` | 매우 관련 — 세부 주제가 거의 같음 |
| `s` | 애매하면 건너뛰기 (다음에 다시 물어봄) |
| `q` | 저장하고 종료 |

- **답할 때마다 즉시 파일에 저장**됩니다. `Ctrl+C`로 강제 종료해도 그때까지 답한 내용은 안전합니다.
- 다시 실행하면 **비어 있는 것부터** 이어서 물어봅니다.
- 같은 쿼리의 후보들이 묶여서 나오므로, 한 쿼리가 끝나는 지점에서 쉬는 게 편합니다.
- 어떤 방식(M1~R2)이 이 후보를 뽑았는지는 **보여주지 않습니다.** 편향 없이 내용만 보고 판정하기 위해서입니다.

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

## 11. 실제 크리에이터 데이터로 비교

합성 100명 결과가 실제 소개글에서도 유지되는지 보기 위해, 외부 폴더의 실제 YouTube 채널 데이터로 M2·M3·M4를 같은 조건에서 비교하는 별도 도구(`src/real_eval.py`)가 있습니다. 소개글 원문과 정답 라벨은 **저장소에 커밋하지 않고** `--data-dir`(또는 `REAL_DATA_DIR`)의 외부 폴더에서만 읽으며, 산출물은 `results/real/`(git 제외)에 저장합니다.

```bash
export REAL_DATA_DIR=<데이터 폴더>
python3 -m src.real_eval embed && python3 -m src.real_eval tag-llm && python3 -m src.real_eval select-params
python3 -m src.real_eval candidates && python3 -m src.real_eval judge-sheet && python3 -m src.real_eval auto-judge
python3 -m src.real_eval human-sheet   # 사람이 채운 뒤 human-agree, 마지막에 score
```

데이터 폴더 구조, 단계별 설명, **결과를 보기 전에 정한 판정 기준**(M4 − M3 ≥ +0.05이고 95% 구간의 하한이 0보다 클 때만 M4 채택), 비용 추정, 해석 주의는 [`docs/real-data-eval.md`](docs/real-data-eval.md)에, 분야 목록·방식별 GPT 사용·평가 범위·정답 점검·다음 실험은 [`docs/real-data-scope.md`](docs/real-data-scope.md)에 있습니다. 이 도구는 API 비용이 들고(`tag-llm`·`auto-judge` 합쳐 대략 $2~3 추정) 기존 `pipeline` 결과와 테스트에는 영향을 주지 않습니다.

## 12. 가입·즐겨찾기 추천 흐름과 세부 주제 실험

태그 선택·크리에이터 선택·즐겨찾기 기반 추천의 후속 탐색 실험을 완료했습니다.

**처음 읽을 문서: [결론·방법·핵심 결과·다음 단계 요약](docs/recommendation-results.md)**

- [입력 흐름·초기 탐색](docs/recommendation-flow-eval.md): 초기 방법 비교와 표본의 한계.
- [동일 입력 M2·M3·M4 비교](docs/recommendation-m234-eval.md): 468개 입력, 2,801쌍 판정.
- [세부 주제 정의](docs/creator-subtopics.md): 상위 17개·세부 85개·콘텐츠 형식 10개.
- [계층 추천 결과](docs/hierarchical-subtopic-eval.md): 기존 입력 유지, 3,709쌍 판정 및 모든 민감도 조건.
- [최종 실험 설계](docs/final-recommendation-eval.md): 새 표본·사람 평가·정보 부족 처리·채택 규칙.
- [보정·가산점 조건 비교·사람 점검](docs/bonus-ablation-eval.md): 네 조건 비교와 터미널 점검 결과·최종 실험 초안.
- [검증·일관성 보고서](docs/recommendation-validation.md): 테스트 출력, 고정 조건, 조건 변경, 재현 한계.

일반 소개글의 LLM P@5는 기존 M4 0.641, 총 가산점 유지 H4 0.655, 가산점이 최대 1.5배인 주 조건 H4 0.664입니다. 주 조건의 상승을 세부 정보만의 효과로 단정하지 않습니다.
같은 표본을 관찰하며 발전시킨 탐색 결과이며 사람 평가·실제 사용자 효과·확정 채택은 미검증입니다.
실제 소개글·개별 판정은 Git에 포함하지 않으므로 외부 v4 데이터와 로컬 캐시가 있어야 수치를 재현할 수 있습니다.
