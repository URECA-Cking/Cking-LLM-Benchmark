# 관심 분야 M2·M3 블라인드 Judge 평가

[Issue #44](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/44)의 오프라인 평가 도구다.
#41의 같은 v0.2 taxonomyHash·Creator manifestHash에서 생성한 M2/M3 Top-10 합집합을
검증하고 평가한다. 운영 추천 생성·조회·BE 적재에서는 이 모듈을 호출하지 않는다.
Judge 점수를 추천 점수에 더하지 않는다.

## 실행 전에 고정하는 판단 기준

`prepare`가 다음 규칙을 `contract.json`과 `decision-plan.json`에 고정한다.
`run`과 `report`는 계약 해시와 규칙 전문이 현재 정본과 같은지 확인한다.
결과를 본 뒤 기준을 바꿔 같은 실험으로 재집계할 수 없다. 새 기준은 버전과 작업 디렉터리를
분리한 새 실험으로 다뤄야 한다.

- 주 지표는 17개 분야 평균 strict Precision@5다. M3−M2가 +0.03 이상이어야 한다.
- 최소 50개의 유일한 분야·Creator 쌍을 사람이 블라인드로 점검해야 한다.
- 사람 표본의 `relevance>=1` 적합 비율과 잡음 비율은 M2보다 악화되지 않아야 한다.
  Top-5와 Top-10 소속 표본을 각각 집계하고 두 범위에서 모두 확인한다. 한 방식의 검수
  표본이 없는 범위가 있으면 M3를 채택하지 않는다.
- Judge나 사람 점검이 미완료면 **M2 유지**다. 완료했지만 기준이 미충족이면 **추가 실험**으로
  표시하고 M2를 유지한다. 모두 충족하면 **M3 채택**으로 표시한다.
- paired bootstrap의 95% 구간은 불확실성을 보여주는 보조 지표다. 구간 하한을 별도 채택
  문턱으로 추가하지 않는다. 온라인 사용자 반응·개인화 효과까지 입증하는 기준은 아니다.
  `decision.primaryMetricUncertainty`에 주 지표의 평균·95% 구간을 함께 남긴다.
  하한이 0 미만이면 개선을 확정할 수 없다는 해석 문구를 로컬·공개 보고서 모두에 표시한다.
  이 경우에도 다른 사전 기준을 충족하면 M3 채택이 가능하며 통계적 개선 확정으로 해석하지 않는다.

판정 기준 버전은 `interest-decision-v1`이다. 평가 대상 M3는 #41의 `generationConfig`에서
tau·bonus·최대 태그·임베딩 버전을 가져온다. 기본값은 tau `0.4296248555`, bonus `0.1`,
최대 태그 `3`이다. 설정을 최적화한 값으로 주장하지 않는다.

## 입력 검증과 블라인드

입력은 #41의 `evaluation/judge-input.json`, `evaluation/provenance.json`과 원본 manifest다.
Top-N=10 평가 계약과 17개 분야의 M2·M3 완성 세대만 받는다. 분야 설명·이름은 v0.2
정본과, Creator 소개는 manifest와 다시 대조한다. 방식별 inputHash를 현재 설정으로
재계산하며 순위·점수·중복·후보 부족 수·쌍 출처·평가 입력 해시를 검증한다.

Judge 요청에는 분야 이름·설명과 익명 `candidateId`·소개만 포함한다.
Creator ID, 방식, 원본 점수, 순위, tau, bonus, manifest 해시는 보내지 않는다.
후보 순서는 사전 seed와 pair 해시로 결정적으로 섞고 특정 방식의 정렬을 따르지 않는다.
소개문 속 지시는 데이터로 취급하도록 프롬프트에 명시한다.

새 `pairId`와 캐시 파일명은 다음 identity의 SHA-256이다.

| 항목 | 목적 |
| --- | --- |
| 관심 분야 코드, 이름·설명 텍스트 해시 | 분야 정의 변경 시 무효화 |
| Creator ID, 소개 텍스트 해시 | 소개 변경 시 무효화 |
| taxonomyHash | 분류체계 변경 시 무효화 |
| Judge 모델, 프롬프트 버전·전문·해시 | 판정 설정 변경 시 무효화 |
| 응답 schema 해시, 섞기 seed, temperature 정책 | 출력·실행 설정 변경 추적 |

계약 전체 해시에는 manifestHash·양 방식 순위·생성 설정·판단 규칙도 포함한다.
manifest의 다른 Creator가 바뀌어도 현재 쌍의 원문과 Judge 계약이 같으면 그 쌍의 판정은
재사용할 수 있다. 집계는 현재 계약과 현재 순위에서 다시 계산하며 별도 집계 캐시를 쓰지 않는다.
옛 `pairId`만으로 판정을 가져오지 않는다. 다른 입력은 새 작업 디렉터리에서 `prepare`한다.

## 명령

다음 명령에서 모든 원문·판정·캐시는 Git 제외 `results/` 아래에 둔다.
PowerShell에서는 예시를 한 줄로 실행하거나 줄 연속 문자를 바꾼다.

```bash
# 준비만 수행: 외부 Judge 호출 없음
python -m src.interest_judge prepare \
  --evaluation-dir results/recommendation/interests/evaluation \
  --manifest results/recommendation/batch/creator-manifest.json \
  --work-dir results/recommendation/interest-eval-v1
```

기본 Judge 모델은 기존 `src/config.py`의 `OPENAI_JUDGE_MODEL`을 사용한다.
`--judge-model`로 실제 평가할 모델·스냅샷을 명시할 수 있다. 프롬프트는
`interest-relevance-v1`이며 본문은 `contract.json`에 기록한다.
temperature는 모델 제공자 기본값을 쓰고 이 정책도 지문에 기록한다.
동일 설정의 판정을 재사용한다는 뜻이며 외부 모델의 완전한 결정성을 보장하지 않는다.

**실제 Creator 소개와 분야 설명을 외부 API로 전송하는 전체 실행에는 별도의 데이터 전송·비용
승인이 필요하다. 아래 플래그는 그 승인을 이미 얻은 경우에만 사용한다.** 준비·보고서·사람
시트 명령은 유료 API를 호출하지 않는다. 인증은 환경 변수 `OPENAI_API_KEY`를 사용한다.

```bash
# 별도 승인 후에만 실행. --max-requests 1로 요청 건수를 제한할 수도 있다.
python -m src.interest_judge run \
  --work-dir results/recommendation/interest-eval-v1 \
  --approve-data-transfer-and-cost

# 같은 명령 재실행: 성공한 쌍은 재사용하고 미완료 쌍만 다시 요청
python -m src.interest_judge run \
  --work-dir results/recommendation/interest-eval-v1 \
  --approve-data-transfer-and-cost

# Judge 완료 후 최소 50쌍의 빈 사람 시트 생성
python -m src.interest_judge human-sheet \
  --work-dir results/recommendation/interest-eval-v1 --sample-size 50

# human-blank.csv를 human-filled.csv로 복사하고 사람 판정 칸을 작성한 뒤 집계
python -m src.interest_judge report \
  --work-dir results/recommendation/interest-eval-v1 \
  --human-filled results/recommendation/interest-eval-v1/human-filled.csv
```

공유 판정 캐시 기본 경로는 `results/recommendation/interest-judge-cache/`다.
`--cache-dir`로 별도 캐시를 지정할 수 있으며 run/human-sheet/report에 같은 경로를 사용한다.
CLI는 작업·캐시·사람 판정 경로가 resolve 후 `results/` 밖이면 거부한다.
부분 완료 run/report의 종료 코드는 1이다. 사람 입력 없이 Judge 보고서만 만들 수도 있으나
이때 결론은 M2 유지로 제한된다. 표본이 50쌍 미만이면 중복 행으로 채우지 않고 시트 생성을 거부한다.

## 응답 검증과 재시작

분야마다 미완료 후보만 요청한다. 모델 응답은 strict JSON Schema로 요구하며 로컬에서도
검증한다. 후보마다 다음 필드가 모두 있어야 한다.

```json
{
  "candidateId": "익명 후보 ID",
  "relevance": 2,
  "isNoise": false,
  "noiseType": "none",
  "reason": "분야에 직접 적합한 짧은 근거"
}
```

relevance는 bool을 제외한 정수 0/1/2다. 잡음 종류는 `none/ad/giveaway/namesake/other`,
근거는 양끝 공백 제거 후 1~500자다. isNoise와 noiseType의 일관성을 확인한다.
누락 필드·추가 필드·중복 JSON 키·중복 후보·알 수 없는 ID·누락 후보·NaN은 거부한다.
분야 응답 전체가 검증을 통과한 뒤에만 성공 쌍을 같은 디렉터리 임시 파일과 `os.replace`로
저장한다. 실패 판정은 캐시에 넣지 않는다. 저장 중 중단되면 이미 저장한 쌍부터 재사용한다.
손상된 캐시는 miss로 처리한다.

SDK의 자동 재시도는 꺼서 숨은 요청 시도를 만들지 않는다. 실패 분야를 기록하고 다음 분야로
계속하며 재실행으로 실패 쌍을 다시 처리한다. 예외 메시지에 원문·키가 들어갈 수 있어 원장에는
예외 타입만 남긴다. 시작을 기록한 뒤 응답 없이 종료된 시도는 interrupted로 남긴다.

## 지표와 사람 판정

- strict Precision@5/10: `relevance=2` 비율. lenient는 `relevance>=1` 비율이다.
- Precision 분모는 항상 k다. 후보가 부족한 자리는 0으로 계산하고 부족 수를 함께 기록한다.
- graded nDCG@5/10: gain=`2**relevance-1`, discount=`log2(rank+1)`이다.
  ideal은 각 분야의 두 방식 Top-10 합집합을 Judge relevance 순으로 정렬한다.
  전체 Creator 정답으로 계산한 절대 Recall/nDCG가 아니다. ideal gain이 0이면 nDCG도 0이다.
- 분야별 지표·M3−M2 차이, 17개 분야 동일 가중 평균과 paired bootstrap 10,000회 95% 구간을
  계산한다. 재표집 seed는 `20261006`이며 두 방식에 같은 분야 인덱스를 사용한다.
- 주 지표 기준 분야별 승·패·동률, Top-5 교체 사례와 광고·경품·동명이인 등 잡음 사례를
  로컬 보고서에 기록한다. 미완료 분야에는 pending 수만 남기고 전체 평균·구간은 만들지 않는다.

사람 표본은 분야별 최소 한 쌍을 보장하고 방식별 단독 Top-5 후보 간 Judge 관련도 차이,
개선·악화·동률 분야, 잡음, 방식별 소속 그룹을 순환해 선택한다. 한 쌍에 여러 층이 겹칠 수 있다.
`methodOnlyRelevanceDiffers:<method>`는 방식별 단독 후보 간 관련도 차이이며 Judge와 사람의 불일치가 아니다.
존재하지 않는 층을 인위적으로 만들지 않으며 `human-sample.json`에 층별 전체·선정 수를 남긴다.
Judge 점수·근거·층·방식·순위는 CSV에서 숨긴다. 익명 ID와 분야 설명·소개만 보여준다.

사람은 relevance=0/1/2, isNoise=0/1, noiseType을 입력한다. reason은 선택이며 비어 있으면
`사람 판정`으로 기록한다. 완전히 빈 판정 행은 미완료로 처리하고 일부만 잘못 채운 행은 거부한다.
원문·ID·헤더·행 중복·행 누락은 검증한다. 빈 시트는 보존하고 별도 filled 파일만 읽는다.
Judge와의 3단계 정확 일치율, 이진 일치율, 방법별 Top-5/10 검수 표본의 strict/이진 적합·잡음
비율을 계산한다. 표본이 전체 합집합을 덮을 때만 완전한 사람 P@k/nDCG도 계산한다.
이 층화 표본 비율을 전체 추천 결과의 불편 추정으로 해석하지 않는다.
방식별 표본 구성은 비대칭일 수 있으므로 적합·잡음 비율은 반드시
`human.methods.<method>.byCutoff.5/10.pairCount`와 함께 해석한다.
이 표본 수와 비율은 로컬·공개 보고서 모두에 포함되며 점추정 비교만으로 모집단의 비악화를 입증하지 않는다.
태그 이름이 변경되기 전에 만든 `human-sample.json`은 보존하고 별도 작업 디렉터리에서 시트를 생성한다.

## 비용·출력·공개 범위

| 파일 | 내용 |
| --- | --- |
| `contract.json`, `decision-plan.json` | 동결한 입력·Judge·사전 판단 계약 |
| `requests/`, `responses/` | 실제 요청·응답 원문 |
| `attempts.json`, `run-summary.json` | 요청 시도, 토큰, 실패·중단·미완료 상태 |
| 별도 공유 캐시의 pairId JSON | 성공한 개별 판정과 원문·모델 identity |
| `human-blank.csv`, `human-sample.json` | 빈 블라인드 시트, 숨겨진 층화·지문 계약 |
| `human-filled.csv` | 독립적인 사람 입력, Judge 캐시와 분리 |
| `report.json` | 개별 사례를 포함하는 로컬 보고서 |
| `public-summary.json` | ID·소개·개별 근거·청구 확인 문구 없는 공개 집계 |

report에 `--input-price-per-million-usd`와 `--output-price-per-million-usd`를 함께 지정하면
확인한 단가로 관측 토큰의 비용을 추정한다. 최신 가격을 자동으로 가정하지 않는다.
잘못된 JSON을 반환한 호출의 토큰도 포함하며 응답 없는 실패·중단은 사용량 불명으로 표시한다.
사용량 불명 시도가 있으면 총비용을 토큰만으로 확정하지 않는다. 단가 미입력은 비용 미확인이다.

청구서를 확인한 경우 `--billed-usd`와 `--billing-reference`로 실제 청구 금액과 확인 근거를
기록한다. 이 값은 사용자 확인 입력이며 도구가 청구서를 조회·검증한 것으로 표시하지 않는다.
토큰 추정과 실청구를 구분하며 비용 범위는 **현재 작업 디렉터리의 실행**이다.
공유 캐시를 만든 과거 실행의 비용은 제외하고 이 한계를 보고서에 명시한다.
공유 캐시를 포함한 전체 평가 비용을 확인하려면 해당 범위의 청구서를 별도로 확인해야 한다.

원문·개별 판정·API 캐시·사람 시트는 모두 Git 제외 `results/`에 둔다.
문서로 옮길 수 있는 산출물은 `public-summary.json`의 집계와 재현 계약·한계·결론이다.
`report.json`의 사례나 응답 reason에는 개인정보가 포함될 수 있으므로 커밋하지 않는다.

## 현재 결론과 #43 전달 계약

2026-10-06 구현 검증에서는 실제 Creator 전체 Judge 호출·사람 판정·청구 확인을 수행하지 않았다.
따라서 실제 서비스의 잠정 결론은 **M2 유지**다. `INTEREST_M2_V1`, bonus `0`을 사용하고,
tau `0.4296248555`·최대 태그 `3`은 비교 대상 M3 설정으로만 보존한다.
실제 평가 완료 후 report의 `decision.selectedConfig`와 `selectedConfigHash`를 #43에 전달한다.
선택된 method·tau·bonus·최대 태그·임베딩 버전·taxonomyHash·판단 버전을 함께 고정해야 한다.
이 문서는 가짜 Judge 테스트 결과를 실제 M2·M3 성능 비교 결과로 제시하지 않는다.

```bash
python -m pytest -q tests/experiments/test_interest_judge.py tests/recommendation
```

테스트는 가짜 Judge·SDK로 입력 블라인드, 지문 무효화, 재시작·부분 완료, JSON 검증,
지표·bootstrap, 빈 사람 시트·독립 판정, 비용 기록과 공개 집계의 원문 제외를 확인한다.
