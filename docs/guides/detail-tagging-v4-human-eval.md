# v4 실험과 사람 기준 평가

Issue #64 구현. 기본값은 `grounded-detail-candidates-v2`이며 v4는 명시적 실험이다.
v2/v3 프롬프트 바이트, 출력 스키마, `detail-limit-v2` 정책과 정확한 인용 검증을 유지한다.
v4는 세부 근거가 없더라도 명시된 상위는 유지하도록 지시한다. 노래 커버, 남자 코디,
해시태그, 패션·뷰티 상위만 유지 예시와 직접 돌봄·배경 음악 과잉 추정 반례를 포함한다.

## 전체 입력의 사람 검수표 준비

기존 50개 전체의 동일 순서·동일 텍스트 JSON 배열을 사용한다.
원문, 채널별 판정과 원본 응답은 Git 제외 `results/`에만 둔다.

```powershell
python -m src.detail_tagging_human_eval prepare --inputs results/issue58-review-sample-v2/private_inputs.json --cohorts results/issue64-cohorts.private.json --output results/issue64-human-new
```

`--cohorts`는 선택 사항이며 역사적 원본에서 확인한 채널 ID 배열을 받는다.
키는 `v3_unclassified_transition`(기존 전환 21개), `baseline_final_failure`(기존 실패 7개),
`baseline_retry_success`(기존 재시도 성공 집단)이다. 누락 집단은 빈 배열로 표시하며
21/7이라는 숫자나 소속을 추측하지 않는다. 같은 채널의 집단 중복은 허용한다.

출력은 원문·taxonomy·해시·집단을 고정한 `pack.private.json`과
`human-review.private.csv`다. 모든 채널 × 모든 상위/세부 코드 행을 `pending`으로 생성한다.
AI 초안을 자동 수입하지 않는다. 21개 AI 초안은 참고 의견으로만 사용하고,
명확한 생략 8개와 혼합 7개를 먼저 확인한 뒤 전체 50개를 사람이 판정한다.
상위 판정은 기존 자동 상위에 제한되지 않는다.

| 필드 | 입력 방법 |
| --- | --- |
| channel_id / input_hash / axis / code | 고정. 변경·누락·중복 금지 |
| decision | `keep` 유지 / `delete` 삭제·비해당 / `hold` 보류 |
| reviewer / reason | 유지·삭제·보류 모두 실제 검수자와 사유 필수 |
| evidence | 유지에는 원문 연속 인용 필수. 나머지는 선택이며 입력하면 원문 인용이어야 함 |

마술 배치, 암묵적 직접 돌봄, 게임명→장르, 다른 플랫폼 소개 등은 팀 기준을 합의하거나
`hold`로 남긴다. 비해당 코드도 `delete`와 사유를 기록해 과잉 비교의 분모를 확정한다.
유지 세부는 유지 상위가 있어야 하며 상위 3개·상위당 세부 3개의 정책을 따른다.
도구는 검수자 이름의 실제 사람 여부를 인증하지 않는다. AI 판정을 사람 검수로 제출하면 안 된다.

## 사람 기준 고정

```powershell
python -m src.detail_tagging_human_eval freeze --pack results/issue64-human-new/pack.private.json --reviews results/issue64-human-new/human-review.private.csv --output results/issue64-criteria.private.json
```

`pending`이 남으면 거부한다. 미결정은 `hold`로 기록한다. CSV를 한 번 읽어 파싱·검증·
원본 바이트 SHA-256을 생성하고 판정 전체를 새 JSON에 고정한다. 기존 파일은 덮어쓰지 않는다.
기준 수정 시 새 기준 파일과 실행 폴더를 사용한다. 준비 폴더도 임시 폴더에서 완성한 후 게시한다.

## 승인된 외부 전송·유료 실행

사람 기준을 고정하고 이번 실행의 원문 외부 전송·유료 호출을 승인받은 후 실행한다.
v4 CLI는 `--human-criteria`를 요구하고 입력·taxonomy를 대조해 기준 해시를 기록한다.
`run_batch` 함수는 합성 진단을 위해 기준 없는 실행도 허용한다. 실제 평가에는
`human_criteria`를 전달한다. 기준 없는 실행은 사전 고정 기준으로 실행했다는 출처 증거가 되지 않는다.

```powershell
python -m src.detail_tagging_batch --inputs results/issue58-review-sample-v2/private_inputs.json --model gpt-5.4-nano-2026-03-17 --output results/issue64-v4-new --prompt-version grounded-detail-candidates-v4 --human-criteria results/issue64-criteria.private.json --tag-llm
```

temperature=0, timeout=60초, SDK 재시도=0, 최대 시도=3의 기존 조건을 사용한다.
모든 시도의 원본 응답·usage·검증 오류·시도 번호를 보존하고 성공 후 중단한다.
재개는 입력·taxonomy·모델·정책·스키마·프롬프트 버전/해시·호출 조건·시도 상한·
전달한 사람 기준 해시를 대조한다. 기존 automatic/reviewed revision, attempts와 summary는 보존한다.
temperature=0에서도 완전 동일 결과를 요구하지 않는다.

## 사람 기준으로 비교

```powershell
python -m src.detail_tagging_human_eval evaluate --criteria results/issue64-criteria.private.json --summary results/new-v2/summary-<ID>.json --summary results/issue64-v4-new/summary-<ID>.json --output results/issue64-evaluation.private.json
```

`--summary`는 여러 번 지정하며 실제 prompt_version/해시로 결과를 식별한다.
각 실행은 같은 입력 집합·순서·텍스트, taxonomy, 모델, 정책, 스키마, temperature,
SDK 재시도, timeout, 시도 상한을 가져야 한다. 성공 revision의 출처와 후보 재검증 결과를 대조한다.
누락 조건은 추정하지 않고 비교를 거부한다. 기존 36/50(v2), 48/50(v3)은
[역사적 v3 보고서](../results/detail-tagging-v3-regression.md)의 첫 시도 수치이며 엄격 비교로 소급하지 않는다.
v2 조건이 누락됐다면 별도 승인 후 같은 조건의 새 v2 기준선을 실행한다.

전체 및 각 집단의 보고 항목:

- 첫 시도/재시도 후 형식 통과 수와 비율: 실패 포함 전체 집단을 분모로 사용.
- 미분류/상위만 분류 수와 비율: 최종 성공 채널을 분모로 사용하고 실패 수 별도 표시.
- 상위·세부 각각 누락/과잉 태그 수와 채널별 코드, 검수 태그 수, 사람 양성 태그 수,
  예측된 검수 태그 수, 보류 수, 모델 실패로 비교하지 못한 검수 태그 수.
- `hold`는 태그별 정답 비교에서 제외. `fully_reviewed_channels`는 성공 중 해당 축 보류 없는 채널 수.
- 사람 미분류/상위만 분류는 모든 축의 보류 없는 채널을 분모로 별도 제공.
- 중복 집단의 채널별 소속. 집단 합계를 전체 합계로 사용하지 않음.

빈 배열로 형식 통과만 높아지는 효과와 성공 결과 선택 편향을 함께 해석한다.
사람 기준 파일과 실행 summary의 해시·조건을 기록하고 채널별 보고서는 비공개로 유지한다.
자동 승격하지 않고 `v2_retained_pending_human_adoption_decision`을 기록한다.
팀은 누락과 과잉을 함께 검토해 기본값 유지/추가 실험/승격을 결정한다.
형식 통과나 합성 후보 테스트를 의미 품질, 추천 순위 개선, 서비스 품질 검증으로 표현하지 않는다.

## 현재 상태

코드·합성 테스트 구현 완료. 실제 21개/전체 50개 사람 검수, 유료 v4 실행,
사람 기준에 대한 실제 품질 비교와 채택 결정은 미수행이다.
