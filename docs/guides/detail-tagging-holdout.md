# v2/v4 개발 표본과 독립 홀드아웃 평가

Issue #66의 보조 도구는 `src.detail_tagging_holdout`이다. #64의 사람 검수·freeze·evaluate와
기존 `run_batch`를 재사용한다. 기본 프롬프트는 v2다. 원문·채널별 기록은 Git 제외
`results/`에만 보관한다. 아래의 준비·선정·고정·보고 명령은 모델을 호출하지 않는다.
`run`만 원문을 외부로 전송하고 비용을 발생시킨다.

## 역사적 집단과 검수표

```powershell
python -m src.detail_tagging_holdout cohorts --inputs results/issue58-review-sample-v2/private_inputs.json --first results/classify50-20261013/first-new-key-summary.json --final results/classify50-20261013/execution-summary.json --index results/classify50-20261013/revision-index.json --revisions results/classify50-20261013/revisions --v3-summary results/issue62-v3-approved/summary-dca0896b2e20430ea9f3b1046d60b609.json --output results/issue66-development
```

첫/최종 역사적 snapshot의 실패 목록과 성공 revision을 대조한다. v3는 원본 attempts,
요청, summary와 성공 후보를 재검증한다. 출력 `historical.private.json`은 집단·중복·
출처 해시를 포함한다. `development/`에는 고정 입력, `pack.private.json`, 검수 CSV가 있다.
역사적 v3의 누락 timeout을 보충하지 않으며, 이 경로는 **집단 추출 전용**이다.
새 엄격 비교는 실제 사람 기준 해시를 기록한 새 v2/v4 실행만 허용한다.

실제 2026-10-14 대조 결과는 전환 21개, 최종 실패 7개, 재시도 성공 7개,
중복 채널 3개다. 사람 검수표는 50×102=5,100행이며 모두 `pending`이다.
기존 36/50(v2), 48/50(v3)은 역사적 첫 시도 수치로만 취급한다.

사람은 [검수 정책](detail-tagging-v4-human-eval.md)에 따라 전체 코드에 keep/delete/hold,
실제 검수자, keep의 연속 인용, keep/hold 사유를 기록한다. AI 초안을 가져오는 기능은 없다.
21개 AI 의견의 명확한 생략 8개·혼합 7개는 검수 우선순위 참고일 뿐이다.
마술·직접 돌봄·게임 장르·다른 플랫폼 소개는 합의하거나 hold로 남긴다.
상위 hold와 세부 keep은 기존 freeze가 거부한다.

```powershell
python -m src.detail_tagging_human_eval freeze --pack results/issue66-development/development/pack.private.json --reviews results/issue66-development/development/human-review.private.csv --output results/issue66-development-criteria.private.json
```

## 모델 실행 전에 독립 표본 선정

팀이 분야/경계 사례별 수량과 표본 규모를 먼저 결정한다. 각 후보는
`{"channel_id":"...","text":"...","stratum":"분야 또는 경계 집단"}` 형식이다.
`quotas.private.json`은 `{"분야 또는 경계 집단": 양의정수}`이며, 각 후보는 하나의 집단만 갖는다.
집단은 사람의 표본 구성 정보이며 모델 예측이나 사람 정답을 의미하지 않는다.

`examples.private.json`은 프롬프트 개발에 쓰인 모든 예시 채널의 `{channel_id,text}` 배열이다.
팀이 예시 출처 목록의 완전성을 확인해야 한다. 개발 채널은 별도로 전부 제외하므로
예시가 모두 개발 표본 안에 있다는 사실을 확인한 경우에만 이 배열을 비워 둘 수 있다.
도구는 목록에 없는 예시 채널이나 원문을 바꿔 쓴 중복을 찾아내지 못한다.

```powershell
python -m src.detail_tagging_holdout select --candidates results/candidates.private.json --development-inputs results/issue66-development/development/inputs.private.json --examples results/examples.private.json --quotas results/quotas.private.json --seed team-agreed-seed --output results/issue66-holdout
```

개발·예시와 같은 채널 ID 또는 같은 원문의 해시를 제외하고,
각 집단 안에서 SHA-256(seed, channel_id) 순서로 필요한 수량을 선택한다.
후보 순서를 바꿔도 결과는 같다. 부족하면 오류로 중단하며 수량을 자동 축소하지 않는다.
선정된 원문 중복도 거부한다. `selection.private.json`에 후보 풀 해시, 제외 수,
가용 수, 수량·선정법·seed·입력 해시를 고정한다. `holdout/` 검수표는 모두 pending이다.
표본의 분야 범위와 부족·편향은 팀이 별도로 기록한다.

홀드아웃도 실제 사람이 검수하고, 모델 결과를 보기 전에 freeze한다.
가능하면 모델 버전을 숨겨 판단한다. 원문 검수표에는 모델 출력이 포함되지 않는다.

```powershell
python -m src.detail_tagging_human_eval freeze --pack results/issue66-holdout/holdout/pack.private.json --reviews results/issue66-holdout/holdout/human-review.private.csv --output results/issue66-holdout-criteria.private.json
```

## 사람 기준과 채택 기준 사전 고정

팀이 `acceptance.private.json`에 다음 다섯 정수 기준을 사전 합의한다.
숫자는 태그 **개수 차이** 기준이며 자동 통계적 유의성 검정은 아니다.

| 키 | 의미 |
| --- | --- |
| min_holdout_channels | 필요한 최소 홀드아웃 채널 수, 1 이상 |
| min_topic_missing_reduction | v2 대비 줄어야 하는 세부 누락 수, 1 이상 |
| max_topic_excess_increase | 허용하는 세부 과잉 증가 수, 0 이상 |
| max_parent_missing_increase | 허용하는 상위 누락 증가 수, 0 이상 |
| max_parent_excess_increase | 허용하는 상위 과잉 증가 수, 0 이상 |

```powershell
python -m src.detail_tagging_holdout register --development results/issue66-development-criteria.private.json --holdout results/issue66-holdout-criteria.private.json --selection results/issue66-holdout/selection.private.json --acceptance results/acceptance.private.json --output results/issue66-protocol.private.json
```

pending 기준, 선정 입력·개발 입력·taxonomy 불일치, 겹치는 표본, 최소 수량 미달은 거부한다.
기준 해시, 표본, 채택 기준, 실제 프롬프트 해시와 다음 조건을 새 protocol에 고정한다:
모델 `gpt-5.4-nano-2026-03-17`, temperature=0, SDK 재시도=0, timeout=60초, 최대 시도=3.
버전 외의 조건은 동일하다. protocol 생성은 사람 신원 인증이나 타임스탬프 공증이 아니다.
모델 결과를 본 뒤 기준을 바꾸고 사전 합의라고 표시하면 안 된다.

## 승인 범위·키·예산을 확인한 뒤 비교 실행

이번 두 버전 × 두 표본의 원문 외부 전송과 유료 호출에 대한 승인을 별도로 받아야 한다.
호출 상한은 `(개발 수 + 홀드아웃 수) × 2 × 3`회다. 이 숫자는 금액 예산이 아니다.
팀은 키 사용 가능 여부, 사용 예산과 공급자 한도를 확인한다. API 키는 환경변수/.env로
주입하며 명령행 인자·결과·Git에 기록하지 않는다. 아래 명령은 승인 후에만 실행한다.

```powershell
python -m src.detail_tagging_holdout run --protocol results/issue66-protocol.private.json --split development --output results/issue66-new-development-runs --approved-paid-transfer
python -m src.detail_tagging_holdout run --protocol results/issue66-protocol.private.json --split holdout --output results/issue66-new-holdout-runs --approved-paid-transfer
```

두 버전의 조건을 요청 전에 모두 대조한다. 새 실행 폴더는 protocol/split 해시로 묶는다.
재실행은 같은 폴더의 기존 attempts/revision을 재사용하며 완료된 채널을 호출하지 않는다.
요청·원본 응답·usage·오류·시도 번호는 기존 배치 도구로 보존한다.
공급자 오류는 해당 시도를 보존하고 중단한다. 재개 시 기존 시도 상한을 유지한다.
protocol·사람 기준·프롬프트가 바뀌면 새 폴더와 새 사전 고정 절차가 필요하다.

## 표본별 보고와 팀 결정

각 폴더의 실제 `summary-<ID>.json` 경로를 지정한다. 재개 후에는 최신 완료 summary를 고른다.
개발과 홀드아웃을 합산하지 않고 각각 보고한다.

```powershell
python -m src.detail_tagging_holdout report --protocol results/issue66-protocol.private.json --split holdout --summary results/issue66-new-holdout-runs/grounded-detail-candidates-v2/summary-<ID>.json --summary results/issue66-new-holdout-runs/grounded-detail-candidates-v4/summary-<ID>.json --output results/issue66-holdout-report.private.json
```

개발 보고도 `--split development`와 해당 실행 경로를 사용한다. 보고는 원본 attempts와
summary를 다시 대조한다. 정확히 v2/v4 하나씩, 사전 고정 사람 기준 해시와 실행 조건을 요구한다.
첫 시도·재시도, 미분류·상위만 분류, 상위/세부 누락·과잉, hold·실패 분모와 집단 중복은
기존 evaluate가 보고한다. usage 합계에는 실패 시도도 포함하며 누락 usage 수를 별도 표시한다.
실제 청구액은 확인되지 않았다고 표시한다. 빈 배열로 형식 통과만 개선돼도 채택 대상이 되지 않는다.

홀드아웃이 사전 기준을 모두 만족하고 hold·모델 실패가 없어야 `eligible_for_team_review`다.
개발 결과는 기준을 만족해도 `retain_v2`다. 자동 승격은 하지 않는다. 최종 사람 결정 기록은:

```powershell
python -m src.detail_tagging_holdout decide --protocol results/issue66-protocol.private.json --development-summary <개발-v2-summary> --development-summary <개발-v4-summary> --holdout-summary <홀드아웃-v2-summary> --holdout-summary <홀드아웃-v4-summary> --choice retain_v2 --reviewer <실제-결정자> --reason <팀-판단-근거> --output results/issue66-decision.private.json
```

선택은 `retain_v2`/`additional_experiment`/`adopt_v4`다. 두 표본을 재검증하고 보고 해시를
기록한다. 채택은 개발 사람 검수 완료와 홀드아웃 기준 통과를 요구한다.
홀드아웃을 보고 프롬프트를 수정했다면 `--prompts-changed`로 재분류를 기록하며 채택은 거부한다.
새 프롬프트에 대해서는 새 홀드아웃이 필요하다. 기본값 변경은 평가 근거와 연결한 별도 코드
검토로 진행한다. taxonomy/코드 확정, BE 시드, 추천 순위, 서비스 E2E·배포는 별도 작업이다.
도구 구현이나 합성 테스트만으로 이슈를 완료 처리하지 않는다.
