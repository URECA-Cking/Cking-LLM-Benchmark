# v5 정의 기반 후보 보존·중복 억제 실험

`grounded-detail-candidates-v5`는 명시적 실험 옵션이다. 기본값 v2, 기존 v2/v3/v4
프롬프트 바이트, 출력 스키마, `detail-limit-v2`, 인용·중복·부모 검증은 유지한다.
결과 보정이나 중복 자동 제거는 하지 않는다.

## 조건 분리

| 조건 | 변경 |
| --- | --- |
| grounded-detail-candidates-v4 | 기존 기준선 |
| grounded-detail-candidates-v4-dedup | v4에 코드당 객체 하나·복수 근거 처리 정상/반례만 추가 |
| grounded-detail-candidates-v5 | 같은 중복 지침 + 정의 판단/인용 분리·core 없는 후보 보존·합성 경계 예시 |

세 조건의 이름과 실제 prompt_hash를 요청·manifest·revision에 보존한다. 대조 조건을
실행하지 않았다면 각 지침의 기여 요인을 분리한 효과라고 주장하지 않는다.
짧은 표현·해시태그도 정의를 충족하면 후보가 될 수 있지만 원문 evidence는 비어 있을 수 없다.
core_evidence와 repetition_evidence는 각각 명시된 주된 활동·독립 입력 행이 없으면 비워 둔다.
상위만 명확하면 상위만, 모두 부족하면 미분류한다. 경계 보류는 근거 없는 세부의 생략이며
모델 스키마에 새 hold 필드를 추가한 것이 아니다. 검수표의 hold는 평가에서 별도로 제외한다.

경계 합성 예시는 개발 표본에서 유래했다. LIFE의 일상 기록/경험 사연, 의류 코디/취미 제작,
영화 작품/일반 편집을 대비한다. 자기소개·이름·직업만으로 LIFE를 부여하거나 배경 음악·
동물 소재·일반 영상 편집을 작곡·직접 돌봄·영화로 확장하지 않는다. 실제 사람 합의는 미수행이다.

## 승인된 v5만 실행하고 기존 결과 재사용

아래는 2026-10-14 실행을 재현하는 명령 형식이다. 이미 완료한 출력 폴더로 실행하면
기존 시도와 성공 revision을 재사용하고 성공 채널을 재호출하지 않는다. 새 폴더에서 다시
실행하는 것은 새 비용이므로 별도 승인 범위를 확인해야 한다. `.env`와 원문은 Git에 추가하지 않는다.

```powershell
.venv\Scripts\python.exe -m src.detail_tagging_batch --inputs results/issue66-development/development/inputs.private.json --output results/issue68-approved-v5-network --prompt-version grounded-detail-candidates-v5 --ai-proxy-criteria results/issue66-ai-proxy-review-v2/criteria.ai-proxy.private.json --max-attempts 3 --tag-llm
```

개발 50개 × 최대 3시도로 모델·온도·SDK 재시도·timeout·스키마·정책을 기존 실행과 고정했다.
이번 사용자는 이전 버전 재호출을 금지하고 v5만 승인했다. 대조 조건도 실행하지 않았다.
API 연결 오류 기록은 별도 폴더에 보존했고 응답 없는 시도를 모델 응답 통계와 구분했다.

기존 `human_eval evaluate`는 버전 제한 없이 v5도 평가한다. 추가한
`detail_tagging_experiment`는 이 검증기를 재사용하고, 원본 응답/성공 revision·실패 검증 오류·
usage를 대조하며 쌍별 공통 성공 집합과 P/R/F1을 보고한다. 모든 비교는 오프라인이다.
summary는 경로를 명시적으로 선택하고 여러 재개 snapshot을 동시에 비교하지 않는다.

```powershell
$v2Summary = (Get-ChildItem results/issue66-ai-approved-v2/summary-*.json).FullName
$v4Summary = (Get-ChildItem results/issue66-ai-approved-v4/summary-*.json).FullName
$v5Summary = (Get-ChildItem results/issue68-approved-v5-network/summary-*.json).FullName
.venv\Scripts\python.exe -m src.detail_tagging_experiment --criteria results/issue66-ai-proxy-review-v2/criteria.ai-proxy.private.json --summary $v2Summary --summary $v4Summary --summary $v5Summary --output results/issue68-comparison/new-evaluation.private.json --ai-proxy
```

기존 파일은 배타적 생성으로 덮어쓰기를 거부한다. 출력은 results/ 내부만 허용하며
stdout에는 표본 수·버전·기본값 유지 결정만 출력한다. 채널별 진단/누락·과잉은 비공개 JSON에 있다.
원본 기준 해시·출처, 입력·taxonomy·모델·정책·스키마·온도·SDK 재시도·timeout·최대 시도를
대조하고, 각 프롬프트의 실제 해시와 요청 본문도 확인한다. 조건이 부족한 역사적 결과에
누락된 provenance를 추정해 채우지 않는다. AI 기준은 `--ai-proxy`이며 사람 기준으로 승격하지 않는다.

## 사람 기준의 독립 검증

[기존 홀드아웃 절차](detail-tagging-holdout.md)의 select/freeze/register/run/report/decide를 재사용한다.
개발·예시 채널 ID와 같은 원문을 제외하고 팀이 실제 사람 기준과 허용 누락 감소·과잉 증가·
최소 표본 수를 결과 확인 전에 합의해야 한다. 기존 채택 검사는 모델 실패 0개와 hold 0개도 요구한다.
이번에 새 사람 기준·독립 표본을 마련하거나 사후 임계값을 만들지 않았다.

`register`에 `--v4-v5`를 추가하면 v4/v5 해시를 고정한다. 생략하면 기존 v2/v4 계약을 유지한다.
허용 차이의 비교 기준은 등록한 첫 버전(v4), 후보는 두 번째 버전(v5)이다.
`run`은 등록한 두 버전을 모두 실행하므로 이번 v5 단독 승인으로 실행하면 안 된다.
사람/AI 기준의 격리는 기존대로 유지하며 AI 기준은 홀드아웃 protocol에 등록할 수 없다.
`decide --choice adopt_v5`는 v4/v5 protocol에서만 가능하고, 실제 사람 검수 완료·독립 홀드아웃
통과·프롬프트 미변경을 요구한다. 결정 기록 자체는 기본값을 변경하지 않는다.

## 실제 결과

[v5 개발 표본 보고서](../results/detail-tagging-v5-development.md)에 조건·집계·한계를 기록했다.
이번 v5는 개선되지 않았다. 대조 조건 실제 실행, 사람 경계 합의, 독립 검증과 팀 채택은 남아 있다.
