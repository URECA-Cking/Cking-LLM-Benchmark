# 콘텐츠 태그 작업자 운영 및 BE #654 연동 계약

이 구현은 BE #654의 `feat/654-creator-ai-tag-api` 구현과 공유하는 `content-tags-v1` 계약이다.
운영 활성화 전 승인 taxonomy/configHash 등록과 전용 토큰·공급자 지출 한도 설정이 필요하다.
운영 초안 taxonomy를 자동 승인하거나 시드로 넣지 않는다.

## 입력·적재 계약

별도 최소 권한 `CKING_CONTENT_TAG_ACCESS_TOKEN` Bearer 인증을 사용한다.
원격 BE는 HTTPS만 허용하며, HTTP 예외는 localhost/숫자 loopback 주소에 한정한다.
기존 추천 API Key의 권한은 확대하지 않는다. 모든 경로는 `/api/internal/content-tags` 아래다.

- `GET /inputs[?cursor=<opaque>]`: `contractVersion=content-tags-v1`, `taxonomyApproved=true`,
  `taxonomyHash`, `generationConfigHash`, `items`, `nextCursor`를 반환한다. items는 전체 활성 크리에이터의
  `creatorId`(양의 정수), `introduction`(문자열/null), `inputRevision`(양의 증가 정수),
  `inputHash`다. 페이지 순회는 일관된 snapshot이어야 한다. draft 85개를 자동 승인하지 않는다.
- 정본은 BE Creator/Space 소개 저장 필드로 합의해야 한다. CRLF/CR→LF 후 NFC를 적용하고
  trim 없이 UTF-8 바이트 SHA-256을 계산한다. null은 빈 문자열이다. 공백만 있으면 모델 없이
  정상 미분류를 저장한다. `fixtures/content-tags-v1.json`을 Java에서도 동일하게 검증한다.
  기존 로컬 분류기의 JSON 문자열 digest는 `record.input_hash`이며 wire `inputHash`와 다르다.
- 입력의 `generationConfigHash`와 로컬 fingerprint가 다르면 공급자 호출 전에 실패한다.
- `PUT /creators/{id}/automatic`: 입력 revision/hash, 자동 revision/멱등 키,
  provenance, UTC 생성 시각, 자동 결과와 후보 근거를 전송한다. provenance는 모델,
  프롬프트/스키마/정책/분류체계/기준 해시를 포함한다. 응답은
  `{status: applied|idempotent, automaticRevision: <요청과 동일>}`다.
- BE는 소개 수정과 동일 트랜잭션 잠금 하에 현재 revision/hash 및 활성 taxonomy를 대조한다.
  변경된 입력/오래된 세대는 409, 같은 멱등 키의 다른 payload는 409, 인증은 401/403이다.
  동일 입력이어도 오래된 실행은 최신 세대를 덮어쓰지 못하도록 BE가 발급한 입력 세대와
  활성 실행 설정 순서를 검증해야 한다. 이 작업자는 하나의 영속 루트/단일 호스트만 지원한다.
- 자동 결과는 본인 확인 결과와 별도 보존한다. `creatorConfirmed=false`이며 공개나
  본인 확인 해제를 요청하지 않는다. 공급자 실패는 빈 결과로 적재하지 않는다.
- 최초의 409는 `refresh_required`로 구분해 정상 편집 경합만으로 exit 1을 만들지 않는다.
  같은 recordKey에서 충돌 관측이 **3회** 또는 최초 관측 후 **300초**에 도달하면
  `PERSISTENT_INPUT_CONFLICT` 실패로 승격하고 exit 1로 알린다. event/daily가 같은 카운터를 쓴다.
  `conflict.json`의 관측 횟수·최초/최근 관측 시각·승격 상태는 재시작 뒤에도 유지한다.
  승격 뒤에는 last-success.json도 갱신하지 않는다. 새 revision/config는 별도 key로 정상 처리한다.
  과거의 단순 conflict flag는 시간 이력을 추정하지 않고 최초 재관측부터 카운트한다.
409를 받은 payload는 로컬에서 fence하고 다시 전송하지 않는다. 다음 순회는 입력을
  다시 조회한다. 입력이 같은데 충돌하면 BE 계약/실행 설정을 조사하고 새 입력 세대를 발급한다.
  401/403은 해당 순회의 추가 태그 적재를 중단한다. 네트워크/5xx는 저장한 동일 payload로 재개한다.

## 변경 이벤트와 정합성

BE의 Space 생성/프로필 수정 Service가 소개와 outbox를 같은 트랜잭션에 저장한다. DB trigger나 SUPER 권한을 쓰지 않는다. 수정 요청 안에서는 모델을 호출하지 않는다.
작업은 creatorId + inputRevision/inputHash로 식별하며 BE에서 최신 세대로 병합한다.

- `PUT /jobs/claim {leaseSeconds:900}` → 작업 없으면 `{}`, 있으면
  `{jobId, leaseToken, generation}`. lease는 전달 보호용이고 데이터 쓰기 허가가 아니다. 작업자는 60초마다 `/jobs/{id}/renew`로 연장한다.
- `PUT /jobs/{jobId}/ack {leaseToken,generation,status,summary}`에서 status는
  `completed|retry`. BE는 토큰과 세대를 원자적으로 비교하며 최신 후속 작업을 지우지 않는다.
- 프로세스 중단/lease 만료 시 BE가 재전달한다. 중복·역순 이벤트도 현재 전체 입력을 다시 읽는다.
  처리 중 수정은 outbox에 남고 만료 lease의 ack는 최신 작업을 완료할 수 없다.
- 이벤트 실행도 기존 전체 manifest 기반 daily apply를 실행하므로 변경 크리에이터를
  후보로 사용하는 모든 유사 추천과 관심 분야 추천을 재계산한다. 추천 가중치는 바꾸지 않는다.
- claim한 Creator의 태그와 추천 단계가 성공하면 그 작업을 완료한다. 다른 Creator의 실패는
해당 Creator의 ack를 막지 않는다. daily는 성공한 행만 reconcile로 전송하며 실패 행이 있으면 exit 1을 유지한다. daily는 작업 유무와 관계없이 전체를 재확인한다.
  추천 완료 마커가 있어도 태그는 실행한다. 잠금 경합은 claim 전에 종료하며 작업을 소비하지 않는다.

## 배포·수동 실행·복구

Linux의 전용 사용자 `cking`, Python 3.11+, 고정 checkout `/opt/cking-llm`을 사용한다.
`deploy/content-worker`의 unit을 `/etc/systemd/system`에 설치한다. 승인한 운영자가
환경 예제를 `/etc/cking/content-worker.env`에 복사하고 권한 0640으로 설정한다.
`StateDirectory=cking-content`와 `StateDirectoryMode=0700`으로 `/var/lib/cking-content`를 생성한다.
컨테이너/볼륨 환경에서도 이 경로의 영속성을 보장한다.
`systemctl daemon-reload` 후 event.timer와 daily.timer를 enable/start한다.
이 문서 추가만으로 배포나 유료 호출은 실행되지 않는다.

일일 보정은 매일 **03:00 Asia/Seoul**, 이벤트 polling은 이전 실행 종료 후 **60초**다.
수동 실행은 `systemctl start cking-content-daily.service`를 사용해 같은 환경·잠금·경로를 쓴다.
성공/무변경은 조용히 종료하며 `last-success.json`에 source/status/completedAt을 원자적으로 기록한다.
작업 없는 event poll도 마커를 갱신한다. 운영 모니터는 타이머 활성 상태와 이 시각을 함께 감시한다.
초기 경보 기준은 마지막 성공 20분 초과이며, 부하 측정 후 처리 시간에 맞춰 조정한다. 실패는 exit 1과 OnFailure의 daemon.err syslog로 알린다.
운영 로그 수집 시스템에서 해당 메시지에 경보를 연결한다. 원문·키·외부 예외 메시지는 알리지 않는다.
일시 중단은 SIGTERM/서비스 중지 후 재실행한다. host 재시작 시 timer가 다시 시작된다.
잠금 경합 exit 75는 실패가 아니며 event timer가 재시도한다. daily 경합 때도 후속 event와
다음 daily가 보정한다. 서비스 자체 강제 중단/timeout도 systemd OnFailure 대상이다.

캐시 models.json, application ledger, recommendations/, records/, budget 파일을 **함께**
백업·복원한다. 임시 runner와 여러 호스트에서 독립 복사본을 실행하지 않는다.
영속 볼륨을 잃으면 실행을 중지하고 BE 최대 applicationSequence와 결과 세대를 먼저 확인한다.
적재 직후 중단해도 payload가 먼저 저장되어 있으므로 같은 멱등 키로 재전송한다.
`summary.json`의 행별 creatorId/recordKey/code로 실패 입력과 보호된 artifact 위치를 찾는다.
`EXTERNAL_TRANSFER_NOT_APPROVED`, `DAILY_BUDGET_EXHAUSTED`, `PROVIDER_FAILURE`,
`FINAL_PROVIDER_FAILURE`, `CONFIGURATION_MISMATCH`, `BACKEND_AUTHORIZATION_FAILED`를 구분한다.
외부 예외 메시지나 소개 원문은 summary/stdout에 기록하지 않는다.
승격된 409를 **같은 입력으로** 복구해야 한다면 두 timer/worker를 중지하고 실행 종료·잠금 해제를
확인한다. BE의 충돌 원인이 해결되어 저장 payload/멱등 키를 수용할 수 있는지 확인한 뒤, 해당
`records/<recordKey>/conflict.json`만 고유한 `conflict.resolved.<UTC시각>.json`으로 이동해 보존하고
수동 실행한다. 저장 payload를 그대로 재전송하므로 모델 재호출은 없다. BE 작업이 FAILED라면
`/jobs/{id}/retry`도 수행하고 검증 후 timer를 재개한다. payload/receipt/attempts/budget/ledger는
삭제·변경하지 않는다. BE의 payload 불일치가 남아 있다면 flag만 해제하지 말고 먼저 원인을 해결한다.

실패 횟수 한도(입력·설정당 3회)에 도달한 입력의 수동 재시도는 운영자가 실패 원인을
해결한 뒤 해당 records/<key>/attempts.json count를 승인하에 조정한다. payload/ledger는 삭제하지 않는다.
롤백은 timer 중지→이전 코드 복귀다. 이미 저장한 자동·확인 결과를 삭제하지 않는다.

## 승인·호출량·측정

외부 전송은 소개 원문과 taxonomy이며 OpenAI 세부 태깅과 기존 추천 OpenAI/DeepInfra를 포함한다.
운영자는 이 범위와 모델/공급자/프로젝트 지출 한도를 승인한 뒤 unit의 명시적 승인 플래그를 유지한다.
미승인 CLI 실행은 차단한다. 태그는 UTC 날짜당 `CKING_TAG_DAILY_CALLS`로 영속 선예약한다.
SDK 재시도는 0, 호출 timeout은 60초다. 중단된 호출도 예산에 포함한다.
호출 횟수 제한은 토큰 비용 상한이 아니므로 **공급자 프로젝트 지출 한도**를 별도로 설정한다.
기존 추천 생성 호출은 태그 호출량 카운터에 포함되지 않는다. 추천 모델도 공급자 지출 한도 및
전용 계정 제한을 적용해야 한다. 테스트에서 유료 호출은 하지 않는다.

운영 목표는 변경→시작 p95 2분, 변경→완료 p95 15분(초기 목표, 부하에 따라 측정 후 조정)이다.
BE outbox의 committedAt/claimedAt/completedAt와 attemptCount/nextRetryAt/finalFailed를 기록한다.
작업자 summary의 startedAt/completedAt와 단계 실패, 409 refreshRequired를 연결해
대기/처리 지연·재시도·최종 실패를 측정한다. BE는 지수 backoff로 최대 5회 재시도하고 최종 실패는 오류 로그로 알린다.
`PUT /jobs/{id}/retry`로 수동 재등록한다. 일일 보정이 성공하면 `/jobs/reconcile`로
현재 revision/hash/config와 맞는 PENDING/FAILED 작업을 완료한다. RUNNING 작업은 건드리지 않는다. 긴 실행은 lease가 만료될 수 있으며 세대 검사가 필수다.

## 검증 경계

Python mock 테스트는 입력 변경·모델 변경·빈 입력·성공 재사용·네트워크 재전송·409 fence·
공급자 실패와 영속 예산·독립 단계 실패·공유 잠금·정규화 fixture를 검증한다.
실제 HTTP/MySQL 연동 검증은 `python -m src.recommendation.e2e.runner --suite content --be-dir ../Cking-BE`로 실행한다.
Java fixture와 소개 수정·적재·본인 확인 경쟁, 이벤트/lease/재시작/실패 복구를 합성 provider로 검증한다.
운영 systemd 설치·유료 모델 품질·사람 검수·독립 holdout 및 운영 taxonomy 승인은 별도다.

## BE 재전송 계약

동일 Creator·키·payload의 재전송은 소개/설정 변경 뒤에도 멱등 성공한다. 이 응답은 BE의 최신 포인터를
되돌리지 않는다. 새 키로 뒤처진 입력/설정을 보내면 기존처럼 409다. HTTP E2E는 두 경우를 구분한다.
공용 content-tags-v1 fixture에 제어문자 JSON 소문자 hex, 이모지와 짝 없는 surrogate 거부를 추가했다.
입력은 여전히 전체 일관 snapshot이며 BE가 행당 한 번 정규화한 본문으로 해시를 산출한다.


## 정책 지문 변경과 운영 갱신

policyHash는 `policy-tokens-v1` 형식, POLICY_VERSION 및 classify/_rank/validate_llm_topics의
정규화된 토큰을 포함한다. AST는 f-string 금지 검사와 docstring 위치 확인에만 쓰며 ast.dump나 노드 필드는 해시하지 않는다.
정책 함수 세 개에는 f-string을 허용하지 않는다. 발견 시 `POLICY_FSTRING_NOT_SUPPORTED`로
지문 생성을 중단한다. f-string을 도입하려면 minor 간 정규화 계약과 검증을 먼저 보완해야 한다.
주석·docstring·공백·무관한 실험 코드는 제외하지만 실제 코드·문자열 리터럴·블록 구조는 보존한다.
따라서 같은 정책 소스는 지원 Python minor 버전의 AST 필드 차이로 configHash가 바뀌지 않는다.
실제 순위/상한/검증 정책 변경은 새 hash가 된다.
promptHash/schemaHash/criteriaHash는 계속 별도로 검증한다. 기존 모듈 전체 hash에서 이 계약으로
전환했던 이력과 마찬가지로, AST dump에서 토큰 계약으로 전환하는 **이번 변경도** configHash가 바뀐다. 운영자는 배포 런타임에서 fingerprint를 산출해
BE 승인 설정을 갱신해야 한다. 기존 캐시/시도 이력을 새 key로 덮어쓰거나 승인 hash를 추론하지 않는다.

## 추천 재계산 범위와 부하 측정 계획

현재 event 한 건도 전체 manifest 기반 추천을 재계산한다. 한 Creator의 수정이 다른 Creator의
유사 추천·관심분야 추천 후보에 영향을 주므로 이 범위를 유지했다. 같은 Creator의 연속 변경은
BE outbox가 최신 세대로 합치지만 서로 다른 Creator N건은 최대 N회 직렬 실행한다.
여러 Creator 간 coalescing은 이번 변경에서 구현하지 않았다. p95 목표의 부하 충족 여부도 미검증이다.

운영 활성화 전 합성 데이터로 Creator 100/1,000/10,000명 × 수정 burst 1/10/100건을 측정한다.
초기/캐시 재사용 조건을 분리해 BE committed_at→claimed_at/completed_at p50/p95/p99,
처리 중 backlog 크기, summary 단계별 durationSeconds, 추천 호출 수·전체 실행 횟수를 기록한다.
변경→시작 p95 2분 또는 완료 p95 15분을 넘는 규모에서는 활성화하지 않고, 여러 job claim과
세대별 추천 coalescing을 후속 설계한다. 실제 유료 공급자로 측정하려면 별도 전송·예산 승인이 필요하다.

## 검증 기록 위치

시나리오·실행 ID·실행 시점 SHA는 `data/summaries/content-tags-http-e2e*.json`에 보존한다.
운영 가이드에는 과거 실행별 숫자를 누적하지 않는다. 최신 검증 링크/범위는 PR #74 본문을 확인한다.
BE 의존 계약은 [BE PR #655](https://github.com/URECA-Cking/Cking-BE/pull/655)의 diff와 비교한다.
최신 공개 집계는 PR URL과 소스 파일별 SHA-256도 기록해 squash 이후에도 내용으로 대조할 수 있다.
합성 taxonomy·모델 검증은 운영 taxonomy 승인이나 모델 의미 품질 검증이 아니다.
