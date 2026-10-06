# manifest 변경 감지와 일일 추천 배치

[문서 안내](../README.md) · [Creator 배치](batch-recommendation.md) · [관심 분야 배치](interest-recommendation.md)

## 매일 실행할 명령

저장소 루트에서 같은 환경·출력 경로로 다음 명령을 매일 실행한다.
cron, Windows 작업 스케줄러 등의 배포 리소스 생성은 #43 범위 밖이다.

```bash
python -m src.recommendation.daily_cli apply --be-base-url https://api.example.com
```

필수 환경 변수는 `CKING_RECOMMENDATION_API_KEY`다. `.env`도 로드하지만 키는 CLI 인자로 받지 않는다.
추천 적재에서는 `X-Cking-Recommendation-Key`만 전송하며 ADMIN JWT를 함께 보내지 않는다.
Creator 목록 GET에는 인증 헤더를 보내지 않는다. 키는 두 추천 적재 경로에서만 사용하며
HTTP 리다이렉트는 거부한다. 키 없는 apply는 모델 생성·BE 호출 전에 실패한다.

| 환경 변수 | 용도 |
| --- | --- |
| `CKING_RECOMMENDATION_API_KEY` | 두 추천 적재 경로의 BE 전용 키, apply 필수 |
| `DEEPINFRA_API_KEY` | 기본 DeepInfra BGE-M3의 임베딩 캐시 미스 처리 |
| `OPENAI_API_KEY` | 소개가 15자 이상인 Creator의 M4 태그 캐시 미스 처리 |

키는 실행 요약·체크포인트·payload에 기록하지 않는다. 모델 오류는 알려진 키 값을 치환하고,
통합 단계 초기화 오류는 예외 타입만 남긴다. HTTP 오류 본문·원래 예외 체인은 출력하지 않는다.
401은 잘못된 키, 403은 권한 부족으로 구분되는 최종 HTTP 상태이며 재시도하지 않는다.
미설정 키는 HTTP 이전 로컬 오류다. timeout·네트워크 오류·불완전한 응답과
408/425/429/500/502/503/504만 기본 2회 추가 시도한다. 대기 시간은 0.5초부터 지수 증가한다.
`--timeout`, `--retries`, `--retry-backoff`로 변경한다.

## 실행과 완료 판단

1. 출력 상태와 공유 모델 캐시의 잠금을 획득한다. 다른 실행이 사용 중이면 즉시 거부한다.
2. 공개 Creator 목록을 끝까지 읽어 ID·소개를 정규화하고 ID 오름차순 manifest를 만든다.
   페이지 크기·목록 순서·실행 시간은 내용 hash에 영향을 주지 않는다.
3. 미완료 실행이 없고 마지막 완전 적용 manifest·생성 설정·BE 대상이 모두 같으면
   `skipped`로 끝낸다. 이때 모델 클라이언트·캐시를 생성하지 않고 BE PUT도 호출하지 않는다.
4. 변경되면 hash와 설정별 독립 디렉터리에서 모든 Creator 유사 추천과 17개 관심 분야 후보를
   기본 Top-20으로 생성·적재한다. 빈 소개·빈 manifest도 정상 빈 세대로 처리한다.
5. 두 작업의 모든 생성과 적재가 성공 또는 정상 멱등 완료일 때만 `last-applied.json`을
   임시 파일과 원자 교체로 기록하고 활성 실행 상태를 해제한다.

유사 추천은 기존 짧은 소개 M2·일반 소개 M4 분기를 따른다. 관심 분야 기본값은
#44의 현재 결론인 `INTEREST_M2_V1`, bonus `0`이다. tau `0.4296248555`, 최대 태그 `3`은
선택 설정의 버전 추적을 위해 보존한다. 실제 Judge·사람 검증을 실행한 결과로 주장하지 않는다.

#44 평가가 끝나면 `report.json` 또는 `public-summary.json`을 다음처럼 전달한다.
`decision.selectedConfig` 전문과 `selectedConfigHash`를 검증하고 method·tau·bonus·최대 태그·
모델 버전·taxonomyHash·판단 버전을 생성 identity에 포함한다. 잠정 M3 설정은 거부한다.

```bash
python -m src.recommendation.daily_cli apply --be-base-url https://api.example.com --decision-report results/recommendation/interest-eval-v1/public-summary.json
```

보고서의 모델 버전은 선택한 제공자와 일치해야 한다. 보고서가 `BAAI/bge-m3@local-v1`이면
`--embedding-provider local`을 함께 지정한다. 모델·런타임 변경 시 보고서와
`--embedding-model-version`에 같은 새 버전을 사용한다. 로컬 임베딩을 사용해도 M4 태깅은
OpenAI 캐시 미스에서 외부 호출한다. Top-N은 `--top-n`으로 두 작업에 동일하게 적용한다.

## 캐시와 출력 경로

기본 출력은 Git 제외 `results/recommendation/daily/`, 공유 캐시는
`results/recommendation/model-cache.json`이다. `--output-dir`, `--cache`로 변경할 수 있다.
공유 캐시는 일일 출력 디렉터리 밖에 둔다. 분류·평가 입력 파일도 출력·캐시와 분리한다.

| 경로 | 내용 |
| --- | --- |
| `current-manifest.json` | 이번 실행에서 조회한 입력, skip에서도 갱신 |
| `last-applied.json` | 마지막 완전 적용 hash·설정 hash·BE 대상 |
| `active-run.json` | 부분 실행과 적용 상태 초기화 여부 |
| `summary.json` | skipped/generated/completed/partial, 실패 단계 수 |
| `runs/{manifestHash}/{configHash}/creator-manifest.json` | 해당 실행의 고정 입력 |
| 같은 경로의 `generation-config.json` | 유사 추천 설정 지문·관심 분야 설정·선택 계약 |
| 같은 경로의 `similar/` | seed별 payload·체크포인트·요약 |
| 같은 경로의 `interests/` | 분야별 양 방식 payload·체크포인트·요약·평가 입력 |

관심 분야는 기존 #41 비교 자료와 경로를 보존한다. M3 payload는 `payloads/`, M2는
`baselines/`에 있지만 실제 적재 방식은 통합 배치의 `selectedMethod`가 결정한다.

두 생성기는 같은 캐시 객체·임베딩 제공자·모델 버전을 공유한다. 소개·분야 설명 텍스트와
모델 버전이 같으면 벡터를 재사용한다. M4 태그는 소개·모델·프롬프트 전문·버전·taxonomyHash에
묶여 재사용한다. Creator 소개가 하나 바뀌면 전체 순위는 재생성하지만 바뀐 모델 입력만 다시 호출한다.

## 부분 실패·복구·잠금

부분 실패는 종료 코드 1, 완전 성공·skip은 0이다. 오류가 없는 단계도 계속 처리한다.
seed·분야별 상세 실패는 각 하위 디렉터리의 `summary.json`에서 확인한다.
통합 요약의 failureCount는 실패한 단계 수이며 seed·분야 실패 건수와 구분한다.

동일 명령을 재실행하면 같은 hash·설정의 성공 payload와 적재 체크포인트를 재사용하고
실패 부분만 재처리한다. payload hash가 맞지 않으면 해당 결과를 재생성한다.
이전 마지막 완료 마커는 부분 실패로 교체하지 않는다.

미완료 실행 도중 새 manifest·설정·BE 대상이 나타나면 **현재 입력으로 전환**한다.
이전 결과는 보존하지만 현재 실행과 섞지 않는다. 재사용하는 현재 실행의 적재 상태는
초기화해 모두 적재한다. 이전 완료 manifest로 돌아와도 부분 적용된 다른 세대를 덮어쓴다.
전환 초기화 중 종료되면 `resetRequired` 상태로 다음 실행에서 초기화를 마친다.

출력 `.batch.lock`과 캐시 옆 `.{cache파일명}.lock`은 Windows/Linux OS 파일 잠금이다.
기존 `batch_cli`·`interest_cli`도 동일한 공유 캐시 잠금을 사용한다. 정상·예외 종료와
프로세스 종료 시 OS가 잠금을 해제한다. 잠금 파일 자체는 남겨 inode 분리 경쟁을 막는다.
실행 중 잠금 파일을 삭제하지 않는다. 직접 Python 생성기를 호출하는 별도 프로그램도
`batch_locks`로 같은 캐시를 보호해야 한다. 잠금을 지원하는 로컬 파일시스템에서 실행한다.

이 완료 마커는 두 배치에 대한 로컬 완료 기록이다. BE의 모든 Creator·분야 세대를 한 번에
전환하는 전역 트랜잭션은 아니므로 부분 실행 중에는 이전·새 세대가 함께 조회될 수 있다.
공개 목록 또한 DB snapshot이 아니며 페이지 순회 중 같은 건수의 프로필 변경은 감지하지 못할 수 있다.

## 외부 전송·비용과 사전 확인

`dry-run`은 BE 쓰기와 완료 마커 갱신 없이 현재 입력의 결과를 생성한다.
캐시 미스에는 여전히 외부 전송·비용이 발생하므로 운영 데이터와 비용 범위를 확인한 후 실행한다.

```bash
python -m src.recommendation.daily_cli dry-run --be-base-url https://api.example.com
python -m pytest -q tests/recommendation
```

테스트는 가짜 모델·HTTP로 skip, 전체 재생성, 두 단계의 부분 실패와 재개, 완료 마커 중단,
hash 전환·이전 hash 복귀, 설정·대상 변경, API Key 전용 인증, 재시도·키 마스킹과
다른 프로세스의 잠금·종료 후 회수를 검증한다. 실제 운영 적재·유료 모델 호출은 수행하지 않는다.

### 공유 임베딩 캐시 계약

두 추천 생성기는 모델 원벡터를 float64 JSON 값으로 저장하고 같은 함수에서 float64 단위 벡터로 한 번 정규화합니다. 생성기 실행 순서나 디스크 캐시 재사용 여부가 점수에 영향을 주지 않습니다.

캐시 schemaVersion은 2입니다. 저장 표현이 불명확한 이전 v1 캐시는 재사용하지 않으며 최초 실행에서 모델 결과를 다시 채웁니다. 태그 캐시도 함께 초기화되므로 최초 실행 비용이 발생할 수 있습니다. 임베딩 계약 버전을 입력 해시와 생성 설정 해시에 포함하므로 이전 payload 및 체크포인트와 완료 상태를 재사용하지 않고 새 세대를 생성합니다.
