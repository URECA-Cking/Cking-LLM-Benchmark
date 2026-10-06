# 전체 크리에이터 유사 추천 배치 생성·BE 적재

[문서 안내](../README.md) · [단일 생성 계약](service-recommendation.md) · [현재 방식 결정](../results/recommendation-decision.md)

## 범위와 안전 경계

이 배치는 Cking-BE의 공개 Creator 목록을 한 번 고정한 뒤 모든 seed에 단일 생성기의 M4/M2 분기를 반복 적용한다.
사용자 추천 조회 중에는 실행하지 않으며, 스케줄러 배포·프로필 변경 이벤트·회원별 개인화·인기순 fallback·추천 UI는 범위 밖이다.

- `manifest`: `GET /api/creators?page={page}&size=100`을 끝까지 읽어 결정적 입력을 저장한다.
- `dry-run`: 외부 모델 캐시를 재사용해 seed별 BE payload와 실행 요약을 저장한다. BE 쓰기 요청은 하지 않는다.
- `apply`: 같은 생성 과정을 거친 뒤에만 `PUT /api/admin/creators/{creatorId}/similar`을 호출한다.
- ADMIN Access Token과 모델 API Key는 환경 변수에서만 읽으며 manifest, payload, 체크포인트, 요약에 기록하지 않는다.

manifest와 배치 결과는 기본적으로 `results/recommendation/batch/`에 저장되며 Git에 포함되지 않는다.

## 1. Creator manifest 고정

```bash
python3 -m src.recommendation.batch_cli manifest \
  --be-base-url https://api.example.com \
  --output results/recommendation/batch/creator-manifest.json
```

모든 페이지의 `page`, `size`, `totalElements`, `totalPages`, `hasNext`를 검사한다. 항목의 `creatorId`는 양의 정수여야 하고
중복될 수 없으며 `introText`는 문자열이어야 한다. 전체를 `creatorId ASC`로 정렬하고 줄바꿈·앞뒤 공백을 정규화한 후
건수와 lowercase SHA-256을 기록한다. 페이지를 읽는 사이 전체 건수가 바뀌면 중단하므로 다시 실행해 일관된 snapshot을 만든다.

기존 manifest는 기본적으로 덮어쓰지 않는다. 운영 Creator 목록 또는 소개가 바뀐 뒤 새 입력을 의도적으로 고정할 때만
`--refresh`를 사용한다.

```bash
python3 -m src.recommendation.batch_cli manifest \
  --be-base-url https://api.example.com \
  --output results/recommendation/batch/creator-manifest.json \
  --refresh
```

manifest는 트랜잭션 DB snapshot이 아니다. 페이지 순회가 끝난 시점의 검증된 목록이지만, 순회 중 이름 정렬 경계에서
건수 변화 없이 프로필이 수정되면 이를 탐지하지 못할 수 있다. 실행 직전 최신성이 중요하면 새 manifest를 만들고 해시를 확인한다.

## 2. dry-run

서비스 기본 분류는 `data/categories_v2.csv`의 17개 분야, `taxonomyVersion=v0.2`,
`tagPromptVersion=creator-category-v2`다. 초기 10개 `data/categories.csv`는 과거 실험용으로 보존한다.
[공통 taxonomyHash 계약](../reference/taxonomy-contract.md)에 따라 분류 내용·행 순서 변경도 태그 캐시와 생성 설정에 반영한다.
v0.1 체크포인트는 새 기본값과 호환되지 않으므로 v0.2 실행에는 새 `--output-dir`을 사용한다.


```bash
python3 -m src.recommendation.batch_cli dry-run \
  --manifest results/recommendation/batch/creator-manifest.json \
  --output-dir results/recommendation/batch \
  --cache results/recommendation/model-cache.json \
  --top-n 5
```

출력은 다음과 같다.

| 경로 | 내용 |
| --- | --- |
| `payloads/creator-{id}.json` | 기존 BE 적재 계약과 같은 seed별 완결 payload |
| `checkpoint.json` | manifest·생성 설정 해시, payload 해시, 정규화된 BE 적용 대상, 생성·적재 상태 |
| `summary.json` | 비어 있지 않은 세대, 빈 세대, 신규·재사용 생성, 적재·멱등·실패 건수와 seed별 오류 |

체크포인트와 각 payload는 임시 파일을 완전히 쓴 뒤 원자 교체한다. 중간에 실패하면 이미 성공한 payload와 공용 모델 캐시는
남고, 같은 manifest와 설정으로 다시 실행할 때 성공 seed는 모델을 다시 호출하지 않는다. 실패 seed만 다시 처리한다.
payload가 없거나 내용 SHA-256이 체크포인트와 다르면 해당 seed를 안전하게 다시 생성한다.

manifest 또는 Top-N, 모델, 프롬프트, 분류체계, M4 설정이 바뀌면 기존 체크포인트를 재사용하지 않고 중단한다.
새 입력·설정은 별도 `--output-dir`에서 실행하거나 기존 로컬 결과를 보관한 뒤 새 디렉터리를 사용한다.

## 3. apply

먼저 본인 환경에 필요한 키를 `.env` 또는 셸 환경 변수로 설정한다. 애플리케이션은 `.env` 값을 환경으로 로드하지만
키 값 자체를 출력 파일에 복사하지 않는다.

```bash
export CKING_ADMIN_ACCESS_TOKEN='...'
export DEEPINFRA_API_KEY='...'
export OPENAI_API_KEY='...'

python3 -m src.recommendation.batch_cli apply \
  --manifest results/recommendation/batch/creator-manifest.json \
  --output-dir results/recommendation/batch \
  --cache results/recommendation/model-cache.json \
  --be-base-url https://api.example.com \
  --top-n 5 \
  --timeout 10 \
  --retries 2 \
  --retry-backoff 0.5
```

Windows PowerShell에서는 `$env:CKING_ADMIN_ACCESS_TOKEN='...'` 형식으로 설정한다. `CKING_ADMIN_ACCESS_TOKEN`은 apply에 항상
필요하다. `DEEPINFRA_API_KEY`는 임베딩 캐시 미스, `OPENAI_API_KEY`는 M4 태그 캐시 미스가 있을 때 필요하다.

BE 응답의 `creatorId`, `inputHash`, `candidateCount`, `applied`를 요청과 대조한다. `applied=false`는 정상 멱등 결과로 기록한다.
신규 적용과 멱등 완료 seed는 체크포인트에서 건너뛰고, 적용 실패 seed만 재호출한다. 빈 세대도 정상 payload로 보내 이전 활성 추천을 비운다.

체크포인트의 `applyTarget`에는 scheme과 host의 대소문자, HTTP(S) 기본 port, 끝 슬래시를 정규화한 BE base URL을 기록한다.
같은 대상으로 재실행하면 적용 완료 또는 멱등 seed를 건너뛴다. 다른 대상으로 변경하면 생성 payload와 모델 캐시는 재사용하되
기존 적용 상태만 `pending`으로 초기화하고 모든 payload를 새 대상에 다시 적재한다. `applyTarget`이 없는 구형 체크포인트는 기존 적용
대상을 확인할 수 없으므로 같은 안전 정책을 적용해 생성 결과는 재사용하고 모든 payload를 다시 적재한다.

## 재시도와 오류 정책

BE base URL, Top-N, timeout, 재시도 횟수와 첫 backoff는 CLI 설정으로 관리한다. 네트워크·timeout·불완전한 응답 본문과 HTTP
`408`, `425`, `429`, `500`, `502`, `503`, `504`만 지수 backoff로 제한 재시도한다. 인증·검증 등 다른 4xx는 즉시 실패한다.
seed별 오류에는 단계, 예외 종류, 메시지만 남기며 알려진 Access Token과 모델 API Key 값은 치환한다.

배치는 실패가 하나라도 있으면 종료 코드 1을 반환하지만 다른 seed 처리는 계속한다. `summary.json`의 `failures`에서 seed와
`generation` 또는 `apply` 단계를 확인한 뒤 같은 명령을 다시 실행한다.

## 비용과 외부 전송

manifest 조회와 BE 적재 자체는 모델 비용이 없지만, 캐시에 없는 소개는 외부 모델 제공자에 전송된다. M2/M4 모두 전체
후보 풀의 임베딩이 필요하고 M4는 상위 분야 태깅도 필요하므로, 운영 전체 데이터를 처음 실행하면 데이터 외부 전송과 비용이
발생한다. 실행 전 조직의 데이터 전송 정책, 대상 manifest 건수, 캐시 보유 여부와 API 결제 계정을 확인한다.

`dry-run`도 모델 캐시 미스가 있으면 유료 호출을 수행한다. 비용 없는 사전 검증은 가짜 BE 응답과 가짜 모델 클라이언트를 쓰는
테스트로 한다.

```bash
python3 -m pytest -q tests/recommendation
```

테스트는 실제 BE 또는 유료 모델 API를 호출하지 않는다. 페이지네이션, 일시적 오류 재시도, dry-run 쓰기 차단, 재시작,
생성·적재 부분 실패, 빈 세대, 멱등 응답, payload 계약, 체크포인트·manifest 해시 불일치를 검증한다.
