# 관심 분야별 M3 후보 생성과 BE 적재

[LLM #41](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/41)의 오프라인 생성기다.
`data/categories_v2.csv`의 17개 분야를 각각 독립 쿼리로 처리한다.
분야 조합·회원 선택·개인화 혼합·공개 분야별 추천 조회·인기순 fallback은 구현하지 않는다.

## 입력과 고정 설정

입력은 기존 `batch_cli manifest`로 만든 전체 Cking Creator manifest다.
`creatorId`는 양의 정수이고 중복은 같은 소개라도 거부한다. 소개는 CRLF/CR을 LF로 바꾸고
양끝 공백을 제거한다. 빈 소개는 임베딩·후보에서 제외하지만 manifest 해시에는 포함한다.
manifest의 Creator ID 순서·건수·내용 해시는 로드와 생성기 초기화에서 검증한다.

분류체계는 `v0.2`와 [정본 taxonomyHash](../reference/taxonomy-contract.md)를 함께 검증한다.
분야 코드·이름·설명·행 순서 중 하나라도 다르면 시작 전에 거부한다.

| 설정 | 기본값 | 허용 범위 |
| --- | --- | --- |
| 생성 Top-N | 20 | 정수 1~100 |
| 평가 Top-N | 10 | 정수 1~생성 Top-N |
| zero-shot tau | 0.4296248555 | 유한한 수 -1~1 |
| zero-shot 최대 태그 | 3 | 정수 1~17 |
| M3 가산점 | 0.1 | 유한한 수 0~1 |
| 점수 소수 자릿수 | 8 | 정수 0~15 |
| 임베딩 배치 크기 | 96 | 정수 1~10000 |
| 임베딩 모델 | BAAI/bge-m3 | 모델은 고정, 제공자·버전은 구분 |
| 분류체계 버전 | v0.2 | 이번 생성기는 v0.2만 지원 |

tau·가산점은 기존 dev 실험의 전이 설정을 명시적으로 고정한 값이다.
현재 Cking 데이터에 최적화한 값으로 주장하지 않는다. 실제 채택은 후속 #44의 평가 대상이다.
설정에는 bool·NaN·Infinity·잘못된 범위·빈 모델 버전을 허용하지 않는다.

## 점수와 결정성

1. 17개 분야 설명과 빈 소개를 제외한 Creator 소개를 BGE-M3 dense 벡터로 만든다.
2. 행 수·클라이언트 기대 차원·유한성·양의 norm을 검증하고 단위 벡터로 정규화한다.
3. 후보 소개와 같은 17개 분야 설명의 코사인을 계산한다. 코사인 내림차순,
   동점이면 분야 CSV의 노출 순서로 정렬해 상위 최대 3개 중 `cosine >= tau`를 태그로 확정한다.
4. `INTEREST_M3_V1` 점수는 쿼리 분야 코사인에 해당 분야 태그가 있으면 가산점 0.1을 더한다.
   `INTEREST_M2_V1`은 같은 코사인만 쓴다. tau 미달 후보도 기본 코사인으로 순위에 참여한다.
5. Python `round`의 ties-to-even 규칙으로 점수를 먼저 반올림하고,
   반올림 점수 내림차순·숫자 Creator ID 오름차순으로 Top-N을 고른다. rank는 1부터 연속이다.

후보가 20명보다 적으면 가능한 전원을 반환한다. 소개가 모두 비거나 manifest가 비면
17개 분야 모두 메타데이터를 유지한 `candidates: []`를 만든다. 빈 세대도 apply 대상으로 삼는다.
M2와 M3는 같은 manifest·Top-N·설명 벡터를 공유하며 BE에는 M3만 보낸다.

`inputHash`는 compact JSON의 SHA-256 소문자 hex다. 분야별 해시에는 분류 버전·해시,
분야 코드·이름·설명, 전체 manifest 해시, Top-N, 방식, 임베딩 모델 버전,
tau·최대 태그 수·가산점·자릿수·반올림·정렬·빈 소개 정책을 넣는다.
모델 실행 배치 크기는 결과 계약에 영향을 주지 않으므로 해시에서 제외한다.
설정 전문은 `summary.json`의 `generationConfig`와 평가 `provenance.json`에도 기록한다.

모델 캐시는 기존 `JsonModelCache`를 공유한다. 소개/설명 텍스트 해시와 모델 버전이 같으면
분야 간·재실행 간·기존 Creator 추천과 임베딩을 재사용한다. 깨진 캐시 벡터는 다시 생성하며
비정상 모델 응답은 저장하지 않는다. 첫 실행과 캐시 실행에서 같은 원벡터를 정규화해 payload가 동일하다.
zero-shot 태그는 저장 벡터와 현재 설정으로 다시 계산하고 LLM 태거를 호출하지 않는다.

## 실행

```bash
# 기존 전체 Creator manifest 생성: 이 명령만 BE 공개 목록을 읽는다.
python -m src.recommendation.batch_cli manifest \
  --be-base-url https://your-cking-be.example \
  --output results/recommendation/batch/creator-manifest.json

# 로컬 BGE-M3. payload·비교 자료·체크포인트·요약만 생성한다.
python -m src.recommendation.interest_cli dry-run \
  --manifest results/recommendation/batch/creator-manifest.json \
  --output-dir results/recommendation/interests \
  --cache results/recommendation/model-cache.json

# DeepInfra를 사용하도록 명시한 경우. 캐시 미스에 외부 전송·비용이 발생한다.
python -m src.recommendation.interest_cli dry-run \
  --embedding-provider deepinfra \
  --output-dir results/recommendation/interests-deepinfra

# 생성한 결과를 동일 설정으로 적재한다. 환경 변수로 인증 정보를 설정한다.
python -m src.recommendation.interest_cli apply \
  --be-base-url https://your-cking-be.example \
  --output-dir results/recommendation/interests
```

PowerShell에서는 줄 연속 기호를 바꾸거나 명령을 한 줄로 실행한다.
로컬 모델 버전은 `BAAI/bge-m3@local-v1`, DeepInfra는 `BAAI/bge-m3@deepinfra-v1`이다.
모델 스냅샷·라이브러리·전처리·제공자 설정이 바뀌면 `--embedding-model-version`에 새 식별자를 넣어
이전 벡터를 혼용하지 않는다. 로컬 실행은 모델 파일이 없으면 다운로드가 필요하다.
비기본 Top-N이 10 미만이면 `--evaluation-top-n`도 함께 줄인다.

`apply`는 `CKING_RECOMMENDATION_API_KEY`를 `X-Cking-Recommendation-Key`로 보낸다.
키가 없으면 모델·BE 쓰기 전에 실패한다. ADMIN JWT를 함께 보내지 않는다.
키/JWT는 공개 manifest GET에 보내지 않으며, 체크포인트와 실패 요약에도 기록하지 않는다.
인증 값은 CLI 인자로 받지 않는다. BE #420의 인증 연결은 해당 적재 경로에도 적용되어야 한다.

공유 캐시와 실행 상태는 OS 파일 잠금으로 보호하며 동시 배치를 거부한다.
이 명령은 #41의 M3 생성·평가 경로다. #44에서 선택한 M2/M3를 실제 적재하는 일일 실행은
[manifest 감지 통합 배치](daily-recommendation.md)를 사용한다.

## 적재 요청·응답 계약

요청은 분야마다 `PUT /api/admin/interests/{interestCode}/recommendations`다.
URL 코드와 body 코드는 동일한 v0.2 정본에서 가져온다. body의 필드는 다음과 같다.

```json
{
  "taxonomyVersion": "v0.2",
  "taxonomyHash": "f77df7a020a8ebb7cb33c1177f098babbc6a58d2d487dbaabcfa8108bf7f8b35",
  "interestCode": "FITNESS",
  "method": "INTEREST_M3_V1",
  "modelVersion": "BAAI/bge-m3@local-v1",
  "inputHash": "64자리 SHA-256 소문자 hex",
  "candidates": [
    {
      "creatorId": 7,
      "rank": 1,
      "score": 0.81234567,
      "taxonomyVersion": "v0.2",
      "taxonomyHash": "f77df7a020a8ebb7cb33c1177f098babbc6a58d2d487dbaabcfa8108bf7f8b35",
      "interestCode": "FITNESS",
      "method": "INTEREST_M3_V1",
      "modelVersion": "BAAI/bge-m3@local-v1",
      "inputHash": "최상위 inputHash와 같은 값"
    }
  ]
}
```

빈 세대도 같은 최상위 메타데이터와 빈 배열을 보낸다. M3 점수 범위는 -1~1+가산점이다.
요청 내 메타데이터·ID 중복·후보 수·순위·정렬·유한한 점수를 적재 전에 검증한다.

기존 Creator 적재 응답 구조를 확장해 다음 `data`를 기대한다.

```json
{
  "taxonomyVersion": "v0.2",
  "interestCode": "FITNESS",
  "inputHash": "요청과 같은 inputHash",
  "candidateCount": 20,
  "applied": true
}
```

`applied: false`는 멱등 성공이다. 응답 분야·분류 버전·입력 해시·후보 건수·boolean을 검증한다.
응답에 `taxonomyHash`가 있으면 요청과 일치하는지도 검증한다.
공통 `ApiResponse.data` 봉투를 해제하며, 일시적 HTTP 상태/timeout/끊긴 응답은 제한 재시도한다.
BE의 멱등 키는 `(taxonomyVersion, interestCode, inputHash)`, 포인터는 `(taxonomyVersion, interestCode)`를 전제로 한다.

BE 적재 API는 Cking-BE #439로 구현·병합됐다. 요청·응답·키 인증·빈 세대 활성화·멱등 동작의 최종 계약은
[BE Interest API](https://github.com/URECA-Cking/Cking-BE/blob/develop/docs/domains/interest/api.md#put-apiadmininterestsinterestcoderecommendations)가 정본이다.
2026-10-06에 이 구현의 요청·응답 검증 조건과 BE DTO·검증 코드를 코드 기준으로 대조했고 서로 맞는다.
이 구현의 fake BE 테스트를 실제 BE 통합 검증으로 해석하지 않는다.

## 출력과 재개

| 파일 | 역할 |
| --- | --- |
| `payloads/interest-CODE.json` | BE 적재용 M3 Top-N |
| `baselines/interest-CODE.json` | 같은 manifest의 M2 Top-N, 평가 전용 |
| `checkpoint.json` | 분야별 `generationStatus`와 `applyStatus`, 두 payload 해시, 적용 대상 |
| `summary.json` | 설정, 후보 수, 성공·빈 세대·생성 실패·적재 실패·멱등·재사용 건수 |
| `evaluation/judge-input.json` | M2/M3 Top-10 합집합의 블라인드 판정 입력 |
| `evaluation/provenance.json` | 방식별 Top-10·순위·점수·inputHash·부족 수·pair 출처·판정 입력 해시 |

모든 JSON은 같은 디렉터리의 임시 파일을 완성한 뒤 원자 교체한다.
실패 분야가 있어도 나머지 분야를 처리한다. 성공·빈 세대는 완료 세대이며 모델을 다시 부르지 않는다.
공통 임베딩 준비에 실패하면 같은 실행에서 반복 모델 호출하지 않고 분야별 생성 실패를 남긴다.
같은 명령을 재실행하면 생성 실패 분야와 적재 실패 분야만 재시도한다.
BE 주소가 바뀌면 저장 payload는 재사용하고 모든 분야의 적재 상태를 초기화한다.
manifest 또는 생성/평가 설정이 바뀌면 기존 체크포인트를 거부하므로 새 `--output-dir`을 쓴다.
payload가 손상되거나 세대 계약이 다르면 해당 분야를 재생성하고 다시 적재한다.
실패가 있으면 CLI 종료 코드는 1이고 모두 성공하면 0이다.

## 블라인드 평가와 한계

17개 분야의 M2/M3 생성이 완료되어야 평가 파일을 만든다. 후보 부족은 가능한 후보만
포함하고 출처 파일에 부족 수를 기록한다. 빈 manifest는 정상적인 0쌍 평가 입력을 만든다.
각 분야에서 동일 Creator가 두 방법에 나타나면 판정 쌍 하나로 합친다.
Judge 입력에는 분야 설명·후보 소개·ID·텍스트 해시만 포함하며 방식·rank·score·가산점은 넣지 않는다.
행은 pair 해시로 정렬해 방법별 순위가 행 순서로 노출되지 않게 한다.
pair ID는 분류 버전/해시, 분야 코드, 쿼리·소개 텍스트 해시, Creator ID에 묶인다.
소개가 바뀐 옛 판정은 그대로 재사용할 수 없다. 별도 #44에서 판정 프롬프트·모델 버전도 고정해야 한다.
같은 계약의 평가 파일은 쓰지 않고 보존하며 다른 내용의 기존 파일은 덮어쓰지 않는다.

로컬 BGE-M3는 소개를 모델 API로 보내지 않는다. DeepInfra 모드는 캐시 미스인 소개와
분야 설명을 외부 API로 전송하고 제공자 과금이 발생한다. API Key는 `DEEPINFRA_API_KEY`다.
BE apply에는 원문 소개가 없는 ID·순위·점수·세대 메타데이터만 보낸다.
평가 파일에는 원문 소개가 포함되므로 `results/`에 보관하며 Git에 커밋하지 않는다.
이 CLI는 LLM Judge를 호출하지 않는다. #44에서 외부 판정을 실행할 때 별도 전송·비용을 확인한다.

자동 판정은 실제 사용자 반응·사람 판정·다중 관심사 혼합 품질을 증명하지 않는다.
모델 제공자의 스냅샷 변경이나 로컬 런타임 변경은 동일 원문에서도 점수를 바꿀 수 있으므로
모델 버전을 갱신해야 한다. 실제 manifest 전체 실행·유료 API 호출·실제 BE 적재는 이번 테스트 범위 밖이다.

```bash
python -m pytest -q tests/recommendation
```

테스트는 가짜 임베딩·가짜 BE로 수행하며 유료 API와 실제 서버에 요청하지 않는다.

적재 실행 번호·409·schema 2 전환·재적용 절차는 [일일 배치 계약](daily-recommendation.md#적용-실행-번호-계약-52)을 따른다. 단독 CLI도 공유 캐시의 ledger를 사용하며 같은 출력의 재개에는 체크포인트 번호를 유지한다.
