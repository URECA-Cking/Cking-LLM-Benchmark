# 세부 태그 고정 표본과 사람 검수 (#58)

이 도구는 분류 타당성을 확인한다. 추천 순위나 서비스 정확도 개선을 입증하지 않는다.
기존 `selected_subtopic` 도구는 선택 관심사에 따른 순위 실험이므로 검수 집계에는 사용하지 않는다.
대신 `flow_eval.load_source`의 캐시 검증, `subtopic_tags.load_taxonomy`, 기존 분류 결과와
`detail_tagging.RevisionStore`를 재사용한다. 외부 API 호출은 없다.

## 실행

저장소 루트에서 실행한다. 기본 입력은 `inputs/real_v4`, `results/real_v4`,
`results/subtopics_v1`이다. 입력/상위 태그/분류 계약 해시가 다르거나 필요한 자동 결과가
없으면 중단한다. 출력은 **존재하지 않는 새 디렉터리**로 지정한다.
모든 파일을 같은 부모의 임시 디렉터리에 기록한 뒤 완성된 디렉터리를 게시한다.
쓰기 실패 시 임시 파일을 정리하므로 같은 출력 경로로 재시도할 수 있다. 기존 출력은 덮어쓰지 않는다.

```powershell
python -m src.sample_review prepare --output-dir results/sample-review-v1
python -m src.sample_review aggregate --manifest results/sample-review-v1/manifest.json --reviews results/sample-review-v1/reviews.csv --output-dir results/sample-report-v1
```

`--size` 기본값은 50이다. 채널 ID 순으로 동률을 해결하면서 아직 확보하지 못한 상위 분야와
경계 유형을 가장 많이 포함하는 채널을 먼저 선정한다. 나머지는 같은 순서로 채운다.
17개 분야, 중복, 미분류, 세부 태그 없음, 상한 초과, 반례의 부족 여부와 표본 수 부족을 manifest에 남긴다.
`unclassified`는 #57과 동일하게 **부모 분야가 없는 경우**다.
`no_subtopic`은 부모 분야는 있지만 세부 태그가 없는 경우다. 두 유형을 별도로 확보·집계한다.
부족한 사례를 합성 채널로 실제 표본에 섞지 않는다. 검증 테스트만 합성 입력을 사용한다.

사람이 지정한 반례는 채널 ID JSON 배열을 `--counterexamples results/counterexamples.json`으로
전달한다. 모델 오류만으로 반례라고 자동 판단하지 않는다. 알려지지 않은 ID는 거부한다.
선정 이유·부모 분야·경계 유형·입력 해시·자동 태그와 인용 근거를 고정한다.

## #57 자동 분류 버전 사용

기존 캐시를 사용할 수도 있고, #57에서 저장한 자동 분류 revision 파일을 사용할 수도 있다.
`--revision-index` 파일은 다음 형식이며, 경로는 index 파일의 디렉터리를 기준으로 해석한다.
이 모드에서는 index에 명시한 실제 채널들만 모집단이 된다. 충분한 revision을 준비해야 한다.

```json
{"실제채널ID": "detail-store/32자리revisionID.json"}
```

입력 해시, taxonomy 기준 해시, 정책 버전, 저장된 결과를 원래 후보로 재계산한 결과까지
검증한다. manifest에는 원본 revision ID를 보존한다. 기본 상한은 상위 분야 3개,
**분야별 세부 태그 3개**이며 `--limit`으로 분야별 상한을 고정할 수 있다.
부모 상한은 고정 3개이며 manifest의 `parent_limit`에 남긴다.
분야별 상한을 3 이외로 바꾸면 검수 정책 버전은 `sample-review-custom-limit-v1`이다.
원본 자동 분류 정책(`classification_contract.automatic_policy_version`)과 검수 정책은 구분한다.
자동 결과의 중복·잘못된 태그·부모 불일치·상한 위반·잘못된 인용은 진단 항목으로 보존한다.

legacy 모드에서는 캐시에 남은 중복·상한 위반을 진단 대상으로 선정할 수 있다.
#57 revision 모드는 `classify`를 통과한 결과만 사용하므로 기본 상한에서 중복·상한 위반은
구조적으로 확보되지 않는다. 이때 `missing_coverage`는 정상적인 입력 경로의 한계다.
검수 상한을 더 낮게 설정한 실험에서는 원본 자동 결과가 검수 상한을 넘을 수 있다.

해시는 다음처럼 용도가 다르며 서로 직접 비교하지 않는다.

| 필드 | 해시 대상/직렬화 |
| --- | --- |
| manifest `hash`, `taxonomy_hash`, `source_hash`, `legacy_plan_hash` | `flow_eval.digest`: 전체 실험 JSON, 키 정렬, 기본 JSON 구분자 |
| #57 `input_hash`, `criteria_hash` | `detail_tagging.digest`: 키 정렬, compact JSON 구분자 |
| #59 `be_hash` | `subtopic_hash`: 정규화한 세부 태그 필드와 배열 순서의 BE canonical 계약 |
| 집계 revision `reviews_hash` | BOM 제거·줄바꿈 정규화 텍스트의 `flow_eval.digest`; 원본 바이트 SHA-256이 아님 |
| 집계 revision `reviews_file_sha256` | CSV 원본 바이트 SHA-256; `sha256sum` 또는 `Get-FileHash`와 비교 가능 |

manifest `hash_scheme`, revision `reviews_hash_scheme`에도 형식을 남긴다.

## 사람 검수

`reviews.csv`는 UTF-8 BOM CSV이다. `channel_id`, `input_hash`, `auto_topics`,
`auto_evidence`는 변경하지 않는다. 다음 열을 사람이 입력한다.

| 열 | 입력 |
| --- | --- |
| status | `pending` / `reviewed` / `undecided` |
| human_topics | 세부 태그 코드의 `\|` 구분 목록. 검수 후 빈 값은 세부 태그 없음 확정 |
| verdict | 동일하면 `agree`, 수정 또는 자동 유효성 오류가 있으면 `corrected` |
| cause | 동일하면 `none`, 수정 시 `model_error` / `criteria_problem` / `both` |
| reason | 검수 또는 미결정 이유 |
| reviewer | 실제 검수자 식별자 |

`pending`에는 검수 열을 채우지 않는다. `reviewed`와 `undecided`는 검수자와 이유가 필수다.
미결정은 정답 비교 집계에서 제외한다. 확정 태그의 유효성·중복·분야별 상한·부모 관계와
판정의 일치 여부를 검사한다. 채널 삭제/추가/중복, 원본 해시와 자동 결과 수정도 거부한다.
코드 앞뒤 공백은 제거하지만 빈 구분 항목·중복 코드는 거부한다.
CSV의 누락·중복·추가 헤더, 짧은 행의 누락 셀과 초과 셀을 먼저 검사하며 오류에 행/채널 ID를 표시한다.
분야 자체를 바꿔야 한다면 원래 표본을 수정하지 말고 새 자동 분류 revision과 표본을 만든다.
따라서 누락·과다는 **고정된 자동 부모 분야 안의 세부 태그 비교**다. 부모 분야 오류는
`undecided`로 남기고 별도 분류 수정을 진행한다. 이 제약은 보고서 `limitations`에도 남긴다.

보고서 `status`는 다음과 같다. 검수 완료와 표본 충분성을 함께 확인한다.

| status | 의미 |
| --- | --- |
| `human_review_pending` | 미검수 또는 미결정 항목이 하나 이상 있음 |
| `reviewed_with_gaps` | 모든 항목 검수 완료, 분야/경계 유형 또는 표본 수 부족 |
| `review_complete` | 모든 항목 검수 완료, 명시한 분야/경계 유형과 표본 수 부족 없음 |

새 표본은 schema version 2다. 과거 표본은 고정된 경계 정의를 그대로 보존하므로
미분류와 세부 태그 없음의 분리 기준으로 비교하려면 새 디렉터리에 표본을 다시 생성한다.

## 산출물과 공유 경계

- `manifest.json`: 표본 및 분류 계약 고정. 내용을 바꾸면 집계가 실패한다.
- `private_inputs.json`: 검수자가 원문을 확인할 로컬 참조.
- `reviews.csv`: 자동 인용과 사람 판정이 함께 있는 로컬 검수표.
- `report.json`, `report.md`: 미검수/완료/미결정 수, 누락·과다 채널 수, 자동/사람 미분류 및 세부 태그 없음,
  오류 원인별 수, 자동 유효성 오류 수, 수정 목록·미결정 항목·분야/경계 구성·한계.
- `revisions/`: 기존 RevisionStore로 저장한 집계 revision. 검수표 스냅샷과 해시도 보존한다.

원문과 인용이 포함되는 산출물은 Git에서 제외되는 `results/`에 보관한다.
**검수표·manifest·private_inputs·revision을 공개 이슈/로그에 붙이지 않는다.**
보고서의 사람이 입력한 reason에도 개인정보나 원문을 쓰지 않고, 공유 전 내용을 확인한다.
미검수 0개라도 부족한 분야나 경계 유형은 해소되지 않으며 대표 정확도 추정으로 사용하지 않는다.

## 일정과 실제 실행 확인

2026-10-13 최초 로컬 실행에서 보고한 '미분류 유형 확보'는 세부 태그 없음도 포함한 구 정의였다.
리뷰 후 schema version 2로 재실행: 모집단 4,743개에서 50개 선정, 17개 분야 확보.
부모 미분류 3개, 부모는 있지만 세부 태그 없음 4개다. 중복·상한 초과·사람 지정 반례는 부족하다.
50개 모두 `pending`이고 보고서 상태는 `human_review_pending`이다. 실제 사람 검수 결과가 아니다.
새 모델 호출이나 기존 분류 결과 수정은 하지 않았다.

- 10/20: 사람이 1차 검수 후 집계하고 결과를 공유한다.
- 10/22: 수정 목록과 미결정 항목, 부족한 표본을 보고 시연 포함 여부를 결정한다.
- 시연 포함 시 10/24: 사람의 승인을 받은 확정 목록/fixture를 전달한다.

코드 구현은 이 일정 공유나 사람 검수의 완료를 대신하지 않는다. PR도 이슈를 자동으로 닫지 않는다.
