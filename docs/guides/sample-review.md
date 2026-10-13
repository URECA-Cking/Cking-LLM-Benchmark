# 세부 태그 고정 표본과 사람 검수 (#58)

이 도구는 분류 타당성을 확인한다. 추천 순위나 서비스 정확도 개선을 입증하지 않는다.
기존 `selected_subtopic` 도구는 선택 관심사에 따른 순위 실험이므로 검수 집계에는 사용하지 않는다.
대신 `flow_eval.load_source`의 캐시 검증, `subtopic_tags.load_taxonomy`, 기존 분류 결과와
`detail_tagging.RevisionStore`를 재사용한다. 외부 API 호출은 없다.

## 실행

저장소 루트에서 실행한다. 기본 입력은 `inputs/real_v4`, `results/real_v4`,
`results/subtopics_v1`이다. 입력/상위 태그/분류 계약 해시가 다르거나 필요한 자동 결과가
없으면 중단한다. 출력은 **존재하지 않는 새 디렉터리**로 지정한다.

```powershell
python -m src.sample_review prepare --output-dir results/sample-review-v1
python -m src.sample_review aggregate --manifest results/sample-review-v1/manifest.json --reviews results/sample-review-v1/reviews.csv --output-dir results/sample-report-v1
```

`--size` 기본값은 50이다. 채널 ID 순으로 동률을 해결하면서 아직 확보하지 못한 상위 분야와
경계 유형을 가장 많이 포함하는 채널을 먼저 선정한다. 나머지는 같은 순서로 채운다.
17개 분야, 중복, 미분류, 상한 초과, 반례의 부족 여부와 표본 수 부족을 manifest에 남긴다.
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
자동 결과의 중복·잘못된 태그·부모 불일치·상한 위반·잘못된 인용은 진단 항목으로 보존한다.

## 사람 검수

`reviews.csv`는 UTF-8 BOM CSV이다. `channel_id`, `input_hash`, `auto_topics`,
`auto_evidence`는 변경하지 않는다. 다음 열을 사람이 입력한다.

| 열 | 입력 |
| --- | --- |
| status | `pending` / `reviewed` / `undecided` |
| human_topics | 세부 태그 코드의 `\|` 구분 목록. 검수 후 빈 값은 미분류 확정 |
| verdict | 동일하면 `agree`, 수정 또는 자동 유효성 오류가 있으면 `corrected` |
| cause | 동일하면 `none`, 수정 시 `model_error` / `criteria_problem` / `both` |
| reason | 검수 또는 미결정 이유 |
| reviewer | 실제 검수자 식별자 |

`pending`에는 검수 열을 채우지 않는다. `reviewed`와 `undecided`는 검수자와 이유가 필수다.
미결정은 정답 비교 집계에서 제외한다. 확정 태그의 유효성·중복·분야별 상한·부모 관계와
판정의 일치 여부를 검사한다. 채널 삭제/추가/중복, 원본 해시와 자동 결과 수정도 거부한다.
분야 자체를 바꿔야 한다면 원래 표본을 수정하지 말고 새 자동 분류 revision과 표본을 만든다.

## 산출물과 공유 경계

- `manifest.json`: 표본 및 분류 계약 고정. 내용을 바꾸면 집계가 실패한다.
- `private_inputs.json`: 검수자가 원문을 확인할 로컬 참조.
- `reviews.csv`: 자동 인용과 사람 판정이 함께 있는 로컬 검수표.
- `report.json`, `report.md`: 미검수/완료/미결정 수, 누락·과다 채널 수, 자동/사람 미분류,
  오류 원인별 수, 자동 유효성 오류 수, 수정 목록·미결정 항목·분야/경계 구성·한계.
- `revisions/`: 기존 RevisionStore로 저장한 집계 revision. 검수표 스냅샷과 해시도 보존한다.

원문과 인용이 포함되는 산출물은 Git에서 제외되는 `results/`에 보관한다.
**검수표·manifest·private_inputs·revision을 공개 이슈/로그에 붙이지 않는다.**
보고서의 사람이 입력한 reason에도 개인정보나 원문을 쓰지 않고, 공유 전 내용을 확인한다.
미검수 0개라도 부족한 분야나 경계 유형은 해소되지 않으며 대표 정확도 추정으로 사용하지 않는다.

## 일정과 실제 실행 확인

2026-10-13 로컬 캐시 실행: 모집단 4,743개에서 표본 50개, 17개 분야와 미분류 유형 확보.
중복·상한 초과·사람 지정 반례는 부족했다. 50개 모두 `pending`이다. 실제 사람 검수 결과가 아니다.
새 모델 호출이나 기존 분류 결과 수정은 하지 않았다.

- 10/20: 사람이 1차 검수 후 집계하고 결과를 공유한다.
- 10/22: 수정 목록과 미결정 항목, 부족한 표본을 보고 시연 포함 여부를 결정한다.
- 시연 포함 시 10/24: 사람의 승인을 받은 확정 목록/fixture를 전달한다.

코드 구현은 이 일정 공유나 사람 검수의 완료를 대신하지 않는다. PR도 이슈를 자동으로 닫지 않는다.
