# 세부 태그 상한 정책과 검수 버전

Issue #57의 오프라인 결과 처리 경로는 `python -m src.detail_tagging`이다.
기존 `src/subtopic_tags.py`의 taxonomy/인용 검증을 재사용한다. 과거 실험은 전체
최대 3개 세부 태그, zero-shot은 상위당 1개였다. selected_subtopic 계열은 사용자가
선택한 태그로 추천을 비교하는 실험이므로 변경하지 않는다. 새 결과는 추천 순위에
반영하거나 BE에 적재하지 않는다. 현재 taxonomy는 여전히 실험 초안이다.

## 분류 계약

모델 후보를 제한 없이 받아 검증한 다음 상위 최대 3개, 각 상위의 세부 최대 3개,
총 최대 9개를 선택한다. 정의와 include/exclude를 충족하는 경우 복수 부여하고
근거가 부족하면 빈 배열을 허용한다. 상위만 있고 세부는 없는 결과도 허용한다.
`unclassified`는 상위도 없는 경우다. 부모 불일치, 미지 코드, 중복 코드,
존재하지 않는 주 근거 인용은 잘라내지 않고 전체 입력을 거부한다.
CLI는 세부 taxonomy의 부모 목록을 독립된 서비스 정본 `data/categories_v2.csv`의
코드 집합과 대조한다. 자기 자신의 부모 목록을 기준으로 통과시키지 않는다.

초과 후보의 우선순위는 다음 튜플의 내림차순, 마지막 코드 오름차순이다.

1. `core_evidence`: 채널의 주된 활동이라고 명시한 원문 인용 존재 여부(1/0).
2. `repetition_evidence`: 서로 다른 원문 행에서 뒷받침되는 활동 인용 수.
   2행 이상만 반복으로 인정한다. 동일 행 중복, 원문에 없는 인용 또는 여러 행에
   모호하게 매칭되는 인용이 있으면 해당 후보의 반복 점수만 0으로 둔다.
   후보와 원본 반복 인용은 보존하고 `repetition_warnings`로 진단한다. 배열/문자열
   형식 오류는 거부한다. 소개글의 반복 언급 수이며 영상 업로드 빈도가 아니다.
3. `evidence`: 원문 연속 인용의 양끝 공백을 제외한 문자 수.
   구체적인 명시적 설명의 길이를 대리 기준으로 사용한다. 의미적 정확도 점수가 아니다.
4. 코드의 사전순. 상위와 각 부모의 세부 후보 모두 같은 기준을 사용한다.

`build_prompt(taxonomy)`는 전체 정의/포함/제외와 반례를 담은 모델 입력 지침을 만든다.
기본 실행은 공급자를 호출하지 않는다. 외부에서 얻은 후보 JSON을 아래 명령에 전달한다.
`--tag-llm`을 추가하면 OpenAI 클라이언트로 소개글과 taxonomy를 전송해 후보를 생성한다.
이 옵션은 API 키와 비용이 필요하다. `json_schema`의 `strict: true`로 parents/topics,
코드 enum, 근거 필드와 추가 필드 금지를 지정한다. 부모-세부 관계와 실제 인용은
응답 후 로컬에서 다시 검증한다. 거절/불완전 응답은 저장하지 않는다.
API 키 누락·타임아웃·400·429 등은 traceback 대신 간결한 오류와 종료 코드 2로 처리한다.
모델 모드의 입력에는 candidates를 생략할 수 있다. 잘못된 응답은 저장 전에 거부한다.
동일 후보/입력에 대한 선택 순서만 결정적이다. temperature=0이나 모델 점수만으로
모델 응답 재현성을 보장하지 않는다. 정확한 인용이어도 의미상 정의 충족 여부와
핵심 활동 판단은 자동 검증할 수 없으며 사람이 검수해야 한다. 후보 원본도 저장한다.

## 실행과 저장

입력 예시(`input.json`):

```json
{
  "channel_id": "sample-channel",
  "text": "케이크 먹방이 주된 활동입니다.",
  "model_version": "provider-model-snapshot",
  "candidates": {
    "parents": [{"code": "FOOD", "evidence": "케이크 먹방", "core_evidence": "케이크 먹방이 주된 활동입니다."}],
    "topics": [{"code": "FOOD_EATING", "parent": "FOOD", "evidence": "케이크 먹방"}]
  }
}
```

```bash
python -m src.detail_tagging input.json --store results/detail-tags
python -m src.detail_tagging reviewed.json --store results/detail-tags --source-revision <출력한-ID> --reviewer reviewer-id --note "정의와 근거 검수 완료"
```

검수 입력도 같은 channel_id/text/candidates 구조다. 확정본 수정 시 해당 확정본의
ID를 source-revision으로 사용한다. 입력이나 taxonomy가 바뀌면 새 자동 결과부터 생성한다.

각 실행은 UUID revision 파일을 배타적으로 생성한다. 자동 결과는 automatic/pending,
검수본은 reviewed/confirmed이며 source_revision, reviewer, note를 추가한다.
파일을 생성하기 전에 JSON 직렬화를 완료하여 NaN 등 직렬화 실패로 빈 revision이
남지 않게 한다. 검수본의 `model_metadata_role=original_automatic_classification`은
복사된 모델/프롬프트 정보가 원본 자동 분류의 출처임을 뜻한다. 사람 검수 모델이 아니다.
재실행은 확정본 파일을 건드리지 않는다. 자동 결과와 검수본은 kind로 구분되며,
검수본을 조회할 때는 명시적 revision을 사용한다. 최신 자동 파일로 확정본을 대체하면 안 된다.
파일 저장소는 로컬 작업용이며 접근 제어/전자서명을 제공하지 않는다.

각 결과에는 channel_id, 입력 SHA-256, taxonomy 버전/스냅샷, BE 정본 해시,
include/exclude/formats까지 포함한 전체 기준 SHA-256, 정책/모델/프롬프트 버전,
프롬프트/출력 스키마 해시, 원본 후보, 선택 태그·근거·우선순위, 검수 상태와 UTC 저장 시각을 기록한다.
반복 점수 완화와 Structured Outputs 전환은 정책 `detail-limit-v2`, 프롬프트
`grounded-detail-candidates-v2`로 구분한다. 기존 revision은 변경하지 않는다.
Issue #62는 프롬프트만 `grounded-detail-candidates-v3`으로 변경한다. 출력 스키마와
정책은 유지한다. JSON 파싱 후 한 연속 구간인 인용만 허용하고 장식 따옴표,
연결/생략/번역/재서술을 금지한다. 근거 없는 후보는 생략하며 짧은 입력도 근거로 판단한다.
모델 버전은 호출자가 실제 사용한 스냅샷을 입력한다. 원문/근거 파일은 Git 제외
`results/` 아래에 저장한다. 순위 변경 및 BE 저장은 후속 단계다.

## 검수 반례 전달표

| 입력 | 허용 예시 | 추정하면 안 되는 태그 |
| --- | --- | --- |
| 케이크 먹방 | FOOD_EATING | 제작 근거 없는 FOOD_BAKING |
| 텐트 제품 리뷰 | TECH_DEVICES | 실제 활동 근거 없는 TRAVEL_CAMPING |
| 축구 게임 | GAME_SPORTS_RACING | SPORTS_FOOTBALL |
| AI 뉴스와 도구 사용 | TECH_AI | 개발 근거 없는 TECH_PROGRAMMING |
| 동물 모음 | 근거 부족 시 미분류 | 직접 돌봄/훈련 추정 |
| 반려견 직접 돌봄과 훈련 | PET_DOG, PET_TRAINING | 다른 동물 추정 |

위 표는 검수용 기대 사례이며 키워드 자동 분류 규칙이 아니다. 테스트는 기대 후보의
처리와 복수 태그/미분류/부모/중복/상한/동률/버전 보존을 검증한다. 실제 모델 품질을
검증했다고 해석하지 않는다. 10/20 1차 검수 공유 시 모델 원본과 확정 revision을 함께 전달한다.
이때 인용 길이 때문에 더 중요한 짧은 근거가 밀린 오선택 사례를 입력 해시,
자동 revision, 검수 revision, 선택/탈락 코드, 인용 길이와 수정 사유로 수집한다.

## PR #60 리뷰 수정 검증 (2026-10-13)

- 관련 회귀 테스트 101개 통과. 엄격한 스키마와 실제 SDK HTTP 직렬화,
  키 누락/타임아웃/400/429 처리, 모호한 반복 점수 완화, NaN/직렬화 실패의
  파일 미생성, 서비스 부모 코드 대조, 검수 출처와 기존 revision 보존을 확인했다.
- 합성 소개글 `케이크 먹방이 주된 활동입니다.` 1건을 실제 API로 호출했다.
  `gpt-5.4-nano-2026-03-17`, timeout 30초, 재시도 0회로 Structured Outputs 응답과
  로컬 검증·저장이 성공했고 FOOD/FOOD_EATING이 선택되었다. 원본은 Git 제외
  `results/pr60-review-smoke/`에 저장했다. 1건 스모크 확인이며 모델 품질 평가가 아니다.
- 전체 테스트는 기존 Windows 미지원 `resource` 모듈로 수집 실패한 이력이 있어
  전체 통과로 주장하지 않는다.
- API 형식 근거: [OpenAI Structured Outputs 공식 문서](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).

## Issue #62 버전 분리 재평가

```powershell
python -m src.detail_tagging_batch --inputs results/issue58-review-sample-v2/private_inputs.json --output results/issue62-v3-regression --tag-llm
python -m src.detail_tagging_compare --summary results/issue62-v3-regression/summary-<ID>.json --baseline results/classify50-20261013 --output results/issue62-comparison.private.json
```

첫 명령은 원문을 외부 API로 전송하며 비용이 발생한다. 모델은 기존과 같은
`gpt-5.4-nano-2026-03-17`, temperature=0, SDK 재시도=0이다. 채널별 최대 3회이며
검증 성공 시 중단한다. 모든 시도에 요청 조건, 원본 응답(usage 포함), 검증 오류,
시도 번호와 UTC 시각을 기록한다. 공급자 오류는 기록 후 실행을 중단한다.
해당 오류도 한 시도로 보존하므로 실제 모델 비교에는 공급자 오류 없는 새 실행 경로를 쓴다.

재개 시 입력 전체/기준/모델/정책/프롬프트 버전·해시/스키마 해시/시도 상한을 대조한다.
기존 v2처럼 조건 manifest가 없는 비어 있지 않은 폴더는 거부한다. 이전 자동·검수
revision과 인덱스는 그대로 두고 새로운 UUID revision 및 summary 파일만 생성한다.
같은 폴더를 동시에 여러 프로세스에서 실행하지 않는다.

로컬 `results/classify50-20261013/run_batch.py`도 추적되는 새 모듈을 호출하도록
교체했다. 이전 코드는 `run_batch.v2-preserved.py`로 보존했다. 로컬 파일은 Git 제외이므로
다른 체크아웃에서는 위 모듈 명령을 사용한다. 기존 v2 폴더를 출력으로 지정하지 않는다.

비교 보고서는 v2 첫 시도 36/50, 최종 실패 7개의 첫 시도/재시도 이후 통과,
기존 성공 43개 중 재시도 성공 7개 표시와 상위/세부 집합 변경을 분리한다.
태그 완전 일치는 합격 기준이 아니다. 성공 revision만 선택한 비교에는 선택 편향이
있으므로 실패를 포함한 모든 attempts도 확인한다. 형식 통과·사람 의미 검수·추천
순위 개선은 별개이며 이번 작업으로 후자의 두 가지를 입증하지 않는다.

2026-10-13 사용자 외부 전송·유료 호출 승인 후 50개 재평가를 완료했다.
결과와 한계는 [v3 회귀 평가](../results/detail-tagging-v3-regression.md)에 기록했다.
승인 전 제한 환경의 연결 오류 실행은 `results/issue62-v3/`에 별도로 보존했다.
