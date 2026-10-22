# 소수 후보군의 관련 후보 선별 (#85)

모델 ID, 17개 taxonomy, 기존 점수·반올림·동점 정렬을 유지한다. **자격 선별 → 기존 점수 정렬 → Top-N** 순서다.
Top-N은 상한이며, 결과는 0~N개다. 후보 수를 채우기 위한 임계값 완화·무관 후보 보충은 없다.

## 선별

- 소개가 빈 크리에이터는 제외한다. seed 소개가 비면 유사 추천도 빈 결과다.
- 관심 분야 M2/M3: 기존 BGE-M3 zero-shot 분류(`tau=0.4296248555`, 최대 3개)에 요청 분야가 있어야 후보로 남긴다.
  M2는 cosine, M3는 기존 cosine+분류 보너스를 유지한다. 선별 뒤 모든 분야 후보가 같은 보너스를 받으므로 단일 분야에서 두 방식의 순서가 같을 수 있다.
- 유사 M4: 기존 GPT 분류의 공통 분야가 있어야 한다. cosine+공통 태그 보너스는 유지한다.
- 짧은 소개·UNCLASSIFIED의 M2: 같은 BGE-M3와 taxonomy 설명의 기존 tau로 자격만 분류하고 공통 분야가 있는 후보만 남긴다.
  점수는 기존 cosine이다. 짧은 소개를 위해 GPT 호출을 추가하지 않는다.
- 태그는 **0~3개**다. zero-shot은 tau를 넘는 것만 남기고, LLM의 빈 분류도 정상이다.
  4개 이상·중복·등록되지 않은 태그는 오류이며, 잘못된 태그 캐시는 재생성한다. 3개로 자르거나 부족한 수를 채우지 않는다.

이 선별은 모델의 의미 판단 자체가 항상 옳다는 보장은 아니다. 고정 모델 테스트는 계약 검증이며 실데이터 의미 품질·인간 평가와 구별한다.
기존 tau를 임의 재조정하거나 다른 모델로 바꾸지 않는다.

## BE 연동

두 payload에 `eligibility: {policyVersion: "RELATED_V1", taxonomyVersion, taxonomyHash, profiles}`를 추가한다.
profiles는 `{creatorId, introductionHash, tags}`의 ID 오름차순 목록이다. tags는 사전순·중복 없는 0~3개 코드다.
선택 후보 전체와 유사 seed를 포함한다. 빈 분야는 profiles도 비며 빈 유사는 seed만 포함한다.
소개 지문은 기존 `CreatorProfile.text`(줄바꿈 통일 → Python strip, NFC 변환 없음)의 UTF-8 SHA-256이다.
추천 이유나 사용자 선택 분야를 실제 크리에이터 태그로 복사하지 않는다.

BE #738은 근거를 세대와 함께 보존하고 요청 분야·공통 태그·현재 소개 지문을 검사한다.
소개가 변경되면 재배치 전까지 이전 결과는 제외한다. 수동 태그 확인을 새 필수 조건으로 도입하지 않는다.
근거 없는 구 세대는 저장 호환하지만 공개하지 않으므로 **BE 배포 → 새 배치 재적재 → FE 배포**가 필요하다.
generation/input/config hash에 `RELATED_V1`과 M2 자격 tau를 포함해 구 결과 캐시와 구분하며, 모델 벡터 캐시는 재사용한다.
기존 applicationSequence·부분 재시도·A→B→A 규칙은 유지한다. 빈 후보도 정상 적재해야 이전 추천이 사라진다.

## 검증

```powershell
python -m pytest tests/recommendation -q
python -m src.recommendation.e2e.runner --be-dir ../Cking-BE --suite small-pool
python -m src.recommendation.e2e.runner --be-dir ../Cking-BE --suite recommendation
```

단위 테스트는 0·1·2·5·19·20·21·540명, 태그 0~3개, 높은 cosine의 무관 후보, 빈 소개, 잘못된 캐시를 다룬다.
`small-pool`은 임시 MySQL/Redis·실제 BE HTTP·고정 합성 모델로 11명 중 관련 2명 반환, 분야 편집, 소개 변경 즉시 제외,
동일 적재 재시도, 구 근거 없는 세대 차단을 검증한다. 외부 모델 API·운영 DB를 사용하지 않는다.
실제 소개를 외부 API로 전송하는 유료 배치는 이 테스트에 포함하지 않는다.
