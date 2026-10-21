# 실제 저장 태그 추천 (#82)

BE #736의 `saved-tags-v1` 경로로 기존 FE 추천 화면의 두 영역을 지원한다. 과거 M2/M3·인기순 배치는 호환용이며 새 화면의 후보를 보충하지 않는다.
자격은 BE가 공개 가능한 본인 확인 태그로 선별한다. 소개/확인 revision·taxonomy·설정 hash가 현재와 맞아야 한다.
정확히 같은 상위/세부 태그만 일치하며 복수 선택은 OR다. 형제 세부 태그·추측 태그를 새로 할당하지 않는다.

## 기존 화면 정책

- 위 목록: `source=INTEREST`. 가입/기존 분야 편집의 저장 관심 분야 중 승인 콘텐츠 부모에 존재하는 같은 코드만 사용한다. 팔로우와 혼합하거나 인기순으로 채우지 않는다.
- 아래 목록: `source=FOLLOW`. 최근 한 명뿐 아니라 전체 팔로우의 현재 공개 확인 태그를 합친다. 같은 세부 또는 같은 상위 분야를 공유한 후보만 선별한다. 서로 다른 상위 분야 간 유사는 추측하지 않는다.
- `recommend_for_member`는 BE 정책의 오프라인 기준이다. 제외 대상(본인/기팔로우)을 Top-N 전에 제거하고 실제 기여 관심 분야/seed만 반환한다. 형제 세부를 태그로 새로 할당하지 않는다.
- 동일한 전체 태그/Creator 임베딩 점수 배치를 두 경로에서 재사용한다. 조회 요청 중 모델 API를 추가 호출하지 않는다. 유효 seed/관심 분야가 없으면 빈 결과, 순위 대기면 PENDING이다.

## 생성 순서와 계약

1. 전용 Bearer로 `/api/internal/content-tags/saved-recommendations/inputs` 전체 snapshot을 받는다.
2. 코드·중복·입력·정규화·snapshotHash를 검증한다. 현재 소개 없는 항목이나 미등록 코드면 모델 호출 전에 실패한다.
3. 실제 확인 태그/Creator 쌍만 선택한다. 임베딩이 유사해도 해당 태그를 갖지 않으면 후보로 만들지 않는다.
4. 설명/소개를 BGE-M3로 임베딩하고 코사인 점수를 소수점 8자리로 만든다. 입력·모델 버전 캐시는 기존 원벡터 규칙을 쓴다.
5. method=`SAVED_TAG_M2_V1`, modelVersion, snapshotHash와 전체 적격 쌍의 점수를 적재한다.
   태그별 Top-20을 미리 자르지 않으므로 회원별 제외 뒤의 21위 후보도 BE가 정상 추천할 수 있다.
6. BE가 조회 시 태그/소개 최신성·본인·기팔로우를 재검증하고 OR 최대 점수 DESC, Creator ID ASC로 정렬한 뒤 size를 적용한다.

2명만 맞으면 2명, 없으면 빈 목록이다. 빈 선택도 빈 목록이다. 점수 snapshot이 과거이면 PENDING/빈 목록으로
알리고 다른 태그/인기/팔로우로 보충하지 않는다. 실제 분류 태그와 추천 근거 interestCodes는 별개다.
FE는 기존 디자인·추천 근거 표시를 유지하고 새 태그 뱃지/탭을 추가하지 않는다. requestId=null이므로 기존 행동 수집에 다른 응답 ID를 쓰지 않는다.

내부 snapshot은 contractVersion/taxonomyVersion/taxonomyHash/generationConfigHash/tags/items와
snapshotHash/lastApplicationSequence다. 지문은 마지막 두 필드를 제외하고 키 정렬 compact JSON,
ensure_ascii=False, UTF-8, BOM/마지막 개행 없는 SHA-256이다. items는 Creator ID ASC, 태그 코드는 ASC다.
items 필드는 creatorId,introduction,inputRevision,confirmationId,parents,topics다.
소개는 CRLF/CR→LF, NFC, trim 없음이다. 태그 확인만 바뀌어도 지문이 달라져 재생성된다.
tags는 type(PARENT/TOPIC),code,name,description이다. 승인 콘텐츠 taxonomy가 정본이며 로컬 초안을 자동 승인하지 않는다.

점수 적재는 `PUT .../rankings`, body는 applicationSequence,snapshotHash,method,modelVersion,scores다.
scores는 type,code,creatorId,score 배열이다. 모든 확인 쌍을 정확히 한 번 보낸다. 빈 snapshot은 scores=[]다.
BE는 같은 번호·payload면 applied=false, 같은 번호의 다른 내용 또는 과거 번호/입력에는 409를 반환한다.
전체 점수는 한 트랜잭션으로 교체하며 소개나 확인 변경 후 과거 점수를 조회하지 않는다.

## 명령

BE #736을 먼저 배포한다. 환경 변수는 `CKING_CONTENT_TAG_ACCESS_TOKEN`, 캐시 미스 시 `DEEPINFRA_API_KEY`다.
추천 API Key·ADMIN JWT를 대신 보내지 않는다. HTTPS만 허용하고 HTTP는 loopback에 한정한다.
명시 승인 후만 실행하며 dry-run도 캐시 미스에서는 소개/태그 설명의 외부 전송·비용이 발생한다.

```powershell
python -m src.recommendation.saved_tags dry-run --be-base-url https://api.example.com --approve-external-transfer
python -m src.recommendation.saved_tags apply --be-base-url https://api.example.com --approve-external-transfer
```

기본 상태 경로는 Git 제외 `results/saved-tags`, 캐시는 기존 공용 모델 캐시다. 두 경로는 기존 배치와 같은 OS 잠금을 사용한다.
dry-run-payload.json은 검토용 점수이며 BE 쓰기는 없다. current.json은 적용 전에 payload·번호를 원자 저장한다.
HTTP 응답이 유실되면 저장 payload와 번호를 그대로 재전송한다. 정상 완료+동일 snapshot은 모델·PUT 없이 skip한다.
409는 성공으로 취급하지 않고 해당 payload를 fence한다. 다음 실행은 서버 snapshot/최대 번호를 다시 읽어 새 적용 의도로 처리한다.
동일 영속 루트의 단일 게시자만 지원한다. 별도 호스트에서 동시에 배치를 게시하지 않는다.
상태 유실은 서버 lastApplicationSequence 이후 번호로 복구하며 서버보다 오래된 번호로 덮어쓰지 않는다.
공급자 실패는 빈 점수로 바꾸지 않는다. 실패 전에 성공한 임베딩만 캐시에 남으며 기존 BE 점수는 유지된다.

콘텐츠 작업자에 `--saved-tag-recommendations`를 추가하면 기존 추천 다음 엄격 경로도 실행한다.
`daily_args`의 `--embedding-provider`와 `--embedding-model-version`을 두 경로에 동일하게 적용한다.
`savedTags`는 `recommendations`와 별도 스테이지다. saved-tags 실패는 summary의 failureCount와
CLI 종료 코드 1로 알리지만, 성공한 기존 추천의 reconcile 및 콘텐츠 job completed ack를 막지 않는다.
따라서 saved-tags 장애 때문에 콘텐츠 job을 retry하여 기존 개인화를 반복하지 않는다.
실패한 saved-tags는 별도 apply 명령 또는 다음 일일 보정으로 재시도한다. 이벤트 idle poll은 이를 재시도하지 않는다.
전체 스테이지가 성공하기 전에는 last-success.json을 갱신하지 않는다.
BE가 없는 환경에서 자동 활성화하지 않는다. 본인 확인만 변경되고 새 소개 작업이 없으면 이벤트 poll은 idle이므로
별도 명령이나 일일 보정에서 반영한다. PENDING 안내가 그 공백을 명시한다.

## 검증

```powershell
python -m pytest tests/recommendation/test_saved_tags.py -q
python -m src.recommendation.e2e.runner --suite content --be-dir ../Cking-BE --fe-dir ../Cking-FE
```

가짜 모델을 사용하며 실제 BE·Security·Flyway·MySQL에서 공개 확인→점수 적재→OR/부족/빈 결과→
기팔로우 제외→소개 변경 무효화·과거 snapshot 거부를 검증한다. 유료 공급자는 호출하지 않는다.
세부 분류는 최대 3개이지 항상 3개가 아니며 0/1/2개 결과도 회귀 테스트한다.
운영 재생성·분류 의미 품질·부하 성능을 이 검증으로 주장하지 않는다.

`--fe-dir`는 선택 사항이다. 지정 시 전용 fan의 실제 JWT로 FE의 기존 화면, 후보 2명, 분야 편집 후 0명/복원, 팔로우 관련 후보, ID 미지원 기록 미전송, 모바일 폭을 실제 Chromium에서 검증한다. 시크릿은 stdin으로 전달하며 원문/키를 콘솔·보고서에 기록하지 않는다.
