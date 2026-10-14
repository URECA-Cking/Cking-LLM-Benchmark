# Issue #64 구현·검증 기록

브랜치: `task/64-v4-prompt-human-eval`. 시작 원격 HEAD: `1a3529b`.
조직 CONTRIBUTING.md의 `<타입>: <한국어 한 줄 요약>` 및 관심사별 커밋 분리를 따른다.

## 코드 구현

- v4를 명시적 프롬프트 옵션으로 추가. 상위/세부 독립 판단, 짧은 명시 주제,
  해시태그, 상위만 유지 예시, 직접 돌봄/배경 음악의 과잉 추정 반례를 포함한다.
- 기본값 v2, 기존 v2/v3 문자열, 정책·스키마·정확한 인용 검증을 유지한다.
- 전체 입력의 모든 상위/세부 코드에 대해 사람이 유지/삭제/보류와 검수자·근거·사유를
  기록하는 CSV 및 고정 기준 파일을 추가한다. AI 초안은 자동 수입하거나 사람 판정으로 취급하지 않는다.
- v4 CLI의 기준 고정 필수화, 입력/taxonomy 대조, 사람 기준 해시 보존 및 재개 혼합 거부.
- 사람 기준 대비 상위/세부 누락·과잉, 실패와 보류 분모, 미분류·상위만 분류,
  첫 시도/재시도 통과, 역사적 집단 및 중복을 보고한다. 기존 비교 도구도 보존한다.
- 새로운 출력 경로와 배타적 파일 게시를 사용하며 기존 자동·검수 revision과 원본 응답은 보존한다.
- 누락된 역사적 조건은 추정하지 않는다. 자동 승격 대신 v2 유지와 사람 채택 판단 대기 상태를 기록한다.

## 검증

```powershell
.venv\Scripts\python.exe -m pytest tests/test_detail_tagging.py tests/test_detail_tagging_batch.py tests/test_detail_tagging_human_eval.py tests/test_sample_review.py tests/test_subtopic_tags.py tests/test_subtopic_taxonomy_hash.py -q --basetemp results/pytest-issue64-verified
git diff --check
```

2026-10-14: **210 passed**, diff 공백 검사 통과.
합성 후보·클라이언트와 임시 파일로 상위 보존, 인용/중복 거부, 프롬프트별 출처,
기존 버전 재개 거부, 검수 CSV/고정 기준/비교 CLI, 보류·실패 분모 및 집단 중복을 검증했다.
기존 SDK HTTP 직렬화·공급자 오류 회귀 테스트도 포함한다.
전체 저장소 테스트와 실제 모델 의미 품질, BE/서비스 E2E를 검증한 결과가 아니다.

## 후속 사람 작업

21개 AI 사전 검토와 전체 50개에 대한 실제 사람 판정은 미수행이다.
마술 배치 등 팀 기준 합의가 필요하며 보류 항목은 정답 비교에서 제외한다.
원문 외부 전송·유료 v4/신규 v2 실행은 이번 구현 요청으로 수행하지 않았다.
따라서 이슈의 사람 검수·재평가·채택 조건 전체를 완료했다고 주장하지 않는다.
실행 순서와 분모 해석은 [가이드](../docs/guides/detail-tagging-v4-human-eval.md)를 따른다.
