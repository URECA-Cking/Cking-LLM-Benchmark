# Issue #66 구현·검증 기록

브랜치: `task/66-v2-v4-holdout-eval`. 시작 HEAD: `8d38a26`.
조직 CONTRIBUTING.md의 `<타입>: <한국어 한 줄 요약>` 및 관심사별 커밋 분리를 따른다.

- 기존 사람 검수/freeze/evaluate와 배치 실행을 재사용하는 `detail_tagging_holdout` 추가.
- 역사적 snapshot·revision·v3 attempts 대조로 개발 집단 추출, 출처·중복 기록.
- 분야/경계별 사전 수량과 seed로 독립 홀드아웃 선정. 개발/예시 채널·동일 원문 제외.
- 실제 사람 기준과 채택 기준을 protocol에 고정. 새 v2/v4 실행과 재개를 조건/기준 해시로 격리.
- 개발/홀드아웃 별도 보고, 실패 포함 usage 집계, 빈 배열·hold·실패를 고려한 채택 검토 조건.
- 실제 팀 판단 기록과 프롬프트 수정 시 홀드아웃 재분류. 기본값 v2 유지.

2026-10-14 실제 비공개 원본 대조: 전환 21개·최종 실패 7개·재시도 성공 7개,
중복 채널 3개. `results/issue66-development/`에 50개 고정 입력과 5,100행 pending 검수표 생성.
과거 v3 timeout 누락을 그대로 보존하며 역사적 집단 추출과 새 엄격 비교를 구분한다.
원문·검수표·채널별 결과는 Git에 포함하지 않는다.

검증 명령:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_detail_tagging_holdout.py tests/test_detail_tagging.py tests/test_detail_tagging_batch.py tests/test_detail_tagging_human_eval.py tests/test_sample_review.py tests/test_subtopic_tags.py tests/test_subtopic_taxonomy_hash.py -q --basetemp results/pytest-issue66-final
git diff --check
```

검증 결과: 관련 합성 테스트 254개 통과. 전체 테스트 스위트와 실제 모델 호출은 실행하지 않았다.

실제 사람 검수, 팀의 표본 구성/예시 출처/채택 기준 합의, 유료 v2/v4 실행,
독립 홀드아웃 실제 평가와 팀 채택 판단은 미수행이다. AI가 사람 reviewer를 대신하지 않았다.
현재 이슈는 코드 보완과 개발 검수 준비 단계이며 전체 완료가 아니다.
재현 절차와 남은 사람 작업은 [평가 가이드](../docs/guides/detail-tagging-holdout.md)에 정리했다.
