# Issue #66 구현·검증 기록

브랜치: `task/66-v2-v4-holdout-eval`. 시작 HEAD: `8d38a26`.
조직 CONTRIBUTING.md의 `<타입>: <한국어 한 줄 요약>` 및 관심사별 커밋 분리를 따른다.

## 현재 구현과 평가 상태

- 기존 검수/freeze/evaluate와 배치 실행을 재사용하는 `detail_tagging_holdout` 추가.
- 역사적 snapshot·revision·v3 attempts 대조로 집단과 중복·출처를 기록했다.
  전환 21개·최종 실패 7개·재시도 성공 7개, 중복 채널 3개를 확인했다.
- 개발/예시 채널·동일 원문을 제외하는 분야/경계별 홀드아웃 선정, 기준 사전 고정,
  v2/v4 실행·재개·별도 보고와 팀 결정 기록 도구를 추가했다.
- 사용자 위임으로 전체 개발 입력 50개와 102코드를 AI 대리 검수했다.
  유지 99·삭제 4,969·보류 32행(15채널), pending 0행이다.
  origin=ai_proxy와 AI: 검수자를 명시하며 실제 사람 검수 완료로 표시하지 않는다.
- AI 전용 freeze·배치·evaluate 경로와 사람 기준 혼합·해시/재개 혼합 거부를 추가했다.
  AI: 표시는 신원 인증이 아닌 선언 관례다. 사람용 홀드아웃 채택 경로는 AI 기준을 거부한다.

## 승인된 개발 표본 실제 비교

사용자가 개발 입력 50개의 v2/v4 원문 외부 전송·유료 호출(최대 300회)을 승인했다.
2026-10-14 모델 `gpt-5.4-nano-2026-03-17`, temperature=0, SDK 재시도=0,
timeout=60초, 채널당 최대 3회, 동일한 고정 AI 기준으로 새 경로에서 실행했다.

- v2 첫 시도 34/50, 최종 42/50. v4 첫 시도 44/50, 최종 48/50.
- 총 133회 호출. 입력 1,008,852·출력 26,789·총 1,035,641토큰. 실제 청구액은 미확인.
- 양쪽 성공 41개에서 세부 과잉 24→3, 누락 4→7, F1 0.641→0.815.
- 원본 133개 시도와 summary·성공 revision·실제 반환 모델·usage를 대조했다.
- 원문·검수표·채널별 결과는 Git 제외 `results/`에 보존했고 기존 역사적 원본은 유지했다.

기본값 v2 유지는 사전 채택 기준과 독립 홀드아웃 검증이 없어 변경을 보류하는 상태다.
v4가 더 나쁘다는 결론은 아니다. 양성 세부 29개와 비블라인드 AI 기준의 한계를
[실제 비교 보고서](../docs/results/detail-tagging-v2-v4-development.md)의 결론 근처에 명시했다.
독립 사람 기준·독립 홀드아웃·최종 팀 판단은 미수행이며 이슈 전체를 닫지 않는다.

## PR #67 리뷰 반영과 최종 검증

- 결론을 v4의 precision/F1 개선과 채택 보류의 근거로 구분했다.
- CLI 내부 고정 메시지와 외부 예외별 안내를 제공하며 민감한 예외 세부는 출력하지 않는다.
- 공통 시도 기록 검증을 공개 함수 `validate_attempt_history`로 호출한다.
- 단계별 과거 테스트 수 대신 아래 최종 검증 결과를 기준으로 삼는다.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_detail_tagging_ai_proxy.py tests/test_detail_tagging_holdout.py tests/test_detail_tagging.py tests/test_detail_tagging_batch.py tests/test_detail_tagging_human_eval.py tests/test_sample_review.py tests/test_subtopic_tags.py tests/test_subtopic_taxonomy_hash.py -q --basetemp results/pytest-pr67-review-final
git diff --check
```

최종 관련 테스트 273개와 `git diff --check`가 통과했다.
기존 실제 평가 133개 시도를 오프라인 재검증해 지표·출처 결과가 그대로임을 확인했다.
전체 저장소 테스트·BE/서비스 E2E는 미실행이다.
리뷰 수정 과정에서 유료 모델을 다시 호출하거나 기존 평가 원본·판정을 수정하지 않았다.
재현 절차는 [평가 가이드](../docs/guides/detail-tagging-holdout.md)에 정리했다.
