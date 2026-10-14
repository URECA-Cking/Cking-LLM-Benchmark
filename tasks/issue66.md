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

## 사용자 위임에 따른 AI 대리 검수

사용자가 Codex에 최대한 사람처럼 판단하도록 위임하여 전체 원문과 taxonomy를 읽고
5,100행을 판정했다. 결과는 `results/issue66-ai-proxy-review-v2/`에 비공개로 보존했다.
유지 99행, 삭제 4,969행, 보류 32행(15채널), pending 0행이다.
AI: 검수자와 origin=ai_proxy를 명시하며 실제 사람 검수 완료로 표시하지 않는다.
기존 사람용 pending 검수표와 역사적 실행은 보존한다.

AI 전용 freeze·배치·evaluate 경로를 추가해 후속 진단에 사용할 수 있게 했다.
AI/사람 기준 혼합, AI 기준 해시 없는 역사적 비교와 재개 조건 혼합을 거부한다.
사람용 홀드아웃 채택 경로는 AI 기준을 거부하며 기본 프롬프트는 v2로 유지한다.
원문의 과거 자동 라벨 일부를 이미 읽어 블라인드가 아니며 독립 사람 정답을 주장하지 않는다.
유료 API 호출, v2/v4 실제 비교와 홀드아웃 모델 실행은 하지 않았다.
추가 검증: AI/사람 모드 격리·인용·재개·혼합 거부를 포함한 관련 합성 테스트 265개 통과,
`git diff --check` 통과. 실제 5,100행도 AI 전용 validator로 대조했다.

## 승인된 개발 표본 실제 비교

사용자가 50개 개발 입력의 v2/v4 원문 외부 전송·유료 호출(최대 300회)을 명시적으로 승인했다.
2026-10-14 모델 `gpt-5.4-nano-2026-03-17`, temperature=0, SDK 재시도=0,
timeout=60초, 채널당 최대 3회, 동일한 고정 AI 기준으로 두 버전을 새 경로에서 실행했다.

- v2 첫 시도 34/50, 최종 42/50. v4 첫 시도 44/50, 최종 48/50.
- 총 133회 호출. 입력 1,008,852·출력 26,789·총 1,035,641토큰. 실제 청구액은 미확인.
- 양쪽 성공 41개에서 세부 과잉 24→3, 누락 4→7. 기본값 v2 유지.
- 원본 133개 시도와 summary·성공 revision·실제 반환 모델·usage를 대조했다.
- 원문과 채널별 결과는 Git 제외 `results/issue66-ai-approved-*`에 보존했다.

집계·분모·오류·사용량과 한계는 [실제 비교 보고서](../docs/results/detail-tagging-v2-v4-development.md)에 기록했다.
실제 사람 기준·독립 홀드아웃·최종 팀 채택 판단은 미수행이므로 이슈 전체를 닫지 않는다.
