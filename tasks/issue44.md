# Issue #44 구현·검증 기록

브랜치: `task/44-m2m3-llm-judge-eval`.
기준: 원격 지정 브랜치에서 최신 `origin/develop`을 fast-forward로 반영해 #40·#41 입력 계약을 확보했다.
조직 `CONTRIBUTING.md`의 `<타입>: <한국어 한 줄 요약>`을 따르고 기능·테스트·문서를 분리한다.

## 구현

- [x] 같은 v0.2 taxonomyHash·manifestHash, 17개 분야, M2/M3 Top-10·합집합·원문·출처 검증.
- [x] 분야·소개 해시, Judge 모델·프롬프트 버전/본문, taxonomyHash를 포함하는 새 pairId·캐시 키.
- [x] 방법·원본 rank/score·Creator ID를 보내지 않는 익명 후보 요청과 결정적 섞기.
- [x] strict JSON Schema와 로컬 요청·응답 검증, 중복·누락·불명 ID·허용되지 않은 라벨 거부.
- [x] 성공 판정의 원자 저장, 실패·중단 기록, 완료 쌍을 재사용하는 재시작.
- [x] strict/lenient Precision@5/10, graded nDCG, M3−M2, 17개 분야 paired bootstrap·승패·잡음 사례.
- [x] 분야·판정 차이·개선/악화·잡음 층화 최소 50쌍의 빈 블라인드 사람 시트.
- [x] 별도 사람 입력의 3단계/이진 일치율, M2/M3 표본 적합·잡음 비율과 미완료 상태.
- [x] 사전 규칙 동결, 미검증 M2 유지, 기준 미충족 추가 실험, 기준 충족 M3 채택 분기.
- [x] #43 전달용 선택 method·tau·bonus·모델·taxonomy·판단 버전과 설정 지문.
- [x] 관측 토큰·실패·중단 비용, 사용자 청구 확인 범위, 공유 캐시 비용 제외 한계 기록.
- [x] CLI 별도 승인 플래그 및 results/ 경로 제한, 개인정보 없는 공개 집계 allowlist.
- [x] 가짜 Judge·SDK 테스트로 유료 호출 없이 검증.

## 실제 평가의 상태

- [ ] 실제 Creator manifest에 대한 전체 Judge API 호출: 별도 데이터 전송·비용 승인 필요.
- [ ] 최소 50쌍 독립 사람 판정.
- [ ] 실제 청구 확인 및 전체 평가 비용 확정.
- [ ] 실제 평가 근거를 반영한 최종 채택·#43 운영 설정 전달.

현재 문서 결론은 **M2 유지**이며 구현 테스트를 실제 품질 검증으로 해석하지 않는다.
현재 #41의 M3 설정 tau `0.4296248555`·bonus `0.1`은 전이 비교 설정으로만 유지한다.

## 검증

- `python -m pytest -q -p no:cacheprovider --basetemp=results/.pytest-issue44-final tests/experiments/test_interest_judge.py tests/recommendation`
  - 325 passed, 84.83s. 기존 추천 288개와 새 평가 37개를 함께 검증했다.
- 이후 추가한 캐시 저장 도중 중단·빈 manifest 테스트:
  `python -m pytest -q -p no:cacheprovider --basetemp=results/.pytest-issue44-extra tests/experiments/test_interest_judge.py -k 'interrupt_during_cache_write or empty_manifest'`
  - 2 passed, 37 deselected, 43.95s.
- 고유한 관련 테스트 총 327개 통과. 테스트에서는 실제 Judge·임베딩 API를 호출하지 않았다.
- `python -m src.interest_judge --help` 실행 확인.
- 변경 파일의 `git diff --check` 확인. 원문·키·판정·캐시는 커밋 대상에서 제외했다.
