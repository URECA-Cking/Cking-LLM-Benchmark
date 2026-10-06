# Issue #41 구현·검증 기록

브랜치: `feat/41-m3-top20-candidates`.
기준: 최신 `origin/develop`의 #40 분류체계 계약을 fast-forward로 반영했다.
조직 `CONTRIBUTING.md`의 `<타입>: <한국어 한 줄 요약>`과 관심사별 커밋 분리를 따른다.

## 구현

- [x] v0.2 정본 17개 분야를 독립 쿼리로 사용해 기본 Top-20을 생성한다.
- [x] 동일 BGE-M3 벡터로 코사인과 zero-shot 상위 3개 분야 태그를 계산한다.
- [x] tau 0.4296248555, M3 가산점 0.1, 모델·분류·자릿수를 고정하고 설정·inputHash에 기록한다.
- [x] 빈 소개·중복 ID·비정상 벡터·후보 부족·점수 동점을 결정적으로 처리한다.
- [x] 모델 캐시를 공유하고 동일 입력 재실행의 payload와 순위를 검증한다.
- [x] 동일 manifest/Top-N의 M2 기준선을 생성하고 Top-10 판정 입력과 방식 출처를 분리한다.
- [x] 분야별 payload·성공/빈 세대/실패 요약·원자 체크포인트를 저장한다.
- [x] M3만 분야별 PUT에 적재하며 성공 분야를 건너뛰고 실패를 재개한다.
- [x] 적재 대상 BE가 바뀌면 생성 결과를 재사용하고 모든 payload를 다시 적재한다.
- [x] 추천 적재 전용 API Key 헤더와 ADMIN JWT 대체 인증을 연결한다.
- [x] 유료 호출 없는 fake 임베딩/BE 테스트와 실행·외부 전송·평가 한계를 문서화한다.

## 검증

변경 전 추천 테스트: 91 passed.
최종 추천 테스트: 180 passed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=results/pytest-issue41-final-recommendation --tb=short tests/recommendation
```

전체 테스트 확인에서는 467 passed, 3 failed를 기록했다. 이후 공통 임베딩 실패/응답 해시/평가
Top-10 관련 테스트 5개를 추가하고 최종 추천 테스트 180개를 통과했다.
아래 3개는 `b6bef4f`의 변경 전 소스를 직접 로드해서도 동일하게 실패했다.

- `test_api_parity.py::test_candidate_dropped_at_cutoff_is_caught`: 부동소수점 컷오프 경계.
- `test_real_eval.py::test_results_dir_can_be_overridden_by_environment_variable`: Windows에서 POSIX 경로 분리로 cwd가 파일 경로가 됨.
- `test_real_eval.py::test_human_agree_file_paths_expand_home_and_ignore_spaces_around_commas`: Windows의 home 확장 동작 차이.

Windows에 없는 `resource`에 의존하는 기존 `test_measure_memory.py`는 전체 검증에서 제외했다.
Git 제외 로컬 실험 파일까지 자동 수집되지 않도록 전체 검증 대상을 `tests`로 명시했다.
`compileall`과 `git diff --check`도 통과했다.

## 연동 경계

현재 BE `develop`에는 분류체계 저장·조회와 추천 적재 키 인증이 구현되어 있으나,
관심 분야 후보 적재 Controller/DTO·최종 응답 명세는 아직 없다.
요청은 #41의 필드, 응답은 기존 Creator 적재의 분야 확장 구조로 구현하고 검증했다.
BE 구현 병합 후 요청·응답·키 권한 연결·멱등·빈 세대 활성화의 실제 통합 확인이 필요하다.
실제 전체 manifest 모델 실행·유료 임베딩/Judge·실제 BE apply는 수행하지 않았다.

입력/출력과 실행 상세: `docs/guides/interest-recommendation.md`.
