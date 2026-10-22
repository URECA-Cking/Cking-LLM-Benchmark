# Issue #42 구현·검증 기록

브랜치: `task/42-hybrid-v1-shared-fixture`.
작업 전 tracked/untracked 변경은 없었다. 요청한 원격 브랜치에서 로컬 브랜치를 만들고
선행 #40·#41이 병합된 `origin/develop`의 `2e85fa6`까지 fast-forward했다.
조직 `CONTRIBUTING.md`의 `<타입>: <한국어 한 줄 요약>`과 관심사별 커밋 분리를 따른다.

## 구현 완료 조건

- [x] Python Decimal 기반 rank RRF, 그룹 평균, 0.5 가중합과 세 단계 8자리 HALF_UP.
- [x] 본인·팔로우·Creator Space 없음 제외 후 유효 source 판정과 분모 계산.
- [x] 제외 전 원래 rank 유지, 반올림 점수 내림차순·Creator ID 숫자 오름차순.
- [x] 두 그룹, 관심 단독, 팔로우 단독, 둘 다 무효의 네 정책 분기.
- [x] 후보별 실제 기여 interestCodes·seedCreatorIds의 중복 제거와 결정적 정렬.
- [x] 설정과 18개 고정 case를 담은 `fixtures/hybrid_personalized_v1.json`.
- [x] 점수·가중치 문자열, 활성 세대·후보 rank·Space 포함 입력, 결과 순서·점수·정책·출처·유효 source 수.
- [x] 무효 세대·후보 부족, 모든 제외 규칙, 중복 후보, 숫자 ID 동점, 각 반올림 경계.
- [x] Python 전체 fixture 비교, 입력 순서·선택 중복·Decimal context 독립성·입력 불변성 검증.
- [x] Java BigDecimal 연산, 필드 의미와 파일 사용 방법 문서화.

기대 점수는 참조 모듈을 import하지 않는 독립 정수·Fraction 계산으로 확인했다.
실행 테스트는 고정 JSON을 읽으며 기대값을 다시 생성하지 않는다.
불명확한 source 중복·한 source 내부 후보/순위 중복·잘못된 타입/범위는 거부한다.
함수는 모델·DB·네트워크를 호출하지 않으며 입력을 변경하지 않는다.

## 검증

새 개인화 테스트: **108 passed**. 최종 fixture에서는 동점 ID 2·10으로 문자열 정렬 오류도 검증한다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=results/pytest-issue42-hybrid-final --tb=short tests/recommendation/test_hybrid.py
```

전체 테스트: **580 passed, 3 failed**.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=results/pytest-issue42-full --tb=short tests --ignore=tests/test_measure_memory.py
```

아래 세 실패는 `origin/develop`의 원본 `src/api_parity.py`, `src/real_eval.py`를
직접 불러 같은 테스트를 실행해 **3 failed**로 재현했다. 이번 변경으로 발생한 실패가 아니다.

- `test_api_parity.py::test_candidate_dropped_at_cutoff_is_caught`: 기존 부동소수점 컷오프 경계.
- `test_real_eval.py::test_results_dir_can_be_overridden_by_environment_variable`: Windows에서 POSIX 경로 분리.
- `test_real_eval.py::test_human_agree_file_paths_expand_home_and_ignore_spaces_around_commas`: Windows home 확장 차이.

Windows에 없는 `resource` 모듈에 의존하는 기존 `test_measure_memory.py`는 제외했다.
`compileall`, 공용 fixture UTF-8/BOM 없음/LF, `git diff --check`도 검증했다.

## 연동 범위

참조 구현·fixture·문서는 이 저장소에만 추가한다. Java 하이브리드 조회와 회원 관심 분야 저장은 범위 밖이다.
Java 테스트나 실제 BE 연동을 실행하지 않았으며, BE 구현은 동일 fixture 전체로 결과를 검증해야 한다.
팔로우 단독 분기도 이슈 #42의 **유효 seed 평균**을 사용한다. 기존 BE의 단순 합산 정책을
이름이 같다는 이유로 그대로 재사용하면 fixture와 일치하지 않는다.
자세한 계약: `docs/reference/hybrid-personalized-contract.md`.
