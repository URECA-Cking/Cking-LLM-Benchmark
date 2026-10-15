# 세부 태그 프롬프트 v2 채택 결정 — 2026-10-15

사용자는 개발 표본의 v2/v3/v4/v5 계열 실험을 검토한 뒤 **`grounded-detail-candidates-v2`를 표준 실행 프롬프트로 채택**했다.
결정 출처는 [이슈 #71](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/71)의 사용자 결정(`user_decision`)이다.
종전의 “실험 채택 대기 중 기본값 v2 유지”와 구분한다. 이 결정은 v2의 일반적인 성능 우위를 입증한 평가 결과가 아니다.
실제 사람 정답·독립 홀드아웃·서비스 품질 개선 검증은 미수행이다.

## 결정 범위와 고정 지문

기준 HEAD는 `315de5631b4140fda618e27ddcc6e1dec0798f02`(develop, PR #69)이다.
프롬프트·배열 스키마·검증/선택 정책·taxonomy 바이트를 변경하지 않고 채택 상태와 진입점을 정리한다.
기계 판독 기록은 [채택 지문 JSON](../reference/detail-tagging-v2-adoption.json)에 있다.

| 대상 | 버전 또는 SHA-256 |
| --- | --- |
| 채택 프롬프트 | `grounded-detail-candidates-v2` |
| prompt_hash | `75c0d36b571201cf8b7245441f652f50ab28c689476181a8ab1a222a73337386` |
| output_schema_hash | `a6c00395dc91580d6669d7344c2e2b9db238b8032c8637cd5047d44137229fc0` |
| 정책 | `detail-limit-v2` |
| policy_source_hash | `ba5f72161786880fd8bb45c0fb0ac015691a03446df6f9d74f9b88d5d8e6f42c` |
| 세부 taxonomy | `subtopic-v1-draft` (세부 85개, 확정 아님) |
| criteria_hash | `c2e7ed5616db11064c6cf1621515007305b1712913591a0c072b4aa478716f60` |
| BE 정본 subtopic 해시 | `7e06e15a711beb6a04599b8a6b81e6c3c1aea2975db40df9bc17cc0157feb0f6` |

prompt/schema/criteria는 기존 `detail_tagging.digest`(키 정렬 compact JSON, UTF-8, ensure_ascii=False)의 지문이다.
정책 지문은 별도 채택 감사용으로, `inspect.getsource(_rank) + inspect.getsource(classify)`를
LF 줄바꿈의 UTF-8로 인코딩한 SHA-256이다. 기존 실행 기록에는 `policy_version`만 저장되므로
과거 기록에 정책 해시가 있었다고 해석하거나 이 값을 소급 추가하지 않는다.
전체 기준(criteria)은 include/exclude/formats까지 포함하며 BE 정본 해시와 다르다.
상위 taxonomy v0.2, 세부 taxonomy, 개수 제한 정책, M2/M3/M4 추천 방식은 프롬프트 v2와 별개다.

## 실험 결과와 결정 근거의 구분

아래는 새 평가가 아닌 [v3 회귀](detail-tagging-v3-regression.md),
[v2/v4 개발 비교](detail-tagging-v2-v4-development.md),
[초기 v5](detail-tagging-v5-development.md), [최종 v5 재구현](detail-tagging-v5-redesign.md)의 관측 요약이다.
과거 보고서의 당시 결론·실패 이력·평가 도구의 채택 대기 기록은 보존한다.

| 개발 50개 형식 통과 | v2 | v3 | v4 | 최초 v5 | 최종 v5-compact |
| --- | ---: | ---: | ---: | ---: | ---: |
| 첫 시도 | 34/50 | 48/50 | 44/50 | 41/50 | 49/50 |
| 최종 | 42/50 | 50/50 | 48/50 | 43/50 | 49/50 |

v3은 최종 형식 통과 50/50이었지만 미분류 32/50으로 늘었다. 빈 출력도 형식 통과할 수 있다.
v3 열은 별도의 역사적 실행이다. 당시 v2 기준은 첫 시도 36/50·최종 43/50이었으며,
위 v2/v4/v5 개발 비교의 새 v2(34/50·42/50)와 섞지 않는다.
역사적 v2의 누락된 timeout 등 실행 조건을 추정하여 엄격 비교를 통과한 결과로 만들지 않는다.
다음 의미 지표는 비블라인드 **AI 대리 기준**(`origin=ai_proxy`, 사람 검수 미수행)으로,
각 비교의 공통 성공 집합에서 hold를 제외했다. 서로 다른 분모를 합치지 않는다.

| 비교 | 공통 성공 | 세부 누락 | 세부 과잉 | 세부 Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 → v4 | 41 | 4 → 7 | 24 → 3 | 0.510 → 0.880 | 0.862 → 0.759 | 0.641 → 0.815 |
| v2 → compact | 42 | 4 → 12 | 25 → 2 | 0.537 → 0.913 | 0.879 → 0.636 | 0.667 → 0.750 |
| v4 → compact | 48 | 10 → 10 | 4 → 2 | 0.886 → 0.939 | 0.756 → 0.756 | 0.816 → 0.838 |

compact의 형식 통과·중복 억제·세부 precision/F1 개선은 관측됐지만 v4 대비 총 세부 누락 감소는 확인되지 않았다.
공통 48개의 상위 누락/과잉은 6/4 → 7/5, 상위 F1은 0.889 → 0.867로 일부 악화됐다.
v2는 이 비교에서 누락이 적었지만 과잉이 많았다. “v2가 모든 지표에서 최고”라고 해석하지 않는다.
사용자는 이 교환 관계와 한계를 검토하고 이후 표준 경로를 v2로 진행하기로 선택했다.

같은 50개 개발 표본에서 v5 계열 6개를 반복 튜닝했고 v4 예시도 개발 표본과 겹친다.
hold 제외, 모델 실패 제외, 공통 성공 집합 선택과 재시도 성공 선택 편향이 남는다.
양성 세부 태그는 비교에 따라 29/33/41개로 작으며 단일 실행·temperature=0은 일반적 우위나 결정성을 보장하지 않는다.
실제 사람 정답·독립 홀드아웃·서비스 품질 개선을 검증한 것으로 표현하지 않는다.

## 표준 실행과 과거 재현

`src.detail_tagging.ADOPTED_DETAIL_PROMPT_VERSION`이 채택 버전의 단일 참조이며
기존 공개 이름 `PROMPT_VERSION`도 같은 값을 가리킨다. 함수·단일 CLI·배치 CLI는 모두 기본 v2다.
단일 CLI의 `--prompt-version grounded-detail-candidates-v2`는 명시적 표준 선택이다.
실험 버전은 기존 함수 인자/배치 `--prompt-version`으로만 명시적으로 선택한다.
v5 슬롯 스키마·변환은 해당 실험 버전에서만 사용하고 v2는 기존 parents/topics 배열을 사용한다.

| 확인한 진입점·설정 | 결과 |
| --- | --- |
| build_prompt / candidate_schema / automatic_record / classify_with_model | 기본·명시 v2 일치 |
| src.detail_tagging / src.detail_tagging_batch / run_batch | 기본 v2, 새 출력 경로 사용 |
| src.detail_tagging_holdout | 과거 v2/v4 비교 프로토콜 보존; 현 채택 결정으로 소급 수정하지 않음 |
| README / docs 실행 가이드 | 표준 v2와 실험/과거 재현 명령 구분 |
| run.cmd / START_HERE / HANDOFF / .env.example / src.config / 추천 CLI | 세부 프롬프트를 자동 선택하는 별도 설정 없음; 추천/상위 분류 흐름과 구분 |
| 로컬 results/classify50-20261013/run_batch.py | 추적 batch.main 위임, 기본 출력만 새 results/issue71-standard-v2로 변경 |

로컬 wrapper는 Git 제외 파일이므로 다른 체크아웃에서는 [표준 모듈 명령](../guides/detail-tagging.md)을 사용한다.
옆의 `run_batch.v2-preserved.py`와 이전 결과 폴더는 역사적 원본으로 보존한다.
실험 실행은 버전별 새 경로를 사용하며 v2 기본값으로 과거 v3/v4/v5 폴더를 재개하면 조건 불일치로 거부한다.
기존 인덱스·revision을 v2로 재라벨링하거나 prompt/schema/criteria 해시가 다른 결과를 혼합하지 않는다.

## 잔여 한계와 후속 작업

v2에는 떨어진 인용 연결·요약·장식 인용·번역 오류와 상위명/게임명/플랫폼에서 세부를 과잉 추정한 이력이 있다.
채택 후에도 정확한 원문 인용·빈 근거·중복·부모 관계·개수 상한·재개 무결성 검증을 유지한다.
인용/부모를 보정하거나 후보를 삭제해 실패를 숨기지 않으며 실제 의미 검수는 별도다.

[#70](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/70)의 명시적 채택 버전 정리는 이번 #71로 다룬다.
compact 사람 홀드아웃 프로토콜·버전별 스키마 해시·과거 r2/r3 옵션 점검은 향후 **재채택 검토가 필요할 때의 후속 작업**으로 남긴다.
#70이나 #66을 자동 종료하지 않고, 이번 결정으로 독립 검증을 완료했다고 처리하지 않는다.
세부 85개 코드 확정·BE 시드·추천 순위 반영·배포는 별도 작업이다.

이번 작업은 합성/오프라인 테스트만 수행한다. 유료 API 호출·전체 재분류·실제 사람 검수·서비스 E2E는 수행하지 않는다.
실제 재생성이 필요하면 새 경로·예산·외부 전송 승인 범위를 별도로 정한다.

## 이번 구현의 검증

변경 전 관련 테스트 364개, 변경 후 채택 회귀 17개를 포함한 **381개 통과**(25.19초).
기본/명시 v2의 단일 CLI·모델 주입 경로·배치 요청과 출처 일치, 고정 지문 보존,
실험 버전/해시 혼합 재개 시 무호출·무변경 거부, 검수 시 원본 실험 출처 보존을 확인했다.
기존 평가·버전 격리·SDK MockTransport·정확한 인용·부모·중복·상한 테스트도 함께 통과했다.
`git diff --check` 통과. 전체 저장소 테스트·실제 API·사람 검수·BE/서비스 E2E는 실행하지 않았다.

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest tests/test_detail_tagging_adoption.py tests/test_detail_tagging.py tests/test_detail_tagging_batch.py tests/test_detail_tagging_human_eval.py tests/test_detail_tagging_holdout.py tests/test_detail_tagging_ai_proxy.py tests/test_detail_tagging_experiment.py tests/test_detail_tagging_v5.py tests/test_detail_tagging_code_map.py tests/test_sample_review.py tests/test_subtopic_tags.py tests/test_subtopic_taxonomy_hash.py -q --basetemp results/pytest-issue71-final
git diff --check
```
