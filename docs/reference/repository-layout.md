# 폴더와 파일 찾기

## 실험 구현

| 실험 | 구현 위치 | 기존 명령 |
| --- | --- | --- |
| 개발 표본·최종 평가 | src/experiments/final_eval/ | src.final_dev_prepare, src.final_eval_prepare, src.final_eval_score |
| 직접 선택 태그·자동 조합 | src/experiments/selected_topics/ | src.selected_subtopic_eval, src.selected_subtopic_score, src.selected_subtopic_auto, src.selected_subtopic_auto_score |
| 모델 호출 | src/clients/ | 공통 파이프라인에서 사용 |
| 서비스 후보 생성 | src/recommendation/ | src.recommendation.cli |
| 선행 실험·공통 계산 | src/의 기존 모듈 | 기존 명령 유지 |

최근 실험 테스트는 tests/experiments/final_eval/과 tests/experiments/selected_topics/에 있다.
루트 src의 최근 실험 파일은 과거 import·python -m 명령을 위한 작은 호환 진입점이다.
실제 구현은 실험별 폴더에서 수정한다. 실행 명령과 계산 동작은 그대로다.

## 공개 집계

| 실험 | 경로 |
| --- | --- |
| 최종 소개 기반 사람 평가 | data/summaries/final-eval/human.json |
| 직접 선택 사람 평가 | data/summaries/selected-topics/human.json |
| 자동 조합 평가 | data/summaries/selected-topics/automatic.json |
| 후보 준비 진단 | data/summaries/selected-topics/preparation.json |
| 선택 기록 정정 감사 | data/summaries/selected-topics/selection-audit.json |

집계 JSON은 계산 결과를 보관한다. 원본 없이 해당 수치를 독립 재계산할 수 있다는 뜻은 아니다.
data/creator-subtopics.json과 CSV 등 기준 입력은 위치를 유지한다.

## 로컬 원본·캐시

results/는 Git 제외 폴더다. 실행 계약과 기본 인자를 유지하기 위해 기존 폴더를 옮기지 않았다.

| 로컬 폴더 | 용도 |
| --- | --- |
| real_v4 | 기존 풀의 벡터·상위 태그 등 공통 소스 |
| flow_v4, flow_m234_v4 | 이전 입력·추천 조건 |
| subtopics_v1 | 세부 태그·벡터 캐시 |
| final_dev_v1 | 개발 20개 점검 |
| final_eval_v2 | 최종 100개·801쌍 원본 |
| selected_subtopics_user_v1 | 직접 선택 7개·53쌍 원본 |
| selected_subtopics_auto_v1 | 170개 입력·1,262쌍 자동 평가 원본 |
| selected_subtopics_v1 | 전체 85개 태그의 준비 진단 |
| selected_subtopics_metadata_verify_v1 | 메타데이터 수정 전후 동일성 확인 |
| recommendation | 서비스 생성기의 임베딩·태그 캐시와 BE 전달 결과 |

v1/v2나 review 이름의 다른 폴더는 이전 준비·검증 버전일 수 있다. 최신 결과를 임의로 판단하지 말고
각 보고서의 실행 경로를 따른다. 정리 과정에서 원문·답변·캐시를 삭제하거나 계약을 바꾸지 않았다.
results/README.md에도 같은 주요 원본 위치를 안내한다.
