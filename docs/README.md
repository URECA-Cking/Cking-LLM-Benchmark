# 문서 안내

## 결론을 먼저 읽기

[현재 추천 방식 결정](results/recommendation-decision.md)은 세 실험의 결과와 적용 범위를 한곳에 정리합니다.
보고서의 사람 평가와 LLM 평가를 같은 성적표로 합치지 않습니다.

## 최근 실험

| 질문 | 보고서 | 평가자 |
| --- | --- | --- |
| 소개 기반 추천에 어떤 방식을 쓸까? | [최종 100개 입력](experiments/final-eval/report.md) | 사람 1명 |
| 직접 세부 관심사를 선택하면 좋아질까? | [7개 관심사 비교](experiments/selected-topics/human.md) | 사람 1명 |
| 선택 조합을 자동으로 비교하면? | [170개 조합 평가](experiments/selected-topics/automatic.md) | GPT |
| 보정 크기와 세부 규칙의 영향을 비교하면? | [가산점 조건 비교](experiments/bonus-ablation/report.md) | LLM·작은 사람 진단 |
| 상위·세부 주제 보정의 차이는? | [계층 태그 실험](experiments/subtopics/report.md) | LLM |

## 선행 실험과 데이터

- [실제 채널 데이터 평가](experiments/real-data/report.md) · [데이터 범위](experiments/real-data/scope.md)
- [가입·즐겨찾기 흐름 파일럿](experiments/recommendation-flow/pilot.md) · [M2/M3/M4 비교](experiments/recommendation-flow/m234.md)
- [태그 사전과 경계](reference/creator-subtopics.md)

## 실행·파일 찾기

- [설치와 준비](guides/installation.md)
- [단일 크리에이터 서비스 후보 생성](guides/service-recommendation.md)
- [서비스 예시 데이터 점검](results/service-recommendation-check.md)
- [기본 합성 벤치마크 실행](guides/pipeline.md)
- [실데이터·후속 실험 실행](guides/experiments.md)
- [지표·데이터·문제 해결](guides/reference.md)
- [코드·공개 집계·로컬 결과 위치](reference/repository-layout.md)

## 과거 기록

- [초기 모델 선정 기록](history/model-selection/index.md): 당시 기준과 수치, API 전환·비용 근거
- [추천 흐름 탐색 요약](history/recommendation-exploration.md)
- [탐색 검증 기록](history/recommendation-validation.md)

과거 문서는 당시 판단을 보존합니다. 현재 채택안은 결과 요약에서 확인합니다.
