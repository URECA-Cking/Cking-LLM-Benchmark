# 과거 임베딩·방식 선정 기록

100명 합성 크리에이터를 사용한 2026년 9월 초기 탐색 기록이다. [현재 사람 평가 결론](../../results/recommendation-decision.md)을 먼저 읽는다.

## 결론

[상세 기록](selection.md)

## 사전 판단 기준 대비 결과

[상세 기록](selection.md)

## 선택 이유와 한계

[상세 기록](selection.md)

## 측정 조건

[상세 기록](quality.md)

## E1. zero-shot 태깅 정확도 (test 70명)

[상세 기록](quality.md)

## E2. 관련도 (LLM 자동 판정, 쿼리 30명 × 상위 5명, 681쌍)

[상세 기록](quality.md)

## E3. 대표 사례 (top-5, 22개 설정 전체, `src/pipeline.py report`)

[상세 기록](quality.md)

## dev 규모 민감도 (이슈 #8, `src/pipeline.py dev-sensitivity`)

[상세 기록](extensions.md)

## 사용자 취향 쿼리 검증 (이슈 #12, `taste-eval`·`taste-judge`)

[상세 기록](extensions.md)

## 로컬 모델 메모리 사용량 (이슈 #13, `python3 -m src.pipeline memory`)

[상세 기록](extensions.md)

## 로컬 ↔ API bge-m3 동등성 (이슈 #20·#22, `python3 -m src.pipeline api-parity`, `api-select-params`)

[상세 기록](operations.md)

## 비용·소요 시간

[상세 기록](operations.md)

## 남은 과제

[상세 기록](operations.md)

## 변경 이력

[상세 기록](operations.md)
