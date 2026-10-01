# 실데이터와 후속 실험 실행

[문서 안내](../README.md) · [현재 결론](../results/recommendation-decision.md)

## 11. 실제 크리에이터 데이터로 비교

합성 100명 결과가 실제 소개글에서도 유지되는지 보기 위해, 외부 폴더의 실제 YouTube 채널 데이터로 M2·M3·M4를 같은 조건에서 비교하는 별도 도구(`src/real_eval.py`)가 있습니다. 소개글 원문과 정답 라벨은 **저장소에 커밋하지 않고** `--data-dir`(또는 `REAL_DATA_DIR`)의 외부 폴더에서만 읽으며, 산출물은 `results/real/`(git 제외)에 저장합니다.

```bash
export REAL_DATA_DIR=<데이터 폴더>
python3 -m src.real_eval embed && python3 -m src.real_eval tag-llm && python3 -m src.real_eval select-params
python3 -m src.real_eval candidates && python3 -m src.real_eval judge-sheet && python3 -m src.real_eval auto-judge
python3 -m src.real_eval human-sheet   # 사람이 채운 뒤 human-agree, 마지막에 score
```

데이터 폴더 구조, 단계별 설명, **결과를 보기 전에 정한 판정 기준**(M4 − M3 ≥ +0.05이고 95% 구간의 하한이 0보다 클 때만 M4 채택), 비용 추정, 해석 주의는 [`docs/real-data-eval.md`](../experiments/real-data/report.md)에, 분야 목록·방식별 GPT 사용·평가 범위·정답 점검·다음 실험은 [`docs/real-data-scope.md`](../experiments/real-data/scope.md)에 있습니다. 이 도구는 API 비용이 들고(`tag-llm`·`auto-judge` 합쳐 대략 $2~3 추정) 기존 `pipeline` 결과와 테스트에는 영향을 주지 않습니다.

## 12. 가입·즐겨찾기 추천 흐름과 세부 주제 실험

태그 선택·크리에이터 선택·즐겨찾기 기반 추천의 후속 탐색 실험을 완료했습니다.

**처음 읽을 문서: [결론·방법·핵심 결과·다음 단계 요약](../history/recommendation-exploration.md)**

- [입력 흐름·초기 탐색](../experiments/recommendation-flow/pilot.md): 초기 방법 비교와 표본의 한계.
- [동일 입력 M2·M3·M4 비교](../experiments/recommendation-flow/m234.md): 468개 입력, 2,801쌍 판정.
- [세부 주제 정의](../reference/creator-subtopics.md): 상위 17개·세부 85개·콘텐츠 형식 10개.
- [계층 추천 결과](../experiments/subtopics/report.md): 기존 입력 유지, 3,709쌍 판정 및 모든 민감도 조건.
- [최종 실험 설계](../experiments/final-eval/report.md): 새 표본·사람 평가·정보 부족 처리·채택 규칙.
- [보정·가산점 조건 비교·사람 점검](../experiments/bonus-ablation/report.md): 네 조건 비교와 터미널 점검 결과·최종 실험 초안.
- [검증·일관성 보고서](../history/recommendation-validation.md): 테스트 출력, 고정 조건, 조건 변경, 재현 한계.

일반 소개글의 LLM P@5는 기존 M4 0.641, 총 가산점 유지 H4 0.655, 가산점이 최대 1.5배인 주 조건 H4 0.664입니다. 주 조건의 상승을 세부 정보만의 효과로 단정하지 않습니다.
같은 표본을 관찰하며 발전시킨 탐색 결과이며 사람 평가·실제 사용자 효과·확정 채택은 미검증입니다.
실제 소개글·개별 판정은 Git에 포함하지 않으므로 외부 v4 데이터와 로컬 캐시가 있어야 수치를 재현할 수 있습니다.
