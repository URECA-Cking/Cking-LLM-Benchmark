# Cking LLM Benchmark

크리에이터 소개글로 비슷한 크리에이터를 추천하는 방식을 비교하는 Python 3.11+ 실험 저장소입니다.

## 지금 결론

**일반 소개는 bge-m3 + M4, 짧은 소개는 M2를 잠정 선택합니다.**
M4는 소개글 유사도에 LLM이 분류한 상위 분야 일치 보정을 더합니다.
자동 세부 태그 H4의 추가 효과는 작고 불확실해 기본 방식으로 채택하지 않습니다.

직접 선택 세부 태그는 후속 선택사항으로 검토합니다. 사람 진단에서는 베이킹에서 개선됐고,
자동 조합 평가에서도 개선 방향이 관찰됐습니다. 다만 자동 판정이 사람과 23/53쌍만 일치해
자동 점수만으로 서비스 적용을 확정하지 않습니다.

## 처음 읽는 순서

1. [현재 결과·선택 이유](docs/results/recommendation-decision.md): 무엇을 쓰고 왜 선택했는지
2. [전체 문서 안내](docs/README.md): 실험별 보고서와 과거 기록
3. [설치](docs/guides/installation.md): 환경과 테스트
4. [실행 안내](docs/guides/experiments.md): 실데이터·후속 실험
5. [폴더와 결과 파일 안내](docs/reference/repository-layout.md): 코드·공개 집계·로컬 원본 위치

서비스 적용 코드는 [단일 크리에이터 추천 후보 생성](docs/guides/service-recommendation.md)과
[전체 크리에이터 배치 생성·적재](docs/guides/batch-recommendation.md)에서 입력·출력 계약과 실행 방법을 확인합니다.
관심 분야 기반 추천의 오프라인 입력은 [17개 분야별 M3 Top-20 생성·적재](docs/guides/interest-recommendation.md)를 참고합니다.

## 모델별 역할

| 구성 | 역할 | 추천 조회 중 호출 |
| --- | --- | --- |
| bge-m3 | 소개글 의미 벡터 생성 | 저장된 벡터를 재사용하도록 설계 |
| 태깅 GPT | M4의 상위 분야 분류 | 저장된 태그를 재사용하도록 설계 |
| 평가 GPT | 실험에서 적합성을 자동 채점 | 서비스 구성에 포함하지 않음 |

서비스 후보 생성은 사용자 조회 경로가 아니라 별도 내부 작업에서 실행합니다. 조회 API는 Cking-BE에 저장된 후보만 읽습니다.

이 저장소는 오프라인 실험입니다. 실제 서비스 구현·API 정합성·비용·전환 성능은 별도 검증 대상입니다.

## 빠른 확인

```bash
python -m pytest -q
```

테스트는 유료 API를 호출하지 않습니다. 결과 재집계에는 각 보고서에 명시된 로컬 원본이 필요합니다.
임베딩·태깅·자동 채점 명령은 외부 전송 및 비용이 발생할 수 있으므로 실행 안내를 먼저 읽으세요.

## 폴더

```text
src/experiments/    최근 실험별 구현
src/clients/        모델 제공자 클라이언트
src/*.py           공통 파이프라인과 기존 명령 호환 진입점
tests/experiments/  최근 실험별 테스트
data/summaries/    공개 집계와 정정 기록
data/              기준 입력·태그 정의
docs/              결론·실험 기록·실행 안내
results/           Git 제외 로컬 캐시·원문·판정
tasks/             작업 계획과 진행 기록
```

기존 python -m src.final_eval_score 등의 명령도 계속 사용할 수 있습니다.
