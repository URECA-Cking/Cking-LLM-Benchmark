# Cking LLM Benchmark

크리에이터 소개글로 비슷한 크리에이터를 추천하는 방식을 비교하는 Python 3.11+ 실험 저장소입니다.

## 지금 결론

**서비스 관심 분야 추천은 LLM 확정 태그로 후보를 제한하고 BGE-M3 코사인 점수로 정렬합니다.**
[후보 자격·캐시·기존 실행 옵션 호환 정책](docs/interest-llm-eligibility.md)을 확인합니다.

**세부 태그 분류 프롬프트는 `grounded-detail-candidates-v2`를 표준 채택했습니다(2026-10-15, 사용자 결정).**
[채택 범위·지문·실험 한계](docs/results/detail-tagging-v2-adoption.md)와
[바로 실행하는 v2 명령](docs/guides/detail-tagging.md#표준-v2-실행)을 확인합니다.
상위 taxonomy v0.2·세부 목록 확정·추천 방식 선택과 별개이며 사람 정답·독립 홀드아웃 검증은 미수행입니다.

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
[세부 태그 분류·검수 버전](docs/guides/detail-tagging.md)은 상위 3개·상위당 세부 3개 정책,
근거 우선순위와 반례, 자동 결과와 검수 확정본의 별도 버전 저장을 설명합니다.
[일일 통합 배치](docs/guides/daily-recommendation.md)는 최신 manifest의 변경을 감지하고 두 추천 작업을 함께 재개·적재합니다.
[실제 BE E2E](docs/guides/recommendation-e2e.md)는 전용 MySQL·Redis에서 HTTP 적재·복귀·개인화·인증 계약을 검증합니다.
관심 분야 기반 추천의 오프라인 입력은 [17개 분야별 M3 Top-20 생성·적재](docs/guides/interest-recommendation.md)를 참고합니다.
분야 추천 방식 비교는 [M2·M3 블라인드 Judge·사람 평가](docs/guides/interest-judge-evaluation.md)에서 사전 판단 기준과 실행 계약을 확인합니다.
저장된 분야·팔로우 후보의 혼합 규칙은 [개인화 참조 구현과 Python·Java 공용 fixture](docs/reference/hybrid-personalized-contract.md)를 참고합니다.

## 모델별 역할

| 구성 | 역할 | 추천 조회 중 호출 |
| --- | --- | --- |
| bge-m3 | 소개글 의미 벡터 생성 | 저장된 벡터를 재사용하도록 설계 |
| 태깅 GPT | M4의 상위 분야 분류 | 저장된 태그를 재사용하도록 설계 |
| 평가 GPT | 실험에서 적합성을 자동 채점 | 서비스 구성에 포함하지 않음 |

서비스 후보 생성은 사용자 조회 경로가 아니라 별도 내부 작업에서 실행합니다. 조회 API는 Cking-BE에 저장된 후보만 읽습니다.

이 저장소는 오프라인 실험입니다. 실제 서비스 구현·API 정합성·비용·전환 성능은 별도 검증 대상입니다.

## 빠른 확인

최소 Python **3.11** 이상이 필요합니다. PR #69 세부 태그 관련 테스트는 Python **3.12**에서 검증했습니다.
시스템 기본 Python이 오래됐다면 3.11 이상으로 만든 `.venv`의 Python을 사용하세요.

```bash
python -m pytest -q
```

테스트는 유료 API를 호출하지 않습니다. 결과 재집계에는 각 보고서에 명시된 로컬 원본이 필요합니다.
임베딩·태깅·자동 채점 명령은 외부 전송 및 비용이 발생할 수 있으므로 실행 안내를 먼저 읽으세요.

## 배포

콘텐츠 작업자는 CI(`.github/workflows/ci.yml`)가 만든 배포 패키지로 서버에 올립니다. 서버 운영은 [콘텐츠 작업자 운영](docs/guides/content-worker.md)을 따릅니다.

- 패키지: `scripts/build-package.sh`가 독립 실행형 Python 3.12와 가상환경, `src`·`data`·`deploy`를 `/opt/cking-llm`에 묶습니다. 서버(Ubuntu 26.04, Python 3.14)의 Python과 관계없이 검증된 3.12로 돕니다
- 의존성: 배포는 `requirements-deploy.txt`(고정 버전)입니다. 서비스 경로는 임베딩을 DeepInfra API로 받아 torch·sentence-transformers를 넣지 않으므로, 배포 패키지에서는 `--embedding-provider local`을 쓸 수 없습니다. 테스트는 `requirements-test.txt`(배포 버전 + sentence-transformers, pytest)입니다
- 확인: `scripts/check-package.sh`가 서버와 같은 Ubuntu 26.04에서 읽기 전용·root가 아닌 사용자로 Python 버전, 서비스 모듈 import(torch 없음), systemd 실행 경로, BE 없이 기동해 BE 요청 단계까지 가는지를 보고 생성 설정 해시(`configHash`)를 출력합니다. taxonomy 승인 등록에 이 값을 씁니다
- CI
  - PR: 테스트(Python 3.12, CPU torch), 워크플로 검사
  - develop 푸시(패키지에 들어가는 파일이 바뀔 때), 매월 1일 10:00(KST) 재빌드, 수동 실행: 패키지 빌드 → 확인 → 개발 배포 버킷 `recommendation/releases/<빌드 시각>-<커밋 12자리>/` 업로드 → 배포 버전 값 `/cking/dev-recommendation/RELEASE` 갱신 → 개발 추천 배치 서버 교체(새 서버가 부팅하며 그 버전을 받음)
  - 실행은 푸시 순서대로 하나씩 돕니다. 배포된 커밋이 이 커밋보다 새로우면 배포 버전을 바꾸지 않고, 이력이 갈라졌으면 실패합니다
  - 매월 재빌드는 Python 3.12 최신 패치를 반영합니다. 라이브러리는 고정 버전이라 Dependabot PR(`.github/dependabot.yml`, 매월, 마이너·패치만)로 올리고, 메이저 업데이트는 따로 확인해 올립니다

로컬에서 패키지를 만들고 확인하려면 다음을 실행합니다(Docker 필요).

```bash
docker run --rm --platform linux/arm64 -v "$PWD:/repo:ro" -v "$PWD/build:/out" -w /repo python:3.12-slim bash -c \
  'apt-get update -qq && apt-get install -y -qq git >/dev/null && pip install -q uv==0.13.4 && git config --global --add safe.directory /repo && scripts/build-package.sh local-$(git rev-parse --short=12 HEAD) /out'
scripts/check-package.sh build/cking-llm.tar.gz
```

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

세부 태그 50개 고정 표본과 사람 검수 집계는 [실행 안내](docs/guides/sample-review.md)를 참고하세요.
