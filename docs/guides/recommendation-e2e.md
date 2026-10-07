# 실제 BE HTTP·MySQL 추천 배치 E2E

[문서 안내](../README.md) · [일일 배치](daily-recommendation.md)

## 실행

Python 3.11+, 이 저장소의 `requirements.txt`, JDK 21, Docker Engine과
최신 계약(#473)이 반영된 Cking-BE checkout이 필요하다. BE의 추적 파일은 수정하지 않는다.
저장소 루트에서 한 번 실행한다.

```bash
python -m src.recommendation.e2e.runner --be-dir ../Cking-BE
```

Windows에서 가상환경 Python은 `.venv\Scripts\python.exe`다.
`JAVA_HOME`을 JDK 21로 지정한다. `GRADLE_USER_HOME`은 선택사항이며,
초기 실행에는 MySQL 8.4·Redis 7.2 이미지와 Gradle·Maven 의존성 다운로드가 필요하다.
API Key, JWT, 운영 DB 주소를 입력받지 않는다. 이미 실행 중인 BE에도 연결하지 않는다.

## 격리 환경과 준비·정리

실행기는 UUID가 붙은 전용 MySQL·Redis 컨테이너를 만들고 각 포트를
`127.0.0.1`의 임의 포트에만 연결한다. MySQL DB명도 `cking_e2e_<UUID>`로 고정하며,
외부 DB 주소나 기존 컨테이너 이름을 지정하는 옵션은 제공하지 않는다.
Docker endpoint도 로컬 Unix socket/Windows named pipe만 허용하며 원격 Docker context는 거부한다.
MySQL root 비밀번호, 추천 API Key, JWT 서명키는 실행별 난수이고 자식 프로세스 환경으로만 전달한다.
Docker 명령 인자에 비밀번호·키 값을 넣지 않는다.

`bridge.gradle`은 BE의 main 클래스·실제 의존성을 사용하는 별도 source set과
`recommendationE2e` 테스트 태스크를 구성한다. LLM 저장소의 Java 테스트만 컴파일하고
BE 전체 테스트나 기본 local 프로필은 실행하지 않는다. 실제 Flyway migration과
Hibernate schema validation을 거쳐 임의 포트의 Spring Boot 웹 서버를 시작한다.
SecurityFilterChain, JWT 발급기·검증기, 추천 Application·Repository는 대체하지 않는다.
스토리지는 BE의 메모리 구현을 쓰고 정기 작업과 구독 인증 recovery는 끈다.

Java 테스트는 DB URL이 이 실행 전용 loopback DB 형식인지 확인하고,
Member·Creator 테이블이 비어 있는지 확인한 뒤에만 고유 이름의 테스트 회원·Creator·Space를 만든다.
분류체계는 실제 Flyway의 v0.2 시드를 쓰며 Python 정본과 taxonomyHash가 같은지 확인한다.
JWT는 테스트 회원·관리자에 대해 BE의 실제 AccessTokenIssuer로 발급한다.
별도 ADMIN 회원의 발급된 JWT를 보존한 채 DB 역할을 USER로 바꿔 권한 재검증을 확인한다.

Java와 Python의 fixture 준비·DB 조회는 부모/자식의 stdin/stdout 제어 채널로만 처리한다.
추가 HTTP 제어 API나 인증 우회 엔드포인트를 만들지 않는다.
관계 정리는 이 실행의 회원 ID로만 한정하며 테이블 전체 DELETE·TRUNCATE는 없다.
시나리오 마지막에 생성된 회원·Creator 수를 다시 확인한다.

정상 종료와 검증 실패 모두 `finally`에서 이 실행이 생성한 두 컨테이너만
`docker rm -f -v`로 제거한다. MySQL의 익명 volume도 제거되므로 DB fixture는 남지 않는다.
공용 개발 컨테이너·DB·volume에는 접근하지 않는다. 프로세스를 강제 종료하거나 전원을 끈 경우
자동 정리가 실행되지 않을 수 있다. 이 경우 `docker ps -a --filter label=cking.e2e`로 남은
실행의 UUID 라벨을 확인하고 해당 실행의 컨테이너만 제거한다. 전역 prune은 사용하지 않는다.

## 실제 HTTP·DB 검증

| 시나리오 | 검증 |
| --- | --- |
| 고정 manifest | 실제 `/api/creators`를 3개씩 마지막 페이지까지 순회하고 다른 페이지 크기와 hash 비교 |
| 최초 적재 | 고정 벡터·태그 모델로 유사 추천과 17개 분야 후보 생성, 실제 두 PUT API 호출 |
| DB 정합성 | 응답 generationId와 활성 포인터·sequence·inputHash·후보 ID·순위·점수·건수 비교 |
| 같은 실행 재적재 | 완료 manifest skip에서 추가 모델·PUT 없음, 같은 payload의 applied=false와 DB 이력 불변 |
| 응답 유실 | 실제 BE 커밋과 응답 수신 후 transport에서 IncompleteRead 주입, 재시도의 멱등 응답 확인 |
| 멱등 충돌 | 같은 적용 번호의 다른 payload는 409 RECOMMENDATION_INPUT_CONFLICT, DB 불변 |
| 부분 실패 재개 | 유사/분야 각 1건의 transport 전송 실패, 혼재된 실제 DB 세대 확인, 같은 번호로 두 실패 대상만 재개 |
| A→B→A | 소개를 변경·복원하고 과거와 같은 inputHash가 더 큰 적용 번호로 활성화되는지 비교 |
| A→B 일부→A | 미완료 B보다 새 A로 전환, 전송된 적 없는 지연 B까지 409 STALE_RECOMMENDATION_INPUT 확인 |
| 빈 세대 | 두 API에서 빈 후보 활성화, DB의 이전 후보 미노출 및 공개 유사 조회의 빈 목록 확인 |
| 인증 | 양 API에서 키 성공, 잘못된 키 401, ADMIN JWT+잘못된 키 401, USER JWT 403, ADMIN JWT 성공 |
| 권한 회수 | JWT를 재발급하지 않고 ADMIN DB 역할 회수 후 양 API 403 및 DB 불변 |

BE는 모델 응답을 만들지 않는다. Python의 모델 클라이언트만 고정 벡터·FOOD 태그를 반환한다.
실제 운영 배치 생성기·체크포인트·적용 번호 원장·HTTP 클라이언트를 사용한다.
HTTP 오류를 가짜 응답으로 만들지 않으며 장애 주입은 전송 차단/응답 유실뿐이다.

## 개인화 공용 fixture와 최신 fallback

`fixtures/hybrid_personalized_v1.json`의 18개 case를 실제 DB fixture와
`GET /api/me/creator-recommendations`로 재생한다. 논리 Creator ID는 실제 생성한 ID에
오름차순으로 매핑하므로 동점 ID 정렬도 보존한다. 정책 버전·전체 순서·8자리 점수·
관심 분야/팔로우 근거, 본인·기팔로우·Space 없음 제외, 후보 부족을 확인한다.

공용 정책 fixture에는 희소 rank, rank 10010·10011, int 최대 rank와 자기 seed 후보처럼
적재 API가 허용하는 연속 Top-100 범위 밖의 입력도 있다. 따라서 이 **조회 정책 검사**에서는
HTTP로 먼저 빈 세대를 활성화하고, 그 세대의 후보를 Java 테스트의 JDBC fixture로 준비한다.
이 구간을 모델 생성 payload의 실제 적재 검증으로 주장하지 않는다.
실제 생성 후보의 HTTP 적재·DB 정합성은 앞선 독립 배치 시나리오가 검증한다.
공용 fixture의 기대 점수는 다시 계산하지 않고 원래 expected와 비교한다.

최신 BE의 인기순 fallback 계약(#466)이 반영되어 있어 공용 fixture의 빈 개인화 결과는
`POPULAR_FALLBACK_V1`로 구분한다. Space가 있고 본인·기팔로우가 아닌 후보만 남기고,
팔로워 수 내림차순·동점 Creator ID 오름차순, 팔로워 수 점수와 빈 추천 근거를 확인한다.
개인화 결과가 하나라도 있으면 결과가 size보다 부족해도 인기순으로 채우지 않는다.
별도 인기순 시나리오에서는 다른 합성 회원들의 팔로우를 준비해 팔로워 수 2·1·0인 후보의
순서·점수·size 제한도 확인한다.

## 결과와 실패 구분

각 실행의 결과는 기본 `results/recommendation/e2e/<UUID>/report.json`에 남긴다.
`--output-dir`로 기본 결과 루트를 바꿀 수 있다. 보고서는 성공한 개별 검사 이름,
실패 단계/검사, 양쪽 checkout commit SHA, taxonomyHash, 합성 모델 입력 수,
외부 연결 시도 수, 정리 결과를 포함한다. 작업 중 실행의 SHA는 그 시점 HEAD이며
아직 커밋하지 않은 변경의 내용 지문은 아니다.

| 종료 코드 | 의미 |
| --- | --- |
| 0 | 모든 시나리오·BE 최종 fixture 감사·컨테이너 정리 성공 |
| 1 | BE와 연결된 시나리오 assertion/HTTP/DB 계약 검증 실패 |
| 2 | Docker·이미지·DB 준비·Java/Gradle 빌드·BE 시작·timeout·정리 실패 |

키·JWT·비밀번호·Creator 소개·HTTP 오류 본문·SQL 원문은 보고서와 콘솔에 기록하지 않는다.
실패는 예외 타입과 소스 파일명·줄 번호로 찾는다. Gradle 출력도 그대로 전달하지 않는다.
BE 테스트 자체의 JUnit 결과는 BE의 `build/`에 생성된다. Spring 로그는 WARN으로 제한하고
토큰·요청 본문을 테스트 출력에 쓰지 않는다.
모델 캐시·manifest·payload는 Git 제외 결과 폴더에 남으며 모두 합성 fixture 내용이다.

부모 실행기는 운영 키·Spring 설정을 제거한 환경으로 자식을 실행한다.
Python 시나리오는 `.env` 자동 로드를 끄고 loopback 외 소켓 연결을 거부·계수한다.
유료 API, 로컬 모델 다운로드, 운영 데이터 전송은 실행하지 않는다.

실행기 자체의 환경 보호·실패 코드·정리 회귀 검사는 Docker 없이 실행할 수 있다.

```bash
python -m pytest -q tests/recommendation/test_e2e_runner.py
```
