# 서비스 추천 후보 예시 데이터 점검

점검일: 2026-10-02

[실행 계약](../guides/service-recommendation.md) · [현재 방식 결정](recommendation-decision.md)

## 목적과 데이터 경계

Issue #36의 서비스 생성기에 실제 크기의 프로필 입력을 넣었을 때 정상 소개, 짧은 소개, 빈 소개, 후보 부족 정책이 동작하는지 확인했다.
로컬 `inputs/real_v4/raw/pool.csv`의 공개 YouTube 채널 프로필 4,743개와 이미 생성돼 있던 `results/real_v4`의 BGE-M3 벡터·GPT 상위 분야 태그를 사용했다.
원문·전체 캐시는 Git 제외 자료이며 이 문서에는 사례 식별자와 동작 결과만 기록한다.

이 점검의 `creatorId`는 서비스의 BIGINT 계약을 검증하기 위해 풀 순서의 양의 정수로 대응시켰다. 아래 채널 ID는 원자료 사례를 다시 찾기 위한 식별자다.
실제 Cking 운영 DB의 프로필 덤프나 운영 creatorId를 사용한 것은 아니다.

## 결과

| 사례 | seed 채널 ID | 소개 길이 | 결과 | 첫 실행 모델 캐시 조회 결과 | 같은 입력 재실행 |
| --- | --- | ---: | --- | --- | --- |
| 정상 소개 | `UC2ntik7Q6x8DxAOIooJeITQ` | 34 | M4, 5명 | 임베딩 44배치, 태그 4,195건 | 추가 호출 0, payload 동일 |
| 짧은 소개 | `UCh5w1Kziu9ZbZBeejZiNojg` | 4 | M2, 5명 | 임베딩 44배치, 태그 0건 | 추가 호출 0, payload 동일 |
| 빈 소개 | `UCYrCnlnTXQoDj0o52qJbdZg` | 0 | 빈 후보 | 임베딩 0, 태그 0 | 추가 호출 0, payload 동일 |
| 후보 부족 | `UC2ntik7Q6x8DxAOIooJeITQ` | 34 | M4, 요청 5명 중 2명 | 임베딩 1배치, 태그 3건 | 추가 호출 0, payload 동일 |

정상 소개 결과의 상위 5개 채널 ID는 아래와 같았다.

1. `UCSR9_FWiORjcq_ZLrN61FQQ`
2. `UCfSxckXGN40ao42PQH_mF5Q`
3. `UCQmMYKjxRTSKWXD2N3avxFg`
4. `UCjd2O1fzT7LRnc8hix9KHFA`
5. `UCu6ynX9XUEnAewUwZMI3k6A`

짧은 소개 결과의 상위 5개 채널 ID는 `UClYV84oSKMXKKegqHP569lA`, `UCagSSh4tXxrZ1p8VCqDt9JQ`,
`UCOUVzwXBHNvPBJvegkzvtnw`, `UCcr6u4oc3UQzcoo3cpPYtYg`, `UCZ-UuK86Bk5sGzQQ2cEglJQ`였다.
후보 부족 사례는 운동 소개 1명과 요리 소개 1명만 후보로 주어 두 명만 반환했고, 무관한 후보를 추가하지 않았다.

## 해석과 한계

- 이 점검은 새 서비스 코드의 분기·정렬·캐시·후보 수 계약을 확인한다. 추천된 채널이 실제 사용자에게 적합하다는 온라인 품질 검증이 아니다.
- 사용한 벡터의 저장 식별자는 `bge-m3@local`이다. 운영 CLI가 쓰는 DeepInfra `BAAI/bge-m3` API의 이번 입력 전체 재호출, 수치 동등성, 비용, 지연은 실행하지 않았다.
- GPT 태그도 기존 캐시를 재사용했다. 새로운 프롬프트·모델 호출의 실시간 가용성을 확인한 결과가 아니다.
- 빈 소개 후보 548개는 모델에 보내지 않아 정상 사례의 태그 대상이 seed 포함 4,195개였다. 운영에서는 입력 manifest를 먼저 고정하고 빈 소개 제외 사유를 함께 기록해야 한다.
- 실제 Cking 운영 프로필과 승인·노출 상태는 BE 통합 전에 별도로 점검해야 한다. 오프라인 평가 점수를 서비스 전환 성능으로 간주하지 않는다.
