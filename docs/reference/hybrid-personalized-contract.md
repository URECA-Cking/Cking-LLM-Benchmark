# HYBRID_PERSONALIZED_V1 참조 구현과 공용 fixture

[LLM Issue #42](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/42)의 정본은
`src/recommendation/hybrid.py`와 저장소 루트 `fixtures/hybrid_personalized_v1.json`이다.
Python과 Java가 **동일한 JSON 파일**을 읽고 입력별 결과 객체 전체를 비교한다.
fixture는 UTF-8, BOM 없음, LF 줄바꿈이며 별도의 모델·DB·네트워크가 필요 없다.

관심 분야 M3 후보와 팔로우 seed의 유사 후보는 각각 BE에 저장된 활성 세대의 **rank만** 사용한다.
M2/M3/M4 raw score, 소개글, 임베딩, 모델 호출은 이 정책의 입력이 아니다.
관심 분야 정본은 [v0.2 분류 계약](taxonomy-contract.md), 후보 생성은
[관심 분야 배치 안내](../guides/interest-recommendation.md)를 참고한다.

## 파일 구조와 입력

최상위는 `schemaVersion: 1`, 고정 `policy` 설정과 `cases` 배열이다.
각 case에는 고유 `id`, 설명 `description`, `input`, 고정 `expected`가 있다.
`policy`는 `rrfK: 60`, `scoreScale: 8`, `roundingMode: "HALF_UP"`,
`interestWeight: "0.5"`, `followWeight: "0.5"`다.
V1에서 설정을 바꾸지 않는다. 변경하려면 새 정책 버전과 fixture를 정의한다.

| input 필드 | 타입 | 의미 |
| --- | --- | --- |
| `memberCreatorId` | 양의 정수 또는 null | 회원 본인의 Creator ID. Creator가 없는 회원은 null |
| `selectedInterestCodes` | 문자열 배열 | 회원이 선택한 분야. 중복 선택은 한 번만 사용 |
| `followedCreatorIds` | 양의 정수 배열 | 현재 팔로우한 Creator. seed 선택과 후보 제외에 모두 사용. 중복은 한 번만 사용 |
| `interestSources` | source 객체 배열 | 분야별 현재 활성 세대. 식별자 키는 `interestCode` |
| `followSources` | source 객체 배열 | 팔로우 seed별 현재 활성 세대. 식별자 키는 `seedCreatorId` |
| `creatorSpaces` | 객체 배열 | `{creatorId, hasCreatorSpace}`. Space 존재 여부는 boolean. 목록에 없는 ID도 Space 없음으로 처리 |

분야 코드는 대문자 ASCII 식별자 `[A-Z][A-Z0-9_]*`다. 관심 분야 표시 순서와 무관하게
기여 출처의 코드는 사전순으로 정렬한다. Creator ID는 Java 양의 `long` 범위,
rank는 양의 `int` 범위다. JSON boolean이나 소수로 ID/rank를 대체할 수 없다.

```json
{
  "interestCode": "FOOD",
  "activeGeneration": {
    "generationId": "generation-FOOD",
    "candidates": [{"creatorId": 101, "rank": 2}]
  }
}
```

팔로우 source도 같은 구조이며 `interestCode` 대신 `seedCreatorId` 정수를 쓴다.
`activeGeneration: null`은 활성 세대 없음, `candidates: []`는 활성 빈 세대다.
`generationId`는 비어 있지 않은 불투명 문자열 식별자이며 점수에는 영향을 주지 않는다.
이미 활성 세대를 선택한 스냅샷을 전달하므로 과거 세대의 후보를 섞지 않는다.
source 행 자체가 없는 선택 분야/팔로우 seed도 무효다.
선택하지 않은 분야와 팔로우하지 않은 seed의 source는 집계에 참여하지 않는다.

하나의 source 식별자는 입력에 한 번만 있어야 한다. 한 세대의 후보 ID와 rank도 각각 고유하다.
다른 source 사이에서 동일한 후보가 나오는 것은 허용하며 기여도를 합친다.
rank에 빈 번호가 있어도 허용하며 후보 배열을 재정렬하거나 제외한 뒤 새 rank를 붙이지 않는다.
`creatorSpaces`의 ID도 고유해야 한다. 참조 함수는 누락된 필수 필드에 `KeyError`,
잘못된 타입·범위·중복 식별자에 `ValueError`를 내어 모호한 스냅샷을 거부한다.

## 제외, 유효 source와 계산 순서

1. 선택한 분야와 팔로우 seed를 각각 중복 제거한다.
2. 각 source의 활성 후보에서 본인, 이미 팔로우한 Creator, Space가 없는 Creator를 제외한다.
3. 제외 후 후보가 한 명 이상 남은 source만 유효하다. 세대 없음, 빈 세대, 전부 제외된 source는
   모두 평균 분모에서 빠진다. 유효 source의 점수가 0으로 반올림되어도 source는 유효하다.
4. 각 남은 후보의 원래 rank에 대해 `1 / (60 + rank)`를 소수점 8자리 `HALF_UP`으로 반올림한다.
5. 같은 후보의 **반올림된** 기여도를 그룹 안에서 합산하고, 그 그룹의 전체 유효 source 수로 나눈 뒤
   소수점 8자리 `HALF_UP`으로 반올림한다. 해당 후보가 나오지 않은 유효 source도 분모에 포함된다.
6. 두 그룹이 유효하면 `관심 평균 × 0.5 + 팔로우 평균 × 0.5`를 다시 소수점 8자리 `HALF_UP`으로
   반올림한다. 후보가 한 그룹에만 있어도 다른 그룹의 평균 0과 함께 계산한다.
   한 그룹만 유효하면 해당 그룹 평균을 전체 점수로 사용하며 0.5를 곱하지 않는다.
7. 최종 반올림 점수 내림차순, 동점이면 숫자 `creatorId` 오름차순으로 정렬한다.

기여도, 그룹 평균, 최종 가중합의 **세 반올림 시점**을 유지한다. 끝에 한 번만 반올림하면 다른 결과다.
예를 들어 rank 1과 11에 나온 후보의 관심 평균은
`(0.01639344 + 0.01408451) / 2 = 0.015238975 → 0.01523898`이다.
원래 분수의 평균을 마지막에만 반올림하면 `0.01523897`이 된다.
rank 452의 기여도 `0.001953125 → 0.00195313`은 `HALF_EVEN`과 차이를 드러낸다.
rank 2의 기여도 `0.01612903`을 유효 source 둘로 나누거나 가중치 0.5를 곱하면
`0.008064515 → 0.00806452`다.

## 기대 결과

| expected 필드 | 의미 |
| --- | --- |
| `policyVersion` | 아래 분기표에 따른 정책 이름 |
| `validSourceCounts.interest` | 모든 제외를 적용한 뒤 유효한 관심 분야 source 수 |
| `validSourceCounts.follow` | 모든 제외를 적용한 뒤 유효한 팔로우 seed source 수 |
| `items` | 점수·ID 규칙으로 정렬한 전체 후보 배열. 순서도 기대값의 일부 |
| `items[].creatorId` | 최종 추천 후보 ID. 여러 source에 나와도 한 번만 기록 |
| `items[].aggregateScore` | 소수점 8자리를 유지한 JSON 문자열. 예: `"0.01562500"` |
| `items[].interestCodes` | 해당 후보에 실제로 기여한 분야의 중복 없는 ASCII 사전순 목록 |
| `items[].seedCreatorIds` | 해당 후보에 실제로 기여한 seed의 중복 없는 숫자 오름차순 목록 |

| 관심 유효 source | 팔로우 유효 source | policyVersion |
| --- | --- | --- |
| 1개 이상 | 1개 이상 | `HYBRID_PERSONALIZED_V1` |
| 1개 이상 | 0개 | `INTEREST_PERSONALIZED_V1` |
| 0개 | 1개 이상 | `FOLLOW_PERSONALIZED_V2` |
| 0개 | 0개 | `FOLLOW_PERSONALIZED_V2`, `items: []` |

`aggregateScore`는 확률이 아니라 이 정책 내부의 정렬 점수다. 서로 다른 정책 버전의 절대값을 비교하지 않는다.
이슈 #42의 팔로우 단독 분기는 **유효 seed 평균**을 쓴다. 기존
[BE Issue #410](https://github.com/URECA-Cking/Cking-BE/issues/410)의 합산 방식과 차이가 있으므로
기존 합산 정책은 `FOLLOW_PERSONALIZED_V1`을 유지하고, 평균 정책은 `FOLLOW_PERSONALIZED_V2`로 구분한다.
두 그룹이 모두 무효인 빈 결과도 `FOLLOW_PERSONALIZED_V2`를 반환한다.
Java 하이브리드 구현은 V2 분기에서 이 fixture의 평균 계약을 따른다.
이번 작업은 BE 코드를 변경하지 않는다.

## Python 실행과 Java 적용

```python
import json
from pathlib import Path
from src.recommendation.hybrid import recommend_personalized

fixture = json.loads(Path("fixtures/hybrid_personalized_v1.json").read_text(encoding="utf-8"))
for case in fixture["cases"]:
    assert recommend_personalized(case["input"]) == case["expected"], case["id"]
```

Python은 중간 연산에 충분한 정밀도의 독립 `Decimal` context를 사용하므로
호출자의 정밀도·rounding·trap 설정에 영향을 받지 않는다. 출력 문자열을 float로 바꾸지 않는다.
함수는 입력을 변경하지 않고 전체 순위를 반환한다. 페이지 크기 제한은 집계·정렬 뒤 조회 계층에서 적용한다.

Java에서는 파일을 그대로 테스트 리소스로 복사하고 Jackson `JsonNode`로 읽을 수 있다.
가중치와 기대 점수는 `new BigDecimal(node.asText())`로 읽는다. `double`, `asDouble()`,
`new BigDecimal(0.5)`나 `stripTrailingZeros()`로 변환하지 않는다.

```java
// 각 source 후보의 기여도: divide 단계에서 첫 반올림
BigDecimal contribution = BigDecimal.ONE.divide(
        BigDecimal.valueOf(60L + rank), 8, RoundingMode.HALF_UP);
// 해당 후보의 기여도를 그룹 안에서 더한 뒤 두 번째 반올림
BigDecimal mean = sum.divide(BigDecimal.valueOf(validSourceCount), 8, RoundingMode.HALF_UP);
// 두 그룹이 유효할 때만 세 번째 반올림. 후보가 없는 그룹은 0
BigDecimal score = interestMean.multiply(new BigDecimal("0.5"))
        .add(followMean.multiply(new BigDecimal("0.5")))
        .setScale(8, RoundingMode.HALF_UP);
```

분모가 0인 그룹은 나누지 않는다. 정렬은 `BigDecimal` 내림차순 비교 후 `long` ID 오름차순이다.
숫자를 문자열로 정렬하거나 후보마다 기여 source 수를 분모로 쓰지 않는다.
Java 결과를 fixture 형식으로 직렬화할 때는 `score.setScale(8).toPlainString()`을 써서
정확한 점수 문자열, 정렬 순서, 정책, 출처, 두 유효 source 수를 모두 비교한다.
fixture의 큰 rank 사례는 반올림 경계를 검증하기 위한 합성 입력이며 서비스의 Top-N 설정을 바꾸지 않는다.

## fixture 검증 범위

18개 case는 기본 네 분기, 그룹 안팎 중복 후보 합산, 동점 ID 정렬, 모든 후보 제외 규칙,
무효 source 분모 제거, 원래 rank 유지, 누락·미선택 source, 중복 선택, null 회원 Creator,
정렬된 기여 출처, 세 반올림 경계, 단계별 반올림, 반올림 후 동점과 0 점수 유효 source를 포함한다.
기대값은 참조 함수에서 생성하지 않고 독립 정수·유리수 계산으로 확인한 고정 문자열이다.
테스트는 모든 case를 읽어 기대 결과 객체 전체를 비교하고, 입력 순서·Decimal context 독립성과 입력 불변성도 검증한다.

```bash
python -m pytest -q tests/recommendation/test_hybrid.py
```

Java 하이브리드 조회, 회원 관심 분야 저장, 클릭·즐겨찾기 지표, 다양성 재정렬, 인기순 fallback은 범위 밖이다.
Java 테스트 실행·서비스 통합 완료를 주장하지 않으며, BE 구현 시 같은 파일 전체를 검증해야 한다.
