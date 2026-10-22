# 상위 관심 분야 v0.2와 taxonomyHash

서비스 정본은 `data/categories_v2.csv`의 17개 `code`, `name`, `description`이다.
실제 평가 입력 `inputs/real_v4/labels/categories_v2.csv`의 필드 내용과 행 순서를 그대로 옮겼으며,
파일 줄바꿈만 LF로 고정했다. [실험 범위](../experiments/real-data/scope.md)의 표는 설명 요약이며 해시 입력으로 사용하지 않는다.
초기 10개 `data/categories.csv`와 `src.config.TAXONOMY_VERSION=v0.1`은 기존 합성 실험 재현용으로 유지한다.

단일 서비스 CLI와 배치는 `taxonomyVersion=v0.2`, `tagPromptVersion=creator-category-v2`를 기본으로 쓴다.
`--categories`, `--taxonomy-version`, `--tag-prompt-version`으로 과거 입력을 명시적으로 선택할 수 있다.
CSV 헤더는 `code,name,description`이어야 하며, 정규화 후 빈 필드·중복 코드·잘못된 행 구조는 거부한다.

## 정규화와 직렬화

1. 각 필드에서 CRLF와 CR을 LF로 변환한다.
2. 앞뒤 ASCII 공백, 탭, LF만 제거한다. Python의 `value.strip(" \t\n")` 범위다.
3. Unicode NFC로 정규화한다. NBSP·EM SPACE와 내부 공백·탭·LF는 유지한다.
4. CSV 행 순서를 유지하고 `{"categories":[{"code":"...","name":"...","description":"..."}]}` 순서로 키를 넣는다.
5. `json.dumps(payload, ensure_ascii=False, separators=(",", ":"))`로 직렬화한다.
6. BOM과 마지막 개행이 없는 UTF-8 바이트를 SHA-256으로 해시하고 64자리 소문자 hex로 기록한다.

CSV 입력 BOM은 허용하지만 canonical JSON에는 포함하지 않는다. 해시는 파일 바이트나 프롬프트를 직접 해시한 값이 아니다.
공용 구현은 `src.recommendation.taxonomy`의 `load_service_categories`, `canonical_taxonomy_json`, `taxonomy_hash`다.

```python
from src.recommendation.taxonomy import DEFAULT_CATEGORIES_CSV, load_service_categories, taxonomy_hash

print(taxonomy_hash(load_service_categories(DEFAULT_CATEGORIES_CSV)))
```

v0.2 기대 해시: `f77df7a020a8ebb7cb33c1177f098babbc6a58d2d487dbaabcfa8108bf7f8b35`

Java 구현에서도 줄바꿈 처리, ASCII trim 범위, NFC, 행·키 순서와 JSON escaping을 같은 순서로 적용해야 한다.
일반 Unicode 공백 제거 함수나 키 정렬 옵션으로 대체하지 않는다. 분류체계 버전과 프롬프트는 해시 입력 밖에서 별도로 관리한다.

## 언어 공통 fixture

`tests/fixtures/taxonomy/`는 Java 테스트에서도 그대로 읽을 수 있는 정본이다.

| 파일 | 역할 |
| --- | --- |
| `input.csv` | BOM, 한글·분해형 한글, ASCII 앞뒤 공백·탭, CRLF·CR·LF, Unicode 공백, 따옴표·역슬래시 원본 |
| `canonical.json` | 원본을 정규화한 기대 UTF-8 바이트, BOM·마지막 개행 없음 |
| `sha256.txt` | canonical JSON 기대 SHA-256 소문자 hex |
| `v02.canonical.json` | `data/categories_v2.csv`의 17개 분야 canonical JSON |
| `v02.sha256.txt` | v0.2 기대 SHA-256 |

`.gitattributes`에서 fixture를 `-text`로 지정해 Git의 줄바꿈 변환을 막는다.
검증 시 JSON을 다시 파싱한 값만 비교하지 말고 canonical UTF-8 바이트와 해시를 함께 비교한다.

## 캐시와 배치 재시작

태그 캐시에는 태그 모델, 프롬프트 버전·본문 해시, 분류체계 버전·`taxonomyHash`를 반영한다.
추천 `inputHash`에는 M2·M4·빈 결과 모두 이 분류체계와 프롬프트 식별자를 반영한다.
소개·임베딩 모델이 같으면 임베딩 캐시는 재사용한다.
배치 생성 설정 해시에도 `taxonomyHash`를 포함하므로 분류 내용이 바뀐 기존 체크포인트는 거부한다.
v0.1 결과 디렉터리를 보관하고 v0.2에는 새 `--output-dir`을 사용한다.

## BE 적재 payload와의 관계

`taxonomyHash` 필드는 적재 payload마다 다르다.

- 기존 Creator 유사 추천 적재(`PUT /api/admin/creators/{creatorId}/similar`) payload에는 `taxonomyHash` 필드를 추가하지 않는다.
  이 추천에서 분류체계 버전·해시는 payload 필드가 아니라 `inputHash`에만 반영된다.
- 관심 분야 추천 적재(`PUT /api/admin/interests/{interestCode}/recommendations`) payload는 최상위에 `taxonomyVersion`과
  `taxonomyHash`를 포함한다. [관심 분야 추천 가이드](../guides/interest-recommendation.md)의 요청 예시를 따른다.
  BE는 분류체계 버전과 해시를 `interest_taxonomy`에 등록해 두고(Cking-BE #431), 적재 시 등록된 버전이고 해시가 같은지 대조해
  다르면 묶음 전체를 거부한다(Cking-BE #439).

이 문서의 계약은 두 언어가 같은 정본을 검증하는 함수·fixture를 제공한다.
