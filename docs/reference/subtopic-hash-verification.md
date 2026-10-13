# 세부 태그 해시 교차 확인

2026-10-13 확인. BE PR #539의 머지 커밋 `401117f98c5c941de9bfaf064b8947a48476d87c`에서
`InterestSubtopicHash`, `InterestTaxonomyHash`, canonical JSON 및 SHA-256 fixture를 확인했다.
fixture 두 파일은 해당 커밋에서 가져왔다. Java 테스트를 이번 작업에서 재실행하지는 않았다.

## 결과

- 로컬 `data/creator-subtopics.json`의 초안 85개를 Python으로 변환했다.
- BE canonical fixture와 UTF-8 바이트 전체 일치: 9,624바이트.
- SHA-256 일치: `7e06e15a711beb6a04599b8a6b81e6c3c1aea2975db40df9bc17cc0157feb0f6`.
- 초안의 서비스 확정이나 시드 적용을 의미하지 않는다.

## 계약

1. 정본 `subtopics` 배열 순서를 유지한다. BE는 이 순서를 `display_order`로 저장한다. 코드로 정렬하지 않는다.
2. 키 순서는 `subtopics`, 각 행은 `code`, `parent`, `name`, `description`이다.
   정본의 `definition`을 `description`으로 매핑한다.
3. 각 필드에서 CRLF/CR을 LF로 통일하고 양끝 ASCII 공백/탭/LF만 제거한 뒤 NFC 정규화한다.
   NBSP, 전각 공백, 내부 공백은 보존한다.
4. `json.dumps(ensure_ascii=False, separators=(",", ":"))`를 사용한다.
5. BOM과 마지막 개행 없는 UTF-8 바이트에 SHA-256을 적용해 소문자 hex를 얻는다.

`version`, `status`, `include`, `exclude`, `formats`, `active`는 해시 대상이 아니다.
따라서 포함/제외 기준만 바뀌면 BE 해시는 바뀌지 않는다. 분류 캐시·결과의 재현성을 위해서는
이 BE 저장 해시와 별도로 분류 기준 전체를 포함한 입력/설정 해시를 관리해야 한다.

## 재실행

Cking-LLM 디렉터리에서:

```powershell
.\.venv\Scripts\python.exe -m src.recommendation.subtopic_taxonomy data/creator-subtopics.json --fixture-dir tests/fixtures/taxonomy
.\.venv\Scripts\python.exe -m pytest tests/test_subtopic_taxonomy_hash.py tests/recommendation/test_taxonomy.py -q
```

확정 목록이 변경되면 기존 초안과 비교가 실패하는 것이 정상이다.
확정본 후보는 별도 JSON 파일로 준비하고 다음 명령으로 출력한다. 원본과 같은
`subtopics` 배열 및 `code`, `parent`, `name`, `definition` 필드를 사용한다.

```powershell
.\.venv\Scripts\python.exe -m src.recommendation.subtopic_taxonomy results/subtopic-approved.json --output-dir results/subtopic-approved-fixture
.\.venv\Scripts\python.exe -m src.recommendation.subtopic_taxonomy results/subtopic-approved.json --fixture-dir results/subtopic-approved-fixture
```

출력은 `subtopic-v1.canonical.json`, `subtopic-v1.sha256.txt` 두 파일이다.
두 파일 모두 BOM·마지막 개행 없이 기록한다. 출력 디렉터리는 새 경로여야 하며,
기존 경로는 덮어쓰지 않는다. 비교는 JSON 바이트와 해시를 각각 검사하고 불일치 시 종료 코드 1을 반환한다.
`--output-dir`와 `--fixture-dir`는 동시에 사용할 수 없다.
행 추가·삭제·순서 변경 및 계약 필드 변경은 해시를 변경한다.

출력 성공은 서비스 승인이나 BE 시드 적용을 의미하지 않는다. 팀의 확정본 검토와 승인 후
BE 담당자에게 두 파일을 전달하고, BE가 제공한 fixture 디렉터리를 `--fixture-dir`로 지정해
다시 비교한다. 승인 전에는 `tests/fixtures/taxonomy/subtopic-v1.*` 초안을 교체하지 않는다.
fixture 파일은 `.gitattributes`의 `tests/fixtures/taxonomy/* -text`로 바이트를 보존한다.

`test_be_draft_fixture_matches_source_bytes_and_hash`는 실제 `data/creator-subtopics.json`과
공유 fixture의 일치를 검사한다. 승인 후 이 원본을 확정 목록으로 변경할 때는 fixture와
해당 테스트의 초안 명칭·기대값도 같은 변경에서 함께 갱신해야 pytest/CI가 통과한다.

생성 전 비어 있지 않은 `subtopics` 배열과 각 행의 필수 문자열 필드를 확인한다.
정규화 후 빈 값 및 중복 `code`는 거부한다. 이는 후보 생성기의 입력 검증이며
BE의 원시 해시 직렬화 규칙을 변경하지 않는다.
입력/파일 오류는 traceback 없이 stderr의 `ERROR:` 메시지와 종료 코드 2로 알린다.
정상 비교는 0, 바이트 또는 해시 불일치는 1이다.

분류 실행의 입력/설정 해시에는 최소한 원본 목록의 `version`, `status`, `include`, `exclude`,
`formats`와 프롬프트·모델 버전 등 실제 분류 설정을 포함하고 결과에 함께 기록해야 한다.
이 도구의 SHA-256은 BE 저장 계약 확인용이며 분류 설정 전체 해시를 대신하지 않는다.
