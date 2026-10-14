# v5 코드별 고정 슬롯과 계층 슬롯 실험

최신 구현 `grounded-detail-candidates-v5-compact`는 v2/v4의 오류를 다시 분석한 계층 슬롯·공유 스키마 실험이다.
기본값 v2와 기존 모든 프롬프트·배열 스키마는 유지한다. taxonomy와 `detail-limit-v2` 정책을 바꾸지 않는다.

## 원인과 변경

| 관찰한 문제 | 구현한 대응 |
| --- | --- |
| v2가 상위 키워드에서 인접 세부를 확장 | 평면/grouped는 슬롯 description에 정의를 연결; compact는 전체 taxonomy를 프롬프트에 유지하고 무관한 슬롯은 null |
| v4가 짧은 생활·사연·제작·작품 표현을 누락 | 문자 일치와 core를 자격 조건으로 삼지 않고 정의를 충족하는 실제 활동을 인정 |
| v4/v5가 같은 상위를 소재·세부별로 여러 객체로 반환 | 상위/세부 각각 code당 nullable 객체 슬롯 하나 |
| 모델이 topics.parent를 잘못 선택 | 슬롯의 부모는 taxonomy에 고정하고 모델이 별도 parent 값을 작성하지 않음 |
| 잘못된 인용·누락된 상위 | 기존 검증으로 그대로 거부, 수정하거나 부모를 추가하지 않음 |
| JSON의 같은 키가 반복되면 파서가 덮어쓸 위험 | object_pairs_hook로 중복 키를 명시적 오류로 거부 |

먼저 17개 상위 슬롯 각각에 null 또는 `evidence/core_evidence/repetition_evidence/topics` 객체를 출력한다.
선택된 상위의 topics 안에서만 소속 세부 코드의 고정 슬롯들을 판단한다. 세부 슬롯의 값은
null 또는 `evidence/core_evidence/repetition_evidence` 객체다. 선택되지 않은 상위는 null로 끝나므로
그 아래의 무관한 세부 슬롯을 모두 출력할 필요가 없다. 상위만 명확하면 topics는 전부 null로 둘 수 있다.
선택된 값은 기존 배열 후보 형식으로 일대일 변환한다. 이 변환은 모든 선택된 후보와 인용을 보존한다.
같은 코드를 삭제하거나 근거를 고치지 않는다. `_rank`와 기존 상위/세부·제한 정책 검증을 통과해야 revision을 저장한다.

원본 응답은 attempts에, 파싱한 원본 객체는 revision의 `provider_candidates`에 저장한다.
revision의 `candidates`는 기존 검증기로 전달한 배열이다. 계층 슬롯은 `provider_schema_version=code-map-grouped-v1`,
초기 평면 슬롯은 `code-map-v1`로 구분한다.
실제 `prompt_version/prompt_hash/output_schema_hash`를 보존한다. 감사는 원본 JSON→원본 객체→배열→결과를 대조한다.
재개 시 배열/고정 슬롯 스키마가 섞이면 요청 전에 거부한다.

코드별 스키마는 필수 키·nullable 값·`additionalProperties: false`를 사용한다.
이 설계는 [OpenAI Structured Outputs 지원 범위](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)를 확인해 구현했다.
이 스키마는 의미상 정답이나 정확한 인용을 보증하지 않으므로 로컬 검증과 실제 평가가 필요하다.

## 실험 격리와 평가

이 버전은 출력 스키마까지 바뀌므로 **프롬프트만 개선한 실험이라고 해석하지 않는다**.
기본 `human_eval evaluate`와 `detail_tagging_experiment`는 스키마가 다른 실행의 비교를 거부한다.
명시적인 `detail_tagging_experiment --schema-experiment`만 스키마 차이를 허용하고,
`comparison_type=schema_and_prompt_experiment_not_prompt_only`와 스키마 해시들을 기록한다.
입력·taxonomy·정책·모델·온도·SDK 재시도·timeout·시도 상한·기준 출처/해시는 여전히 같아야 한다.

API 실행은 원문 전송·추가 호출 예산의 승인이 필요하다. 아래는 승인 후 실행하는 명령 형식이다.

```powershell
.venv\Scripts\python.exe -m src.detail_tagging_batch --inputs results/issue66-development/development/inputs.private.json --output results/issue68-approved-v5-compact --prompt-version grounded-detail-candidates-v5-compact --ai-proxy-criteria results/issue66-ai-proxy-review-v2/criteria.ai-proxy.private.json --max-attempts 3 --tag-llm
```

기존 v2/v4·실패한 v5·r2/r3·평면 고정 슬롯은 저장된 summary와 attempts/revision을 재사용한다.
과거 결과의 부족한 provenance를 만들어 채우거나 이전 버전을 재호출하지 않는다.
비공개 비교 명령은 다음 형식이며 `--summary`는 명시적으로 고른 기존 결과 경로를 반복한다.

```powershell
.venv\Scripts\python.exe -m src.detail_tagging_experiment --criteria results/issue66-ai-proxy-review-v2/criteria.ai-proxy.private.json --summary <기존-v4-summary> --summary <새-code-map-summary> --output results/issue68-code-map-comparison/evaluation.private.json --ai-proxy --schema-experiment
```

AI 기준은 원본 그대로 유지하며 사람 정답으로 취급하지 않는다. 개발 표본과 반복 수정으로 생긴
선택 편향이 있으므로 독립 사람 검증·홀드아웃·팀 채택을 대체하지 않는다. 고정 슬롯의 null은
해당 후보 미선택을 뜻하며 사람 검수의 hold와 다르다. 배열 버전의 v4/v5 사람 홀드아웃 도구는
`--v4-v5`, `--v4-v5-r2`, `--v4-v5-r3`로 명시적으로 등록한다. 고정 슬롯 실험은 이 채택 경로에
자동 편입하지 않는다. 새 schema에 대한 사람 기준·독립 검증 프로토콜은 별도 검토가 필요하다.

## 보존한 수정 이력

| 이름 | 변경 | 상태 |
| --- | --- | --- |
| grounded-detail-candidates-v5 | 최초 지침 추가 | 기존 실패 결과와 해시 보존 |
| grounded-detail-candidates-v5-r2 | 독립 판단 순서·12개 완전한 합성 예시 | 50개 실제 실행, 과잉 억제 개선 미확인 |
| grounded-detail-candidates-v5-r3 | v4 바이트를 기반으로 경계·7개 합성 예시 추가 | 50개 실제 실행, 누락은 줄었으나 과잉 증가 |
| grounded-detail-candidates-v5-code-map | 코드별 평면 고정 슬롯과 엄격한 변환·감사 | 중복 오류는 없었으나 미분류 24개로 누락 악화 |
| grounded-detail-candidates-v5-grouped | 선택한 상위 아래에서만 세부 고정 슬롯 판단 | 계층 출력 구조 실험 |
| grounded-detail-candidates-v5-compact | 계층 슬롯의 반복 정의를 $defs/$ref로 공유 | grouped와 같은 프롬프트, description 제거·참조 공유를 함께 적용 |

과거 이름의 프롬프트 바이트를 덮어쓰지 않는다. 각 수정본의 실패도 사용량·원본 응답과 함께 보존한다.

compact는 계층 슬롯에서 반복되던 근거 객체 정의를 공유한다. 검증 형태는 참조를 펼치고 description을 제외하면 기존 계층 슬롯과 같다.
스키마 안에서 반복된 taxonomy 설명도 제거하지만 프롬프트에는 전체 taxonomy를 그대로 유지한다.
스키마 문자 수는 55,958→description 제거 36,381→공유 적용 14,866이다. 총 감소의 47.6%는
description 제거다. 상위만 선택하는 지침 등도 빠지므로 모델이 받는 지침이 달라진다. grouped와 prompt_hash는 같다.
이는 모델의 정확성 향상을 보장하지 않는다. 실제 입력 토큰·첫 시도·최종 성공·누락·과잉으로 검증해야 한다.
비교 보고서의 쌍별 changed_factors에 실제로 달라진 프롬프트/스키마 해시를 기록한다.

실제 개발 표본 결과는 [재구현 보고서](../results/detail-tagging-v5-redesign.md)를 참고한다.
최종 compact는 첫 시도·최종 49/50, 중복 오류 0회였다. v4와 공통 성공한 48개에서
세부 누락 10→10·과잉 4→2·F1 0.816→0.838이었다. 상위 지표는 소폭 악화됐고
독립 사람 검증은 미수행이므로 기본값 승격이나 모든 품질 지표 개선을 주장하지 않는다.

최종 compact의 독립 사람 홀드아웃 등록과 버전별 스키마·채택 결정 정비는 [#70](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/70) 후속 범위다.
