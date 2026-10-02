# 단일 크리에이터 서비스 추천 후보 생성

[문서 안내](../README.md) · [현재 방식 결정](../results/recommendation-decision.md) · [예시 데이터 점검](../results/service-recommendation-check.md)

## 적용 범위

`src.recommendation`은 한 Cking `creatorId`를 기준으로 Cking-BE가 저장할 유사 후보를 만드는 오프라인 생성기다.
사용자 추천 조회 중 실행하는 코드가 아니다. 후보 생성이 완료된 뒤 BE가 묶음 전체를 검증·저장하고, 조회 API는 저장된 결과만 읽는다.

- 소개 15자 이상: BGE-M3 코사인 유사도에 GPT 상위 분야 일치 가산점 0.2를 더하는 M4
- 소개 1~14자: BGE-M3 임베딩만 쓰는 M2
- 빈 소개: 모델이 내용을 추측하지 않고 후보를 만들지 않음
- 정상 소개가 `UNCLASSIFIED`: 유효한 분류 결과로 보고 M2로 폴백
- 모델 API 예외·형식 오류: 부분 후보를 반환하지 않고 생성 작업 실패로 전파. BE는 기존 정상 묶음을 유지하는 운영 정책을 적용

회원별 관심사·여러 즐겨찾기 개인화, 추천 UI, 인기순 보완, 온라인 전환 검증은 이 생성기의 범위가 아니다.

## 입력 계약

```json
{
  "creatorId": 10,
  "introduction": "초보자를 위한 홈트레이닝을 소개합니다.",
  "topN": 5,
  "candidateCreators": [
    {
      "creatorId": 20,
      "introduction": "집에서 따라 하는 근력 운동과 스트레칭을 소개합니다."
    },
    {
      "creatorId": 30,
      "introduction": "간단한 한 끼 요리와 자취 레시피를 소개합니다."
    }
  ]
}
```

`creatorId`는 양의 JSON 정수다. 같은 후보 ID와 같은 소개가 반복되면 하나만 사용하고, 같은 ID에 서로 다른 소개가 오면 입력 오류로 중단한다.
seed 자신과 빈 소개 후보는 후보 풀에서 제외한다. `topN`은 선택이며 없으면 CLI의 `--top-n` 기본값 5를 쓴다.

## 실행

`.env`에 본인의 `DEEPINFRA_API_KEY`와 `OPENAI_API_KEY`를 설정한다. 캐시에 없는 입력은 실제 외부 API 호출과 비용이 발생한다.

```bash
python3 -m src.recommendation.cli \
  --input inputs/recommendation-request.json \
  --output results/recommendation/creator-10.json
```

주요 설정은 CLI에서 바꿀 수 있다.

```bash
python3 -m src.recommendation.cli \
  --input inputs/recommendation-request.json \
  --output results/recommendation/creator-10.json \
  --cache results/recommendation/model-cache.json \
  --short-introduction-chars 15 \
  --top-n 5 \
  --m4-bonus 0.2
```

상위 분야는 기본 `data/categories.csv`의 `code`, `name`, `description`을 사용한다. 다른 파일은 `--categories`로 지정한다.

## BE 전달 계약

```json
{
  "creatorId": 10,
  "candidates": [
    {
      "creatorId": 10,
      "similarCreatorId": 20,
      "score": 1.08341234,
      "rank": 1,
      "method": "M4",
      "modelVersion": "BAAI/bge-m3@deepinfra-v1+gpt-5.4-nano-2026-03-17@creator-category-v1",
      "inputHash": "64자리 SHA-256"
    }
  ]
}
```

- 자기 자신을 제외하고 같은 `similarCreatorId`를 한 번만 보낸다.
- 점수는 소수점 8자리로 고정한 뒤 내림차순, 동점이면 `similarCreatorId` 오름차순으로 정렬한다.
- `rank`는 1부터 연속이다. 후보가 부족하면 요청 개수보다 적게 보내며 빈 배열도 정상이다.
- 한 묶음의 후보는 같은 `method`, `modelVersion`, `inputHash`를 갖는다.
- `inputHash`는 seed 소개, 정렬된 후보 ID·소개, Top-N, 방식, 모델·프롬프트·분류체계·가산점 설정을 정규 JSON으로 만든 SHA-256이다.

## 캐시와 재실행

기본 캐시는 `results/recommendation/model-cache.json`이며 Git에 포함하지 않는다.

- 임베딩: 정규화한 소개의 SHA-256과 임베딩 모델 버전으로 조회
- GPT 태그: 소개 SHA-256, 태그 모델 버전, 프롬프트 버전, 프롬프트 본문 SHA-256, 분류체계 버전으로 조회
- 소개·모델·프롬프트 본문·분류체계 중 하나라도 바뀌면 해당 캐시를 재사용하지 않는다.
- 성공한 모델 결과만 저장한다. 작업이 중간에 실패해도 후보 묶음은 출력하지 않고, 다음 실행은 이미 성공한 개별 캐시부터 이어 쓴다.
- 같은 입력과 설정으로 재실행하면 외부 API를 다시 호출하지 않고 같은 BE payload를 만든다.

임베딩은 기본 96개씩 API에 보내고, 태그 결과는 기본 100개마다 캐시에 반영한다. 캐시 파일 갱신은 임시 파일을 같은 디렉터리에 완전히 쓴 뒤 원자적으로 교체한다.
여러 프로세스가 같은 캐시 파일을 동시에 쓰도록 설계한 것은 아니므로 생성 작업은 캐시 파일당 하나만 실행한다.

## 검증

단위 테스트는 가짜 임베딩·태그 응답만 사용하므로 유료 API를 호출하지 않는다.

```bash
python3 -m pytest -q tests/recommendation
```

검증 범위는 M4/M2/빈 소개 분기, `UNCLASSIFIED` 폴백, 자기 자신 제외, 중복 제거, 동점 정렬, 후보 부족, BE 필드, 동일 실행 재현, 입력·모델·프롬프트 캐시 무효화, 외부 API 실패 전파다.
