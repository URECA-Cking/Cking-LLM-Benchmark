# 실제 크리에이터 데이터로 M2·M3·M4 비교

합성 100명 결과([`embedding-method-selection.md`](embedding-method-selection.md))가 실제 소개글에서도 유지되는지 확인하는 실행 도구(`src/real_eval.py`, 이슈 [#28](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/28))의 사용법과 사전 판정 기준이다. **이 문서는 도구와 기준을 기록하며, 실행 결과는 아직 없다.** 결과가 나오면 결과 문서에 반영한다(이슈 [#14](https://github.com/URECA-Cking/Cking-LLM-Benchmark/issues/14)).

## 왜 별도 도구인가
기존 `pipeline`은 합성 100명에 묶여 있다(데이터 해시 고정, 정확히 100명, 특정 ID 기반 대표 사례). 실데이터는 규모·분야 수·라벨 체계가 달라서 `similarity`·`tagging`·`metrics`·`judge`·`clients`는 재사용하고 실행 흐름만 새로 두었다. 기존 결과와 테스트는 바뀌지 않는다.

## 데이터와 취급 원칙
- 채널 소개글 원문과 정답 라벨은 **저장소에 커밋하지 않는다.** YouTube API 약관(저장 기간·재배포) 확인 전까지 외부 폴더에서만 읽고, 산출물은 `results/real/`(git 제외)에 쓴다.
- 외부 폴더는 `--data-dir` 또는 환경변수 `REAL_DATA_DIR`로 지정한다.

```
<데이터 폴더>/
├─ raw/pool.csv                 channel_id, name, bio, source_query   추천 후보 풀
└─ labels/
   ├─ gold_v2.csv               번호, gold_v2, use_gold, ...            정답 분야 (use_gold=1만 사용)
   ├─ reference_hidden.csv      번호, channel_id, ...                   번호 ↔ 채널 ID
   └─ categories_v2.csv         code, name, description                분야 목록(설명문은 zero-shot·LLM 태깅에 쓴다)
```
- 모델에 넣는 텍스트는 소개글이고, 소개글이 비어 있으면 채널명이다(서비스가 가진 정보만 쓴다. 구독자 수·영상 제목 등은 넣지 않는다).

## 실행 순서
```bash
export REAL_DATA_DIR=<데이터 폴더>
python3 -m src.real_eval embed            # 후보 풀·분야 설명문 임베딩 (로컬 bge-m3 기본, --api로 DeepInfra). 4,743명 약 2분
python3 -m src.real_eval tag-llm          # 풀 전체 LLM 태깅 (분야 설명 포함 프롬프트, 동시 8개, 끊겨도 이어서)
python3 -m src.real_eval select-params    # dev/test 분할(분야별 층화)·tau·bonus 선택·쿼리 추출·test 태깅 정확도
python3 -m src.real_eval candidates       # 쿼리별 M2·M3·M4 상위 후보
python3 -m src.real_eval judge-sheet      # 방식별 상위 5의 합집합 = 판정할 쌍
python3 -m src.real_eval auto-judge       # LLM이 "추천에 넣을 만한가"(0/1) 판정
python3 -m src.real_eval human-sheet      # 무작위 150쌍 사람 채점 시트 (results/real/human_sheet.csv)
python3 -m src.real_eval human-agree      # 사람이 채운 시트와 LLM 판정의 일치율 (기준 80%)
python3 -m src.real_eval score            # 정밀도@5, M4 − M3 짝 차이, 사전 기준 판정
```

| 방식 | 유사도 |
| --- | --- |
| M2 | 소개글 벡터 코사인 |
| M3 | 코사인 + (zero-shot 태그가 하나라도 겹치면 +bonus_m3). 태그는 분야 설명문과의 코사인이 tau 이상이면 부여 |
| M4 | 코사인 + (LLM 태그가 하나라도 겹치면 +bonus_m4) |

- tau·bonus는 **dev 채널만**으로 고른다(정답 분야 기준). test 채널의 벡터·라벨이 바뀌어도 선택값이 같은지 테스트로 확인한다.
- 쿼리는 dev를 뺀 풀에서 무작위로 뽑는다: 일반(소개글 15자 이상) 300개, 짧은 소개글(15자 미만) 100개. 후보는 풀 전체(쿼리 자신 제외)에서 고른다.
- LLM 태깅 프롬프트에도 분야 설명문을 넣어 M3(설명문 임베딩)와 정보량을 맞췄다.

## 사전 판정 기준 (결과를 보기 전에 확정)
- **M4가 M3보다 정밀도@5가 +0.05 이상 높고, 쿼리별 짝 차이의 95% 신뢰구간(부트스트랩 2,000회)이 0을 넘지 않으면 M4를 채택하고, 아니면 M3를 채택한다.**
- 결정은 **일반 채널** 기준이다. 짧은 소개글 결과는 같은 규칙을 적용해 참고로만 함께 적는다.
- 정밀도@5는 쿼리의 상위 후보 5개 중 "추천에 넣을 만하다"(1)로 판정된 비율이다.
- 사람 150쌍과 LLM 판정의 일치율이 **80% 미만이면** 판정 기준을 고치고 사람 채점을 늘린 뒤 다시 판정한다. `score`는 이 확인이 없거나 미달이면 결과에 경고를 붙인다.

## 결정을 가르는 쌍의 사람 확인과 보정 (결과를 본 뒤에 추가한 분석)
사전 기준(위)은 그대로이고 바꾸지 않았다. 다만 첫 실행에서 사람(대체) 채점용 무작위 150쌍의 64%가 M2·M3·M4가 모두 뽑은 쌍이라 M4 − M3와 무관했고, 실제로 둘을 가르는 쌍(M4만·M3만 뽑은 쌍)은 전체의 30%뿐이었다. 그래서 일치율 확인을 가르는 쌍에서 다시 하는 절차를 결과를 본 뒤에 추가했다.

```bash
python3 -m src.real_eval human-sheet --targeted              # M4만·M3만 뽑은 쌍 50개씩, 섞어서 human_sheet_targeted.csv
python3 -m src.real_eval human-agree --targeted --source human   # 채운 시트: 일치율, 방식 조합별 LLM 판정 보정값(offset ± 표준오차)
python3 -m src.real_eval score --calibrated                  # 채점한 쌍은 그 값, 나머지는 조합별 offset만큼 보정한 정밀도@5와 M4 − M3
```
- 보정값은 (채점 긍정률 − LLM 긍정률)이다. 표준오차만큼 틀릴 때의 M4 − M3 범위도 함께 나온다.
- 이 보정은 사전 기준의 공식 결정을 대체하지 않는다. 사전 기준의 "일치율 80% 미만이면 판정 기준을 고치고 사람 채점을 늘린 뒤 다시 판정한다"를 실행하는 방법의 하나로 쓴다.
- `--source`로 채점 출처를 남긴다. 사람이 아니면(`claude`) `score`가 독립성이 약하다고 경고한다.

## 비용 (추정, 실제 청구액은 미확인)
- `embed`: 로컬이면 0원, `--api`면 1센트 미만.
- `tag-llm`: 4,743호출. 분야 설명문이 들어가 프롬프트가 기존보다 길어서 대략 $1~2.
- `auto-judge`: 쿼리 400개 x 후보 합집합(약 7쌍) 약 2,800쌍, 대략 $0.7~1.5.
- 캐시·이어하기가 있어서 중간에 끊겨도 처음부터 다시 내지 않는다. 이 값은 이전 실행 단가(판정 681쌍 약 $0.17)와 프롬프트 길이로 어림한 것이다.

## 해석할 때 주의
- 정답 라벨은 Claude가 붙이고 사람이 애매한 것과 무작위 표본만 검수했다. 분야 정확도는 "이 라벨 기준"이다.
- 분야 목록(17개)은 수집한 채널을 본 뒤에 확장했다. 분야 설명문이 데이터를 본 뒤에 쓰인 한계가 있다.
- 후보 풀은 검색어와 인기 동영상으로 모은 표본이라 검색어가 소개글에 그대로 들어 있는 채널이 많다.
- M4 태깅(gpt-5.4-nano)과 판정(gpt-5.4-mini)이 같은 계열이라 판정이 완전히 독립적이지 않다. 사람 150쌍 확인이 이를 점검한다.
- 이 결과는 소개글 텍스트 유사도의 경향이다. 사용자가 실제로 좋아하는지는 서비스 지표(클릭·응모 전환)로만 확인할 수 있다.
