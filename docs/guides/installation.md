# 설치와 준비

[문서 안내](../README.md) · [현재 결론](../results/recommendation-decision.md)

아래 비용·모델 다운로드 안내는 초기 합성 벤치마크 기준이다. 최근 실험 전체의 청구액을 뜻하지 않는다.

## 1. 사전 준비물

| 항목 | 확인 방법 | 없으면 |
| --- | --- | --- |
| Python 3.11 이상 | `python3 --version` | `brew install python@3.12` (macOS) |
| OpenAI API 키 (결제 등록 완료) | — | [platform.openai.com](https://platform.openai.com/api-keys)에서 발급, Billing에서 결제수단 등록 |
| 디스크 여유 공간 약 9GB | — | `bge-m3`·`KURE-v1`·`Qwen3-Embedding-0.6B`·`bge-reranker-v2-m3` 모델을 로컬에 내려받는 데 필요 |
| (선택) 인터넷 | — | 임베딩·태깅·판정은 실제 OpenAI API를 호출합니다 |

> ⚠️ **비용 안내**: 이 저장소의 기본 파이프라인(embed ~ auto-judge)을 한 번 실행하면 **API 비용이 약 $0.19 (260원 안팎, 약 16~17분)** 발생할 것으로 예상합니다(실제 청구액은 미확인이며, 이 프로젝트에서 여러 차례 재판정하며 쌓인 누적 비용은 약 $0.22입니다). 선택 실험(`dev-sensitivity`, `taste-eval`·`taste-judge`)까지 모두 돌리면 약 $0.8까지 늘고 시간도 수십 분 더 걸립니다. 상세 내역은 [비용·소요 시간](../history/model-selection/index.md#비용소요-시간)을 참고하세요. 큰돈은 아니지만 **본인 API 키에서 실제로 빠져나가는 돈**이니, `.env`에 다른 사람 키를 쓰지 말고 본인 키로 실행하세요.

## 2. 설치

```bash
git clone https://github.com/URECA-Cking/Cking-LLM-Benchmark.git
cd Cking-LLM-Benchmark

python3 -m venv .venv
source .venv/bin/activate        # Windows는 .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

`.env` 파일을 열어 `OPENAI_API_KEY=` 뒤에 본인 키를 붙여넣습니다.

```
OPENAI_API_KEY=sk-...본인_키...
```

설치가 끝났는지 아래 명령으로 확인합니다.

```bash
python3 -m pytest -q
```

```
.................................................................  [100%]
N passed in ...s  # 개수·시간은 변경에 따라 달라집니다
```

**모든 테스트가 통과하면 준비 완료입니다. 테스트 개수는 변경에 따라 달라집니다.** 이 테스트들은 API를 호출하지 않는 순수 로직 검증이라 비용이 들지 않습니다.

> 💡 이후 모든 명령은 `source .venv/bin/activate`로 가상환경을 켠 상태에서 실행한다고 가정합니다. 터미널을 새로 열었다면 저장소 폴더에서 이 명령을 다시 실행하세요.
