"""E2 관련도 판정으로 설정 간 차이를 쿼리별 짝 차이로 계산한다 (리뷰 P2: 새 통계 결과를 저장소만으로 재현).

설정별 평균 관련도의 신뢰구간이 겹친다는 사실만으로는 두 설정의 차이가 없다고 할 수 없다. 그래서 같은 쿼리에 대한
두 설정의 관련도 차이를 쿼리별로 구하고, 쿼리를 다시 뽑는(부트스트랩) 방식으로 차이의 평균에 대한 신뢰구간을 계산한다.

입력은 커밋된 `data/e2_judgments.json`(쿼리 30명의 설정별 상위 5 후보와 판정 점수)이고 해시로 고정한다. 이 파일은
`results/candidates.json`과 `results/judge_sheet.csv`(둘 다 커밋하지 않음)에서 `paired-diff --export`로 만든다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config import E2_JUDGMENTS_JSON, E2_JUDGMENTS_JSON_SHA256, JUDGE_TOP_K, OPENAI_JUDGE_MODEL, RESULTS_DIR
from src.data import _sha256, load_creators, query_creators
from src.metrics import bootstrap_ci, mean_relevance_at_k, paired_win_counts

EMBEDDINGS = ("text-embedding-3-small", "bge-m3", "kure-v1", "qwen3-embedding-0.6b")
# 결과 문서 "방식 간 차이"의 두 표: M4 − M3(임베딩별)와 M4_bge-m3 − 상위권 설정
M4_MINUS_M3 = [(f"M4_{key}", f"M3_{key}") for key in EMBEDDINGS]
TOP_SETTINGS = ["M3_bge-m3", "M2_bge-m3", "M4_qwen3-embedding-0.6b", "R2_kure-v1", "M4_kure-v1", "M5_bge-m3"]
M4_BGE_MINUS_TOP = [("M4_bge-m3", other) for other in TOP_SETTINGS]


def export_artifact(results_dir: Path = RESULTS_DIR) -> dict:
    """results/의 판정 시트와 후보 목록에서 짝 차이 계산에 필요한 부분만 뽑아 입력 아티팩트를 만든다."""
    query_ids = [c.id for c in query_creators(load_creators())]
    judgments = []
    with (results_dir / "judge_sheet.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["score"].strip():
                raise ValueError(f"판정이 비어 있는 행이 있습니다: {row['pair_id']}")
            judgments.append([row["query_id"], row["candidate_id"], int(row["score"])])
    candidates = json.loads((results_dir / "candidates.json").read_text(encoding="utf-8"))
    return {
        "description": "E2 관련도 판정(LLM 자동 판정)과 쿼리별 설정별 상위 후보. paired-diff의 입력이며 paired-diff --export로 만든다.",
        "judge_model": OPENAI_JUDGE_MODEL,
        "top_k": JUDGE_TOP_K,
        "query_ids": query_ids,
        "judgments": sorted(judgments),
        "top_ids_by_method": {
            method: {qid: [item[0] for item in by_creator[qid][:JUDGE_TOP_K]] for qid in query_ids}
            for method, by_creator in sorted(candidates.items())
        },
    }


def write_artifact(artifact: dict, path: Path = E2_JUDGMENTS_JSON) -> str:
    """아티팩트를 항상 같은 형식(키 정렬, LF)으로 저장하고 sha256을 돌려준다."""
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return _sha256(path)


def load_artifact(path: Path = E2_JUDGMENTS_JSON, expected_sha256: str | None = E2_JUDGMENTS_JSON_SHA256) -> dict:
    """고정된 해시와 같을 때만 입력 아티팩트를 읽는다. expected_sha256이 None이면 검사하지 않는다(테스트용)."""
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError(
            f"{path.name}이 고정된 내용과 다릅니다. 의도한 변경이면 src/config.py의 E2_JUDGMENTS_JSON_SHA256과 docs를 함께 갱신하세요."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def per_query_relevance(artifact: dict) -> dict[str, dict[str, float]]:
    """설정별·쿼리별 관련도@5(상위 5 후보의 판정 점수 평균)를 계산한다."""
    judged = {(q, c): score for q, c, score in artifact["judgments"]}
    return {
        method: {qid: mean_relevance_at_k(ids, qid, judged) for qid, ids in by_query.items()}
        for method, by_query in artifact["top_ids_by_method"].items()
    }


def paired_difference(rel_a: dict[str, float], rel_b: dict[str, float], query_ids: list[str]) -> dict:
    """같은 쿼리에 대한 A − B의 쿼리별 차이를 구해 평균, 부트스트랩 95% 신뢰구간, 승·패·동점 수를 돌려준다.

    부트스트랩은 값 목록의 순서가 다르면 구간 끝값이 조금 달라지므로, 쿼리 순서(query_ids)를 인자로 명시해 고정한다.
    """
    diffs = [rel_a[qid] - rel_b[qid] for qid in query_ids]
    lower, upper = bootstrap_ci(diffs)
    wins, losses, ties = paired_win_counts({q: rel_a[q] for q in query_ids}, {q: rel_b[q] for q in query_ids})
    return {"mean": sum(diffs) / len(diffs), "ci_lower": lower, "ci_upper": upper, "wins": wins, "losses": losses, "ties": ties}


def compute_all(artifact: dict) -> dict[str, dict]:
    """결과 문서 "방식 간 차이"의 두 표에 나오는 모든 비교를 계산한다. 키는 "A - B"다."""
    rel = per_query_relevance(artifact)
    return {f"{a} - {b}": paired_difference(rel[a], rel[b], artifact["query_ids"]) for a, b in [*M4_MINUS_M3, *M4_BGE_MINUS_TOP]}


def format_row(result: dict) -> str:
    """결과를 문서 표와 같은 표기(부호, 유니코드 마이너스)로 만든다."""
    text = f"{result['mean']:+.3f} | [{result['ci_lower']:+.3f}, {result['ci_upper']:+.3f}] | {result['wins']} / {result['losses']} / {result['ties']}"
    return text.replace("-", "\u2212")


def cmd_paired_diff(args: argparse.Namespace) -> None:
    """`--export`면 results/에서 입력 아티팩트를 만들고, 아니면 아티팩트로 짝 차이를 계산해 results/paired_diff.json에 남긴다."""
    if getattr(args, "export", False):
        digest = write_artifact(export_artifact())
        print(f"[paired-diff] {E2_JUDGMENTS_JSON} 저장. sha256={digest}")
        print("  내용이 바뀌었다면 src/config.py의 E2_JUDGMENTS_JSON_SHA256과 결과 문서를 함께 갱신하세요.")
        return
    artifact = load_artifact()
    results = compute_all(artifact)
    with (RESULTS_DIR / "paired_diff.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[paired-diff] 쿼리 {len(artifact['query_ids'])}명, 쿼리를 다시 뽑는 부트스트랩 95% 신뢰구간 (0을 포함하면 차이를 확인하지 못한 것)")
    for name, result in results.items():
        print(f"  {name}: {format_row(result)}  {'0 포함' if result['ci_lower'] <= 0 <= result['ci_upper'] else '0 제외'}")
