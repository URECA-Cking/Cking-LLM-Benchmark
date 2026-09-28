"""실험 단계를 순서대로 실행하는 CLI다. `python3 -m src.pipeline <단계>`로 호출한다.

단계: embed -> tag-llm -> select-params -> candidates -> judge-sheet -> report -> score-judgments
report까지는 사람 판정 없이 자동으로 끝난다 (E1, E3, E4). judge-sheet 이후 score-judgments 전에
`auto-judge`(LLM 자동 판정, 보조 수단) 또는 `src.judge_cli`(사람 판정)로 judge_sheet.csv의
score 열을 채운다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.clients import BgeEmbeddingClient, OpenAIEmbeddingClient, OpenAIJudge, OpenAITagger
from src.config import (
    CACHE_DIR,
    JUDGE_SHUFFLE_SEED,
    JUDGE_TOP_K,
    LLM_CONSISTENCY_RUNS,
    LLM_TAG_MAX,
    OPENAI_JUDGE_MODEL,
    OPENAI_JUDGE_MODEL_PRICE,
    OPENAI_LLM_MODEL_CANDIDATES,
    RESULTS_DIR,
    TOP_N_STORED,
)
from src.data import Creator, dev_creators, load_categories, load_creators, query_creators
from src.judge import build_judge_pairs, shuffle_rows, write_judge_sheet, write_provenance
from src.similarity import cosine_matrix, cosine_with_tag_bonus, jaccard_matrix, select_bonus, top_n
from src.spot_check import agreement_stats, select_disagreement_pairs
from src.tagging import (
    confusion_pairs,
    consistency_rate,
    llm_hit_rate,
    llm_mean_f1,
    pick_llm_model,
    rank_all,
    select_tau,
    top1_accuracy,
    top3_inclusion_rate,
    unclassified_rate,
)

EMBEDDING_MODEL_KEYS = ("text-embedding-3-small", "bge-m3")
BONUS_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _embedding_client(key: str):
    """설정 키로 임베딩 클라이언트를 만든다."""
    if key == "text-embedding-3-small":
        return OpenAIEmbeddingClient()
    if key == "bge-m3":
        return BgeEmbeddingClient()
    raise ValueError(f"알 수 없는 임베딩 모델 키: {key}")


def _save_vectors(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    """id 순서와 벡터를 함께 저장해, 나중에 순서가 어긋나지 않게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), vectors=vectors)


def _load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    """저장된 id 순서와 벡터를 함께 불러온다."""
    data = np.load(path)
    return list(data["ids"]), data["vectors"]


def cmd_embed(_: argparse.Namespace) -> None:
    """카테고리 설명문과 크리에이터 100명의 입력 텍스트를 두 임베딩 모델로 인코딩한다.

    카테고리 이름 자체는 크리에이터 입력 텍스트에 없으므로, zero-shot 태깅이 자기
    분야를 되맞히는 순환이 생기지 않는다. 소요 시간·벡터 크기·비용은 E4(운영 지표)로
    `results/e4_embedding.json`에 남긴다.
    """
    import time

    from src.config import OPENAI_EMBEDDING_PRICE_PER_1M

    categories = load_categories()
    creators = load_creators()
    creator_ids = [c.id for c in creators]
    creator_texts = [c.input_text() for c in creators]
    category_codes = [c.code for c in categories]
    category_texts = [c.description for c in categories]

    e4_report: dict[str, dict] = {}
    for key in EMBEDDING_MODEL_KEYS:
        client = _embedding_client(key)
        started = time.perf_counter()
        creator_vectors = client.embed(creator_texts)
        creator_tokens = getattr(client, "last_input_tokens", None)
        category_vectors = client.embed(category_texts)
        category_tokens = getattr(client, "last_input_tokens", None)
        elapsed_seconds = time.perf_counter() - started

        _save_vectors(CACHE_DIR / f"creators_{key}.npz", creator_ids, creator_vectors)
        _save_vectors(CACHE_DIR / f"categories_{key}.npz", category_codes, category_vectors)

        total_tokens = None if creator_tokens is None or category_tokens is None else creator_tokens + category_tokens
        cost_usd = None if total_tokens is None else total_tokens / 1_000_000 * OPENAI_EMBEDDING_PRICE_PER_1M
        e4_report[key] = {
            "elapsed_seconds": elapsed_seconds,
            "dim": int(creator_vectors.shape[1]),
            "bytes_per_creator": int(creator_vectors.dtype.itemsize * creator_vectors.shape[1]),
            "total_input_tokens": total_tokens,
            "estimated_cost_usd": cost_usd,
        }
        print(
            f"[embed] {key}: 크리에이터 {len(creator_ids)}명, 카테고리 {len(category_codes)}개, "
            f"dim={creator_vectors.shape[1]}, {elapsed_seconds:.1f}초"
        )

    with (RESULTS_DIR / "e4_embedding.json").open("w", encoding="utf-8") as f:
        json.dump(e4_report, f, ensure_ascii=False, indent=2)


def cmd_tag_llm(_: argparse.Namespace) -> None:
    """LLM 태깅 후보 모델들을 크리에이터 전원에게 2회씩 돌리고, dev 정확도로 하나를 고른다.

    두 번 실행하는 것은 비결정성(consistency)을 재기 위함이며, 승자 모델의 태그는
    이후 M4(임베딩 + LLM 태그 보정)의 입력으로 전원에게 쓰인다.
    """
    categories = load_categories()
    creators = load_creators()
    category_codes = [c.code for c in categories]
    dev_ids = {c.id for c in dev_creators(creators)}
    gold_by_id = {c.id: frozenset(c.gold) for c in creators}

    dev_metrics: dict[str, dict[str, float]] = {}
    usage_summary: dict[str, dict[str, int]] = {}
    for model_name in OPENAI_LLM_MODEL_CANDIDATES:
        tagger = OpenAITagger(model=model_name, category_codes=category_codes)
        runs: list[dict[str, list[str]]] = []
        total_input_tokens = total_output_tokens = 0
        for run_index in range(LLM_CONSISTENCY_RUNS):
            tags_by_id: dict[str, list[str]] = {}
            for creator in creators:
                result = tagger.tag(creator.input_text())
                tags_by_id[creator.id] = list(result.tags)
                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
            runs.append(tags_by_id)
            print(f"[tag-llm] {model_name} run {run_index + 1}/{LLM_CONSISTENCY_RUNS} 완료")

        safe_name = model_name.replace("/", "_")
        for run_index, tags_by_id in enumerate(runs):
            with (CACHE_DIR / f"llm_tags_{safe_name}_run{run_index}.json").open("w", encoding="utf-8") as f:
                json.dump(tags_by_id, f, ensure_ascii=False, indent=2)

        first_run = {cid: frozenset(tags) for cid, tags in runs[0].items()}
        dev_tags = {cid: tags for cid, tags in first_run.items() if cid in dev_ids}
        dev_gold = {cid: gold_by_id[cid] for cid in dev_tags}
        dev_metrics[model_name] = {
            "f1": llm_mean_f1(dev_tags, dev_gold),
            "hit_rate": llm_hit_rate(dev_tags, dev_gold),
            "unclassified_rate": unclassified_rate(list(dev_tags.values())),
        }
        second_run = {cid: frozenset(tags) for cid, tags in runs[1].items()}
        dev_metrics[model_name]["consistency"] = consistency_rate(first_run, second_run)
        usage_summary[model_name] = {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}

    selected = pick_llm_model(dev_metrics)
    selection = {"selected_model": selected, "dev_metrics": dev_metrics, "token_usage": usage_summary}
    with (RESULTS_DIR / "llm_model_selection.json").open("w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)
    print(f"[tag-llm] dev 지표로 선택된 모델: {selected}")
    print(json.dumps(dev_metrics, ensure_ascii=False, indent=2))


def _zero_shot_tags_for_key(key: str, creators: list[Creator], categories, tau_candidates: list[float] | None = None):
    """저장된 임베딩으로 zero-shot 순위를 매기고, dev F1로 tau를 골라 태그 집합을 만든다."""
    creator_ids, creator_vectors = _load_vectors(CACHE_DIR / f"creators_{key}.npz")
    _, category_vectors = _load_vectors(CACHE_DIR / f"categories_{key}.npz")
    category_codes = [c.code for c in categories]
    ranked_list = rank_all(creator_vectors, category_vectors, category_codes)
    ranked_by_id = dict(zip(creator_ids, ranked_list))

    dev_ids = {c.id for c in dev_creators(creators)}
    gold_by_id = {c.id: frozenset(c.gold) for c in creators}
    dev_ranked = {cid: r for cid, r in ranked_by_id.items() if cid in dev_ids}
    dev_gold = {cid: gold_by_id[cid] for cid in dev_ranked}
    tau = select_tau(dev_ranked, dev_gold, LLM_TAG_MAX, candidate_taus=tau_candidates)

    tags_by_id = {cid: ranked.assigned(tau, LLM_TAG_MAX) for cid, ranked in ranked_by_id.items()}
    return creator_ids, ranked_by_id, tags_by_id, tau


def cmd_select_params(_: argparse.Namespace) -> None:
    """zero-shot tau와, 세 태그 소스(zero-shot, LLM, 입력분야) 각각의 bonus를 dev로 고정한다."""
    categories = load_categories()
    creators = load_creators()
    gold_by_id = {c.id: frozenset(c.gold) for c in creators}
    declared_by_id = {c.id: frozenset(c.declared) for c in creators}
    dev_ids_ordered = [c.id for c in dev_creators(creators)]

    with (RESULTS_DIR / "llm_model_selection.json").open(encoding="utf-8") as f:
        llm_selection = json.load(f)
    llm_model = llm_selection["selected_model"]
    safe_name = llm_model.replace("/", "_")
    with (CACHE_DIR / f"llm_tags_{safe_name}_run0.json").open(encoding="utf-8") as f:
        llm_tags_raw = json.load(f)
    llm_tags_by_id = {cid: frozenset(tags) for cid, tags in llm_tags_raw.items()}

    params: dict[str, dict] = {"llm_model": llm_model, "per_embedding": {}}
    for key in EMBEDDING_MODEL_KEYS:
        ids, _, zero_shot_tags, tau = _zero_shot_tags_for_key(key, creators, categories)
        _, vectors = _load_vectors(CACHE_DIR / f"creators_{key}.npz")
        cosine = cosine_matrix(vectors)

        zero_shot_tag_sets = [zero_shot_tags[cid] for cid in ids]
        llm_tag_sets = [llm_tags_by_id[cid] for cid in ids]
        declared_sets = [declared_by_id[cid] for cid in ids]

        bonus_m3 = select_bonus(cosine, ids, zero_shot_tag_sets, gold_by_id, dev_ids_ordered, BONUS_GRID)
        bonus_m4 = select_bonus(cosine, ids, llm_tag_sets, gold_by_id, dev_ids_ordered, BONUS_GRID)
        bonus_r2 = select_bonus(cosine, ids, declared_sets, gold_by_id, dev_ids_ordered, BONUS_GRID)

        params["per_embedding"][key] = {"tau": tau, "bonus_m3": bonus_m3, "bonus_m4": bonus_m4, "bonus_r2": bonus_r2}
        print(f"[select-params] {key}: tau={tau:.4f} bonus_m3={bonus_m3} bonus_m4={bonus_m4} bonus_r2={bonus_r2}")

    with (RESULTS_DIR / "selected_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def cmd_candidates(_: argparse.Namespace) -> None:
    """11개 설정 각각에 대해 크리에이터 100명의 상위 20명 후보를 계산해 저장한다."""
    categories = load_categories()
    creators = load_creators()
    declared_by_id = {c.id: frozenset(c.declared) for c in creators}

    with (RESULTS_DIR / "selected_params.json").open(encoding="utf-8") as f:
        params = json.load(f)
    llm_model = params["llm_model"]
    safe_name = llm_model.replace("/", "_")
    with (CACHE_DIR / f"llm_tags_{safe_name}_run0.json").open(encoding="utf-8") as f:
        llm_tags_raw = json.load(f)
    llm_tags_by_id = {cid: frozenset(tags) for cid, tags in llm_tags_raw.items()}

    all_candidates: dict[str, dict[str, list[list]]] = {}

    for key in EMBEDDING_MODEL_KEYS:
        p = params["per_embedding"][key]
        ids, _, zero_shot_tags, _ = _zero_shot_tags_for_key(key, creators, categories, tau_candidates=[p["tau"]])
        _, vectors = _load_vectors(CACHE_DIR / f"creators_{key}.npz")
        cosine = cosine_matrix(vectors)

        zero_shot_tag_sets = [zero_shot_tags[cid] for cid in ids]
        llm_tag_sets = [llm_tags_by_id[cid] for cid in ids]
        declared_sets = [declared_by_id[cid] for cid in ids]

        matrices = {
            f"M1_{key}": jaccard_matrix(zero_shot_tag_sets),
            f"M2_{key}": cosine,
            f"M3_{key}": cosine_with_tag_bonus(cosine, zero_shot_tag_sets, p["bonus_m3"]),
            f"M4_{key}": cosine_with_tag_bonus(cosine, llm_tag_sets, p["bonus_m4"]),
            f"R2_{key}": cosine_with_tag_bonus(cosine, declared_sets, p["bonus_r2"]),
        }
        if key == EMBEDDING_MODEL_KEYS[0]:
            matrices["R1"] = jaccard_matrix(declared_sets)  # 모델 무관, 한 번만

        for method_id, matrix in matrices.items():
            all_candidates[method_id] = {cid: top_n(matrix, ids, i, TOP_N_STORED) for i, cid in enumerate(ids)}

    with (RESULTS_DIR / "candidates.json").open("w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)
    print(f"[candidates] {len(all_candidates)}개 설정 저장 완료")


def cmd_judge_sheet(_: argparse.Namespace) -> None:
    """쿼리 30명의 설정별 상위 5명을 합집합으로 모아 블라인드 판정 시트를 만든다."""
    creators = load_creators()
    query_ids = [c.id for c in query_creators(creators)]

    with (RESULTS_DIR / "candidates.json").open(encoding="utf-8") as f:
        all_candidates = json.load(f)

    top5_by_query: dict[str, dict[str, list[tuple[str, float]]]] = {qid: {} for qid in query_ids}
    for method_id, by_creator in all_candidates.items():
        for qid in query_ids:
            top5_by_query[qid][method_id] = [tuple(item) for item in by_creator[qid][:JUDGE_TOP_K]]

    rows, provenance = build_judge_pairs(top5_by_query)
    rows = shuffle_rows(rows, seed=JUDGE_SHUFFLE_SEED)
    write_judge_sheet(rows, RESULTS_DIR / "judge_sheet.csv")
    write_provenance(provenance, RESULTS_DIR / "judge_provenance.csv")
    print(f"[judge-sheet] {len(rows)}쌍. results/judge_sheet.csv의 score 열(0/1/2)을 채운 뒤 score-judgments를 실행하세요.")


def cmd_auto_judge(_: argparse.Namespace) -> None:
    """judge_sheet.csv의 빈 score를 LLM으로 자동 채운다. 사람 판정의 보조·대체 수단이다.

    M4 태깅에 쓴 모델보다 강한 모델(OPENAI_JUDGE_MODEL)을 쓴다. 답을 받을 때마다 즉시
    파일에 저장해 중단·재개가 가능하며, 결과는 `results/auto_judge_report.json`에
    비용·건수와 함께 남긴다. 자동 판정은 사람 판정을 대체하는 보조 수단이므로, 신뢰도
    확인을 위해 일부를 직접 재판정해보는 것을 권장한다.
    """
    path = RESULTS_DIR / "judge_sheet.csv"
    fieldnames = ["pair_id", "query_id", "candidate_id", "score"]
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    creators_by_id = {c.id: c for c in load_creators()}
    judge = OpenAIJudge(model=OPENAI_JUDGE_MODEL)

    pending = [row for row in rows if not row["score"].strip()]
    total_all = len(rows)
    print(f"[auto-judge] 전체 {total_all}쌍 중 {total_all - len(pending)}쌍 완료, {len(pending)}쌍 자동 판정 시작 (모델: {OPENAI_JUDGE_MODEL})")

    total_input_tokens = total_output_tokens = 0
    for i, row in enumerate(pending, start=1):
        query = creators_by_id[row["query_id"]]
        candidate = creators_by_id[row["candidate_id"]]
        result = judge.judge(query.input_text(), candidate.input_text())
        row["score"] = str(result.score)
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        if i % 50 == 0 or i == len(pending):
            print(f"[auto-judge] {i}/{len(pending)} 완료")

    price = OPENAI_JUDGE_MODEL_PRICE
    cost = total_input_tokens / 1_000_000 * price["input_price"] + total_output_tokens / 1_000_000 * price["output_price"]
    report = {
        "model": OPENAI_JUDGE_MODEL,
        "judged_count": len(pending),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": cost,
    }
    with (RESULTS_DIR / "auto_judge_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[auto-judge] 완료. 비용 약 ${cost:.4f}. 일부를 src.judge_cli로 직접 재판정해 일치율을 확인하는 것을 권장합니다.")


def cmd_report(_: argparse.Namespace) -> None:
    """사람 판정 없이 계산되는 지표(E1 태깅 정확도, E3 대표 사례)를 test 기준으로 출력한다."""
    from src.metrics import check_case

    categories = load_categories()
    creators = load_creators()
    gold_by_id = {c.id: frozenset(c.gold) for c in creators}
    test_ids = {c.id for c in creators if c.split == "test"}

    print("=== E1. zero-shot 태깅 정확도 (test) ===")
    for key in EMBEDDING_MODEL_KEYS:
        with (RESULTS_DIR / "selected_params.json").open(encoding="utf-8") as f:
            tau = json.load(f)["per_embedding"][key]["tau"]
        ids, ranked_by_id, _, _ = _zero_shot_tags_for_key(key, creators, categories, tau_candidates=[tau])
        test_ranked = {cid: r for cid, r in ranked_by_id.items() if cid in test_ids}
        test_gold = {cid: gold_by_id[cid] for cid in test_ranked}
        print(f"-- {key} (tau={tau:.4f})")
        print(f"   Top-1 정확도: {top1_accuracy(test_ranked, test_gold):.3f}")
        print(f"   Top-3 포함률: {top3_inclusion_rate(test_ranked, test_gold):.3f}")
        confusions = sorted(confusion_pairs(test_ranked, test_gold).items(), key=lambda kv: -kv[1])[:5]
        print(f"   주요 혼동 쌍(정답->예측1등): {confusions}")

    print("\n=== E3. 대표 사례 (top-5, 11개 설정 전체) ===")
    with (RESULTS_DIR / "candidates.json").open(encoding="utf-8") as f:
        all_candidates = json.load(f)

    game_ids = {"G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09"}
    cases = [
        ("① 동의어 F01<->F02", "F01", {"F02"}, None),
        ("① 동의어 X12<->K03", "X12", {"K03"}, None),
        ("① 동의어 X13<->S01", "X13", {"S01"}, None),
        ("② 경품잡음 X01: 게임 크리에이터 없어야 함", "X01", None, game_ids),
        ("② 경품잡음 X02: 게임 크리에이터 없어야 함", "X02", None, game_ids),
        ("③ 분야넘기 X14->K07 목록에 있어야 함", "K07", {"X14"}, None),
        ("③ 분야넘기 X15->F06 목록에 있어야 함", "F06", {"X15"}, None),
    ]
    for method in all_candidates:
        print(f"-- {method}")

        def top5(cid: str, method: str = method) -> list[str]:
            return [item[0] for item in all_candidates[method][cid][:5]]

        for label, query_id, must_include, must_exclude in cases:
            result = check_case(top5(query_id), must_include, must_exclude)
            print(f"   [{'PASS' if result else 'FAIL'}] {label} -> top5={top5(query_id)}")


def cmd_score_judgments(_: argparse.Namespace) -> None:
    """사람이 채운 judge_sheet.csv로 E2 지표(관련도, nDCG, 무관 비율, 짝비교)를 계산한다."""
    from src.metrics import bootstrap_ci, irrelevant_rate_at_k, mean_relevance_at_k, ndcg_at_k, paired_win_counts

    creators = load_creators()
    query_ids = [c.id for c in query_creators(creators)]

    judged: dict[tuple[str, str], int] = {}
    pool_by_query: dict[str, list[int]] = {qid: [] for qid in query_ids}
    with (RESULTS_DIR / "judge_sheet.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["score"].strip():
                raise ValueError(f"판정이 비어 있는 행이 있습니다: {row['pair_id']}")
            score = int(row["score"])
            judged[(row["query_id"], row["candidate_id"])] = score
            pool_by_query[row["query_id"]].append(score)

    with (RESULTS_DIR / "candidates.json").open(encoding="utf-8") as f:
        all_candidates = json.load(f)

    per_method_query_relevance: dict[str, dict[str, float]] = {}
    for method_id, by_creator in all_candidates.items():
        per_query = {}
        for qid in query_ids:
            top5_ids = [item[0] for item in by_creator[qid][:JUDGE_TOP_K]]
            per_query[qid] = mean_relevance_at_k(top5_ids, qid, judged)
        per_method_query_relevance[method_id] = per_query

    print("=== E2. 방식별 평균 관련도@5 / nDCG@5 / 무관 비율@5 (test 쿼리 30명) ===")
    for method_id, by_creator in all_candidates.items():
        relevances, ndcgs, irrelevants = [], [], []
        for qid in query_ids:
            top5_ids = [item[0] for item in by_creator[qid][:JUDGE_TOP_K]]
            relevances.append(mean_relevance_at_k(top5_ids, qid, judged))
            ndcgs.append(ndcg_at_k(top5_ids, qid, pool_by_query[qid], judged, k=JUDGE_TOP_K))
            irrelevants.append(irrelevant_rate_at_k(top5_ids, qid, judged))
        lower, upper = bootstrap_ci(relevances)
        print(
            f"   {method_id}: 관련도={sum(relevances)/len(relevances):.3f} "
            f"[{lower:.3f}, {upper:.3f}]  nDCG={sum(ndcgs)/len(ndcgs):.3f}  무관비율={sum(irrelevants)/len(irrelevants):.3f}"
        )

    print("\n=== 짝비교 (M3 vs M2, 임베딩별) ===")
    for key in EMBEDDING_MODEL_KEYS:
        a, b = f"M3_{key}", f"M2_{key}"
        wins_a, wins_b, ties = paired_win_counts(per_method_query_relevance[a], per_method_query_relevance[b])
        print(f"   {a} 승 {wins_a} / {b} 승 {wins_b} / 동점 {ties} (쿼리 {len(query_ids)}개 중)")


SPOT_CHECK_TARGET = "M4_bge-m3"
SPOT_CHECK_BASELINES = ["M3_bge-m3", "R2_bge-m3"]


def cmd_spot_check(_: argparse.Namespace) -> None:
    """자동 판정 전체(556쌍) 대신, target_method가 baseline과 다르게 고른 후보만 골라

    소규모 시트(results/spot_check.csv)를 만든다. 이 쌍들이 target_method 점수를
    baseline보다 높게 만드는 실제 원인이라, 여기만 사람이 다시 봐도 결정에 필요한
    확인은 대부분 된다. `python3 -m src.judge_cli --file results/spot_check.csv`로 채운다.
    """
    creators = load_creators()
    query_ids = [c.id for c in query_creators(creators)]

    with (RESULTS_DIR / "candidates.json").open(encoding="utf-8") as f:
        all_candidates = json.load(f)

    pairs = select_disagreement_pairs(all_candidates, query_ids, SPOT_CHECK_TARGET, SPOT_CHECK_BASELINES, k=JUDGE_TOP_K)
    rows = [
        {"pair_id": f"{qid}::{cid}", "query_id": qid, "candidate_id": cid, "score": ""}
        for qid, cid in pairs
    ]
    write_judge_sheet_rows(rows, RESULTS_DIR / "spot_check.csv")
    print(
        f"[spot-check] {SPOT_CHECK_TARGET}가 {SPOT_CHECK_BASELINES}와 다르게 고른 {len(rows)}쌍을 "
        f"results/spot_check.csv에 저장했습니다.\n"
        f"python3 -m src.judge_cli --file results/spot_check.csv 로 채운 뒤 "
        f"python3 -m src.pipeline spot-check-report 를 실행하세요."
    )


def cmd_spot_check_report(_: argparse.Namespace) -> None:
    """spot_check.csv의 사람 점수와 judge_sheet.csv의 자동 판정 점수를 같은 쌍끼리 비교한다."""
    spot_path = RESULTS_DIR / "spot_check.csv"
    with spot_path.open(encoding="utf-8") as f:
        spot_rows = list(csv.DictReader(f))
    unscored = [r["pair_id"] for r in spot_rows if not r["score"].strip()]
    if unscored:
        raise ValueError(f"아직 판정이 안 된 쌍이 있습니다: {unscored}. src.judge_cli --file results/spot_check.csv 로 먼저 채우세요.")
    human = {(r["query_id"], r["candidate_id"]): int(r["score"]) for r in spot_rows}

    with (RESULTS_DIR / "judge_sheet.csv").open(encoding="utf-8") as f:
        auto_rows = list(csv.DictReader(f))
    auto = {(r["query_id"], r["candidate_id"]): int(r["score"]) for r in auto_rows if r["score"].strip()}

    stats = agreement_stats(human, auto)
    print(f"=== spot-check 일치율 ({SPOT_CHECK_TARGET} vs {SPOT_CHECK_BASELINES}만 고른 후보 {stats['count']}쌍) ===")
    print(f"   완전 일치율: {stats['exact_match_rate']:.2f}")
    print(f"   ±1 이내 일치율: {stats['within_1_rate']:.2f}")
    print(f"   평균 절대 오차: {stats['mean_abs_diff']:.2f}")
    for (qid, cid), auto_score in auto.items():
        if (qid, cid) in human and human[(qid, cid)] != auto_score:
            print(f"   불일치: {qid}::{cid}  사람={human[(qid, cid)]}  자동={auto_score}")


def write_judge_sheet_rows(rows: list[dict[str, str]], path: Path) -> None:
    """spot_check.csv 등, judge_sheet.csv와 같은 4열 형식으로 판정 시트를 새로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "query_id", "candidate_id", "score"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """서브커맨드를 파싱해 해당 단계 함수를 실행한다."""
    parser = argparse.ArgumentParser(description="추천 방식 비교 실험 파이프라인")
    sub = parser.add_subparsers(dest="stage", required=True)
    stages = {
        "embed": cmd_embed,
        "tag-llm": cmd_tag_llm,
        "select-params": cmd_select_params,
        "candidates": cmd_candidates,
        "judge-sheet": cmd_judge_sheet,
        "auto-judge": cmd_auto_judge,
        "report": cmd_report,
        "score-judgments": cmd_score_judgments,
        "spot-check": cmd_spot_check,
        "spot-check-report": cmd_spot_check_report,
    }
    for name in stages:
        sub.add_parser(name)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stages[args.stage](args)


if __name__ == "__main__":
    main()
