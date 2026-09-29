"""실험 단계를 순서대로 실행하는 CLI다. `python3 -m src.pipeline <단계>`로 호출한다.

단계: embed -> tag-llm -> select-params -> candidates -> judge-sheet -> report -> score-judgments
report까지는 사람 판정 없이 자동으로 끝난다 (E1, E3, E4). judge-sheet 이후 score-judgments 전에
`auto-judge`(LLM 자동 판정, 보조 수단) 또는 `src.judge_cli`(사람 판정)로 judge_sheet.csv의
score 열을 채운다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from src.clients import LocalEmbeddingClient, OpenAIEmbeddingClient, OpenAIJudge, OpenAITagger, RerankerClient
from src.config import (
    CACHE_DIR,
    JUDGE_SHUFFLE_SEED,
    JUDGE_TOP_K,
    LLM_CONSISTENCY_RUNS,
    LLM_TAG_MAX,
    LLM_TEMPERATURE,
    LOCAL_EMBEDDING_MODELS,
    OPENAI_JUDGE_MODEL,
    OPENAI_JUDGE_MODEL_PRICE,
    OPENAI_LLM_MODEL_CANDIDATES,
    RESULTS_DIR,
    TOP_N_STORED,
)
from src.data import Creator, dev_creators, load_categories, load_creators, query_creators, test_creators
from src.judge import build_judge_pairs, load_existing_scores, pair_text_hash, shuffle_rows, write_judge_sheet, write_provenance
from src.similarity import cosine_matrix, cosine_with_tag_bonus, jaccard_matrix, select_bonus, top_n
from src.spot_check import agreement_stats, select_pairwise_symmetric_disagreement
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

EMBEDDING_MODEL_KEYS = ("text-embedding-3-small", "bge-m3", "kure-v1", "qwen3-embedding-0.6b")
BONUS_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _embedding_client(key: str):
    """설정 키로 임베딩 클라이언트를 만든다."""
    if key == "text-embedding-3-small":
        return OpenAIEmbeddingClient()
    if key in LOCAL_EMBEDDING_MODELS:
        return LocalEmbeddingClient(key)
    raise ValueError(f"알 수 없는 임베딩 모델 키: {key}")


def _embedding_model_identity(key: str) -> str:
    """캐시 무효화 해시에 포함할, 이 키가 실제로 가리키는 모델·차원 설정이다.

    (PR #5 재리뷰로 발견: 캐시 키가 "bge-m3" 같은 이름뿐이라, config.py에서 그 이름이
    가리키는 실제 HF 모델·차원을 바꿔도 텍스트가 그대로면 오래된 캐시를 계속 썼다.)
    """
    if key == "text-embedding-3-small":
        from src.config import OPENAI_EMBEDDING_DIM, OPENAI_EMBEDDING_MODEL

        return f"{OPENAI_EMBEDDING_MODEL}:{OPENAI_EMBEDDING_DIM}"
    spec = LOCAL_EMBEDDING_MODELS[key]
    return f"{spec['model_name']}:{spec['dim']}"


def _save_vectors(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    """id 순서와 벡터를 함께 저장해, 나중에 순서가 어긋나지 않게 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), vectors=vectors)


def _load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    """저장된 id 순서와 벡터를 함께 불러온다."""
    data = np.load(path)
    return list(data["ids"]), data["vectors"]


def _content_hash(*parts: str) -> str:
    """입력 텍스트들의 해시를 만든다. 캐시 파일명이 같아도 입력이 바뀌면 재계산하도록 쓴다

    (PR #5 리뷰로 발견: 캐시 파일 존재 여부만 보고 재사용해, data/creators.csv를 바꿔도
    이전 실행의 results/cache가 남아 있으면 새 입력이 실제로 반영되지 않을 수 있었다).
    """
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _cached_hash_matches(hash_path: Path, expected: str) -> bool:
    return hash_path.exists() and hash_path.read_text(encoding="utf-8").strip() == expected


def cmd_embed(args: argparse.Namespace) -> None:
    """카테고리 설명문과 크리에이터 100명의 입력 텍스트를 EMBEDDING_MODEL_KEYS의 모델들로 인코딩한다.

    모델별로 저장된 벡터 파일이 있고 입력 텍스트 해시도 그대로면 API를 다시 부르지 않고
    건너뛴다 (리뷰 P2: 재실행 시 무조건 다시 호출하던 문제 수정. 리뷰 P1: 파일 존재만
    보고 건너뛰면 data/creators.csv를 바꿔도 오래된 캐시를 그대로 쓸 수 있었던 문제 수정).
    `--force`를 주면 강제로 다시 계산한다.
    카테고리 이름 자체는 크리에이터 입력 텍스트에 없으므로, zero-shot 태깅이 자기
    분야를 되맞히는 순환이 생기지 않는다. 소요 시간·벡터 크기·비용은 E4(운영 지표)로
    `results/e4_embedding.json`에 남긴다(이번 실행에서 새로 계산한 모델만 기록됨).
    """
    import time

    from src.config import OPENAI_EMBEDDING_PRICE_PER_1M

    categories = load_categories()
    creators = load_creators()
    creator_ids = [c.id for c in creators]
    creator_texts = [c.input_text() for c in creators]
    category_codes = [c.code for c in categories]
    category_texts = [c.description for c in categories]

    e4_path = RESULTS_DIR / "e4_embedding.json"
    e4_report: dict[str, dict] = json.loads(e4_path.read_text(encoding="utf-8")) if e4_path.exists() else {}
    for key in EMBEDDING_MODEL_KEYS:
        # 키별로 해시가 다르다 — 어떤 실제 모델·차원을 가리키는지도 해시에 넣어, config.py에서
        # 같은 키가 다른 모델을 가리키게 바꿔도(텍스트는 그대로라도) 캐시를 다시 계산하게 한다.
        input_hash = _content_hash(_embedding_model_identity(key), *creator_texts, *category_texts)
        creators_path = CACHE_DIR / f"creators_{key}.npz"
        categories_path = CACHE_DIR / f"categories_{key}.npz"
        hash_path = CACHE_DIR / f"embed_{key}.input_hash"
        if not args.force and creators_path.exists() and categories_path.exists() and _cached_hash_matches(hash_path, input_hash):
            print(f"[embed] {key}: 캐시된 벡터 사용 (재계산하려면 --force)")
            continue

        client = _embedding_client(key)
        started = time.perf_counter()
        creator_vectors = client.embed(creator_texts)
        creator_tokens = getattr(client, "last_input_tokens", None)
        category_vectors = client.embed(category_texts)
        category_tokens = getattr(client, "last_input_tokens", None)
        elapsed_seconds = time.perf_counter() - started

        _save_vectors(creators_path, creator_ids, creator_vectors)
        _save_vectors(categories_path, category_codes, category_vectors)
        hash_path.write_text(input_hash, encoding="utf-8")

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

    with e4_path.open("w", encoding="utf-8") as f:
        json.dump(e4_report, f, ensure_ascii=False, indent=2)


def cmd_tag_llm(args: argparse.Namespace) -> None:
    """LLM 태깅 후보 모델들을 크리에이터 전원에게 2회씩 돌리고, dev 정확도로 하나를 고른다.

    run 파일(`llm_tags_{model}_run{n}.json`)이 있고 입력 텍스트 해시도 그대로면 그 run은
    다시 호출하지 않고 읽어서 재사용한다 (리뷰 P2: 재실행 시 전체를 다시 태깅하던 문제
    수정. 크리에이터 단위 재개는 아니고 run 단위다. 리뷰 P1: 파일 존재만 보고 건너뛰면
    data/creators.csv를 바꿔도 오래된 캐시를 그대로 쓸 수 있었던 문제 수정 —
    `--force`를 주면 전부 다시 계산한다). 해시에는 실제 요청에 쓰이는 시스템 프롬프트·
    temperature도 포함한다(재리뷰로 발견: LLM_TAG_MAX·LLM_TEMPERATURE를 바꿔도 캐시가
    무효화되지 않았다 — SYSTEM_PROMPT가 LLM_TAG_MAX를 그대로 담고 있어 이거 하나로 충분).
    토큰 사용량은 run 단위 사이드카(`*.token_usage.json`)에 저장해두고, 캐시를 쓰든 새로
    계산하든 매 run의 사용량을 항상 더한다 (재리뷰로 발견: run 중 일부만 재계산하면
    캐시로 읽은 run의 과거 사용량이 통째로 빠져 기록이 실제보다 줄어들었다 — "이번에
    새로 쓴 토큰만 더한다"는 이전 방식은 완전 캐시 히트든 부분 캐시 히트든 둘 다 부정확
    했다. run별로 저장해두면 어느 조합이든 합산만 하면 항상 정확하다).
    두 번 실행하는 것은 비결정성(consistency)을 재기 위함이며, 승자 모델의 태그는
    이후 M4(임베딩 + LLM 태그 보정)의 입력으로 전원에게 쓰인다.
    """
    from src.clients.openai_tagger import SYSTEM_PROMPT

    categories = load_categories()
    creators = load_creators()
    category_codes = [c.code for c in categories]
    dev_ids = {c.id for c in dev_creators(creators)}
    gold_by_id = {c.id: frozenset(c.gold) for c in creators}
    input_hash = _content_hash(SYSTEM_PROMPT, str(LLM_TEMPERATURE), *(c.input_text() for c in creators), *category_codes)

    dev_metrics: dict[str, dict[str, float]] = {}
    usage_summary: dict[str, dict[str, int]] = {}
    for model_name in OPENAI_LLM_MODEL_CANDIDATES:
        tagger = OpenAITagger(model=model_name, category_codes=category_codes)
        safe_name = model_name.replace("/", "_")
        runs: list[dict[str, list[str]]] = []
        total_input_tokens = total_output_tokens = 0
        for run_index in range(LLM_CONSISTENCY_RUNS):
            run_path = CACHE_DIR / f"llm_tags_{safe_name}_run{run_index}.json"
            hash_path = CACHE_DIR / f"llm_tags_{safe_name}_run{run_index}.input_hash"
            usage_path = CACHE_DIR / f"llm_tags_{safe_name}_run{run_index}.token_usage.json"
            if not args.force and run_path.exists() and _cached_hash_matches(hash_path, input_hash):
                tags_by_id = json.loads(run_path.read_text(encoding="utf-8"))
                run_usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else {"input_tokens": 0, "output_tokens": 0}
                print(f"[tag-llm] {model_name} run {run_index + 1}/{LLM_CONSISTENCY_RUNS} 캐시 사용 (재계산하려면 --force)")
            else:
                tags_by_id = {}
                run_input_tokens = run_output_tokens = 0
                for creator in creators:
                    result = tagger.tag(creator.input_text())
                    tags_by_id[creator.id] = list(result.tags)
                    run_input_tokens += result.input_tokens
                    run_output_tokens += result.output_tokens
                run_usage = {"input_tokens": run_input_tokens, "output_tokens": run_output_tokens}
                with run_path.open("w", encoding="utf-8") as f:
                    json.dump(tags_by_id, f, ensure_ascii=False, indent=2)
                hash_path.write_text(input_hash, encoding="utf-8")
                usage_path.write_text(json.dumps(run_usage), encoding="utf-8")
                print(f"[tag-llm] {model_name} run {run_index + 1}/{LLM_CONSISTENCY_RUNS} 완료")
            runs.append(tags_by_id)
            total_input_tokens += run_usage["input_tokens"]
            total_output_tokens += run_usage["output_tokens"]

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
    """zero-shot tau와, 세 태그 소스(zero-shot, LLM, 입력분야) 각각의 bonus를 dev로만 고정한다.

    bonus 선택은 dev 크리에이터끼리의 후보 pool·정답만 사용한다 (dev×dev 부분 행렬).
    test 크리에이터가 후보나 relevance 정답으로 섞이면, 쿼리를 dev로 제한해도 test 라벨이
    선택에 영향을 주는 누수가 생기기 때문이다 (2026-09-28 리뷰로 발견, dev 입력은 그대로
    두고 test gold만 바꿔도 선택 bonus가 달라지는 것으로 재현됨).
    """
    categories = load_categories()
    creators = load_creators()
    gold_by_id = {c.id: frozenset(c.gold) for c in creators}
    declared_by_id = {c.id: frozenset(c.declared) for c in creators}
    dev_id_set = {c.id for c in dev_creators(creators)}

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

        # dev만 남긴 부분 행렬로 bonus를 고른다. test 벡터·gold는 이 시점에 전혀 등장하지 않는다.
        dev_index = [i for i, cid in enumerate(ids) if cid in dev_id_set]
        dev_ids_ordered = [ids[i] for i in dev_index]
        # 평가(candidates)와 같은 조건으로 고르도록, 프롬프트가 있는 모델은 dev 행에도 쿼리 프롬프트를 적용한다
        dev_query_vectors = _query_prompted_vectors(key, ids, vectors, creators, target_ids=dev_id_set)[dev_index]
        dev_cosine = cosine_matrix(vectors[dev_index], query_vectors=dev_query_vectors)

        dev_zero_shot_tag_sets = [zero_shot_tags[cid] for cid in dev_ids_ordered]
        dev_llm_tag_sets = [llm_tags_by_id[cid] for cid in dev_ids_ordered]
        dev_declared_sets = [declared_by_id[cid] for cid in dev_ids_ordered]
        dev_gold_by_id = {cid: gold_by_id[cid] for cid in dev_ids_ordered}

        bonus_m3 = select_bonus(dev_cosine, dev_ids_ordered, dev_zero_shot_tag_sets, dev_gold_by_id, dev_ids_ordered, BONUS_GRID)
        bonus_m4 = select_bonus(dev_cosine, dev_ids_ordered, dev_llm_tag_sets, dev_gold_by_id, dev_ids_ordered, BONUS_GRID)
        bonus_r2 = select_bonus(dev_cosine, dev_ids_ordered, dev_declared_sets, dev_gold_by_id, dev_ids_ordered, BONUS_GRID)

        params["per_embedding"][key] = {"tau": tau, "bonus_m3": bonus_m3, "bonus_m4": bonus_m4, "bonus_r2": bonus_r2}
        print(f"[select-params] {key}: tau={tau:.4f} bonus_m3={bonus_m3} bonus_m4={bonus_m4} bonus_r2={bonus_r2}")

    with (RESULTS_DIR / "selected_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def _rerank_bge_m2_candidates(
    ids: list[str], cosine: np.ndarray, text_by_id: dict[str, str], reranker: RerankerClient, pool_size: int
) -> dict[str, list[list]]:
    """M2(bge-m3) 코사인으로 추린 후보(pool_size명)를 cross-encoder로 다시 채점·정렬한다 (M5).

    리랭커는 100x100 행렬을 만들지 않는다 — 이미 추린 소수 후보에만 쌍별로 추론한다.
    """
    result: dict[str, list[list]] = {}
    for i, qid in enumerate(ids):
        pool = top_n(cosine, ids, i, pool_size)
        pairs = [(text_by_id[qid], text_by_id[cid]) for cid, _ in pool]
        scores = reranker.score(pairs)
        ranked = sorted(zip((cid for cid, _ in pool), scores), key=lambda item: -item[1])
        result[qid] = [[cid, score] for cid, score in ranked]
    return result


def _query_prompted_vectors(
    key: str, ids: list[str], vectors: np.ndarray, creators: list[Creator], target_ids: set[str] | None = None
) -> np.ndarray:
    """쿼리 쪽(행) 벡터를 반환한다. 원래 벡터 배열은 후보 쪽(열)으로 그대로 두고 바꾸지 않는다.

    query_prompt_name이 등록된 모델(Qwen3)이면 평가 쿼리 30명의 행만 그 프롬프트로 다시
    인코딩해 바꿔치기한 복사본을 돌려준다 — M2~M4/R2의 "쿼리가 후보를 찾는" 방향 유사도용이다
    (이슈 #8, 리뷰로 발견: 공식 사용법은 검색 쿼리 쪽에 instruct 프롬프트를 쓰길 권장하는데
    기존엔 접두어 없이 측정했다). 이 반환값을 `cosine_matrix(vectors, query_vectors=...)`의 행에만
    쓰고 열에는 원래 vectors를 써야 한다 — 쿼리끼리 서로의 후보가 될 때도 후보로 참조되는 쪽은
    프롬프트 없는 원래 임베딩이어야 하기 때문이다(재리뷰로 발견: 예전엔 하나의 배열을 덮어써
    후보 쪽 벡터까지 바뀌었다). target_ids로 프롬프트를 적용할 행을 바꿀 수 있다 — 기본은 평가 쿼리 30명이고, select-params는 dev 크리에이터를 준다(평가와 같은 표현으로 bonus를 고르기 위함). query_prompt_name이 없는 모델은 vectors를 그대로 반환한다.
    """
    prompt_name = LOCAL_EMBEDDING_MODELS.get(key, {}).get("query_prompt_name")
    if not prompt_name:
        return vectors
    query_ids = target_ids if target_ids is not None else {c.id for c in query_creators(creators)}
    text_by_id = {c.id: c.input_text() for c in creators}
    targets = [(i, cid) for i, cid in enumerate(ids) if cid in query_ids]
    if not targets:
        return vectors
    query_vectors = LocalEmbeddingClient(key).embed([text_by_id[cid] for _, cid in targets], prompt_name=prompt_name)
    vectors = vectors.copy()
    for (i, _cid), qv in zip(targets, query_vectors):
        vectors[i] = qv
    return vectors


def cmd_candidates(_: argparse.Namespace) -> None:
    """설정(임베딩 모델 x 방식) 각각에 대해 크리에이터 100명의 상위 20명 후보를 계산해 저장한다."""
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
        cosine = cosine_matrix(vectors, query_vectors=_query_prompted_vectors(key, ids, vectors, creators))

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

        if key == "bge-m3":
            text_by_id = {c.id: c.input_text() for c in creators}
            reranker = RerankerClient()
            all_candidates["M5_bge-m3"] = _rerank_bge_m2_candidates(ids, cosine, text_by_id, reranker, TOP_N_STORED)

    with (RESULTS_DIR / "candidates.json").open("w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)
    print(f"[candidates] {len(all_candidates)}개 설정 저장 완료")


def cmd_judge_sheet(_: argparse.Namespace) -> None:
    """쿼리 30명의 설정별 상위 5명을 합집합으로 모아 블라인드 판정 시트를 만든다."""
    creators = load_creators()
    query_ids = [c.id for c in query_creators(creators)]
    text_by_id = {c.id: c.input_text() for c in creators}

    with (RESULTS_DIR / "candidates.json").open(encoding="utf-8") as f:
        all_candidates = json.load(f)

    top5_by_query: dict[str, dict[str, list[tuple[str, float]]]] = {qid: {} for qid in query_ids}
    for method_id, by_creator in all_candidates.items():
        for qid in query_ids:
            top5_by_query[qid][method_id] = [tuple(item) for item in by_creator[qid][:JUDGE_TOP_K]]

    rows, provenance = build_judge_pairs(top5_by_query, text_by_id)
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
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # 파일의 실제 헤더를 그대로 써서 text_hash 열을 지우지 않는다
        rows = list(reader)

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

    print("\n=== E3. 대표 사례 (top-5, 설정 전체) ===")
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
SPOT_CHECK_SAMPLE_SIZE = 30  # 141쌍 전부는 부담이 커서 무작위 표본만 사람이 본다 (2026-09-28 결정)
SPOT_CHECK_SAMPLE_SEED = 20260928

# bge-m3와 KURE-v1 중 하나를 고를 때 실제로 추천이 갈리는 쌍만 사람이 보게 한다 (이슈 #8).
# 각 임베딩의 현재 1순위 후보를 골랐다: M3_bge-m3(재현성 우선 픽), M2_kure-v1(태그 보정 없이도
# 견고했던 KURE-v1의 대표 픽). 다른 조합을 보고 싶으면 이 두 값만 바꾸면 된다.
MODEL_SPOT_CHECK_TARGET = "M3_bge-m3"
MODEL_SPOT_CHECK_BASELINES = ["M2_kure-v1"]


def _build_spot_check_rows(
    target: str, baselines: list[str], sample_size: int, seed: int, out_path: Path
) -> tuple[int, int]:
    """target과 각 baseline을 양쪽 차집합(대칭차집합)으로 비교해 다른 후보만 골라

    소규모 시트를 만든다. target−baseline 합집합만 보던 이전 방식은 "A=X, B=Y, C=X"처럼
    다른 baseline이 같은 후보를 갖고 있으면 실제 차이를 놓쳤다(리뷰 P2). 지금은 각 baseline과
    정확히 한 쌍씩 비교해 두 방향(더한 것·뺀 것)을 모두 잡는다. 전체 쌍이 sample_size보다
    많으면 고정 시드로 무작위 표본만 남긴다(전수 조사가 아니라 표본 조사임을 결과에 함께
    적어야 한다). 반환값은 (실제 저장한 쌍 수, 표본 뽑기 전 전체 쌍 수)다.
    """
    creators = load_creators()
    query_ids = [c.id for c in query_creators(creators)]
    text_by_id = {c.id: c.input_text() for c in creators}

    with (RESULTS_DIR / "candidates.json").open(encoding="utf-8") as f:
        all_candidates = json.load(f)

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for baseline in baselines:
        for pair in select_pairwise_symmetric_disagreement(all_candidates, query_ids, target, baseline, k=JUDGE_TOP_K):
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)

    total_found = len(pairs)
    if total_found > sample_size:
        # 정렬 후 샘플링해야 고정 시드가 실행마다 같은 쌍을 뽑는다는 보장이 생긴다
        # (PR #5 리뷰로 발견: set 순회 순서가 해시 시드에 따라 달라져 pairs 자체가 이미
        # 비결정적이었다 — select_pairwise_symmetric_disagreement에서 sorted로 고쳤지만,
        # 여기서도 한 번 더 정렬해 이 함수만 보고도 재현성이 보장됨을 알 수 있게 한다).
        pairs = sorted(pairs)
        pairs = random.Random(seed).sample(pairs, sample_size)

    rows = [
        {
            "pair_id": f"{qid}::{cid}",
            "query_id": qid,
            "candidate_id": cid,
            "score": "",
            "text_hash": pair_text_hash(text_by_id[qid], text_by_id[cid]),
        }
        for qid, cid in pairs
    ]
    write_judge_sheet_rows(rows, out_path)
    return len(rows), total_found


def _print_spot_check_report(target: str, baselines: list[str], spot_path: Path) -> None:
    """spot_path의 사람 점수와 judge_sheet.csv의 자동 판정 점수를 같은 쌍끼리 비교한다."""
    with spot_path.open(encoding="utf-8") as f:
        spot_rows = list(csv.DictReader(f))
    unscored = [r["pair_id"] for r in spot_rows if not r["score"].strip()]
    if unscored:
        raise ValueError(f"아직 판정이 안 된 쌍이 있습니다: {unscored}. src.judge_cli --file {spot_path} 로 먼저 채우세요.")
    human = {(r["query_id"], r["candidate_id"]): int(r["score"]) for r in spot_rows}

    with (RESULTS_DIR / "judge_sheet.csv").open(encoding="utf-8") as f:
        auto_rows = list(csv.DictReader(f))
    auto = {(r["query_id"], r["candidate_id"]): int(r["score"]) for r in auto_rows if r["score"].strip()}

    stats = agreement_stats(human, auto)
    print(f"=== spot-check 일치율 ({target} vs {baselines}만 고른 후보 {stats['count']}쌍) ===")
    print(f"   완전 일치율: {stats['exact_match_rate']:.2f}")
    print(f"   ±1 이내 일치율: {stats['within_1_rate']:.2f}")
    print(f"   평균 절대 오차: {stats['mean_abs_diff']:.2f}")
    for (qid, cid), auto_score in auto.items():
        if (qid, cid) in human and human[(qid, cid)] != auto_score:
            print(f"   불일치: {qid}::{cid}  사람={human[(qid, cid)]}  자동={auto_score}")


def cmd_spot_check(_: argparse.Namespace) -> None:
    """SPOT_CHECK_TARGET·SPOT_CHECK_BASELINES 기준으로 results/spot_check.csv를 만든다.

    `python3 -m src.judge_cli --file results/spot_check.csv`로 채운다.
    """
    saved, total_found = _build_spot_check_rows(
        SPOT_CHECK_TARGET, SPOT_CHECK_BASELINES, SPOT_CHECK_SAMPLE_SIZE, SPOT_CHECK_SAMPLE_SEED, RESULTS_DIR / "spot_check.csv"
    )
    sample_note = f" (전체 {total_found}쌍 중 무작위 표본)" if total_found > saved else ""
    print(
        f"[spot-check] {SPOT_CHECK_TARGET}를 {SPOT_CHECK_BASELINES}와 각각 양쪽 차집합으로 비교해 "
        f"다른 {saved}쌍{sample_note}을 results/spot_check.csv에 저장했습니다.\n"
        f"python3 -m src.judge_cli --file results/spot_check.csv 로 채운 뒤 "
        f"python3 -m src.pipeline spot-check-report 를 실행하세요."
    )


def cmd_spot_check_report(_: argparse.Namespace) -> None:
    """results/spot_check.csv 기준으로 SPOT_CHECK_TARGET·SPOT_CHECK_BASELINES 일치율을 출력한다."""
    _print_spot_check_report(SPOT_CHECK_TARGET, SPOT_CHECK_BASELINES, RESULTS_DIR / "spot_check.csv")


def _model_spot_check_args(args: argparse.Namespace) -> tuple[str, list[str], Path]:
    """--target/--baseline/--out이 없으면 기본값(M3_bge-m3 vs M2_kure-v1, spot_check_models.csv)을 쓴다."""
    target = getattr(args, "target", None) or MODEL_SPOT_CHECK_TARGET
    baselines = getattr(args, "baseline", None) or MODEL_SPOT_CHECK_BASELINES
    out = getattr(args, "out", None)
    return target, baselines, RESULTS_DIR / (out or "spot_check_models.csv")


def cmd_spot_check_models(args: argparse.Namespace) -> None:
    """두 설정이 실제로 다르게 추천한 쌍만 뽑아 사람 판정 시트를 만든다 (이슈 #8).

    기본은 bge-m3 vs KURE-v1(MODEL_SPOT_CHECK_TARGET·_BASELINES → results/spot_check_models.csv).
    `--target M4_bge-m3 --baseline M4_qwen3-embedding-0.6b --out spot_check_m4_qwen3.csv`처럼
    다른 조합도 지정할 수 있다. `python3 -m src.judge_cli --file results/<out>`으로 채운다.
    """
    target, baselines, out_path = _model_spot_check_args(args)
    saved, total_found = _build_spot_check_rows(target, baselines, SPOT_CHECK_SAMPLE_SIZE, SPOT_CHECK_SAMPLE_SEED, out_path)
    sample_note = f" (전체 {total_found}쌍 중 무작위 표본)" if total_found > saved else ""
    flags = "" if out_path.name == "spot_check_models.csv" else f" --target {target} --baseline {' --baseline '.join(baselines)} --out {out_path.name}"
    print(
        f"[spot-check-models] {target}를 {baselines}와 각각 양쪽 차집합으로 비교해 "
        f"다른 {saved}쌍{sample_note}을 results/{out_path.name}에 저장했습니다.\n"
        f"python3 -m src.judge_cli --file results/{out_path.name} 로 채운 뒤 "
        f"python3 -m src.pipeline spot-check-models-report{flags} 를 실행하세요."
    )


def cmd_spot_check_models_report(args: argparse.Namespace) -> None:
    """spot-check-models와 같은 --target/--baseline/--out으로 사람·자동 판정 일치율을 출력한다."""
    target, baselines, out_path = _model_spot_check_args(args)
    _print_spot_check_report(target, baselines, out_path)


def write_judge_sheet_rows(rows: list[dict[str, str]], path: Path) -> None:
    """spot_check.csv 등, judge_sheet.csv와 같은 5열 형식(text_hash 포함)으로 판정 시트를 새로 쓴다.

    같은 경로에 이미 채워진 판정이 있고 pair_id·text_hash가 둘 다 같으면 점수를 이어받는다
    (리뷰 P1과 동일한 문제: spot-check 재실행 시 기존 판정이 사라지던 것 수정. 재리뷰 지적:
    pair_id만 같다고 보존하면 크리에이터 텍스트가 바뀐 경우 예전 텍스트로 매긴 점수가 남을
    수 있어, text_hash가 다르거나 비어 있으면 점수를 비워 재판정하게 한다). rows의 각 항목은
    "text_hash" 키를 포함해야 한다.
    """
    existing = load_existing_scores(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "query_id", "candidate_id", "score", "text_hash"])
        writer.writeheader()
        for row in rows:
            text_hash = row.get("text_hash", "")
            prev = existing.get(row["pair_id"])
            keep = bool(text_hash) and prev is not None and prev[1] == text_hash
            writer.writerow({**row, "score": prev[0] if keep else row["score"], "text_hash": text_hash})


SENSITIVITY_SIZES = [30, 60, 120, 240]
SENSITIVITY_REPS = 20
SENSITIVITY_SEED = 20260929


def _large_llm_tags(creators: list[Creator], llm_model: str, category_codes: list[str]) -> dict[str, frozenset[str]]:
    """합성 크리에이터를 선택된 LLM 태거로 1회 태깅한다. 입력·프롬프트 해시가 같으면 캐시를 쓴다."""
    from src.clients.openai_tagger import SYSTEM_PROMPT

    input_hash = _content_hash(llm_model, SYSTEM_PROMPT, str(LLM_TEMPERATURE), *(c.input_text() for c in creators), *category_codes)
    path = CACHE_DIR / "large_llm_tags.json"
    hash_path = CACHE_DIR / "large_llm_tags.input_hash"
    if path.exists() and _cached_hash_matches(hash_path, input_hash):
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        tagger = OpenAITagger(model=llm_model, category_codes=category_codes)
        raw = {c.id: list(tagger.tag(c.input_text()).tags) for c in creators}
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        hash_path.write_text(input_hash, encoding="utf-8")
    return {cid: frozenset(tags) for cid, tags in raw.items()}


def _large_vectors(key: str, creators: list[Creator]) -> np.ndarray:
    """합성 크리에이터 벡터를 모델별로 캐시해 만든다. 텍스트·모델 설정이 바뀌면 다시 계산한다."""
    texts = [c.input_text() for c in creators]
    input_hash = _content_hash(_embedding_model_identity(key), *texts)
    path = CACHE_DIR / f"large_creators_{key}.npz"
    hash_path = CACHE_DIR / f"large_creators_{key}.input_hash"
    if path.exists() and _cached_hash_matches(hash_path, input_hash):
        return _load_vectors(path)[1]
    vectors = _embedding_client(key).embed(texts)
    _save_vectors(path, [c.id for c in creators], vectors)
    hash_path.write_text(input_hash, encoding="utf-8")
    return vectors


def cmd_dev_sensitivity(_: argparse.Namespace) -> None:
    """dev 크기(30/60/120/240)별로 tau·bonus 선택이 얼마나 흔들리는지 잰다 (이슈 #8).

    dev 후보 풀 = 기존 dev 30명 + 합성 300명. test = 기존 100명(쿼리 30명 포함)이며, test 성적은
    LLM 판정 없는 gold 기준 대리 지표다. 쿼리 프롬프트가 있는 모델(Qwen3)은 평가와 같은 조건으로 dev·쿼리 행에 프롬프트를 적용한다.
    기존 dev 30명 그대로 고른 값이 selected_params.json과 같은지도 확인해 함께 기록한다.
    """
    from src.data import load_large_creators
    from src.sensitivity import CreatorSet, run_sensitivity, select_on_subset, summarize

    categories = load_categories()
    codes = [c.code for c in categories]
    creators = load_creators()
    large = load_large_creators()
    dev_ids = {c.id for c in dev_creators(creators)}
    query_ids = [c.id for c in query_creators(creators)]
    test_ids = [c.id for c in test_creators(creators)]

    selected = json.loads((RESULTS_DIR / "selected_params.json").read_text(encoding="utf-8"))
    llm_model = selected["llm_model"]
    safe_name = llm_model.replace("/", "_")
    orig_llm = {cid: frozenset(t) for cid, t in json.loads((CACHE_DIR / f"llm_tags_{safe_name}_run0.json").read_text(encoding="utf-8")).items()}
    large_llm = _large_llm_tags(large, llm_model, codes)

    orig_gold = {c.id: frozenset(c.gold) for c in creators}
    orig_declared = {c.id: frozenset(c.declared) for c in creators}
    large_gold = {c.id: frozenset(c.gold) for c in large}
    large_declared = {c.id: frozenset(c.declared) for c in large}

    output: dict = {"sizes": SENSITIVITY_SIZES, "reps": SENSITIVITY_REPS, "seed": SENSITIVITY_SEED, "per_embedding": {}}
    for key in EMBEDDING_MODEL_KEYS:
        ids, vectors = _load_vectors(CACHE_DIR / f"creators_{key}.npz")
        _, category_vectors = _load_vectors(CACHE_DIR / f"categories_{key}.npz")
        # 쿼리 프롬프트가 있는 모델(Qwen3)은 평가와 같은 조건으로 test 쿼리 행·dev 행에 프롬프트를 적용한다
        prompted = bool(LOCAL_EMBEDDING_MODELS.get(key, {}).get("query_prompt_name"))
        test_query_vectors = _query_prompted_vectors(key, list(ids), vectors, creators) if prompted else None
        test = CreatorSet(list(ids), vectors, rank_all(vectors, category_vectors, codes), orig_gold, orig_llm, orig_declared, test_query_vectors)

        dev_index = [i for i, cid in enumerate(ids) if cid in dev_ids]
        large_vectors = _large_vectors(key, large)
        pool_ids = [ids[i] for i in dev_index] + [c.id for c in large]
        pool_vectors = np.vstack([vectors[dev_index], large_vectors])
        pool_creators = [c for c in creators if c.id in dev_ids] + large
        pool_query_vectors = _query_prompted_vectors(key, pool_ids, pool_vectors, pool_creators, target_ids=set(pool_ids)) if prompted else None
        pool = CreatorSet(
            pool_ids,
            pool_vectors,
            rank_all(pool_vectors, category_vectors, codes),
            {**orig_gold, **large_gold},
            {**orig_llm, **large_llm},
            {**orig_declared, **large_declared},
            pool_query_vectors,
        )

        tau, bonuses = select_on_subset(pool, list(range(len(dev_index))), BONUS_GRID, LLM_TAG_MAX)
        expected = selected["per_embedding"][key]
        matches = abs(tau - expected["tau"]) < 1e-9 and all(bonuses[n] == expected[n] for n in bonuses)
        records = run_sensitivity(pool, test, test_ids, query_ids, SENSITIVITY_SIZES, SENSITIVITY_REPS, BONUS_GRID, LLM_TAG_MAX, SENSITIVITY_SEED)
        output["per_embedding"][key] = {
            "original_dev30": {"tau": tau, **bonuses, "matches_selected_params": matches},
            "summary": summarize(records),
        }
        print(f"[dev-sensitivity] {key}: 기존 dev 30 재현={'OK' if matches else 'MISMATCH'}")
        for row in output["per_embedding"][key]["summary"]:
            print(
                f"  size={row['size']:>3} tau={row['tau_mean']:.3f}±{row['tau_std']:.3f} "
                f"m3={row['bonus_m3']['mode']}({row['bonus_m3']['mode_share']:.0%}) "
                f"m4={row['bonus_m4']['mode']}({row['bonus_m4']['mode_share']:.0%}) "
                f"r2={row['bonus_r2']['mode']}({row['bonus_r2']['mode_share']:.0%}) "
                f"F1={row['test_tag_f1']['mean']:.3f}±{row['test_tag_f1']['std']:.3f} "
                f"P@5 m3={row['p5_m3']['mean']:.3f} m4={row['p5_m4']['mean']:.3f} r2={row['p5_r2']['mean']:.3f}"
            )
    with (RESULTS_DIR / "dev_sensitivity.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


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
        "spot-check-models": cmd_spot_check_models,
        "spot-check-models-report": cmd_spot_check_models_report,
        "dev-sensitivity": cmd_dev_sensitivity,
    }
    for name in stages:
        stage_parser = sub.add_parser(name)
        if name in ("spot-check-models", "spot-check-models-report"):
            stage_parser.add_argument("--target", help="비교의 기준 설정 (예: M4_bge-m3)")
            stage_parser.add_argument("--baseline", action="append", help="비교 대상 설정, 여러 번 지정 가능 (예: M4_qwen3-embedding-0.6b)")
            stage_parser.add_argument("--out", help="results/ 아래에 저장할 파일명 (기본 spot_check_models.csv)")
        if name in ("embed", "tag-llm"):
            stage_parser.add_argument(
                "--force", action="store_true", help="캐시된 결과가 있어도 API를 다시 호출해 새로 계산한다"
            )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stages[args.stage](args)


if __name__ == "__main__":
    main()
