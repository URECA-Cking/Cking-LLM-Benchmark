"""사용자 취향 요약문 → 크리에이터 추천 검증 (이슈 #12).

기존 벤치마크는 "크리에이터 소개글 → 유사 크리에이터"만 검증했다. 여기서는 서비스가 사용자 취향을 요약한
텍스트를 쿼리로 크리에이터를 추천할 때를 가정해, 같은 임베딩 모델·방식(M2/M3/M4/M5)으로 쿼리 표현(키워드형·
문장형·서술형)과 Qwen3 쿼리 프롬프트 조건을 바꿔 가며 잰다. 이 파일의 앞부분은 순수 계산이고, 뒷부분은
`taste-eval`·`taste-judge` 단계다.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.config import (
    CACHE_DIR,
    LLM_TAG_MAX,
    LOCAL_EMBEDDING_MODELS,
    OPENAI_JUDGE_MODEL,
    OPENAI_JUDGE_MODEL_PRICE,
    RESULTS_DIR,
    TASTE_JUDGE_STYLES,
    TASTE_QWEN3_INSTRUCT,
    TASTE_TOP_K,
    TOP_N_STORED,
)


def score_matrix(
    query_vectors: np.ndarray,
    creator_vectors: np.ndarray,
    query_tags: list[frozenset[str]],
    creator_tags: list[frozenset[str]],
    bonus: float,
) -> np.ndarray:
    """(쿼리 수 x 크리에이터 수) 점수 행렬이다. 코사인에 태그가 하나라도 겹치면 +bonus를 더한다(bonus 0이면 M2)."""
    cosine = query_vectors @ creator_vectors.T
    if bonus == 0.0:
        return cosine
    overlap = np.array([[1.0 if q & c else 0.0 for c in creator_tags] for q in query_tags], dtype=np.float32)
    return cosine + bonus * overlap


def top_k(row: np.ndarray, creator_ids: list[str], k: int) -> list[tuple[str, float]]:
    """점수 내림차순 상위 k명을 (id, 점수)로 돌려준다. 동점은 id 오름차순으로 안정적으로 정한다."""
    order = sorted(range(len(creator_ids)), key=lambda j: (-row[j], creator_ids[j]))
    return [(creator_ids[j], float(row[j])) for j in order[:k]]


def gold_precision(top_ids: list[str], query_gold: tuple[str, ...] | frozenset[str], gold_by_id: dict[str, frozenset[str]]) -> float:
    """상위 후보 중 정답 분야를 하나라도 공유하는 비율이다."""
    if not top_ids:
        return 0.0
    gold = frozenset(query_gold)
    return sum(1 for cid in top_ids if gold_by_id[cid] & gold) / len(top_ids)


def summarize_gold(
    rankings: dict[str, dict[str, list[list]]],
    style_by_query: dict[str, str],
    gold_by_query: dict[str, tuple[str, ...]],
    gold_by_id: dict[str, frozenset[str]],
) -> dict[str, dict[str, float]]:
    """설정별로 표현(style)마다, 그리고 전체의 정답 분야 기준 P@k 평균을 계산한다."""
    summary: dict[str, dict[str, float]] = {}
    for setting, by_query in rankings.items():
        per_style: dict[str, list[float]] = {}
        for qid, top in by_query.items():
            value = gold_precision([cid for cid, _ in top], gold_by_query[qid], gold_by_id)
            per_style.setdefault(style_by_query[qid], []).append(value)
        summary[setting] = {style: sum(v) / len(v) for style, v in per_style.items()}
        summary[setting]["all"] = sum(sum(v) for v in per_style.values()) / sum(len(v) for v in per_style.values())
    return summary


def judge_config_hash(model: str, system_prompt: str) -> str:
    """판정 모델과 판정 프롬프트의 해시다. 이 값이 바뀌면 저장된 판정은 새 조건의 결과가 아니므로 다시 판정한다."""
    import hashlib

    return hashlib.sha256(f"{model}\x1f{system_prompt}".encode("utf-8")).hexdigest()[:16]


def pending_pairs(
    pairs: list[tuple[str, str]], saved: dict[str, dict], text_hash: dict[tuple[str, str], str], config_hash: str
) -> list[tuple[str, str]]:
    """저장된 판정이 없거나 텍스트·판정 조건(모델·프롬프트)이 달라진 쌍만 골라 다시 판정 대상으로 돌려준다."""
    return [
        pair
        for pair in pairs
        if saved.get(f"{pair[0]}::{pair[1]}", {}).get("hash") != text_hash[pair]
        or saved.get(f"{pair[0]}::{pair[1]}", {}).get("judge") != config_hash
    ]


def build_judged_pool(
    pairs: list[tuple[str, str]], saved: dict[str, dict]
) -> tuple[dict[tuple[str, str], int], dict[str, list[int]]]:
    """현재 후보 쌍(pairs)의 판정만 모아 (쌍별 점수, 쿼리별 점수 목록)을 만든다. 저장 파일에 남은 과거 후보의 판정은 nDCG 풀에서 뺀다."""
    judged = {pair: saved[f"{pair[0]}::{pair[1]}"]["score"] for pair in pairs}
    pool: dict[str, list[int]] = {}
    for (qid, _cid), score in judged.items():
        pool.setdefault(qid, []).append(score)
    return judged, pool


def setting_name(method: str, key: str, variant: str) -> str:
    """설정 이름이다. 기본(plain) 조건이면 접미사 없이 M2_bge-m3처럼, 프롬프트 조건이면 +variant를 붙인다."""
    return f"{method}_{key}" if variant == "plain" else f"{method}_{key}+{variant}"


def variants_for(key: str) -> list[str]:
    """모델이 지원하는 쿼리 인코딩 조건이다. 쿼리 프롬프트가 등록된 모델(Qwen3)만 셋이고 나머지는 plain뿐이다."""
    if LOCAL_EMBEDDING_MODELS.get(key, {}).get("query_prompt_name"):
        return ["plain", "prompt-query", "prompt-taste"]
    return ["plain"]


def _encode_queries(key: str, variant: str, texts: list[str]) -> np.ndarray:
    """쿼리 텍스트를 조건에 맞게 인코딩하고 캐시한다. 텍스트·모델 설정·조건이 바뀌면 다시 계산한다."""
    from src.pipeline import _cached_hash_matches, _content_hash, _embedding_client, _embedding_model_identity, _load_vectors, _save_vectors

    input_hash = _content_hash(_embedding_model_identity(key), variant, TASTE_QWEN3_INSTRUCT, *texts)
    path = CACHE_DIR / f"taste_queries_{key}_{variant}.npz"
    hash_path = CACHE_DIR / f"taste_queries_{key}_{variant}.input_hash"
    if path.exists() and _cached_hash_matches(hash_path, input_hash):
        return _load_vectors(path)[1]
    client = _embedding_client(key)
    if variant == "plain":
        vectors = client.embed(texts)
    elif variant == "prompt-query":
        vectors = client.embed(texts, prompt_name=LOCAL_EMBEDDING_MODELS[key]["query_prompt_name"])
    else:
        vectors = client.embed(texts, prompt=TASTE_QWEN3_INSTRUCT)
    _save_vectors(path, [str(i) for i in range(len(texts))], vectors)
    hash_path.write_text(input_hash, encoding="utf-8")
    return vectors


def _query_llm_tags(texts: list[str], llm_model: str, category_codes: list[str]) -> list[frozenset[str]]:
    """취향 쿼리를 M4용 LLM 태거로 분류한다. 입력·프롬프트·모델이 같으면 캐시를 쓴다."""
    from src.clients import OpenAITagger
    from src.clients.openai_tagger import SYSTEM_PROMPT
    from src.config import LLM_TEMPERATURE
    from src.pipeline import _cached_hash_matches, _content_hash

    input_hash = _content_hash(llm_model, SYSTEM_PROMPT, str(LLM_TEMPERATURE), *texts, *category_codes)
    path = CACHE_DIR / "taste_llm_tags.json"
    hash_path = CACHE_DIR / "taste_llm_tags.input_hash"
    if path.exists() and _cached_hash_matches(hash_path, input_hash):
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        tagger = OpenAITagger(model=llm_model, category_codes=category_codes)
        raw = [list(tagger.tag(text).tags) for text in texts]
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        hash_path.write_text(input_hash, encoding="utf-8")
    return [frozenset(tags) for tags in raw]


def cmd_taste_eval(_: argparse.Namespace) -> None:
    """취향 쿼리 90개로 설정별 상위 5 추천을 만들고 정답 분야 기준 P@5를 출력한다 (LLM 판정 없음).

    M5는 bge-m3 M2 상위 20명을 리랭커로 쿼리 텍스트와 다시 채점해 재정렬한다. 결과는
    results/taste_candidates.json, 요약은 results/taste_gold_summary.json에 저장한다.
    """
    from src.clients import RerankerClient
    from src.data import load_categories, load_creators, load_taste_queries
    from src.pipeline import EMBEDDING_MODEL_KEYS, _load_vectors, _rerank_bge_m2_candidates_for_queries, _zero_shot_tags_for_key
    from src.tagging import rank_all

    categories = load_categories()
    codes = [c.code for c in categories]
    creators = load_creators()
    queries = load_taste_queries()
    creator_ids = [c.id for c in creators]
    text_by_id = {c.id: c.input_text() for c in creators}
    gold_by_id = {c.id: frozenset(c.gold) for c in creators}
    query_texts = [q.text for q in queries]

    params = json.loads((RESULTS_DIR / "selected_params.json").read_text(encoding="utf-8"))
    llm_model = params["llm_model"]
    llm_creator_tags = {
        cid: frozenset(tags)
        for cid, tags in json.loads((CACHE_DIR / f"llm_tags_{llm_model.replace('/', '_')}_run0.json").read_text(encoding="utf-8")).items()
    }
    query_llm_tags = _query_llm_tags(query_texts, llm_model, codes)

    rankings: dict[str, dict[str, list[list]]] = {}
    for key in EMBEDDING_MODEL_KEYS:
        p = params["per_embedding"][key]
        ids, creator_vectors = _load_vectors(CACHE_DIR / f"creators_{key}.npz")
        assert list(ids) == creator_ids
        _, category_vectors = _load_vectors(CACHE_DIR / f"categories_{key}.npz")
        _, _, zero_shot_tags, _ = _zero_shot_tags_for_key(key, creators, categories, tau_candidates=[p["tau"]])
        creator_zero_shot = [zero_shot_tags[cid] for cid in ids]
        creator_llm = [llm_creator_tags[cid] for cid in ids]

        plain_vectors = _encode_queries(key, "plain", query_texts)
        # 쿼리의 zero-shot 태그는 임베딩 조건과 무관하게 plain 벡터로 붙인다(tau를 plain 임베딩에서 골랐기 때문)
        query_zero_shot = [r.assigned(p["tau"], LLM_TAG_MAX) for r in rank_all(plain_vectors, category_vectors, codes)]
        for variant in variants_for(key):
            vectors = plain_vectors if variant == "plain" else _encode_queries(key, variant, query_texts)
            for method, q_tags, c_tags, bonus in (
                ("M2", query_zero_shot, creator_zero_shot, 0.0),
                ("M3", query_zero_shot, creator_zero_shot, p["bonus_m3"]),
                ("M4", query_llm_tags, creator_llm, p["bonus_m4"]),
            ):
                matrix = score_matrix(vectors, creator_vectors, q_tags, c_tags, bonus)
                rankings[setting_name(method, key, variant)] = {
                    q.id: [list(item) for item in top_k(matrix[i], creator_ids, TASTE_TOP_K)] for i, q in enumerate(queries)
                }
            if key == "bge-m3" and variant == "plain":
                cosine = plain_vectors @ creator_vectors.T
                reranked = _rerank_bge_m2_candidates_for_queries(
                    [q.id for q in queries], query_texts, cosine, creator_ids, text_by_id, RerankerClient(), TOP_N_STORED
                )
                rankings[setting_name("M5", key, variant)] = {qid: items[:TASTE_TOP_K] for qid, items in reranked.items()}
        print(f"[taste-eval] {key} 완료")

    with (RESULTS_DIR / "taste_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(rankings, f, ensure_ascii=False, indent=2)
    summary = summarize_gold(rankings, {q.id: q.style for q in queries}, {q.id: q.gold for q in queries}, gold_by_id)
    with (RESULTS_DIR / "taste_gold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== 취향 쿼리 정답 분야 기준 P@5 (쿼리 표현별 / 전체) ===")
    for setting, row in summary.items():
        print(f"   {setting}: keyword={row['keyword']:.3f} sentence={row['sentence']:.3f} long={row['long']:.3f} 전체={row['all']:.3f}")


def cmd_taste_judge(_: argparse.Namespace) -> None:
    """문장형 취향 쿼리의 상위 5 후보 합집합을 LLM으로 판정하고 설정별 관련도·nDCG를 출력한다 (API 비용 발생).

    판정은 results/taste_judgments.json에 쌍마다 저장해 중단·재개가 가능하며, 쿼리·크리에이터 텍스트가
    바뀌면(해시 불일치) 다시 판정한다.
    """
    from src.clients import OpenAIJudge
    from src.clients.openai_judge import TASTE_SYSTEM_PROMPT
    from src.data import load_creators, load_taste_queries
    from src.judge import pair_text_hash
    from src.metrics import bootstrap_ci, mean_relevance_at_k, ndcg_at_k

    creators = {c.id: c for c in load_creators()}
    queries = {q.id: q for q in load_taste_queries() if q.style in TASTE_JUDGE_STYLES}
    rankings = json.loads((RESULTS_DIR / "taste_candidates.json").read_text(encoding="utf-8"))

    pairs = sorted({(qid, cid) for by_query in rankings.values() for qid, top in by_query.items() if qid in queries for cid, _ in top})
    path = RESULTS_DIR / "taste_judgments.json"
    saved: dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    hash_of = {(q, c): pair_text_hash(queries[q].text, creators[c].input_text()) for q, c in pairs}
    config_hash = judge_config_hash(OPENAI_JUDGE_MODEL, TASTE_SYSTEM_PROMPT)
    pending = pending_pairs(pairs, saved, hash_of, config_hash)
    print(f"[taste-judge] 판정 대상 {len(pairs)}쌍 중 {len(pairs) - len(pending)}쌍 완료, {len(pending)}쌍 판정 시작 (모델: {OPENAI_JUDGE_MODEL})")

    judge = OpenAIJudge(model=OPENAI_JUDGE_MODEL, system_prompt=TASTE_SYSTEM_PROMPT)
    in_tokens = out_tokens = 0
    for i, (qid, cid) in enumerate(pending, start=1):
        result = judge.judge(queries[qid].text, creators[cid].input_text())
        saved[f"{qid}::{cid}"] = {"score": result.score, "hash": hash_of[(qid, cid)], "judge": config_hash}
        in_tokens += result.input_tokens
        out_tokens += result.output_tokens
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        if i % 50 == 0:
            print(f"[taste-judge] {i}/{len(pending)} 완료")
    cost = (in_tokens * OPENAI_JUDGE_MODEL_PRICE["input_price"] + out_tokens * OPENAI_JUDGE_MODEL_PRICE["output_price"]) / 1_000_000
    if pending:
        print(f"[taste-judge] 완료. 이번 실행 비용 약 ${cost:.4f}")

    judged, pool = build_judged_pool(pairs, saved)
    print(f"\n=== 취향 쿼리(문장형 {len(queries)}개) 설정별 관련도@5 / nDCG@5 ===")
    results = {}
    for setting, by_query in rankings.items():
        rel, ndcg = [], []
        for qid in queries:
            ranked = [cid for cid, _ in by_query[qid]]
            rel.append(mean_relevance_at_k(ranked, qid, judged))
            ndcg.append(ndcg_at_k(ranked, qid, pool.get(qid, []), judged, TASTE_TOP_K))
        lo, hi = bootstrap_ci(rel)
        results[setting] = {"relevance": sum(rel) / len(rel), "ci": [lo, hi], "ndcg": sum(ndcg) / len(ndcg)}
        print(f"   {setting}: 관련도={results[setting]['relevance']:.3f} [{lo:.3f}, {hi:.3f}]  nDCG={results[setting]['ndcg']:.3f}")
    with (RESULTS_DIR / "taste_judge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
