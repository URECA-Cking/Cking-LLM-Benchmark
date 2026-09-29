"""dev 규모에 따라 tau·bonus 선택이 얼마나 흔들리는지 재는 민감도 실험의 순수 계산부다 (이슈 #8).

현재 파이프라인은 dev 30명으로 tau·bonus를 고른다. 여기서는 더 큰 dev 후보 풀(기존 dev 30명 +
합성 300명)에서 크기별(30/60/120/240)로 무작위 부분표본을 여러 번 뽑아 같은 절차로 tau·bonus를
고르고, 고른 값이 얼마나 일정한지와 기존 test에서의 성적이 얼마나 달라지는지를 본다. test 성적은
LLM 판정 없이 gold 라벨로 계산하는 대리 지표(태그 F1, 상위 5 분야 일치율)다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, pstdev

import numpy as np

from src.similarity import cosine_matrix, cosine_with_tag_bonus, precision_at_k_by_gold, select_bonus
from src.tagging import RankedTags, f1_against_gold, select_tau


@dataclass(frozen=True)
class CreatorSet:
    """한 임베딩 모델 기준으로 준비한 크리에이터 묶음이다. 순서는 ids·vectors·ranked가 모두 같다."""

    ids: list[str]
    vectors: np.ndarray
    ranked: list[RankedTags]
    gold: dict[str, frozenset[str]]
    llm_tags: dict[str, frozenset[str]]
    declared: dict[str, frozenset[str]]
    # 쿼리 프롬프트를 적용한 쿼리 쪽(행) 벡터. None이면 vectors를 그대로 쓴다(프롬프트 없는 모델)
    query_vectors: np.ndarray | None = None


def select_on_subset(
    pool: CreatorSet, subset: list[int], bonus_grid: list[float], max_tags: int
) -> tuple[float, dict[str, float]]:
    """pool의 subset(인덱스)만 dev로 보고 select-params와 같은 절차로 tau와 세 bonus를 고른다."""
    sub_ids = [pool.ids[i] for i in subset]
    gold = {cid: pool.gold[cid] for cid in sub_ids}
    ranked_by_id = {pool.ids[i]: pool.ranked[i] for i in subset}
    tau = select_tau(ranked_by_id, gold, max_tags)

    query_rows = None if pool.query_vectors is None else pool.query_vectors[subset]
    cosine = cosine_matrix(pool.vectors[subset], query_vectors=query_rows)
    zero_shot_sets = [ranked_by_id[cid].assigned(tau, max_tags) for cid in sub_ids]
    llm_sets = [pool.llm_tags[cid] for cid in sub_ids]
    declared_sets = [pool.declared[cid] for cid in sub_ids]
    bonuses = {
        "bonus_m3": select_bonus(cosine, sub_ids, zero_shot_sets, gold, sub_ids, bonus_grid),
        "bonus_m4": select_bonus(cosine, sub_ids, llm_sets, gold, sub_ids, bonus_grid),
        "bonus_r2": select_bonus(cosine, sub_ids, declared_sets, gold, sub_ids, bonus_grid),
    }
    return tau, bonuses


def evaluate_on_test(
    test: CreatorSet, test_ids: list[str], query_ids: list[str], tau: float, bonuses: dict[str, float], max_tags: int, k: int = 5
) -> dict[str, float]:
    """고른 tau·bonus를 기존 100명(후보 풀)에 적용해 test 성적을 gold 기준 대리 지표로 계산한다."""
    index = {cid: i for i, cid in enumerate(test.ids)}
    zero_shot_sets = [ranked.assigned(tau, max_tags) for ranked in test.ranked]
    tag_f1 = mean(f1_against_gold(zero_shot_sets[index[cid]], test.gold[cid]) for cid in test_ids)

    cosine = cosine_matrix(test.vectors, query_vectors=test.query_vectors)
    sources = {
        "bonus_m3": zero_shot_sets,
        "bonus_m4": [test.llm_tags[cid] for cid in test.ids],
        "bonus_r2": [test.declared[cid] for cid in test.ids],
    }
    result = {"test_tag_f1": tag_f1}
    for name, tag_sets in sources.items():
        combined = cosine_with_tag_bonus(cosine, tag_sets, bonuses[name])
        result[f"p5_{name.removeprefix('bonus_')}"] = precision_at_k_by_gold(combined, test.ids, test.gold, query_ids, k)
    return result


def run_sensitivity(
    pool: CreatorSet,
    test: CreatorSet,
    test_ids: list[str],
    query_ids: list[str],
    sizes: list[int],
    reps: int,
    bonus_grid: list[float],
    max_tags: int,
    seed: int,
) -> list[dict]:
    """크기별로 reps번 무작위 부분표본을 뽑아 선택·평가하고 기록을 돌려준다. 같은 seed면 같은 결과다."""
    records = []
    for size in sizes:
        if size > len(pool.ids):
            raise ValueError(f"dev 후보 풀({len(pool.ids)}명)보다 큰 크기({size})는 뽑을 수 없습니다.")
        for rep in range(reps):
            rng = np.random.default_rng([seed, size, rep])
            subset = sorted(rng.choice(len(pool.ids), size=size, replace=False).tolist())
            tau, bonuses = select_on_subset(pool, subset, bonus_grid, max_tags)
            records.append(
                {"size": size, "rep": rep, "tau": tau, **bonuses, **evaluate_on_test(test, test_ids, query_ids, tau, bonuses, max_tags)}
            )
    return records


def summarize(records: list[dict]) -> list[dict]:
    """크기별로 tau 분포, bonus 최빈값·일치율, test 지표 평균·표준편차를 요약한다."""
    rows = []
    for size in sorted({r["size"] for r in records}):
        group = [r for r in records if r["size"] == size]
        row = {"size": size, "reps": len(group)}
        taus = [r["tau"] for r in group]
        row.update(tau_mean=mean(taus), tau_std=pstdev(taus), tau_min=min(taus), tau_max=max(taus))
        for name in ("bonus_m3", "bonus_m4", "bonus_r2"):
            counts = Counter(r[name] for r in group)
            mode, mode_count = counts.most_common(1)[0]
            row[name] = {"mode": mode, "mode_share": mode_count / len(group), "distribution": dict(sorted(counts.items()))}
        for metric in ("test_tag_f1", "p5_m3", "p5_m4", "p5_r2"):
            values = [r[metric] for r in group]
            row[metric] = {"mean": mean(values), "std": pstdev(values)}
        rows.append(row)
    return rows
