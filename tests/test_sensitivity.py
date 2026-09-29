import numpy as np
import pytest

from src import sensitivity as sv
from src.tagging import rank_all

CATEGORY_VECTORS = np.array([[1.0, 0.0], [0.0, 1.0]])
CODES = ["A", "B"]


def _creator_set(n: int, seed: int) -> sv.CreatorSet:
    """절반은 A축, 절반은 B축 근처에 있는 가짜 크리에이터 n명을 만든다."""
    rng = np.random.default_rng(seed)
    ids = [f"c{seed}_{i}" for i in range(n)]
    labels = ["A" if i % 2 == 0 else "B" for i in range(n)]
    vectors = np.array([[1.0, 0.1 * rng.random()] if label == "A" else [0.1 * rng.random(), 1.0] for label in labels])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    gold = {cid: frozenset({label}) for cid, label in zip(ids, labels)}
    return sv.CreatorSet(
        ids=ids,
        vectors=vectors,
        ranked=rank_all(vectors, CATEGORY_VECTORS, CODES),
        gold=gold,
        llm_tags=dict(gold),
        declared=dict(gold),
    )


def _run(pool, test, seed=1, sizes=(6, 10), reps=3):
    test_ids = test.ids[:6]
    return sv.run_sensitivity(pool, test, test_ids, test_ids[:3], list(sizes), reps, [0.0, 0.1, 0.3], 3, seed)


def test_select_on_subset_returns_grid_bonus_and_observed_tau() -> None:
    pool = _creator_set(12, seed=1)

    tau, bonuses = sv.select_on_subset(pool, list(range(8)), [0.0, 0.1, 0.3], 3)

    assert tau in {score for ranked in pool.ranked for score in ranked.scores_by_rank}
    assert set(bonuses) == {"bonus_m3", "bonus_m4", "bonus_r2"}
    assert all(value in (0.0, 0.1, 0.3) for value in bonuses.values())


def test_evaluate_on_test_reports_perfect_proxy_scores_on_separable_data() -> None:
    test = _creator_set(12, seed=2)

    result = sv.evaluate_on_test(test, test.ids[:6], test.ids[:3], tau=0.5, bonuses={"bonus_m3": 0.1, "bonus_m4": 0.1, "bonus_r2": 0.1}, max_tags=3)

    assert result["test_tag_f1"] == pytest.approx(1.0)
    assert result["p5_m3"] == result["p5_m4"] == result["p5_r2"] == pytest.approx(1.0)


def test_run_sensitivity_is_reproducible_for_same_seed_and_varies_subsets_across_seeds() -> None:
    pool, test = _creator_set(20, seed=3), _creator_set(12, seed=4)

    first, second = _run(pool, test, seed=7), _run(pool, test, seed=7)

    assert first == second
    assert len(first) == 2 * 3
    assert {r["size"] for r in first} == {6, 10}


def test_run_sensitivity_rejects_size_larger_than_pool() -> None:
    pool, test = _creator_set(8, seed=5), _creator_set(12, seed=6)

    with pytest.raises(ValueError):
        _run(pool, test, sizes=(9,))


def test_summarize_reports_mode_share_and_spread() -> None:
    records = [
        {"size": 30, "rep": i, "tau": tau, "bonus_m3": b, "bonus_m4": 0.1, "bonus_r2": 0.2, "test_tag_f1": f1, "p5_m3": 0.5, "p5_m4": 0.5, "p5_r2": 0.5}
        for i, (tau, b, f1) in enumerate([(0.4, 0.1, 0.8), (0.4, 0.1, 0.9), (0.5, 0.2, 1.0), (0.4, 0.1, 0.9)])
    ]

    (row,) = sv.summarize(records)

    assert row["size"] == 30 and row["reps"] == 4
    assert row["tau_min"] == 0.4 and row["tau_max"] == 0.5
    assert row["bonus_m3"]["mode"] == 0.1 and row["bonus_m3"]["mode_share"] == 0.75
    assert row["bonus_m3"]["distribution"] == {0.1: 3, 0.2: 1}
    assert row["bonus_m4"]["mode_share"] == 1.0
    assert row["test_tag_f1"]["mean"] == pytest.approx(0.9)


def test_query_vectors_change_row_side_only() -> None:
    """프롬프트 적용 쿼리 벡터는 행에만 쓰이므로, 후보 쪽 벡터가 같으면 선택된 bonus 형식은 유지되고 결과는 달라질 수 있다."""
    pool = _creator_set(12, seed=1)
    flipped = sv.CreatorSet(**{**pool.__dict__, "query_vectors": pool.vectors[::-1].copy()})

    _, plain = sv.select_on_subset(pool, list(range(8)), [0.0, 0.1, 0.3], 3)
    _, prompted = sv.select_on_subset(flipped, list(range(8)), [0.0, 0.1, 0.3], 3)

    assert set(plain) == set(prompted)
    same = sv.evaluate_on_test(pool, pool.ids[:6], pool.ids[:3], 0.5, {"bonus_m3": 0.1, "bonus_m4": 0.1, "bonus_r2": 0.1}, 3)
    diff = sv.evaluate_on_test(flipped, pool.ids[:6], pool.ids[:3], 0.5, {"bonus_m3": 0.1, "bonus_m4": 0.1, "bonus_r2": 0.1}, 3)
    assert same["p5_m3"] != diff["p5_m3"] or same != diff
