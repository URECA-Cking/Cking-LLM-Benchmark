import numpy as np
import pytest

from src.tagging import (
    RankedTags,
    confusion_pairs,
    consistency_rate,
    f1_against_gold,
    llm_hit_rate,
    llm_mean_f1,
    pick_llm_model,
    rank_categories,
    select_tau,
    top1_accuracy,
    top3_inclusion_rate,
    unclassified_rate,
)


def test_rank_categories_orders_by_cosine_descending() -> None:
    creator = np.array([1.0, 0.0], dtype=np.float32)
    categories = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32)
    ranked = rank_categories(creator, categories, ["B", "A", "C"])

    assert ranked.codes_by_rank[0] == "A"
    assert ranked.scores_by_rank[0] == 1.0
    assert ranked.codes_by_rank[-1] == "B"


def test_ranked_tags_assigned_respects_tau_and_max_tags() -> None:
    ranked = RankedTags(codes_by_rank=("A", "B", "C"), scores_by_rank=(0.9, 0.5, 0.4))

    assert ranked.assigned(tau=0.5, max_tags=3) == frozenset({"A", "B"})
    assert ranked.assigned(tau=0.5, max_tags=1) == frozenset({"A"})
    assert ranked.assigned(tau=0.95, max_tags=3) == frozenset()


def test_f1_against_gold_handles_empty_sets() -> None:
    assert f1_against_gold(frozenset(), frozenset()) == 1.0
    assert f1_against_gold(frozenset({"A"}), frozenset()) == 0.0
    assert f1_against_gold(frozenset({"A", "B"}), frozenset({"A"})) == pytest.approx(2 / 3)


def test_select_tau_maximizes_mean_f1_on_dev() -> None:
    ranked = {
        "c1": RankedTags(codes_by_rank=("A", "B"), scores_by_rank=(0.9, 0.3)),
        "c2": RankedTags(codes_by_rank=("B", "A"), scores_by_rank=(0.8, 0.2)),
    }
    gold = {"c1": frozenset({"A"}), "c2": frozenset({"B"})}

    tau = select_tau(ranked, gold, max_tags=3)

    assert 0.3 < tau <= 0.8


def test_top1_and_top3_metrics() -> None:
    ranked = {
        "c1": RankedTags(codes_by_rank=("A", "B", "C"), scores_by_rank=(0.9, 0.5, 0.1)),
        "c2": RankedTags(codes_by_rank=("B", "C", "A"), scores_by_rank=(0.9, 0.5, 0.1)),
    }
    gold = {"c1": frozenset({"A"}), "c2": frozenset({"A"})}

    assert top1_accuracy(ranked, gold) == 0.5
    assert top3_inclusion_rate(ranked, gold) == 1.0


def test_confusion_pairs_only_counts_misses() -> None:
    ranked = {
        "c1": RankedTags(codes_by_rank=("GAME", "FOOD"), scores_by_rank=(0.9, 0.1)),
        "c2": RankedTags(codes_by_rank=("FOOD", "GAME"), scores_by_rank=(0.9, 0.1)),
    }
    gold = {"c1": frozenset({"FOOD"}), "c2": frozenset({"FOOD"})}

    pairs = confusion_pairs(ranked, gold)

    assert pairs == {("FOOD", "GAME"): 1}


def test_llm_hit_rate_and_f1() -> None:
    tags = {"c1": frozenset({"FOOD", "GAME"}), "c2": frozenset()}
    gold = {"c1": frozenset({"FOOD"}), "c2": frozenset({"PET"})}

    assert llm_hit_rate(tags, gold) == 0.5
    assert 0.0 < llm_mean_f1(tags, gold) < 1.0


def test_unclassified_rate() -> None:
    assert unclassified_rate([frozenset({"A"}), frozenset(), frozenset()]) == pytest.approx(2 / 3)


def test_pick_llm_model_prefers_higher_f1_then_hit_rate() -> None:
    metrics = {
        "model-a": {"f1": 0.8, "hit_rate": 0.9},
        "model-b": {"f1": 0.9, "hit_rate": 0.5},
    }
    assert pick_llm_model(metrics) == "model-b"

    tie_broken_by_hit_rate = {
        "model-a": {"f1": 0.8, "hit_rate": 0.9},
        "model-b": {"f1": 0.8, "hit_rate": 0.95},
    }
    assert pick_llm_model(tie_broken_by_hit_rate) == "model-b"


def test_consistency_rate_perfect_and_partial() -> None:
    run1 = {"c1": frozenset({"A", "B"}), "c2": frozenset()}
    run2 = {"c1": frozenset({"A"}), "c2": frozenset()}

    assert consistency_rate(run1, run1) == 1.0
    assert consistency_rate(run1, run2) == pytest.approx((0.5 + 1.0) / 2)
