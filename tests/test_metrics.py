import pytest

from src.metrics import (
    bootstrap_ci,
    check_case,
    irrelevant_rate_at_k,
    mean_relevance_at_k,
    ndcg_at_k,
    paired_win_counts,
)


def test_mean_relevance_and_irrelevant_rate() -> None:
    judged = {("q1", "a"): 2, ("q1", "b"): 0, ("q1", "c"): 1}

    assert mean_relevance_at_k(["a", "b", "c"], "q1", judged) == 1.0
    assert irrelevant_rate_at_k(["a", "b", "c"], "q1", judged) == 1 / 3


def test_mean_relevance_unjudged_candidate_counts_as_zero() -> None:
    judged = {("q1", "a"): 2}

    assert mean_relevance_at_k(["a", "unseen"], "q1", judged) == 1.0


def test_ndcg_perfect_order_is_one() -> None:
    judged = {("q1", "a"): 2, ("q1", "b"): 1, ("q1", "c"): 0}
    pool = [2, 1, 0]

    assert ndcg_at_k(["a", "b", "c"], "q1", pool, judged, k=3) == 1.0


def test_ndcg_reversed_order_is_less_than_one() -> None:
    judged = {("q1", "a"): 2, ("q1", "b"): 1, ("q1", "c"): 0}
    pool = [2, 1, 0]

    assert ndcg_at_k(["c", "b", "a"], "q1", pool, judged, k=3) < 1.0


def test_paired_win_counts() -> None:
    a = {"q1": 2.0, "q2": 1.0, "q3": 0.5}
    b = {"q1": 1.0, "q2": 1.0, "q3": 1.0}

    assert paired_win_counts(a, b) == (1, 1, 1)


def test_bootstrap_ci_tight_for_constant_values() -> None:
    lower, upper = bootstrap_ci([1.0] * 20, n_boot=200)

    assert lower == upper == 1.0


def test_bootstrap_ci_widens_with_variance() -> None:
    tight_lower, tight_upper = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_boot=500)
    wide_lower, wide_upper = bootstrap_ci([0.0, 1.0, 0.0, 2.0], n_boot=500)

    assert (wide_upper - wide_lower) > (tight_upper - tight_lower)


def test_check_case_include_and_exclude() -> None:
    assert check_case(["a", "b"], must_include={"b", "c"}) is True
    assert check_case(["a", "b"], must_include={"c"}) is False
    assert check_case(["a", "b"], must_exclude={"c"}) is True
    assert check_case(["a", "b"], must_exclude={"a"}) is False


def test_cutoff_effect_counts_removed_unrelated_kept_related_and_empty_queries() -> None:
    from src.metrics import cutoff_effect

    gold = {"q1": frozenset({"X"}), "q2": frozenset({"Y"}), "a": frozenset({"X"}), "b": frozenset({"Z"}), "c": frozenset({"Z"})}
    top_lists = {"q1": [["a", 0.9], ["b", 0.3]], "q2": [["b", 0.3], ["c", 0.2]]}

    effect = cutoff_effect(top_lists, 0.5, gold, ["q1", "q2"], k=2)

    assert effect["unrelated_removed_rate"] == pytest.approx(1.0)  # b·b·c 3쌍 모두 제거
    assert effect["related_kept_rate"] == pytest.approx(1.0)  # a 유지
    assert effect["empty_result_rate"] == pytest.approx(0.5)  # q2는 전부 걸러져 빈 결과


def test_cutoff_effect_returns_none_when_no_pairs_of_kind() -> None:
    from src.metrics import cutoff_effect

    gold = {"q": frozenset({"X"}), "a": frozenset({"X"})}

    effect = cutoff_effect({"q": [["a", 0.9]]}, 0.5, gold, ["q"], k=1)

    assert effect["unrelated_removed_rate"] is None
