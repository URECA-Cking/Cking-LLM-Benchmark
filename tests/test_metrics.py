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
