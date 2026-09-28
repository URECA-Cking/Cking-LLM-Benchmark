from src.spot_check import agreement_stats, select_disagreement_pairs


def _candidates() -> dict[str, dict[str, list[list]]]:
    return {
        "M4": {
            "q1": [["a", 0.9], ["b", 0.8], ["c", 0.7]],
            "q2": [["x", 0.9]],
        },
        "M3": {
            "q1": [["b", 0.8], ["d", 0.7], ["e", 0.6]],
            "q2": [["x", 0.9]],
        },
        "R2": {
            "q1": [["c", 0.7], ["d", 0.7], ["f", 0.5]],
            "q2": [["y", 0.5]],
        },
    }


def test_select_disagreement_pairs_finds_target_only_picks() -> None:
    pairs = select_disagreement_pairs(_candidates(), ["q1", "q2"], "M4", ["M3", "R2"], k=3)

    # q1: M4=abc, baseline(M3∪R2)=bcdef -> M4만 가진 건 a
    # q2: M4=x, baseline=xy -> M4만 가진 건 없음
    assert pairs == [("q1", "a")]


def test_select_disagreement_pairs_deduplicates_across_queries() -> None:
    candidates = {
        "M4": {"q1": [["a", 1.0]], "q2": [["a", 1.0]]},
        "M3": {"q1": [["z", 1.0]], "q2": [["z", 1.0]]},
    }
    pairs = select_disagreement_pairs(candidates, ["q1", "q2"], "M4", ["M3"], k=1)

    assert pairs == [("q1", "a"), ("q2", "a")]  # 같은 후보 id라도 쿼리가 다르면 별개 쌍


def test_agreement_stats_perfect_match() -> None:
    human = {("q1", "a"): 2, ("q1", "b"): 0}
    auto = {("q1", "a"): 2, ("q1", "b"): 0}

    stats = agreement_stats(human, auto)

    assert stats["count"] == 2
    assert stats["exact_match_rate"] == 1.0
    assert stats["within_1_rate"] == 1.0
    assert stats["mean_abs_diff"] == 0.0


def test_agreement_stats_partial_match_and_ignores_unmatched_keys() -> None:
    human = {("q1", "a"): 2, ("q1", "b"): 0, ("q1", "unscored_in_auto"): 1}
    auto = {("q1", "a"): 1, ("q1", "b"): 0}

    stats = agreement_stats(human, auto)

    assert stats["count"] == 2  # unscored_in_auto는 비교 대상에서 빠짐
    assert stats["exact_match_rate"] == 0.5
    assert stats["within_1_rate"] == 1.0
    assert stats["mean_abs_diff"] == 0.5


def test_agreement_stats_empty_returns_zeros() -> None:
    stats = agreement_stats({}, {})

    assert stats == {"count": 0, "exact_match_rate": 0.0, "within_1_rate": 0.0, "mean_abs_diff": 0.0}
