"""Verify explicit topics change ranks without increasing the bonus ceiling."""

import numpy as np

from src.selected_subtopic_eval import rank_selected


def test_explicit_topic_uses_equal_bonus_budget_and_deterministic_ties():
    ids = ["a", "b", "c", "d", "e", "f"]
    vectors = np.array(
        [[0.8, 0.6], [0.75, 0.66], [0.7, 0.7], [0.7, 0.7], [0.6, 0.8], [0.5, 0.86]]
    )
    parents = {cid: {"FITNESS"} for cid in ids}
    topics = {cid: {"RUN"} if cid == "b" else set() for cid in ids}
    rows = rank_selected(
        "FITNESS", "RUN", ids, vectors, np.array([1.0, 0.0]), parents, topics
    )
    assert rows["parent_only"][0]["id"] == "a"
    assert rows["selected_subtopic"][0]["id"] == "b"
    assert rows["parent_only"][1]["score"] == rows["selected_subtopic"][0]["score"]
    assert [r["id"] for r in rows["selected_subtopic"]][2:4] == ["c", "d"]


def test_topic_without_parent_does_not_receive_bonus():
    ids = ["a"]
    rows = rank_selected(
        "FITNESS",
        "RUN",
        ids,
        np.array([[1.0, 0.0]]),
        np.array([1.0, 0.0]),
        {"a": set()},
        {"a": {"RUN"}},
    )
    assert rows["parent_only"] == rows["selected_subtopic"]
