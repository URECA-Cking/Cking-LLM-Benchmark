import numpy as np
import pytest

from src import pipeline, taste_eval as te


def test_score_matrix_adds_bonus_only_for_shared_tags() -> None:
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    c = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    plain = te.score_matrix(q, c, [frozenset({"A"})], [frozenset({"A"}), frozenset({"B"})], 0.0)
    boosted = te.score_matrix(q, c, [frozenset({"A"})], [frozenset({"A"}), frozenset({"B"})], 0.3)

    assert plain.tolist() == [[0.0, 1.0]]
    assert boosted[0, 0] == pytest.approx(0.3) and boosted[0, 1] == pytest.approx(1.0)  # 태그가 겹친 첫 후보만 +0.3


def test_top_k_orders_by_score_then_id() -> None:
    row = np.array([0.5, 0.9, 0.5], dtype=np.float32)

    assert [cid for cid, _ in te.top_k(row, ["b", "a", "c"], 3)] == ["a", "b", "c"]
    assert len(te.top_k(row, ["b", "a", "c"], 2)) == 2


def test_gold_precision_counts_shared_gold_fraction() -> None:
    gold = {"a": frozenset({"X"}), "b": frozenset({"Y"}), "c": frozenset({"X", "Y"})}

    assert te.gold_precision(["a", "b", "c"], ("X",), gold) == pytest.approx(2 / 3)
    assert te.gold_precision([], ("X",), gold) == 0.0


def test_summarize_gold_averages_per_style_and_overall() -> None:
    gold_by_id = {"a": frozenset({"X"}), "b": frozenset({"Y"})}
    rankings = {"S": {"q1": [["a", 1.0]], "q2": [["b", 1.0]], "q3": [["a", 1.0]]}}

    summary = te.summarize_gold(
        rankings, {"q1": "keyword", "q2": "keyword", "q3": "long"}, {"q1": ("X",), "q2": ("X",), "q3": ("X",)}, gold_by_id
    )["S"]

    assert summary["keyword"] == pytest.approx(0.5) and summary["long"] == pytest.approx(1.0) and summary["all"] == pytest.approx(2 / 3)


def test_setting_name_and_variants(monkeypatch) -> None:
    monkeypatch.setattr(te, "LOCAL_EMBEDDING_MODELS", {"q": {"query_prompt_name": "query"}, "b": {}})

    assert te.setting_name("M2", "b", "plain") == "M2_b" and te.setting_name("M2", "q", "prompt-taste") == "M2_q+prompt-taste"
    assert te.variants_for("q") == ["plain", "prompt-query", "prompt-taste"]
    assert te.variants_for("b") == ["plain"] and te.variants_for("text-embedding-3-small") == ["plain"]


def test_rerank_for_queries_reorders_pool_with_query_text() -> None:
    class FakeReranker:
        def score(self, pairs):
            return [1.0 if text == "이 크리에이터" else 0.0 for _query, text in pairs]

    cosine = np.array([[0.9, 0.8, 0.1]], dtype=np.float32)

    result = pipeline._rerank_bge_m2_candidates_for_queries(
        ["q"], ["쿼리"], cosine, ["a", "b", "c"], {"a": "다른", "b": "이 크리에이터", "c": "다른"}, FakeReranker(), 2
    )

    assert [cid for cid, _ in result["q"]] == ["b", "a"]  # 풀은 코사인 상위 2명(a, b), 리랭커가 b를 앞으로
