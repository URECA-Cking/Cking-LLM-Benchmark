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


def test_pending_pairs_redoes_when_text_or_judge_config_changes() -> None:
    saved = {
        "q::a": {"score": 1, "hash": "h1", "judge": "cfg"},
        "q::b": {"score": 2, "hash": "old", "judge": "cfg"},  # 텍스트 해시가 달라짐
        "q::c": {"score": 0, "hash": "h3", "judge": "other"},  # 모델·프롬프트가 달라짐
        "q::d": {"score": 0, "hash": "h4"},  # 판정 조건 정보가 없는 예전 저장분
    }
    text_hash = {("q", "a"): "h1", ("q", "b"): "h2", ("q", "c"): "h3", ("q", "d"): "h4", ("q", "e"): "h5"}

    pending = te.pending_pairs(list(text_hash), saved, text_hash, "cfg")

    assert pending == [("q", "b"), ("q", "c"), ("q", "d"), ("q", "e")]  # 조건이 모두 같은 a만 재사용


def test_judge_config_hash_changes_with_model_or_prompt() -> None:
    base = te.judge_config_hash("m1", "p")

    assert base == te.judge_config_hash("m1", "p")
    assert base != te.judge_config_hash("m2", "p") and base != te.judge_config_hash("m1", "p2")


def test_build_judged_pool_ignores_judgments_of_pairs_no_longer_candidates() -> None:
    saved = {"q::a": {"score": 2}, "q::b": {"score": 0}, "q::gone": {"score": 2}}  # gone은 지금 후보가 아님

    judged, pool = te.build_judged_pool([("q", "a"), ("q", "b")], saved)

    assert judged == {("q", "a"): 2, ("q", "b"): 0} and pool == {"q": [2, 0]}
