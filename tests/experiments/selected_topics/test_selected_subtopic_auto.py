"""Check synthetic pair sampling, bonus ceilings and any/all metrics."""

import json

import numpy as np
import pytest

from src.flow_eval import digest
from src.selected_subtopic_auto import combinations, rank_combination
from src.selected_subtopic_auto_score import score


def test_combinations_cover_singles_and_respect_pair_groups():
    taxonomy = {
        "parents": ["A", "B"],
        "subtopics": [
            {"code": "A1", "parent": "A"},
            {"code": "A2", "parent": "A"},
            {"code": "B1", "parent": "B"},
            {"code": "B2", "parent": "B"},
        ],
    }
    result = combinations(taxonomy)
    assert result == combinations(taxonomy)
    assert len([c for g, c in result if g == "single"]) == 4
    assert len([c for g, c in result if g == "same_parent_pair"]) == 2
    assert len([c for g, c in result if g == "different_parent_pair"]) == 4
    assert len({tuple(c) for _, c in result}) == len(result)


def test_multi_parent_bonus_is_maximum_not_sum():
    taxonomy = {
        "subtopics": [{"code": "A1", "parent": "A"}, {"code": "B1", "parent": "B"}]
    }
    result = rank_combination(
        ["A1", "B1"],
        ["c"],
        np.array([[1.0, 0.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        ["A", "B"],
        {"c": {"A", "B"}},
        {"c": {"A1", "B1"}},
        taxonomy,
    )
    assert result["parent_only"][0]["score"] == pytest.approx(1.2)
    assert result["selected_subtopic"][0]["score"] == pytest.approx(1.2)


def fixture_files(path):
    candidates = {
        "q": {
            m: [{"id": str(i)} for i in range(5)]
            for m in ("parent_only", "selected_subtopic")
        }
    }
    pairs = {
        "q::"
        + str(i): {"topics": ["A", "B"], "hash": str(i), "short_candidate_bio": False}
        for i in range(5)
    }
    plan = {
        "cases": [{"id": "q", "group": "different_parent_pair", "topics": ["A", "B"]}],
        "candidate_hash": digest(candidates),
        "pairs_hash": digest(pairs),
        "agreement_targets": {},
    }
    raw = [[1, 0], [0, 1], [1, 1], [-1, 0], [0, 0]]
    scores = {
        key: {
            "hash": pair["hash"],
            "topic_scores": raw[i],
            "reason": "",
            "input_tokens": 10,
            "output_tokens": 2,
        }
        for i, (key, pair) in enumerate(pairs.items())
    }
    judged = {"contract_hash": digest(plan), "scores": scores}
    for name, data in [
        ("auto_plan.json", plan),
        ("candidates.json", candidates),
        ("evaluation_pairs.json", pairs),
        ("auto_judgments.json", judged),
    ]:
        (path / name).write_text(json.dumps(data))
    return judged


def test_any_all_coverage_and_unknown_are_distinct(tmp_path):
    fixture_files(tmp_path)
    group = score(tmp_path)["groups"]["different_parent_pair"]["selected_subtopic"]
    assert group["p_at_5_any"] == 0.6
    assert group["p_at_5_all"] == 0.2
    assert group["interest_coverage"] == 1.0
    assert group["unknown_at_5"] == 0.2


def test_missing_auto_judgment_is_rejected(tmp_path):
    judged = fixture_files(tmp_path)
    del judged["scores"]["q::0"]
    (tmp_path / "auto_judgments.json").write_text(json.dumps(judged))
    with pytest.raises(ValueError, match="Incomplete"):
        score(tmp_path)


def test_paid_judge_hides_methods_and_resumes_without_extra_calls(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    import openai
    from src.selected_subtopic_auto import PROMPT, SCHEMA, judge

    pairs = {
        "q::a": {
            "query": json.dumps([{"name": "baking", "definition": "bread"}]),
            "candidate": "I bake bread.",
            "topics": ["BAKING"],
            "hash": "a",
        },
        "q::b": {
            "query": json.dumps([{"name": "coding", "definition": "software"}]),
            "candidate": "I write software.",
            "topics": ["CODING"],
            "hash": "b",
        },
    }
    candidates = {
        "q": {"parent_only": [{"id": "a"}], "selected_subtopic": [{"id": "b"}]}
    }
    plan = {
        "pairs_hash": digest(pairs),
        "candidate_hash": digest(candidates),
        "judge_model": "test-model",
        "judge_temperature": 0,
        "judge_prompt": PROMPT,
        "judge_schema": SCHEMA,
    }
    for name, data in [
        ("auto_plan.json", plan),
        ("evaluation_pairs.json", pairs),
        ("candidates.json", candidates),
    ]:
        (tmp_path / name).write_text(json.dumps(data))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        assert set(payload) == {"interests", "candidate"}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"topic_scores": [1], "reason": "fits"})
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            model="test-model",
        )

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    args = SimpleNamespace(output_dir=tmp_path)
    judge(args)
    judge(args)
    assert len(calls) == 2
    state = json.loads((tmp_path / "auto_judgments.json").read_text())
    assert len(state["scores"]) == 2
    assert sum(r["input_tokens"] for r in state["scores"].values()) == 20
