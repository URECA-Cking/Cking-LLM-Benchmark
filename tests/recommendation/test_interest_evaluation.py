from __future__ import annotations

import json

import pytest

from src.recommendation.batch import _json_hash
from src.recommendation.interest import METHODS
from src.recommendation.interest_evaluation import write_interest_evaluation
from tests.recommendation.test_interest import categories, recommender


def bundles(service):
    return {row.code: {method: service.recommend(row.code, method).to_backend_payload() for method in METHODS}
            for row in service.categories}


def test_same_manifest_top10_rankings_and_union_are_blind_and_reproducible(tmp_path):
    service = recommender(tmp_path)
    generated = bundles(service)
    output = tmp_path / "evaluation"
    result = write_interest_evaluation(output, service, generated)
    judge_text = (output / "judge-input.json").read_text(encoding="utf-8")
    judge = json.loads(judge_text)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert result["pairCount"] == 51
    assert len({pair["pairId"] for pair in judge["pairs"]}) == 51
    assert provenance["manifestHash"] == service.manifest.manifest_hash
    assert provenance["judgeInputHash"] == _json_hash(judge)
    assert provenance["evaluationTopN"] == 10
    assert all(forbidden not in judge_text for forbidden in ("method", "score", "rank", "INTEREST_M2", "INTEREST_M3"))
    for pair in judge["pairs"]:
        assert len(pair["queryTextHash"]) == len(pair["candidateTextHash"]) == 64
        assert len(provenance["pairSources"][pair["pairId"]]) == 2
    for code, by_method in provenance["rankings"].items():
        for method, ranking in by_method.items():
            assert ranking["candidates"] == generated[code][method]["candidates"][:10]
            assert ranking["candidateCount"] == 3
            assert ranking["shortageCount"] == 7
    before = {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in output.iterdir()}
    write_interest_evaluation(output, service, generated)
    assert before == {file.name: (file.read_bytes(), file.stat().st_mtime_ns) for file in output.iterdir()}


def test_changed_candidate_text_changes_pair_ids_and_refuses_old_evaluation(tmp_path):
    from src.recommendation.models import CreatorProfile

    first = recommender(tmp_path)
    output = tmp_path / "evaluation"
    write_interest_evaluation(output, first, bundles(first))
    old = (output / "judge-input.json").read_bytes()
    changed = recommender(tmp_path, [CreatorProfile(1, "changed")], {"changed": [1] + [0] * 16})
    with pytest.raises(ValueError, match="기존 평가"):
        write_interest_evaluation(output, changed, bundles(changed))
    assert (output / "judge-input.json").read_bytes() == old
    other = tmp_path / "new-evaluation"
    write_interest_evaluation(other, changed, bundles(changed))
    assert set(pair["pairId"] for pair in json.loads(old)["pairs"]).isdisjoint(
        pair["pairId"] for pair in json.loads((other / "judge-input.json").read_text(encoding="utf-8"))["pairs"])


def test_incomplete_generation_cannot_produce_judge_input(tmp_path):
    service = recommender(tmp_path)
    data = bundles(service)
    del data[categories()[0].code]
    with pytest.raises(ValueError, match="17개"):
        write_interest_evaluation(tmp_path / "eval", service, data)
    assert not (tmp_path / "eval").exists()


def test_full_top20_generations_produce_exact_top10_evaluation(tmp_path):
    from src.recommendation.models import CreatorProfile
    from tests.recommendation.test_interest import vector

    service = recommender(tmp_path, [CreatorProfile(index, "same") for index in range(1, 26)],
                          {"same": vector(1)})
    data = bundles(service)
    output = tmp_path / "evaluation"
    result = write_interest_evaluation(output, service, data)
    assert result["pairCount"] == 170
    plan = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    for code in plan["rankings"]:
        for method, ranking in plan["rankings"][code].items():
            assert len(data[code][method]["candidates"]) == 20
            assert [row["creatorId"] for row in ranking["candidates"]] == list(range(1, 11))
            assert ranking["shortageCount"] == 0


def test_different_m2_m3_winners_are_both_in_blind_union(tmp_path):
    from src.recommendation.models import CreatorProfile
    from tests.recommendation.test_interest import vector

    service = recommender(tmp_path, [CreatorProfile(1, "broad"), CreatorProfile(2, "focused")],
                          {"broad": vector(.45, .51, .51, .51), "focused": vector(.44, .9)})
    output = tmp_path / "evaluation"
    write_interest_evaluation(output, service, bundles(service), top_n=1)
    pairs = json.loads((output / "judge-input.json").read_text(encoding="utf-8"))["pairs"]
    assert {row["creatorId"] for row in pairs if row["interestCode"] == categories()[0].code} == {1, 2}
