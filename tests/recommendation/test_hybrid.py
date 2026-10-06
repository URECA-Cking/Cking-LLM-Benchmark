"""고정 공용 fixture와 순서·정밀도 독립성으로 개인화 참조 정책을 검증한다."""

import copy
import json
import re
from decimal import Inexact, ROUND_DOWN, Rounded, localcontext
from pathlib import Path

import pytest

from src.recommendation.hybrid import policy_settings, recommend_personalized


FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "hybrid_personalized_v1.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]


def test_shared_fixture_policy_and_decimal_wire_format():
    assert FIXTURE["schemaVersion"] == 1
    assert FIXTURE["policy"] == policy_settings()
    assert FIXTURE["policy"]["interestWeight"] == "0.5"
    assert FIXTURE["policy"]["followWeight"] == "0.5"
    assert len({case["id"] for case in CASES}) == len(CASES) == 18
    assert {case["expected"]["policyVersion"] for case in CASES} == {
        "HYBRID_PERSONALIZED_V1", "INTEREST_PERSONALIZED_V1", "FOLLOW_PERSONALIZED_V2",
    }
    for case in CASES:
        for item in case["expected"]["items"]:
            assert isinstance(item["aggregateScore"], str)
            assert re.fullmatch(r"0\.[0-9]{8}", item["aggregateScore"])
            assert item["interestCodes"] == sorted(set(item["interestCodes"]))
            assert item["seedCreatorIds"] == sorted(set(item["seedCreatorIds"]))
    # 입력은 저장된 정수 순위, 소수 점수·가중치는 문자열만 사용한다.
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8"), parse_float=lambda value: pytest.fail(value))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_every_shared_case_matches_exact_expected_output(case):
    before = copy.deepcopy(case["input"])
    assert recommend_personalized(case["input"]) == case["expected"]
    assert case["input"] == before


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_source_candidate_selection_order_does_not_change_result(case):
    data = copy.deepcopy(case["input"])
    for field in ("selectedInterestCodes", "followedCreatorIds", "creatorSpaces",
                  "interestSources", "followSources"):
        data[field].reverse()
    for source in data["interestSources"] + data["followSources"]:
        if source["activeGeneration"] is not None:
            source["activeGeneration"]["candidates"].reverse()
    assert recommend_personalized(data) == case["expected"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_caller_decimal_context_cannot_change_contract(case):
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        assert recommend_personalized(case["input"]) == case["expected"]
        assert context.prec == 2 and context.rounding == ROUND_DOWN


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_duplicate_user_selections_do_not_add_sources_or_contributions(case):
    data = copy.deepcopy(case["input"])
    data["selectedInterestCodes"] *= 2
    data["followedCreatorIds"] *= 2
    assert recommend_personalized(data) == case["expected"]


def test_rounding_stages_have_independent_known_values():
    results = {case["id"]: recommend_personalized(case["input"]) for case in CASES}
    assert results["rrf_half_up_boundary"]["items"][0]["aggregateScore"] == "0.00195313"
    assert results["group_mean_half_up_boundary"]["items"][1]["aggregateScore"] == "0.00806452"
    assert results["final_weight_half_up_boundary"]["items"][1]["aggregateScore"] == "0.00806452"
    # round((1/61 + 1/71)/2) = 0.01523897이다. 기여도를 먼저 반올림하면 마지막 자리가 다르다.
    assert results["round_contributions_before_mean"]["items"][0]["aggregateScore"] == "0.01523898"


@pytest.fixture
def valid_input():
    return copy.deepcopy(CASES[0]["input"])


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1", 2**63])
def test_invalid_creator_ids_are_rejected(valid_input, value):
    valid_input["interestSources"][0]["activeGeneration"]["candidates"][0]["creatorId"] = value
    with pytest.raises(ValueError, match="creatorId"):
        recommend_personalized(valid_input)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1", 2**31])
def test_invalid_ranks_are_rejected(valid_input, value):
    valid_input["interestSources"][0]["activeGeneration"]["candidates"][0]["rank"] = value
    with pytest.raises(ValueError, match="rank"):
        recommend_personalized(valid_input)


@pytest.mark.parametrize("field", ["creatorId", "rank"])
def test_duplicate_candidate_or_rank_in_one_source_is_rejected(valid_input, field):
    candidates = valid_input["interestSources"][0]["activeGeneration"]["candidates"]
    candidates[1][field] = candidates[0][field]
    with pytest.raises(ValueError, match="고유"):
        recommend_personalized(valid_input)


@pytest.mark.parametrize("field", ["interestSources", "followSources"])
def test_duplicate_source_is_rejected_instead_of_counted_twice(valid_input, field):
    valid_input[field].append(copy.deepcopy(valid_input[field][0]))
    with pytest.raises(ValueError, match="source 식별자"):
        recommend_personalized(valid_input)


@pytest.mark.parametrize("value", [None, 1, " "])
def test_active_generation_requires_identifier(valid_input, value):
    valid_input["interestSources"][0]["activeGeneration"]["generationId"] = value
    with pytest.raises(ValueError, match="generationId"):
        recommend_personalized(valid_input)


def test_conflicting_space_records_are_rejected(valid_input):
    valid_input["creatorSpaces"].append({"creatorId": 101, "hasCreatorSpace": False})
    with pytest.raises(ValueError, match="creatorSpaces"):
        recommend_personalized(valid_input)


@pytest.mark.parametrize("value", [1, "true", None])
def test_space_presence_requires_boolean(valid_input, value):
    valid_input["creatorSpaces"][0]["hasCreatorSpace"] = value
    with pytest.raises(ValueError, match="boolean"):
        recommend_personalized(valid_input)


@pytest.mark.parametrize("value", ["food", " FOOD", "요리", "", 1])
def test_interest_codes_have_cross_language_ascii_order(valid_input, value):
    valid_input["selectedInterestCodes"] = [value]
    with pytest.raises(ValueError, match="ASCII"):
        recommend_personalized(valid_input)


@pytest.mark.parametrize("field", ["selectedInterestCodes", "followedCreatorIds", "creatorSpaces",
                                   "interestSources", "followSources"])
def test_required_collections_are_not_silently_treated_as_empty(valid_input, field):
    valid_input[field] = None
    with pytest.raises(ValueError, match="배열"):
        recommend_personalized(valid_input)
