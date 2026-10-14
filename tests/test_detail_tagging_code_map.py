import copy
import json

import httpx
from openai import OpenAI
import pytest

from src.detail_tagging import (V4_PROMPT_VERSION, V5_CODE_MAP_PROMPT_VERSION, V5_GROUPED_PROMPT_VERSION,
                                V5_COMPACT_PROMPT_VERSION, GROUPED_VERSIONS, candidate_schema, build_prompt,
                                classify, classify_with_model)
from src.detail_tagging_batch import read, run_batch
from src.detail_tagging_code_map import normalize, normalize_grouped, parse, schema, grouped_schema, compact_grouped_schema
from src.detail_tagging_experiment import compare_experiments
from tests.test_detail_tagging_ai_proxy import ai_criteria
from tests.test_detail_tagging_experiment import candidate, client


@pytest.fixture
def taxonomy():
    return read("data/creator-subtopics.json")


def mapped(taxonomy, candidates=None):
    value = {"parents": dict.fromkeys(taxonomy["parents"]),
             "topics": dict.fromkeys(t["code"] for t in taxonomy["subtopics"])}
    for axis, rows in (candidates or {}).items():
        for row in rows:
            value[axis][row["code"]] = {k: row[k] for k in ("evidence", "core_evidence", "repetition_evidence")}
    return value


def grouped(taxonomy, candidates):
    flat = mapped(taxonomy, candidates)
    raw = {"parents": dict.fromkeys(taxonomy["parents"])}
    for parent, value in flat["parents"].items():
        if value is not None:
            raw["parents"][parent] = {**value, "topics": {r["code"]: flat["topics"][r["code"]]
                for r in taxonomy["subtopics"] if r["parent"] == parent}}
    return raw


def test_grouped_slots_preserve_all_values_and_bound_parent_relationship(taxonomy):
    data = {"parents": [candidate(c, "주제") for c in taxonomy["parents"]],
            "topics": [candidate(r["code"], "주제", r["parent"]) for r in taxonomy["subtopics"]]}
    raw = grouped(taxonomy, data)
    original = copy.deepcopy(raw)
    assert normalize_grouped(raw, taxonomy) == data
    assert raw == original
    assert grouped_schema(taxonomy) == candidate_schema(taxonomy, V5_GROUPED_PROMPT_VERSION)
    assert set(grouped_schema(taxonomy)["properties"]) == {"parents"}
    for parent, slot in grouped_schema(taxonomy)["properties"]["parents"]["properties"].items():
        child = slot["anyOf"][0]["properties"]["topics"]
        assert set(child["properties"]) == {r["code"] for r in taxonomy["subtopics"] if r["parent"] == parent}


def test_shared_definitions_preserve_shape_and_prompt_while_removing_repetition(taxonomy):
    compact = compact_grouped_schema(taxonomy)
    def expand(value):
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return expand(compact["$defs"][value["$ref"].split("/")[-1]])
        return {k: expand(v) for k, v in value.items() if k not in ("$defs", "description")}
    assert expand(compact) == expand(grouped_schema(taxonomy))
    assert len(json.dumps(compact)) < len(json.dumps(grouped_schema(taxonomy))) / 3
    assert build_prompt(taxonomy, V5_GROUPED_PROMPT_VERSION) == build_prompt(taxonomy, V5_COMPACT_PROMPT_VERSION)


@pytest.mark.parametrize("fault", ["missing_parent", "missing_child", "wrong_child", "root_topics", "unknown_field"])
def test_grouped_malformed_handoffs_are_not_repaired(taxonomy, fault):
    raw = grouped(taxonomy, {"parents": [candidate("MUSIC", "주제")]})
    if fault == "missing_parent":
        del raw["parents"]["MUSIC"]
    elif fault == "missing_child":
        del raw["parents"]["MUSIC"]["topics"]["MUSIC_VOCAL"]
    elif fault == "wrong_child":
        raw["parents"]["MUSIC"]["topics"]["LIFE_PERSONAL"] = None
    elif fault == "root_topics":
        raw["topics"] = {}
    else:
        raw["parents"]["MUSIC"]["parent"] = "LIFE"
    with pytest.raises(ValueError):
        normalize_grouped(raw, taxonomy)


def test_schema_uses_one_nullable_slot_per_code(taxonomy):
    value = schema(taxonomy)
    assert value == candidate_schema(taxonomy, V5_CODE_MAP_PROMPT_VERSION)
    assert value != candidate_schema(taxonomy, V4_PROMPT_VERSION)
    assert value["additionalProperties"] is False
    for axis in ("parents", "topics"):
        node = value["properties"][axis]
        assert set(node["required"]) == set(node["properties"])
        assert node["additionalProperties"] is False
        for slot in node["properties"].values():
            assert slot["anyOf"][1] == {"type": "null"}
            assert set(slot["anyOf"][0]["required"]) == {"evidence", "core_evidence", "repetition_evidence"}


def test_conversion_retains_every_candidate_and_auxiliary_quote(taxonomy):
    data = {"parents": [candidate(c, "주제") for c in taxonomy["parents"]],
            "topics": [candidate(r["code"], "주제", r["parent"]) for r in taxonomy["subtopics"]]}
    data["topics"][0].update(core_evidence="주제", repetition_evidence=["첫 행", "다음 행"])
    raw = mapped(taxonomy, data)
    before = copy.deepcopy(raw)
    assert normalize(raw, taxonomy) == data
    assert raw == before
    assert len(normalize(raw, taxonomy)["topics"]) == len(taxonomy["subtopics"])


@pytest.mark.parametrize("content", ['{"parents":{},"parents":{},"topics":{}}',
                                     '{"parents":{"MUSIC":null,"MUSIC":null},"topics":{}}'])
def test_duplicate_json_keys_are_rejected_instead_of_collapsed(content):
    with pytest.raises(ValueError, match="Duplicate provider"):
        parse(content)


@pytest.mark.parametrize("fault", ["missing_slot", "unknown_slot", "extra_field", "missing_field", "wrong_type", "array"])
def test_malformed_maps_are_rejected(taxonomy, fault):
    raw = mapped(taxonomy)
    slot = dict(evidence="주제", core_evidence="", repetition_evidence=[])
    raw["parents"]["MUSIC"] = slot
    if fault == "missing_slot":
        del raw["parents"]["MUSIC"]
    elif fault == "unknown_slot":
        raw["parents"]["UNKNOWN"] = slot
    elif fault == "extra_field":
        slot["code"] = "MUSIC"
    elif fault == "missing_field":
        del slot["core_evidence"]
    elif fault == "wrong_type":
        slot["repetition_evidence"] = "주제"
    else:
        raw["parents"] = []
    with pytest.raises(ValueError):
        normalize(raw, taxonomy)


@pytest.mark.parametrize("quote", ["", "song cover", "노래...커버", '"노래 커버"'])
def test_map_schema_does_not_relax_exact_evidence(taxonomy, quote):
    raw = mapped(taxonomy, {"parents": [candidate("MUSIC", quote)]})
    with pytest.raises(ValueError, match="exact source quote"):
        classify("노래 커버", normalize(raw, taxonomy), taxonomy)


def test_missing_parent_is_not_added_automatically(taxonomy):
    raw = mapped(taxonomy, {"topics": [candidate("MUSIC_VOCAL", "노래 커버", "MUSIC")]})
    assert normalize(raw, taxonomy)["parents"] == []
    with pytest.raises(ValueError):
        classify("노래 커버", normalize(raw, taxonomy), taxonomy)


@pytest.mark.parametrize("version", [V5_CODE_MAP_PROMPT_VERSION, *GROUPED_VERSIONS])
def test_real_sdk_map_schema_and_raw_response_preserved(taxonomy, version):
    data = {"parents": [candidate("MUSIC", "노래 커버")],
            "topics": [candidate("MUSIC_VOCAL", "노래 커버", "MUSIC")]}
    raw = (grouped if version in GROUPED_VERSIONS else mapped)(taxonomy, data)
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(raw)}}]})
    with OpenAI(api_key="test", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
        record = classify_with_model(sdk, "a", "노래 커버", taxonomy, "model", version)
    assert requests[0]["response_format"]["json_schema"]["schema"] == candidate_schema(taxonomy, version)
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert record["provider_candidates"] == raw
    assert record["provider_schema_version"] == ("code-map-grouped-v1" if version in GROUPED_VERSIONS else "code-map-v1")
    assert record["candidates"] == data


@pytest.mark.parametrize("new_version", [V5_CODE_MAP_PROMPT_VERSION, *GROUPED_VERSIONS])
def test_schema_comparison_is_explicit_and_resume_preserves_raw(tmp_path, ai_criteria, new_version):
    manifest = ai_criteria["criteria"]["pack"]["manifest"]
    taxonomy = manifest["taxonomy"]
    data = {"parents": [candidate("MUSIC", "노래 커버")],
            "topics": [candidate("MUSIC_VOCAL", "노래 커버", "MUSIC")]}
    paths = []
    converter = grouped if new_version in GROUPED_VERSIONS else mapped
    for version, response in [(V4_PROMPT_VERSION, data), (new_version, converter(taxonomy, data))]:
        folder = tmp_path / version
        run_batch(client([response]), manifest["inputs"], taxonomy, "model", folder,
                  max_attempts=3, prompt_version=version, ai_proxy_criteria=ai_criteria)
        paths.append(next(folder.glob("summary-*.json")))
    with pytest.raises(ValueError, match="conditions differ"):
        compare_experiments(ai_criteria, paths)
    report = compare_experiments(ai_criteria, paths, schema_experiment=True)
    assert report["comparison_type"] == "schema_and_prompt_experiment_not_prompt_only"
    assert len(report["schema_variants"]) == 2
    assert report["human_review"] == "not_performed"
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), manifest["inputs"], taxonomy, "model", paths[0].parent,
                  prompt_version=new_version, ai_proxy_criteria=ai_criteria)
    snapshots = {p: p.read_bytes() for p in paths[-1].parent.rglob("*.json")}
    run_batch(client([]), manifest["inputs"], taxonomy, "model", paths[-1].parent,
              prompt_version=new_version, ai_proxy_criteria=ai_criteria)
    assert all(p.read_bytes() == value for p, value in snapshots.items())
    revision = next((paths[-1].parent / "revisions").glob("*.json"))
    value = read(revision)
    value["provider_candidates"]["parents"]["MUSIC"] = None
    revision.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="code-map differs"):
        compare_experiments(ai_criteria, paths, schema_experiment=True)


@pytest.mark.parametrize("kind", ["refusal", "incomplete"])
def test_provider_refusal_and_incomplete_response_are_not_normalized(taxonomy, kind):
    def handler(request):
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "model", "choices": [{"index": 0, "finish_reason": "stop" if kind == "refusal" else "length",
                "message": {"role": "assistant", "content": None, "refusal": "refused" if kind == "refusal" else None}}]})
    with OpenAI(api_key="test", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
        with pytest.raises(ValueError, match="refused|incomplete"):
            classify_with_model(sdk, "a", "노래 커버", taxonomy, "model", V5_CODE_MAP_PROMPT_VERSION)
