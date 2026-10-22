"""Verify synthetic example handoff and provenance, not model semantic quality."""
import copy
import json

import httpx
from openai import OpenAI
import pytest

from src.detail_tagging import (V5_PROMPT_VERSION, V5_REVISED_PROMPT_VERSION, V5_REFINED_PROMPT_VERSION,
                                automatic_record, build_prompt, candidate_schema,
                                classify_with_model, digest)
from src.detail_tagging_batch import read, run_batch, main as batch_main
from src.detail_tagging_v5 import examples
from tests.test_detail_tagging_experiment import client


@pytest.fixture
def taxonomy():
    return read("data/creator-subtopics.json")


@pytest.mark.parametrize("example", examples(), ids=[str(i) for i in range(len(examples()))])
def test_complete_examples_validate_without_repairs(taxonomy, example):
    original = copy.deepcopy(example)
    record = automatic_record("synthetic", example["creator"], example["output"], taxonomy,
                              "model", V5_REVISED_PROMPT_VERSION)
    for axis in ("parents", "topics"):
        rows = example["output"][axis]
        assert len({r["code"] for r in rows}) == len(rows)
        assert {r["code"] for r in record["result"][axis]} == {r["code"] for r in rows}
        required = set(candidate_schema(taxonomy)["properties"][axis]["items"]["required"])
        assert all(set(row) == required for row in rows)
        assert all(row["core_evidence"] == "" and row["repetition_evidence"] == [] for row in rows)
    assert original == example


def test_initial_v5_bytes_preserved_and_new_revision_has_its_own_hash(taxonomy):
    assert digest(build_prompt(taxonomy, V5_PROMPT_VERSION)) == "8ae319e778826b0eaa7e0ad26ddb97f8799dde27a340cc2c49807f6dcd2f4d41"
    prompt = build_prompt(taxonomy, V5_REVISED_PROMPT_VERSION)
    assert digest(prompt) != digest(build_prompt(taxonomy, V5_PROMPT_VERSION))
    assert prompt.index("판단 절차") < prompt.index("taxonomy(분류 기준)") < prompt.index("정상 출력 예시") < prompt.index("최종 출력 점검")
    assert "각 code를 한 번씩 판단" in prompt
    assert "코디라는 단어를 요구하지 않는다" in prompt
    assert "세부가 없다는 이유로 상위를 지우지 않는다" in prompt


@pytest.mark.parametrize("version", [V5_REVISED_PROMPT_VERSION, V5_REFINED_PROMPT_VERSION])
def test_real_sdk_serializes_revised_prompt_and_unchanged_schema(taxonomy, version):
    example = examples()[0]
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "model", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(example["output"])}}]})
    with OpenAI(api_key="test-key", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as sdk:
        record = classify_with_model(sdk, "synthetic", example["creator"], taxonomy, "model",
                                     version)
    request = calls[0]
    assert request["messages"][0]["content"] == build_prompt(taxonomy, version)
    assert request["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "detail_tag_candidates", "strict": True, "schema": candidate_schema(taxonomy)}}
    assert request["temperature"] == 0
    assert record["prompt_hash"] == digest(request["messages"][0]["content"])


@pytest.mark.parametrize("axis", ["parents", "topics"])
@pytest.mark.parametrize("version", [V5_REVISED_PROMPT_VERSION, V5_REFINED_PROMPT_VERSION])
def test_revised_prompt_cannot_bypass_duplicate_validator(taxonomy, axis, version):
    example = examples()[0]
    example["output"][axis] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        automatic_record("synthetic", example["creator"], example["output"], taxonomy, "m", version)


@pytest.mark.parametrize("quote", ["", "스마트폰...조립", "desktop assembly", '"데스크톱 조립"'])
@pytest.mark.parametrize("version", [V5_REVISED_PROMPT_VERSION, V5_REFINED_PROMPT_VERSION])
def test_revised_prompt_cannot_repair_invalid_quotes(taxonomy, quote, version):
    example = examples()[0]
    example["output"]["topics"][1]["evidence"] = quote
    with pytest.raises(ValueError, match="exact source quote"):
        automatic_record("synthetic", example["creator"], example["output"], taxonomy, "m", version)


@pytest.mark.parametrize("version", [V5_REVISED_PROMPT_VERSION, V5_REFINED_PROMPT_VERSION])
def test_new_revision_cannot_resume_old_v5_attempts(tmp_path, taxonomy, version):
    inputs = [dict(channel_id="a", text="")]
    run_batch(client([dict(parents=[], topics=[])]), inputs, taxonomy, "m", tmp_path, prompt_version=V5_PROMPT_VERSION)
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), inputs, taxonomy, "m", tmp_path, prompt_version=version)
    assert snapshots == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}


@pytest.mark.parametrize("version", [V5_REVISED_PROMPT_VERSION, V5_REFINED_PROMPT_VERSION])
def test_cli_still_requires_frozen_reference(capsys, version):
    with pytest.raises(SystemExit) as error:
        batch_main(["--inputs", "unused", "--output", "unused", "--tag-llm", "--prompt-version", version])
    assert error.value.code == 2
    assert "requires --human-criteria" in capsys.readouterr().err


def test_refined_prompt_preserves_v4_and_all_complete_examples(taxonomy):
    from src.detail_tagging import V4_PROMPT_VERSION
    base = build_prompt(taxonomy, V4_PROMPT_VERSION)
    prompt = build_prompt(taxonomy, V5_REFINED_PROMPT_VERSION)
    assert prompt.startswith(base)
    example_start = prompt.index("\n", prompt.index("다음은 개발 오류")) + 1
    decoder = json.JSONDecoder()
    cases, _ = decoder.raw_decode(prompt[example_start:])
    assert len(cases) == 7
    for case in cases:
        record = automatic_record("synthetic", case["creator"], case["output"], taxonomy, "m", V5_REFINED_PROMPT_VERSION)
        assert record["candidates"] == case["output"]
    assert {t["code"] for t in cases[0]["output"]["parents"]} == {"ENTERTAIN"}
    assert {t["code"] for t in cases[0]["output"]["topics"]} == {"ENTERTAIN_VARIETY", "ENTERTAIN_DRAMA"}
