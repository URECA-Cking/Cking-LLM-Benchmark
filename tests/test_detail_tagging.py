import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
import openai

from src.detail_tagging import (RevisionStore, automatic_record, candidate_schema,
                                classify, classify_with_model, main)


@pytest.fixture
def taxonomy():
    return json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))


def candidate(code, evidence="주제", **kwargs):
    return {"code": code, "evidence": evidence, **kwargs}


def payload(parents, topics):
    return {"parents": [candidate(p) for p in parents],
            "topics": [candidate(t, parent=t.split("_")[0]) for t in topics]}


def test_limits_and_tie_order(taxonomy):
    parents = ["FOOD", "GAME", "TECH", "PET"]
    codes = [r["code"] for r in taxonomy["subtopics"] if r["parent"] in parents]
    result = classify("주제", payload(parents[::-1], codes[::-1]), taxonomy)
    assert [r["code"] for r in result["parents"]] == ["FOOD", "GAME", "PET"]
    assert len(result["topics"]) == 9
    for parent in ("FOOD", "GAME", "PET"):
        assert [r["code"] for r in result["topics"] if r["parent"] == parent] == sorted(
            c for c in codes if c.startswith(parent + "_"))[:3]


def test_priority_uses_grounded_rows(taxonomy):
    data = payload(["TECH", "FOOD", "GAME", "PET"], [])
    data["parents"][0].update(core_evidence="주제")
    data["parents"][3].update(repetition_evidence=["첫 활동", "다음 활동"])
    data["parents"][2]["evidence"] = "긴 주제 인용"
    result = classify("주제\n첫 활동\n다음 활동\n긴 주제 인용", data, taxonomy)
    assert [r["code"] for r in result["parents"]] == ["TECH", "PET", "GAME"]
    data["parents"][3]["repetition_evidence"] = ["활동"]
    result = classify("주제\n첫 활동\n다음 활동\n긴 주제 인용", data, taxonomy)
    assert [r["code"] for r in result["parents"]] == ["TECH", "GAME", "FOOD"]


@pytest.mark.parametrize("case", ["duplicate", "parent", "unknown", "evidence", "core"])
def test_reject_invalid(taxonomy, case):
    data = payload(["FOOD"], ["FOOD_EATING"])
    if case == "duplicate":
        data["topics"] *= 2
    elif case == "parent":
        data["topics"][0]["parent"] = "GAME"
    elif case == "unknown":
        data["topics"][0]["code"] = "NO_CODE"
    elif case == "evidence":
        data["topics"][0]["evidence"] = "없는 근거"
    elif case == "core":
        data["topics"][0]["core_evidence"] = "없는 근거"
    with pytest.raises(ValueError):
        classify("주제", data, taxonomy)


def test_empty_and_missing_parent(taxonomy):
    assert classify("", payload([], []), taxonomy)["unclassified"]
    with pytest.raises(ValueError):
        classify("주제", payload([], ["FOOD_EATING"]), taxonomy)


# These fixtures validate the agreed candidate handoff, not model semantic accuracy.
@pytest.mark.parametrize("text,parents,codes,excluded", [
    ("케이크 먹방", ["FOOD"], ["FOOD_EATING"], "FOOD_BAKING"),
    ("텐트 리뷰", ["TECH"], ["TECH_DEVICES"], "TRAVEL_CAMPING"),
    ("축구 게임", ["GAME"], ["GAME_SPORTS_RACING"], "SPORTS_FOOTBALL"),
    ("AI 뉴스와 도구 사용", ["TECH"], ["TECH_AI"], "TECH_PROGRAMMING"),
    ("동물 모음", [], [], "PET_TRAINING"),
    ("반려견 직접 돌봄과 훈련", ["PET"], ["PET_DOG", "PET_TRAINING"], "PET_CAT"),
])
def test_counterexamples(taxonomy, text, parents, codes, excluded):
    data = payload(parents, codes)
    for row in data["parents"] + data["topics"]:
        row["evidence"] = text
    result = classify(text, data, taxonomy)
    assert {r["code"] for r in result["topics"]} == set(codes)
    assert excluded not in {r["code"] for r in result["topics"]}


def test_revision_and_hash_contract(tmp_path, taxonomy):
    data = payload(["FOOD"], ["FOOD_EATING"])
    record = automatic_record("channel", "주제", data, taxonomy, "model-snapshot")
    changed = copy.deepcopy(taxonomy)
    changed["subtopics"][0]["exclude"].append("새 제외 기준")
    other = automatic_record("channel", "주제", data, changed, "model-snapshot")
    assert record["be_hash"] == other["be_hash"]
    assert record["criteria_hash"] != other["criteria_hash"]
    store = RevisionStore(tmp_path)
    auto = store.save(record)
    reviewed = store.review(auto["revision"], "주제", payload([], []), taxonomy, "reviewer", "근거 부족")
    original_bytes = (tmp_path / f"{reviewed['revision']}.json").read_bytes()
    rerun = store.save(record)
    corrected = store.review(reviewed["revision"], "주제", data, taxonomy, "reviewer", "수정")
    assert len({r["revision"] for r in (auto, reviewed, rerun, corrected)}) == 4
    assert corrected["source_revision"] == reviewed["revision"]
    assert reviewed["review_status"] == "confirmed"
    assert reviewed["model_metadata_role"] == "original_automatic_classification"
    assert reviewed["model_version"] == auto["model_version"]
    assert reviewed["prompt_hash"] == auto["prompt_hash"]
    assert (tmp_path / f"{reviewed['revision']}.json").read_bytes() == original_bytes
    for text, tax in (("다른 입력", taxonomy), ("주제", changed)):
        with pytest.raises(ValueError):
            store.review(auto["revision"], text, data, tax, "reviewer", "변경")
    with pytest.raises(ValueError):
        store.read("../escape")


def test_cli_roundtrip(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"channel_id": "channel", "text": "",
                                 "candidates": payload([], []), "model_version": "model"}))
    store = tmp_path / "store"
    assert main([str(source), "--store", str(store)]) == 0
    revision = next(store.glob("*.json")).stem
    assert main([str(source), "--store", str(store), "--source-revision", revision,
                 "--reviewer", "human", "--note", "검수 완료"]) == 0


def test_model_path_validates_response(taxonomy):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content=json.dumps(payload(["FOOD"], ["FOOD_EATING"]))))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = classify_with_model(client, "channel", "주제", taxonomy, "model")
    assert result["result"]["topics"][0]["code"] == "FOOD_EATING"
    assert calls[0]["model"] == "model"
    from src.detail_tagging import digest, PROMPT_VERSION
    assert result["prompt_hash"] == digest(calls[0]["messages"][0]["content"])
    assert result["prompt_version"] == PROMPT_VERSION == "grounded-detail-candidates-v2"
    assert "include" in calls[0]["messages"][0]["content"]
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    with pytest.raises(ValueError):
        classify_with_model(client, "channel", "근거 없음", taxonomy, "model")


def test_schema_requires_complete_objects(taxonomy):
    schema = candidate_schema(taxonomy)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"parents", "topics"}
    for axis in ("parents", "topics"):
        item = schema["properties"][axis]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
        assert item["properties"]["repetition_evidence"]["items"] == {"type": "string"}
    assert "parent" in schema["properties"]["topics"]["items"]["required"]


@pytest.mark.parametrize("quotes", [["활동"], ["없는 인용"], ["첫 활동", "첫 활동"],
                                    ["첫 활동", "다음 활동", "활동"]])
def test_bad_repetition_only_clears_auxiliary_score(taxonomy, quotes):
    data = payload(["FOOD"], ["FOOD_EATING"])
    data["topics"][0]["repetition_evidence"] = quotes
    topic = classify("주제\n첫 활동\n다음 활동", data, taxonomy)["topics"][0]
    assert topic["priority"][1] == 0
    assert topic["repetition_warnings"]
    assert topic["repetition_evidence"] == quotes


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object()])
def test_serialization_failure_leaves_no_revision(tmp_path, invalid):
    store = RevisionStore(tmp_path)
    original = store.save({"kind": "reviewed", "value": "confirmed"})
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises((ValueError, TypeError)):
        store.save({"candidate": invalid})
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert store.read(original["revision"])["value"] == "confirmed"


def test_cli_rejects_self_consistent_unknown_parent(tmp_path, taxonomy, capsys):
    wrong = copy.deepcopy(taxonomy)
    wrong["parents"] = ["UNKNOWN" if p == "FOOD" else p for p in wrong["parents"]]
    for row in wrong["subtopics"]:
        if row["parent"] == "FOOD":
            row["parent"] = "UNKNOWN"
            row["code"] = row["code"].replace("FOOD_", "UNKNOWN_")
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing.json"), "--taxonomy", str(path),
              "--store", str(tmp_path / "store")])
    assert error.value.code == 2
    assert "parent list" in capsys.readouterr().err
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize("failure", ["key", "timeout", "400", "429"])
def test_cli_provider_errors_are_concise(tmp_path, monkeypatch, capsys, failure):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"channel_id": "channel", "text": "주제", "model_version": "model"}))
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    if failure == "key":
        exception = openai.OpenAIError("private provider details")
    elif failure == "timeout":
        exception = openai.APITimeoutError(request=request)
    else:
        status = int(failure)
        cls = openai.BadRequestError if status == 400 else openai.RateLimitError
        exception = cls("private provider details", response=httpx.Response(status, request=request), body=None)
    def fail(**kwargs):
        raise exception
    monkeypatch.setattr("src.detail_tagging.OpenAI", fail)
    with pytest.raises(SystemExit) as error:
        main([str(source), "--store", str(tmp_path / "store"), "--tag-llm"])
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "ERROR: OpenAI" in stderr
    assert "Traceback" not in stderr
    assert "private provider details" not in stderr
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize("finish,refusal,content", [
    ("length", None, '{"parents":'), ("stop", "refused", None), ("stop", None, None)])
def test_model_refusal_and_incomplete_output(taxonomy, finish, refusal, content):
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
            message=SimpleNamespace(content=content, refusal=refusal))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(ValueError):
        classify_with_model(client, "channel", "주제", taxonomy, "model")


def test_real_sdk_serializes_strict_schema(taxonomy):
    captured = []
    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        assert body["response_format"]["json_schema"]["schema"] == candidate_schema(taxonomy)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"parents":[],"topics":[]}'}}]})
    with openai.OpenAI(api_key="test-key", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        record = classify_with_model(client, "channel", "", taxonomy, "test-model")
    assert len(captured) == 1
    assert record["result"]["unclassified"]


@pytest.mark.parametrize("quote", ['"케이크 제작"', "케이크/빵", "케이크...굽기",
                                  "cake making", "베이킹을 합니다", "", "   "])
def test_v3_rejects_noncontiguous_evidence(taxonomy, quote):
    with pytest.raises(ValueError, match="exact source quote"):
        classify("케이크 제작과 빵 굽기", {"parents": [candidate("FOOD", quote)], "topics": []}, taxonomy)


@pytest.mark.parametrize("axis", ["parents", "topics"])
def test_v3_duplicates_in_each_array(taxonomy, axis):
    data = payload(["FOOD"], ["FOOD_EATING"])
    data[axis] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        classify("주제", data, taxonomy)


def test_v3_short_grounded_and_ungrounded(taxonomy):
    assert not classify("먹방", {"parents": [candidate("FOOD", "먹방")], "topics": []}, taxonomy)["unclassified"]
    assert classify("안녕", payload([], []), taxonomy)["unclassified"]


def test_default_prompt_is_exact_original_v2(taxonomy):
    from src.detail_tagging import build_prompt, digest
    # Frozen original v2 prompt hash for the repository's taxonomy fixture.
    assert digest(build_prompt(taxonomy)) == "75c0d36b571201cf8b7245441f652f50ab28c689476181a8ab1a222a73337386"
    experimental = build_prompt(taxonomy, "grounded-detail-candidates-v3")
    assert "후보 자체를 생략" in experimental
    assert digest(experimental) != digest(build_prompt(taxonomy))


@pytest.mark.parametrize("version", ["grounded-detail-candidates-v3", "grounded-detail-candidates-v4"])
def test_explicit_experiment_records_actual_prompt(taxonomy, version):
    from src.detail_tagging import digest
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
            message=SimpleNamespace(content='{"parents":[],"topics":[]}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    record = classify_with_model(client, "channel", "", taxonomy, "model",
                                 prompt_version=version)
    assert record["prompt_version"] == version
    assert record["prompt_hash"] == digest(calls[0]["messages"][0]["content"])


@pytest.mark.parametrize("text,parents,topics", [
    ("노래 커버", ["MUSIC"], ["MUSIC_VOCAL"]),
    ("남자 코디", ["FASHION"], ["FASHION_CLOTHING"]),
    ("#노래커버", ["MUSIC"], ["MUSIC_VOCAL"]),
    ("패션·뷰티", ["FASHION", "BEAUTY"], []),
    ("동물 영상 모음", [], []),
    ("배경 음악", [], []),
])
def test_v4_candidate_handoff_preserves_explicit_parents(taxonomy, text, parents, topics):
    from src.detail_tagging import V4_PROMPT_VERSION, build_prompt, digest
    data = payload(parents, topics)
    for row in data["parents"] + data["topics"]:
        row["evidence"] = text
    record = automatic_record("channel", text, data, taxonomy, "model", V4_PROMPT_VERSION)
    assert {r["code"] for r in record["result"]["parents"]} == set(parents)
    assert {r["code"] for r in record["result"]["topics"]} == set(topics)
    assert record["prompt_hash"] == digest(build_prompt(taxonomy, V4_PROMPT_VERSION))


def test_v4_prompt_retains_v3_rules_and_independent_parent_examples(taxonomy):
    from src.detail_tagging import PROMPT_VERSIONS, build_prompt, digest
    prompts = [build_prompt(taxonomy, version) for version in PROMPT_VERSIONS]
    assert len({digest(p) for p in prompts}) == 3
    v3, v4 = prompts[1:]
    for instruction in v3.splitlines():
        assert instruction in v4
    for instruction in ("topics만 비우며", "노래 커버 → MUSIC/MUSIC_VOCAL", "남자 코디 → FASHION/FASHION_CLOTHING",
                        "해시태그", "메이크업·스킨케어", "배경 음악만으로 작곡·댄스·악기"):
        assert instruction in v4


@pytest.mark.parametrize("quote", ['"노래 커버"', "노래...커버", "song cover", "노래를 합니다", ""])
def test_v4_does_not_relax_evidence_validation(taxonomy, quote):
    from src.detail_tagging import V4_PROMPT_VERSION
    with pytest.raises(ValueError, match="exact source quote"):
        automatic_record("channel", "노래 커버", {"parents": [candidate("MUSIC", quote)], "topics": []},
                         taxonomy, "model", V4_PROMPT_VERSION)


@pytest.mark.parametrize("axis", ["parents", "topics"])
def test_v4_duplicate_rejection(taxonomy, axis):
    from src.detail_tagging import V4_PROMPT_VERSION
    data = payload(["MUSIC"], ["MUSIC_VOCAL"])
    data[axis] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        automatic_record("channel", "주제", data, taxonomy, "model", V4_PROMPT_VERSION)
