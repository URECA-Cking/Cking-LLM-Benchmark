import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.detail_tagging import RevisionStore, automatic_record, classify, classify_with_model, main


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
    with pytest.raises(ValueError, match="unambiguous"):
        classify("주제\n첫 활동\n다음 활동\n긴 주제 인용", data, taxonomy)


@pytest.mark.parametrize("case", ["duplicate", "parent", "unknown", "evidence", "core", "repetition"])
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
    else:
        data["topics"][0]["repetition_evidence"] = ["주제", "주제"]
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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(payload(["FOOD"], ["FOOD_EATING"]))))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = classify_with_model(client, "channel", "주제", taxonomy, "model")
    assert result["result"]["topics"][0]["code"] == "FOOD_EATING"
    assert calls[0]["model"] == "model"
    assert "include" in calls[0]["messages"][0]["content"]
    with pytest.raises(ValueError):
        classify_with_model(client, "channel", "근거 없음", taxonomy, "model")
