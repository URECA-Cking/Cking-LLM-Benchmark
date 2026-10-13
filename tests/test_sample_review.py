import copy

import pytest

from src.sample_review import aggregate, prepare, review_rows


@pytest.fixture
def sample():
    taxonomy = {"parents": ["A", "B"], "subtopics": [
        {"code": "A_ONE", "parent": "A"}, {"code": "A_TWO", "parent": "A"},
        {"code": "B_ONE", "parent": "B"}]}
    records = [
        {"channel_id": "one", "text": "source", "parents": ["A"],
         "topics": [{"code": "A_ONE", "evidence": "source"}]},
        {"channel_id": "two", "text": "other", "parents": ["B"], "topics": []},
    ]
    return taxonomy, records


def test_deterministic_sample_and_shortfalls(sample):
    taxonomy, records = sample
    first = prepare(records, taxonomy)
    assert first == prepare(list(reversed(records)), taxonomy)
    assert first["manifest"]["size_shortfall"] == 48
    assert first["manifest"]["missing_coverage"] == ["counterexample", "duplicate", "over_limit"]
    assert "text" not in first["manifest"]["rows"][0]


def test_boundary_violations_are_preserved(sample):
    taxonomy, records = sample
    records[0]["topics"] *= 4
    pack = prepare(records, taxonomy, counterexamples=["one"])
    row = next(r for r in pack["manifest"]["rows"] if r["channel_id"] == "one")
    assert row["boundaries"] == ["duplicate", "over_limit", "counterexample"]


def test_pending_not_completed(sample):
    taxonomy, records = sample
    pack = prepare(records, taxonomy)
    report = aggregate(pack, review_rows(pack), taxonomy)
    assert report["counts"] == {"pending": 2}
    assert report["status"] == "human_review_pending"
    assert len(report["unresolved"]) == 2


def completed(pack):
    rows = review_rows(pack)
    for row in rows:
        row.update(status="reviewed", human_topics=row["auto_topics"], verdict="agree",
                   cause="none", reason="checked", reviewer="human")
    return rows


def test_corrections_and_causes(sample):
    taxonomy, records = sample
    pack = prepare(records, taxonomy)
    rows = completed(pack)
    row = next(r for r in rows if r["channel_id"] == "one")
    row.update(human_topics="A_TWO", verdict="corrected", cause="criteria_problem")
    report = aggregate(pack, rows, taxonomy)
    assert report["status"] == "review_complete"
    assert report["counts"]["missing_channels"] == report["counts"]["excess_channels"] == 1
    assert report["counts"]["auto_unclassified"] == 1
    assert report["corrections"][0]["cause"] == "criteria_problem"


@pytest.mark.parametrize("tags", ["B_ONE", "UNKNOWN", "A_ONE|A_ONE", "A_ONE|A_TWO|A_ONE|A_TWO"])
def test_invalid_human_tags_rejected(sample, tags):
    taxonomy, records = sample
    pack = prepare(records, taxonomy)
    rows = completed(pack)
    next(r for r in rows if r["channel_id"] == "one")["human_topics"] = tags
    with pytest.raises(ValueError):
        aggregate(pack, rows, taxonomy)


@pytest.mark.parametrize("change", ["hash", "taxonomy", "missing", "duplicate", "input", "auto", "status", "reviewer", "verdict", "cause", "pending"])
def test_review_integrity(sample, change):
    taxonomy, records = sample
    pack = prepare(records, taxonomy)
    rows = completed(pack)
    if change == "hash":
        pack["manifest"]["limit"] = 9
    elif change == "taxonomy":
        taxonomy = copy.deepcopy(taxonomy)
        taxonomy["parents"].append("C")
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1] = rows[0].copy()
    else:
        key, value = {"input": ("input_hash", "changed"), "auto": ("auto_topics", "TAMPERED"),
                      "status": ("status", "done"), "reviewer": ("reviewer", ""),
                      "verdict": ("verdict", "corrected"), "cause": ("cause", "model_error"),
                      "pending": ("status", "pending")}[change]
        rows[0][key] = value
    with pytest.raises(ValueError):
        aggregate(pack, rows, taxonomy)


def test_undecided_and_unknown_counterexample(sample):
    taxonomy, records = sample
    pack = prepare(records, taxonomy)
    rows = review_rows(pack)
    rows[0].update(status="undecided", reviewer="human", reason="criteria discussion")
    report = aggregate(pack, rows, taxonomy)
    assert report["counts"] == {"undecided": 1, "pending": 1}
    with pytest.raises(ValueError):
        prepare(records, taxonomy, counterexamples=["absent"])


def test_cap_applies_per_parent():
    taxonomy = {"parents": ["A", "B"], "subtopics": [
        {"code": f"{p}_{i}", "parent": p} for p in ("A", "B") for i in range(3)]}
    records = [{"channel_id": "multi", "text": "source", "parents": ["A", "B"],
                "topics": [{"code": t["code"], "evidence": "source"} for t in taxonomy["subtopics"]]}]
    pack = prepare(records, taxonomy)
    assert pack["manifest"]["rows"][0]["auto_issues"] == []
    assert aggregate(pack, completed(pack), taxonomy)["status"] == "review_complete"


def test_cli_revision_store_and_report(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from src import sample_review as module
    from src.detail_tagging import RevisionStore, automatic_record

    taxonomy_path = Path("data/creator-subtopics.json")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    parent = taxonomy["parents"][0]
    topic = next(t for t in taxonomy["subtopics"] if t["parent"] == parent)["code"]
    channel = SimpleNamespace(id="real-id", text=lambda: "source")
    data = SimpleNamespace(pool=[channel], codes=taxonomy["parents"])
    monkeypatch.setattr(module, "load_source", lambda *args: (data, None, None, {"real-id": {parent}}, None))
    plan = {"taxonomy_hash": module.digest(taxonomy), "source_hash": module.digest({
        "texts": [("real-id", "source")], "llm_parents": {"real-id": [parent]}})}
    (tmp_path / "subtopic_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "subtopics_llm.json").write_text(json.dumps({"contract_hash": module.digest(plan),
        "tags": {"real-id": {"text_hash": module.digest("source"), "topics": []}}}), encoding="utf-8")
    candidate = {"code": parent, "evidence": "source", "core_evidence": "", "repetition_evidence": []}
    record = RevisionStore(tmp_path / "store").save(automatic_record("real-id", "source", {
        "parents": [candidate], "topics": [{**candidate, "code": topic, "parent": parent}]}, taxonomy, "offline"))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"real-id": f"store/{record['revision']}.json"}), encoding="utf-8")
    output = tmp_path / "sample"
    command = ["prepare", "--subtopic-dir", str(tmp_path), "--revision-index", str(index), "--output-dir", str(output)]
    module.main(command)
    pack = module.read_json(output / "manifest.json")
    assert pack["manifest"]["rows"][0]["source_revision"] == record["revision"]
    assert pack["manifest"]["rows"][0]["topics"][0]["code"] == topic
    with pytest.raises(FileExistsError):
        module.main(command)
    report_dir = tmp_path / "report"
    module.main(["aggregate", "--manifest", str(output / "manifest.json"),
                 "--reviews", str(output / "reviews.csv"), "--output-dir", str(report_dir)])
    assert module.read_json(report_dir / "report.json")["counts"] == {"pending": 1}
    assert len(list((report_dir / "revisions").glob("*.json"))) == 1
    record_path = tmp_path / "store" / f"{record['revision']}.json"
    record["input_hash"] = "stale"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Stale"):
        module.main([*command[:-1], str(tmp_path / "new")])
