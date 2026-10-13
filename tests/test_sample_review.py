import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.sample_review import aggregate, prepare, review_rows
from src import sample_review as module


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
    assert first["manifest"]["missing_coverage"] == ["counterexample", "duplicate", "over_limit", "unclassified"]
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
    assert report["status"] == "reviewed_with_gaps"
    assert report["counts"]["missing_channels"] == report["counts"]["excess_channels"] == 1
    assert report["counts"]["auto_unclassified"] == 0
    assert report["counts"]["auto_no_subtopic"] == 1
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
    assert aggregate(pack, completed(pack), taxonomy)["status"] == "reviewed_with_gaps"


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


def test_complete_requires_coverage_and_requested_size(sample):
    taxonomy, records = sample
    records[0]["topics"] *= 4
    records.append({"channel_id": "empty", "text": "empty", "parents": [], "topics": []})
    pack = prepare(records, taxonomy, size=3, counterexamples=["one"])
    rows = completed(pack)
    next(r for r in rows if r["channel_id"] == "one").update(
        human_topics="A_ONE", verdict="corrected", cause="model_error")
    assert pack["manifest"]["missing_coverage"] == []
    assert aggregate(pack, rows, taxonomy)["status"] == "review_complete"
    short = prepare(records, taxonomy, size=50, counterexamples=["one"])
    assert aggregate(short, rows, taxonomy)["status"] == "reviewed_with_gaps"
    rows[0].update(status="undecided", reason="discussion")
    assert aggregate(pack, rows, taxonomy)["status"] == "human_review_pending"


def test_missing_parent_is_not_no_subtopic(sample):
    taxonomy, records = sample
    records.append({"channel_id": "empty", "text": "empty", "parents": [], "topics": []})
    pack = prepare(records, taxonomy)
    by_id = {r["channel_id"]: r for r in pack["manifest"]["rows"]}
    assert by_id["empty"]["boundaries"] == ["unclassified"]
    assert by_id["two"]["boundaries"] == ["no_subtopic"]
    assert by_id["one"]["boundaries"] == []
    report = aggregate(pack, completed(pack), taxonomy)
    assert report["boundary_counts"]["unclassified"] == report["boundary_counts"]["no_subtopic"] == 1
    assert report["counts"]["auto_unclassified"] == report["counts"]["human_unclassified"] == 1
    assert report["counts"]["auto_no_subtopic"] == report["counts"]["human_no_subtopic"] == 1
    assert any("frozen automatic parents" in text for text in report["limitations"])


def test_whitespace_is_normalized_but_empty_code_and_duplicates_rejected(sample):
    taxonomy, records = sample
    pack = prepare(records, taxonomy)
    rows = completed(pack)
    row = next(r for r in rows if r["channel_id"] == "one")
    row["human_topics"] = " A_ONE "
    assert aggregate(pack, rows, taxonomy)["corrections"] == []
    for tags, message in [("A_ONE | A_ONE", "duplicate"), ("A_ONE | ", "unknown_tag"), ("B_ONE", "parent_mismatch")]:
        row["human_topics"] = tags
        with pytest.raises(ValueError, match=f"channel_id=one:.*{message}"):
            aggregate(pack, rows, taxonomy)


@pytest.mark.parametrize("change", ["missing", "none", "extra", "non_string"])
def test_malformed_row_is_value_error(sample, change):
    taxonomy, records = sample
    pack = prepare(records, taxonomy)
    rows = completed(pack)
    row = rows[0]
    if change == "missing":
        del row["reviewer"]
    elif change == "extra":
        row[None] = ["extra"]
    else:
        row["reviewer"] = None if change == "none" else 42
    with pytest.raises(ValueError, match=f"CSV row 2, channel_id={row['channel_id']}"):
        aggregate(pack, rows, taxonomy)


@pytest.mark.parametrize("kind", ["missing_header", "duplicate_header", "extra_header", "short_row", "extra_cell", "broken_quote", "broken_header"])
def test_csv_structure_errors(tmp_path, kind):
    header = list(module.FIELDS)
    values = ["id"] + [""] * (len(header)-1)
    if kind == "missing_header":
        header.pop()
    elif kind == "duplicate_header":
        header[-1] = header[0]
    elif kind == "extra_header":
        header.append("extra")
    elif kind == "short_row":
        values.pop()
    elif kind == "extra_cell":
        values.append("extra")
    path = tmp_path / "reviews.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(values)
    if kind == "broken_quote":
        path.write_text(",".join(header) + '\n"unterminated', encoding="utf-8-sig")
    elif kind == "broken_header":
        path.write_text('"unterminated', encoding="utf-8-sig")
    with pytest.raises(ValueError, match="CSV"):
        module.read_reviews(path)


def test_custom_limit_is_not_default_policy(sample):
    taxonomy, records = sample
    default = prepare(records, taxonomy)["manifest"]
    custom = prepare(records, taxonomy, limit=1)["manifest"]
    assert default["parent_limit"] == custom["parent_limit"] == 3
    assert default["policy_version"] != custom["policy_version"]
    assert custom["limit"] == 1


@pytest.fixture
def legacy_cache(tmp_path, monkeypatch):
    taxonomy = module.read_json(Path("data/creator-subtopics.json"))
    parent = taxonomy["parents"][0]
    channel = SimpleNamespace(id="real-id", text=lambda: "source")
    data = SimpleNamespace(pool=[channel], codes=taxonomy["parents"])
    monkeypatch.setattr(module, "load_source", lambda *args: (data, None, None, {"real-id": {parent}}, None))
    plan = {"taxonomy_hash": module.digest(taxonomy), "source_hash": module.digest({
        "texts": [("real-id", "source")], "llm_parents": {"real-id": [parent]}})}
    tags = {"contract_hash": module.digest(plan), "tags": {
        "real-id": {"text_hash": module.digest("source"), "topics": []}}}
    for name, value in [("subtopic_plan", plan), ("subtopics_llm", tags)]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    return tmp_path, plan, tags


@pytest.mark.parametrize("corruption", [None, "taxonomy", "source", "contract", "missing_channel", "stale_text"])
def test_legacy_cli_contract_and_records(legacy_cache, corruption):
    root, plan, tags = legacy_cache
    if corruption == "taxonomy":
        plan["taxonomy_hash"] = "changed"
        tags["contract_hash"] = module.digest(plan)
    elif corruption == "source":
        plan["source_hash"] = "changed"
        tags["contract_hash"] = module.digest(plan)
    elif corruption == "contract":
        tags["contract_hash"] = "changed"
    elif corruption == "missing_channel":
        tags["tags"] = {}
    elif corruption == "stale_text":
        tags["tags"]["real-id"]["text_hash"] = "changed"
    for name, value in [("subtopic_plan", plan), ("subtopics_llm", tags)]:
        (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    output = root / "sample"
    command = ["prepare", "--subtopic-dir", str(root), "--output-dir", str(output)]
    if corruption:
        with pytest.raises(ValueError, match="channel_id=real-id" if corruption in {"missing_channel", "stale_text"} else "Stale classification contract"):
            module.main(command)
        assert not output.exists()
    else:
        module.main(command)
        assert module.read_json(output / "manifest.json")["manifest"]["classification_contract"]["mode"] == "legacy_selected_subtopic_cache"


@pytest.mark.parametrize("step", ["prepare", "aggregate"])
def test_failed_write_leaves_no_partial_output_and_can_retry(legacy_cache, monkeypatch, step):
    root, _, _ = legacy_cache
    sample = root / "sample"
    prepare_command = ["prepare", "--subtopic-dir", str(root), "--output-dir", str(sample)]
    if step == "prepare":
        command = prepare_command
        output = sample
    else:
        module.main(prepare_command)
        output = root / "report"
        command = ["aggregate", "--manifest", str(sample / "manifest.json"),
                   "--reviews", str(sample / "reviews.csv"), "--output-dir", str(output)]
    with monkeypatch.context() as patch:
        if step == "prepare":
            original = Path.write_text
            def fail_private(self, *args, **kwargs):
                if self.name == "private_inputs.json":
                    raise OSError("injected write failure")
                return original(self, *args, **kwargs)
            patch.setattr(Path, "write_text", fail_private)
        else:
            def fail_save(*args, **kwargs):
                raise OSError("injected revision failure")
            patch.setattr(module.RevisionStore, "save", fail_save)
        with pytest.raises(OSError, match="injected"):
            module.main(command)
    assert not output.exists()
    assert not list(root.glob(f".{output.name}-*"))
    module.main(command)
    if step == "aggregate":
        revision = module.read_json(next((output / "revisions").glob("*.json")))
        assert revision["reviews_file_sha256"] == hashlib.sha256((sample / "reviews.csv").read_bytes()).hexdigest()
        assert "normalized-text" in revision["reviews_hash_scheme"]


def test_hash_domains_are_explicit(sample):
    from src.detail_tagging import digest as revision_digest
    from src.recommendation.subtopic_taxonomy import subtopic_hash
    taxonomy = module.read_json(Path("data/creator-subtopics.json"))
    assert len({module.digest(taxonomy), revision_digest(taxonomy), subtopic_hash(taxonomy)}) == 3
    taxonomy, records = sample
    assert prepare(records, taxonomy)["manifest"]["hash_scheme"].startswith("flow_eval.digest:")


@pytest.mark.parametrize("encoding,newline", [("utf-8-sig", "\r\n"), ("utf-8", "\n"), ("utf-8-sig", "\r")])
def test_aggregate_uses_one_csv_snapshot_despite_later_edits(legacy_cache, monkeypatch, encoding, newline):
    import io

    root, _, _ = legacy_cache
    sample = root / "sample"
    module.main(["prepare", "--subtopic-dir", str(root), "--output-dir", str(sample)])
    path = sample / "reviews.csv"
    rows = module.read_reviews(path)
    # Include a multiline field to exercise CSV parsing and newline normalization.
    rows[0].update(status="undecided", reviewer="human", reason=f"기준 확인{newline}추가 논의")
    with io.StringIO(newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=module.FIELDS, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)
        raw = stream.getvalue().encode(encoding)
    path.write_bytes(raw)
    expected_text_hash = module.digest(path.read_text(encoding="utf-8-sig"))
    original_read = Path.read_bytes
    reads = []

    def read_then_edit(self):
        value = original_read(self)
        if self == path:
            reads.append(self)
            self.write_bytes(b"changed after the snapshot")
        return value

    output = root / "report"
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_then_edit)
        module.main(["aggregate", "--manifest", str(sample / "manifest.json"),
                     "--reviews", str(path), "--output-dir", str(output)])
    assert reads == [path]
    assert path.read_bytes() != raw
    report = module.read_json(output / "report.json")
    revision = module.read_json(next((output / "revisions").glob("*.json")))
    assert report["counts"] == {"undecided": 1}
    assert revision["report"] == report
    assert revision["reviews"] == rows
    assert revision["reviews_hash"] == expected_text_hash
    assert revision["reviews_file_sha256"] == hashlib.sha256(raw).hexdigest()
