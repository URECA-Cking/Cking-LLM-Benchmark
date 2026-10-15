"""Offline adoption regressions; no provider calls or semantic-quality verdicts."""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import detail_tagging as detail
from src import detail_tagging_batch as batch
from src.recommendation.subtopic_taxonomy import subtopic_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def client(calls):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model_dump=lambda: {"content": {"parents": [], "topics": []}},
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content='{"parents":[],"topics":[]}'))])
    return SimpleNamespace(timeout=60, max_retries=0,
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_adopted_fingerprints_preserve_baseline():
    snapshot = read("docs/reference/detail-tagging-v2-adoption.json")
    taxonomy = read("data/creator-subtopics.json")
    assert detail.PROMPT_VERSION == detail.ADOPTED_DETAIL_PROMPT_VERSION == snapshot["prompt_version"]
    assert detail.digest(detail.build_prompt(taxonomy)) == snapshot["prompt_hash"]
    assert detail.digest(detail.candidate_schema(taxonomy)) == snapshot["output_schema_hash"]
    assert detail.POLICY_VERSION == snapshot["policy_version"]
    policy_source = (inspect.getsource(detail._rank) + inspect.getsource(detail.classify)
                     + inspect.getsource(detail.validate_llm_topics))
    assert hashlib.sha256(policy_source.encode("utf-8")).hexdigest() == snapshot["policy_source_hash"]
    assert taxonomy["version"] == snapshot["taxonomy_version"]
    assert detail.digest(taxonomy) == snapshot["criteria_hash"]
    assert subtopic_hash(taxonomy) == snapshot["be_hash"]
    schema = detail.candidate_schema(taxonomy)
    assert schema["properties"]["parents"]["type"] == "array"
    assert schema["properties"]["topics"]["type"] == "array"
    assert "$defs" not in schema


@pytest.mark.parametrize("model_mode", [False, True])
def test_single_cli_default_and_explicit_v2_match(tmp_path, monkeypatch, model_mode):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"channel_id": "synthetic", "text": "", "model_version": "model",
                                  "candidates": {"parents": [], "topics": []}}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(detail, "OpenAI", lambda **kwargs: client(calls))
    records = []
    for name, options in (("default", []), ("explicit", ["--prompt-version", detail.PROMPT_VERSION])):
        store = tmp_path / name
        assert detail.main([str(source), "--store", str(store), *options,
                            *(["--tag-llm"] if model_mode else [])]) == 0
        record = read(next(store.glob("*.json")))
        records.append({k: v for k, v in record.items() if k not in ("revision", "created_at")})
    assert records[0] == records[1]
    assert records[0] == detail.automatic_record("synthetic", "", {"parents": [], "topics": []},
                                                read("data/creator-subtopics.json"), "model")
    assert "provider_candidates" not in records[0]
    if model_mode:
        assert calls[0] == calls[1]


def test_batch_cli_default_and_explicit_v2_match(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.json"
    inputs.write_text('[{"channel_id":"synthetic","text":""}]', encoding="utf-8")
    calls = []
    monkeypatch.setattr(batch, "OpenAI", lambda **kwargs: client(calls))
    monkeypatch.setattr(batch, "load_dotenv", lambda: None)
    contracts = []
    records = []
    for name, options in (("default", []), ("explicit", ["--prompt-version", detail.PROMPT_VERSION])):
        output = tmp_path / name
        assert batch.main(["--inputs", str(inputs), "--output", str(output),
                           "--model", "model", "--tag-llm", *options]) == 0
        contracts.append(read(output / "run-conditions.json"))
        records.append(read(next((output / "revisions").glob("*.json"))))
    assert contracts[0] == contracts[1]
    assert calls[0] == calls[1]
    expected = detail.automatic_record("synthetic", "", {"parents": [], "topics": []},
                                       read("data/creator-subtopics.json"), "model")
    for record in records:
        assert {k: record[k] for k in expected} == expected


@pytest.mark.parametrize("key", ["prompt_hash", "output_schema_hash", "criteria_hash"])
def test_v2_resume_rejects_changed_hash_without_writes(tmp_path, key):
    taxonomy = read("data/creator-subtopics.json")
    batch.run_batch(client([]), [], taxonomy, "model", tmp_path)
    manifest = tmp_path / "run-conditions.json"
    contract = read(manifest)
    contract[key] = "different-historical-hash"
    manifest.write_text(json.dumps(contract), encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    calls = []
    with pytest.raises(ValueError, match="mismatch"):
        batch.run_batch(client(calls), [], taxonomy, "model", tmp_path)
    assert calls == []
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


def test_review_keeps_actual_experimental_origin_with_explicit_v2_cli(tmp_path):
    taxonomy = read("data/creator-subtopics.json")
    store = detail.RevisionStore(tmp_path / "store")
    source = store.save(detail.automatic_record("synthetic", "", {"parents": [], "topics": []},
                                                taxonomy, "model", detail.V4_PROMPT_VERSION))
    original_bytes = (store.directory / f'{source["revision"]}.json').read_bytes()
    payload = tmp_path / "review.json"
    payload.write_text(json.dumps({"channel_id": "synthetic", "text": "",
                                   "candidates": {"parents": [], "topics": []}}), encoding="utf-8")
    assert detail.main([str(payload), "--store", str(store.directory),
                        "--source-revision", source["revision"], "--reviewer", "synthetic-reviewer",
                        "--note", "offline test", "--prompt-version", detail.PROMPT_VERSION]) == 0
    records = [read(p) for p in store.directory.glob("*.json")]
    reviewed = next(r for r in records if r["kind"] == "reviewed")
    for key in ("prompt_version", "prompt_hash", "output_schema_hash"):
        assert reviewed[key] == source[key]
    assert reviewed["source_revision"] == source["revision"]
    assert (store.directory / f'{source["revision"]}.json').read_bytes() == original_bytes


@pytest.mark.parametrize("version", [v for v in detail.PROMPT_VERSIONS if v != detail.PROMPT_VERSION])
def test_adopted_default_rejects_experimental_resume_without_mutation(tmp_path, version):
    taxonomy = read("data/creator-subtopics.json")
    # Empty input still freezes a real experiment contract without any request.
    batch.run_batch(client([]), [], taxonomy, "model", tmp_path, prompt_version=version)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    calls = []
    with pytest.raises(ValueError, match="mismatch"):
        batch.run_batch(client(calls), [], taxonomy, "model", tmp_path)
    assert calls == []
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
