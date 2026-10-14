import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.detail_tagging import RevisionStore
from src.detail_tagging_batch import run_batch


def client(outputs):
    def create(**kwargs):
        data = outputs.pop(0)
        return SimpleNamespace(model_dump=lambda: {"usage": {"total_tokens": 10}, "content": data},
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(data)))])
    return SimpleNamespace(max_retries=0, timeout=60,
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_attempts_resume_and_preservation(tmp_path):
    taxonomy = json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))
    inputs = [{"channel_id": "channel", "text": "먹방"}]
    bad = {"parents": [{"code": "FOOD", "evidence": ""}], "topics": []}
    good = {"parents": [{"code": "FOOD", "evidence": "먹방"}], "topics": []}
    summary = run_batch(client([bad, good]), inputs, taxonomy, "model", tmp_path)
    assert summary["first_pass"] == 0
    assert summary["passed_after_retries"] == 1
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    store = RevisionStore(tmp_path / "revisions")
    store.review(summary["rows"][0]["revision"], "먹방", good, taxonomy, "human", "확인")
    assert run_batch(client([]), inputs, taxonomy, "model", tmp_path)["first_pass"] == 0
    assert all(p.read_bytes() == content for p, content in snapshots.items())
    attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "attempts").rglob("*.json")]
    assert sorted(a["attempt"] for a in attempts) == [1, 2]
    assert all(a["response"]["usage"]["total_tokens"] == 10 for a in attempts)


@pytest.mark.parametrize("key", ["prompt_version", "prompt_hash"])
def test_mismatched_resume_rejected_without_writes(tmp_path, key):
    taxonomy = json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))
    inputs = [{"channel_id": "channel", "text": ""}]
    run_batch(client([{"parents": [], "topics": []}]), inputs, taxonomy, "model", tmp_path)
    path = tmp_path / "run-conditions.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[key] = "old"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="mismatch"):
        run_batch(client([]), inputs, taxonomy, "model", tmp_path)
    assert all(p.read_bytes() == content for p, content in snapshots.items())


def test_old_store_rejected(tmp_path):
    (tmp_path / "revision-index.json").write_text("{}")
    with pytest.raises(ValueError, match="Unversioned"):
        run_batch(client([]), [], {"parents": [], "subtopics": []}, "model", tmp_path)


@pytest.mark.parametrize("old", ["grounded-detail-candidates-v2", "grounded-detail-candidates-v3"])
def test_v4_cannot_resume_earlier_prompt(tmp_path, old):
    from src.detail_tagging import V4_PROMPT_VERSION
    taxonomy = json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))
    inputs = [{"channel_id": "a", "text": ""}]
    run_batch(client([{"parents": [], "topics": []}]), inputs, taxonomy, "model", tmp_path, prompt_version=old)
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), inputs, taxonomy, "model", tmp_path, prompt_version=V4_PROMPT_VERSION)
    assert snapshots == {p: p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_compare_separates_retry_cohort(tmp_path):
    from src.detail_tagging_compare import compare
    from src.detail_tagging_batch import write_new
    taxonomy = json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))
    current = tmp_path / "current"
    summary = run_batch(client([{"parents": [], "topics": []}]),
                        [{"channel_id": "channel", "text": ""}], taxonomy, "model", current)
    baseline = tmp_path / "baseline"
    bad = {"parents": [{"code": "FOOD", "evidence": ""}], "topics": []}
    run_batch(client([bad, {"parents": [], "topics": []}]),
              [{"channel_id": "channel", "text": ""}], taxonomy, "model", baseline)
    report = compare(next(current.glob("summary-*.json")), next(baseline.glob("summary-*.json")))
    assert report["v2_first_pass"] == 0
    assert report["rows"][0]["v2_retry_success"]
    assert report["rows"][0]["parents"]["equal"]
    assert report["v2_final_failure_count"] == 0


@pytest.fixture
def taxonomy():
    return json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))


def test_provider_error_consumes_budget_and_stops(tmp_path, taxonomy):
    api = client([])
    def fail(**kwargs):
        raise RuntimeError("provider unavailable")
    api.chat.completions.create = fail
    inputs = [{"channel_id": "a", "text": ""}, {"channel_id": "b", "text": ""}]
    with pytest.raises(RuntimeError, match="Provider failure"):
        run_batch(api, inputs, taxonomy, "model", tmp_path, max_attempts=1)
    entry = json.loads(next((tmp_path / "attempts").rglob("*.json")).read_text(encoding="utf-8"))
    assert entry["status"] == "provider_error" and entry["attempt"] == 1
    assert len(list((tmp_path / "attempts").rglob("*.json"))) == 1
    # The failed channel has no remaining budget. Only b may call the model.
    summary = run_batch(client([{"parents": [], "topics": []}]), inputs, taxonomy,
                        "model", tmp_path, max_attempts=1)
    assert summary["rows"][0]["passed"] is False
    assert summary["rows"][0]["attempts"] == 1
    assert summary["rows"][1]["passed"] is True


@pytest.mark.parametrize("target", ["conditions", "input_hash", "request_temperature"])
def test_attempt_tampering_is_rejected(tmp_path, taxonomy, target):
    inputs = [{"channel_id": "a", "text": ""}]
    run_batch(client([{"parents": [], "topics": []}]), inputs, taxonomy, "model", tmp_path)
    path = next((tmp_path / "attempts").rglob("*.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    if target == "conditions":
        value["conditions"]["prompt_hash"] = "changed"
    elif target == "input_hash":
        value["input_hash"] = "changed"
    else:
        value["request"]["temperature"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        run_batch(client([]), inputs, taxonomy, "model", tmp_path)


def test_exhausted_validation_budget_and_limit_change(tmp_path, taxonomy):
    bad = {"parents": [{"code": "FOOD", "evidence": "absent"}], "topics": []}
    inputs = [{"channel_id": "a", "text": ""}]
    run_batch(client([bad, bad]), inputs, taxonomy, "model", tmp_path, max_attempts=2)
    summary = run_batch(client([]), inputs, taxonomy, "model", tmp_path, max_attempts=2)
    assert summary["passed_after_retries"] == 0 and summary["rows"][0]["attempts"] == 2
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), inputs, taxonomy, "model", tmp_path, max_attempts=3)


def test_duplicate_channels_rejected_before_writing(tmp_path, taxonomy):
    row = {"channel_id": "a", "text": ""}
    with pytest.raises(ValueError, match="Duplicate input"):
        run_batch(client([]), [row, row], taxonomy, "model", tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key,value", [("max_retries", 2), ("timeout", 30)])
def test_actual_client_configuration_is_recorded_and_checked(tmp_path, taxonomy, key, value):
    api = client([{"parents": [], "topics": []}])
    setattr(api, key, value)
    rows = [{"channel_id": "a", "text": ""}]
    summary = run_batch(api, rows, taxonomy, "model", tmp_path, temperature=0.3)
    assert summary["conditions"]["sdk_max_retries"] == api.max_retries
    assert summary["conditions"]["timeout_seconds"] == api.timeout
    assert summary["conditions"]["temperature"] == 0.3
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), rows, taxonomy, "model", tmp_path, temperature=0.3)


def test_atomic_publication_failure_leaves_no_partial_json(tmp_path, monkeypatch):
    from src.detail_tagging_batch import write_new
    def fail(*args):
        raise OSError("publication failed")
    monkeypatch.setattr("src.detail_tagging_batch._publish", fail)
    with pytest.raises(OSError):
        write_new(tmp_path / "value.json", {"value": 1})
    assert not list(tmp_path.iterdir())


def test_atomic_publication_never_replaces_existing(tmp_path):
    from src.detail_tagging_batch import write_new
    path = tmp_path / "value.json"
    write_new(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"original": False})
    assert path.read_bytes() == before


@pytest.mark.parametrize("key", ["temperature", "sdk_max_retries", "timeout_seconds"])
def test_comparison_requires_matching_verified_settings(tmp_path, taxonomy, key):
    from src.detail_tagging_compare import compare
    for name in ("old", "new"):
        run_batch(client([{"parents": [], "topics": []}]), [{"channel_id": "a", "text": ""}],
                  taxonomy, "model", tmp_path / name)
    old = next((tmp_path / "old").glob("summary-*.json"))
    new = next((tmp_path / "new").glob("summary-*.json"))
    value = json.loads(old.read_text(encoding="utf-8"))
    value["conditions"][key] = 99
    old.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="conditions differ"):
        compare(new, old)
    del value["conditions"][key]
    old.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Unverified"):
        compare(new, old)


@pytest.mark.parametrize("drift", ["temperature", "max_retries", "timeout"])
def test_harness_drift_aborts_without_call_or_budget_use(tmp_path, taxonomy, monkeypatch, drift):
    import src.detail_tagging_batch as batch
    original = batch.classify_with_model
    api = client([{"parents": [], "topics": []}])
    calls = []
    provider_create = api.chat.completions.create
    def count_call(**kwargs):
        calls.append(kwargs)
        return provider_create(**kwargs)
    api.chat.completions.create = count_call

    def changed_call(proxy, channel, text, tax, model, prompt_version, temperature):
        if drift == "temperature":
            temperature = 1
        else:
            setattr(api, drift, 2 if drift == "max_retries" else 30)
        return original(proxy, channel, text, tax, model, prompt_version, temperature)
    monkeypatch.setattr(batch, "classify_with_model", changed_call)
    inputs = [{"channel_id": "a", "text": ""}, {"channel_id": "b", "text": ""}]
    with pytest.raises(batch.HarnessConfigurationError, match="Actual call conditions"):
        run_batch(api, inputs, taxonomy, "model", tmp_path, max_attempts=1)
    assert calls == []
    assert not list((tmp_path / "attempts").rglob("*.json"))
    assert not list((tmp_path / "revisions").glob("*.json"))
    errors = list((tmp_path / "harness-errors").glob("*.json"))
    assert len(errors) == 1
    diagnostic = json.loads(errors[0].read_text(encoding="utf-8"))
    assert diagnostic["status"] == "harness_error"
    assert diagnostic["provider_called"] is False
    assert diagnostic["next_attempt"] == 1 and "attempt" not in diagnostic

    # Correcting local settings resumes at attempt 1, even with a one-call budget.
    monkeypatch.setattr(batch, "classify_with_model", original)
    # Preserve the same input manifest and run directory when resuming.
    api = client([{"parents": [], "topics": []}, {"parents": [], "topics": []}])
    result = run_batch(api, inputs, taxonomy, "model", tmp_path, max_attempts=1)
    assert result["first_pass"] == 2
    assert all(row["attempts"] == 1 for row in result["rows"])
    assert errors[0].exists()
