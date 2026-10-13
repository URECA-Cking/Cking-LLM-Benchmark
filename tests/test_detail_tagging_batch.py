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
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


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


def test_compare_separates_retry_cohort(tmp_path):
    from src.detail_tagging_compare import compare
    from src.detail_tagging_batch import write_new
    taxonomy = json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))
    current = tmp_path / "current"
    summary = run_batch(client([{"parents": [], "topics": []}]),
                        [{"channel_id": "channel", "text": ""}], taxonomy, "model", current)
    baseline = tmp_path / "baseline"
    record = json.loads(next((current / "revisions").glob("*.json")).read_text(encoding="utf-8"))
    write_new(baseline / "old.json", record)
    write_new(baseline / "revision-index.json", {"channel": "old.json"})
    write_new(baseline / "first-new-key-summary.json", {"failures": [{"channel_id": "channel"}]})
    report = compare(next(current.glob("summary-*.json")), baseline)
    assert report["rows"][0]["v2_retry_success"]
    assert report["rows"][0]["parents"]["equal"]
    assert report["v2_final_failure_count"] == 0
