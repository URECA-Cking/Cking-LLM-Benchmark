import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.detail_tagging import V4_PROMPT_VERSION, PROMPT_VERSION, digest
from src.detail_tagging_batch import run_batch
from src.detail_tagging_human_eval import (FIELDS, evaluate, freeze, main, prepare,
                                           review_rows, validate)


@pytest.fixture
def taxonomy():
    return json.loads(Path("data/creator-subtopics.json").read_text(encoding="utf-8"))


def client(outputs):
    def create(**kwargs):
        output = outputs.pop(0)
        return SimpleNamespace(model_dump=lambda: {"usage": {"total_tokens": 10}, "content": output},
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(output)))])
    return SimpleNamespace(timeout=60, max_retries=0,
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def candidates(parents=(), topics=()):
    return {"parents": [{"code": p, "evidence": "노래 커버"} for p in parents],
            "topics": [{"code": t, "parent": "MUSIC", "evidence": "노래 커버"} for t in topics]}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def judged(pack):
    rows = review_rows(pack)
    for r in rows:
        r.update(decision="keep" if r["code"] in {"MUSIC", "MUSIC_VOCAL"} else "delete",
                 reviewer="human", reason="원문 확인", evidence="노래 커버" if r["code"] in {"MUSIC", "MUSIC_VOCAL"} else "")
    return rows


@pytest.fixture
def pack(taxonomy):
    return prepare([{"channel_id": "a", "text": "노래 커버"}, {"channel_id": "b", "text": "노래 커버"}], taxonomy,
                   {"v3_unclassified_transition": ["a"], "baseline_retry_success": ["a"], "baseline_final_failure": ["b"]})


def frozen(tmp_path, pack, rows=None):
    path = tmp_path / "review.csv"
    write_csv(path, judged(pack) if rows is None else rows)
    return freeze(pack, path)


def run(tmp_path, pack, outputs, version=V4_PROMPT_VERSION):
    run_batch(client(outputs), pack["manifest"]["inputs"], pack["manifest"]["taxonomy"], "model", tmp_path,
              max_attempts=1, prompt_version=version)
    return next(tmp_path.glob("summary-*.json"))


def test_prepare_never_turns_ai_drafts_into_human_truth(pack):
    assert {r["decision"] for r in review_rows(pack)} == {"pending"}
    assert all(not r["reviewer"] for r in review_rows(pack))
    assert {r["axis"] for r in review_rows(pack)} == {"parents", "topics"}


def test_pending_cannot_be_frozen(tmp_path, pack):
    with pytest.raises(ValueError, match="pending"):
        frozen(tmp_path, pack, review_rows(pack))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "input_hash", "reviewer", "reason", "evidence", "decision", "extra", "cell"])
def test_bad_human_review_rejected(pack, fault):
    rows = judged(pack)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "input_hash":
        rows[0]["input_hash"] = "modified"
    elif fault in {"reviewer", "reason"}:
        rows[0][fault] = ""
    elif fault == "evidence":
        next(r for r in rows if r["decision"] == "keep")["evidence"] = "없는 인용"
    elif fault == "decision":
        rows[0]["decision"] = "ai_reviewed"
    elif fault == "extra":
        rows[0]["unknown"] = ""
    else:
        rows[0]["code"] = None
    with pytest.raises(ValueError):
        validate(pack, rows)


def test_parent_correction_not_constrained_by_automatic_parents(tmp_path, pack):
    criteria = frozen(tmp_path, pack)
    path = run(tmp_path / "run", pack, [candidates(), candidates(["MUSIC"])])
    report = evaluate(criteria, [path])
    metrics = report["runs"][0]["all"]
    assert metrics["parents"]["missing"] == 1
    assert metrics["topics"]["missing"] == 2
    assert metrics["unclassified"] == 1 and metrics["parent_only"] == 1
    assert metrics["human_classification_denominator"] == 2
    assert report["cohort_overlap"] == {"a": ["v3_unclassified_transition", "baseline_retry_success"]}
    assert report["adoption"].startswith("v2_retained")


def test_holds_and_failures_have_explicit_denominators(tmp_path, pack):
    rows = judged(pack)
    next(r for r in rows if r["channel_id"] == "a" and r["code"] == "MUSIC_VOCAL").update(decision="hold", evidence="", reason="팀 기준 미결정")
    criteria = frozen(tmp_path, pack, rows)
    bad = {"parents": [{"code": "MUSIC", "evidence": "없는 인용"}], "topics": []}
    path = run(tmp_path / "run", pack, [candidates(["MUSIC"], ["MUSIC_VOCAL", "MUSIC_INSTRUMENT"]), bad])
    report = evaluate(criteria, [path])
    metrics = report["runs"][0]["all"]
    assert report["human_review"] == "reviewed_with_holds"
    assert metrics["format_denominator"] == 2 and metrics["classification_denominator"] == 1
    assert metrics["failed_channels"] == 1
    assert metrics["topics"]["held_tags"] == 1 and metrics["topics"]["missing"] == 0
    assert metrics["topics"]["excess"] == 1
    assert metrics["topics"]["failed_reviewed_tags"] > 0
    assert metrics["topics"]["fully_reviewed_channels"] == 0
    assert metrics["human_classification_denominator"] == 1


def test_human_parent_consistency(pack):
    rows = judged(pack)
    next(r for r in rows if r["code"] == "MUSIC")["decision"] = "delete"
    with pytest.raises(ValueError, match="kept parent"):
        validate(pack, rows)


@pytest.mark.parametrize("key", ["inputs_hash", "criteria_hash", "model_version", "policy_version", "output_schema_hash",
                                 "temperature", "sdk_max_retries", "timeout_seconds", "max_attempts", "prompt_hash", "prompt_version"])
def test_unverified_or_different_conditions_rejected(tmp_path, pack, key):
    criteria = frozen(tmp_path, pack)
    first = run(tmp_path / "old", pack, [candidates(), candidates()], PROMPT_VERSION)
    second = run(tmp_path / "new", pack, [candidates(), candidates()])
    value = json.loads(second.read_text(encoding="utf-8"))
    del value["conditions"][key]
    second.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Unverified"):
        evaluate(criteria, [first, second])


def test_versions_compared_without_v3_field_labels(tmp_path, pack):
    criteria = frozen(tmp_path, pack)
    old = run(tmp_path / "old", pack, [candidates(), candidates()], PROMPT_VERSION)
    new = run(tmp_path / "new", pack, [candidates(["MUSIC"], ["MUSIC_VOCAL"]), candidates(["MUSIC"])])
    report = evaluate(criteria, [old, new])
    assert [r["conditions"]["prompt_version"] for r in report["runs"]] == [PROMPT_VERSION, V4_PROMPT_VERSION]
    assert report["runs"][0]["all"]["parents"]["missing"] == 2
    assert report["runs"][1]["all"]["parents"]["missing"] == 0


@pytest.mark.parametrize("key", ["temperature", "sdk_max_retries", "timeout_seconds", "max_attempts"])
def test_changed_call_settings_reject_comparison(tmp_path, pack, key):
    criteria = frozen(tmp_path, pack)
    old = run(tmp_path / "old", pack, [candidates(), candidates()], PROMPT_VERSION)
    new = run(tmp_path / "new", pack, [candidates(), candidates()])
    value = json.loads(new.read_text(encoding="utf-8"))
    value["conditions"][key] += 1
    new.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="conditions differ"):
        evaluate(criteria, [old, new])


def test_batch_records_frozen_human_criteria_and_blocks_resume_mixing(tmp_path, pack):
    criteria = frozen(tmp_path, pack)
    inputs, taxonomy = pack["manifest"]["inputs"], pack["manifest"]["taxonomy"]
    output = tmp_path / "run"
    summary = run_batch(client([candidates(), candidates()]), inputs, taxonomy, "model", output,
                        prompt_version=V4_PROMPT_VERSION, human_criteria=criteria)
    assert summary["conditions"]["human_criteria_hash"] == criteria["hash"]
    altered = copy.deepcopy(criteria)
    altered["criteria"]["rows"][0]["reason"] = "다른 검수 기준"
    altered["hash"] = digest(altered["criteria"])
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), inputs, taxonomy, "model", output,
                  prompt_version=V4_PROMPT_VERSION, human_criteria=altered)
    with pytest.raises(ValueError, match="human criteria differ"):
        evaluate(altered, [next(output.glob("summary-*.json"))])
    changed_inputs = copy.deepcopy(inputs)
    changed_inputs[0]["text"] += "변경"
    with pytest.raises(ValueError, match="input/taxonomy differs"):
        run_batch(client([]), changed_inputs, taxonomy, "model", tmp_path / "unused",
                  prompt_version=V4_PROMPT_VERSION, human_criteria=criteria)
    assert not (tmp_path / "unused").exists()


def test_v4_cli_requires_human_criteria_before_client_creation(monkeypatch):
    import src.detail_tagging_batch as batch
    monkeypatch.setattr(batch, "OpenAI", lambda **kwargs: pytest.fail("Must not create provider client"))
    with pytest.raises(SystemExit) as error:
        batch.main(["--inputs", "unused", "--output", "unused", "--tag-llm", "--prompt-version", V4_PROMPT_VERSION])
    assert error.value.code == 2


def test_tampered_criteria_or_revision_rejected(tmp_path, pack):
    criteria = frozen(tmp_path, pack)
    path = run(tmp_path / "run", pack, [candidates(), candidates()])
    changed = copy.deepcopy(criteria)
    changed["criteria"]["rows"][0]["decision"] = "hold"
    with pytest.raises(ValueError, match="criteria changed"):
        evaluate(changed, [path])
    revision = next((path.parent / "revisions").glob("*.json"))
    value = json.loads(revision.read_text(encoding="utf-8"))
    value["result"]["unclassified"] = False
    revision.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="validated candidates"):
        evaluate(criteria, [path])


def test_cli_private_artifacts_and_exclusive_outputs(tmp_path, taxonomy):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([{"channel_id": "a", "text": "노래 커버"}]), encoding="utf-8")
    output = tmp_path / "pack"
    args = ["prepare", "--inputs", str(inputs), "--output", str(output)]
    assert main(args) == 0
    pack = json.loads((output / "pack.private.json").read_text(encoding="utf-8"))
    reviews = output / "human-review.private.csv"
    write_csv(reviews, judged(pack))
    criteria = tmp_path / "criteria.json"
    assert main(["freeze", "--pack", str(output / "pack.private.json"), "--reviews", str(reviews), "--output", str(criteria)]) == 0
    before = criteria.read_bytes()
    with pytest.raises(SystemExit):
        main(["freeze", "--pack", str(output / "pack.private.json"), "--reviews", str(reviews), "--output", str(criteria)])
    assert criteria.read_bytes() == before
    summary = run(tmp_path / "run", pack, [candidates(["MUSIC"])])
    report = tmp_path / "report.json"
    assert main(["evaluate", "--criteria", str(criteria), "--summary", str(summary), "--output", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["human_review"] == "review_complete"
    with pytest.raises(SystemExit):
        main(args)
