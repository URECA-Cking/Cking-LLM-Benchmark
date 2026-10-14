import copy
import json
from pathlib import Path

import pytest

from src.detail_tagging import PROMPT_VERSION, V4_PROMPT_VERSION, digest
from src.detail_tagging_batch import read, write_new
from src.detail_tagging_holdout import (compare_pair, historical_cohorts, main, protocol,
                                       record_decision, run_pair, select_holdout, validated_protocol)
from src.detail_tagging_human_eval import prepare
from tests.test_detail_tagging_human_eval import (candidates, client, frozen, run)


@pytest.fixture
def study(tmp_path):
    taxonomy = read("data/creator-subtopics.json")
    dev = [{"channel_id": "dev", "text": "노래 커버"}]
    pool = [{"channel_id": "test", "text": "새로운 노래 커버", "stratum": "music"}]
    selection = select_holdout(pool, dev, [], {"music": 1}, "fixed-seed")
    (tmp_path / "dev").mkdir()
    (tmp_path / "test").mkdir()
    development = frozen(tmp_path / "dev", prepare(dev, taxonomy))
    holdout = frozen(tmp_path / "test", prepare(selection["manifest"]["inputs"], taxonomy))
    acceptance = {"min_holdout_channels": 1, "min_topic_missing_reduction": 1,
                  "max_topic_excess_increase": 0, "max_parent_missing_increase": 0,
                  "max_parent_excess_increase": 0}
    return protocol(development, holdout, selection, acceptance)


def summaries(output):
    return [next((output / version).glob("summary-*.json")) for version in (PROMPT_VERSION, V4_PROMPT_VERSION)]


def test_preregistered_v4_v5_pair_and_adoption_remain_explicit(study, tmp_path):
    from src.detail_tagging import V5_PROMPT_VERSION
    manifest = study["manifest"]
    versions = (V4_PROMPT_VERSION, V5_PROMPT_VERSION)
    value = protocol(manifest["development"], manifest["holdout"], manifest["selection"],
                     manifest["acceptance"], versions)
    assert tuple(validated_protocol(value)["prompt_hashes"]) == versions
    assert study["manifest"]["prompt_hashes"].keys() == {PROMPT_VERSION, V4_PROMPT_VERSION}
    output = tmp_path / "v5-pair"
    run_pair(client([candidates(), candidates(["MUSIC"], ["MUSIC_VOCAL"])]), value, "holdout", output)
    paths = [next((output / v).glob("summary-*.json")) for v in versions]
    report = compare_pair(value, "holdout", paths)
    assert report["decision"] == "eligible_for_team_review"
    assert [r["conditions"]["prompt_version"] for r in report["runs"]] == list(versions)
    # A report is not an adoption/default change; a v4 decision cannot target this pair.
    with pytest.raises(ValueError, match="Team decision"):
        record_decision(value, [], [], "adopt_v4", "human", "wrong challenger")
    with pytest.raises(ValueError):
        protocol(manifest["development"], manifest["holdout"], manifest["selection"],
                 manifest["acceptance"], (PROMPT_VERSION, V5_PROMPT_VERSION))


def test_v5_protocol_still_rejects_ai_reference(study):
    from src.detail_tagging import V5_PROMPT_VERSION
    manifest = copy.deepcopy(study["manifest"])
    manifest["development"]["criteria"]["origin"] = "ai_proxy"
    manifest["development"]["hash"] = digest(manifest["development"]["criteria"])
    with pytest.raises(ValueError):
        protocol(manifest["development"], manifest["holdout"], manifest["selection"],
                 manifest["acceptance"], (V4_PROMPT_VERSION, V5_PROMPT_VERSION))


def test_selection_is_order_independent_and_excludes_ids_and_text():
    dev = [{"channel_id": "d", "text": "dev"}]
    examples = [{"channel_id": "e", "text": "example"}]
    pool = [{"channel_id": cid, "text": text, "stratum": "x"}
            for cid, text in [("d", "changed"), ("copy", "dev"), ("e", "example"), ("one", "one"), ("two", "two")]]
    a = select_holdout(pool, dev, examples, {"x": 2}, "seed")
    b = select_holdout(pool[::-1], dev, examples, {"x": 2}, "seed")
    assert a == b
    assert a["manifest"]["excluded_overlap"] == 3
    assert {r["channel_id"] for r in a["manifest"]["inputs"]} == {"one", "two"}


@pytest.mark.parametrize("fault", ["shortage", "quota", "bool", "stratum", "duplicate_id", "duplicate_text", "empty_text"])
def test_bad_selection_rejected(fault):
    pool = [{"channel_id": "a", "text": "a", "stratum": "x"}, {"channel_id": "b", "text": "b", "stratum": "x"}]
    quotas = {"x": 2}
    if fault == "shortage":
        quotas["x"] = 3
    elif fault == "quota":
        quotas["x"] = 0
    elif fault == "bool":
        quotas["x"] = True
    elif fault == "stratum":
        pool[0]["stratum"] = "unknown"
    elif fault == "duplicate_id":
        pool[1]["channel_id"] = "a"
    elif fault == "duplicate_text":
        pool[1]["text"] = "a"
    else:
        pool[0]["text"] = ""
    with pytest.raises(ValueError):
        select_holdout(pool, [], [], quotas, "seed")


@pytest.mark.parametrize("fault", ["settings", "prompt", "selection", "criteria", "acceptance"])
def test_protocol_drift_rejected_even_with_updated_outer_hash(study, fault):
    changed = copy.deepcopy(study)
    manifest = changed["manifest"]
    if fault == "settings":
        manifest["settings"] = {**manifest["settings"], "timeout_seconds": 90}
    elif fault == "prompt":
        manifest["prompt_hashes"][V4_PROMPT_VERSION] = "changed"
    elif fault == "selection":
        manifest["selection"]["manifest"]["inputs"][0]["text"] = "changed"
    elif fault == "criteria":
        manifest["holdout"]["criteria"]["rows"][0]["decision"] = "pending"
    else:
        manifest["acceptance"]["min_topic_missing_reduction"] = 0
    changed["hash"] = digest(manifest)
    with pytest.raises(ValueError):
        validated_protocol(changed)


def test_pair_is_bound_to_preregistered_criteria_and_resume_does_not_call(study, tmp_path):
    output = tmp_path / "pair"
    run_pair(client([candidates(), candidates(["MUSIC"], ["MUSIC_VOCAL"])]), study, "holdout", output)
    report = compare_pair(study, "holdout", summaries(output))
    assert report["decision"] == "eligible_for_team_review"
    assert report["adoption"].startswith("v2_retained")
    assert all(report["checks"].values())
    assert report["usage"][0]["attempts"] == 1
    assert report["usage"][0]["tokens"]["total_tokens"] == 10
    before = {p: p.read_bytes() for p in output.rglob("*.json")}
    run_pair(client([]), study, "holdout", output)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    with pytest.raises(ValueError, match="binding"):
        compare_pair(study, "development", summaries(output))


def test_development_improvement_never_authorizes_adoption(study, tmp_path):
    output = tmp_path / "pair"
    run_pair(client([candidates(), candidates(["MUSIC"], ["MUSIC_VOCAL"])]), study, "development", output)
    report = compare_pair(study, "development", summaries(output))
    assert all(report["checks"].values())
    assert report["decision"] == "retain_v2"


def test_empty_v4_does_not_pass_semantic_gate(study, tmp_path):
    output = tmp_path / "pair"
    run_pair(client([candidates(["MUSIC"], ["MUSIC_VOCAL"]), candidates()]), study, "holdout", output)
    report = compare_pair(study, "holdout", summaries(output))
    assert report["decision"] == "retain_v2"
    assert not report["checks"]["topic_missing_reduction"]
    assert report["runs"][1]["all"]["after_retries_rate"] == 1


def test_both_resume_contracts_checked_before_calls(study, tmp_path):
    output = tmp_path / "pair"
    folder = output / V4_PROMPT_VERSION
    write_new(folder / "run-conditions.json", {"wrong": True})
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_pair(client([]), study, "holdout", output)
    assert not (output / PROMPT_VERSION).exists()


def test_report_rejects_legacy_unbound_runs(study, tmp_path):
    pack = study["manifest"]["holdout"]["criteria"]["pack"]
    a = run(tmp_path / "old", pack, [candidates()], PROMPT_VERSION)
    b = run(tmp_path / "new", pack, [candidates()], V4_PROMPT_VERSION)
    with pytest.raises(FileNotFoundError):
        compare_pair(study, "holdout", [a, b])


def test_report_rejects_two_v2_summaries(study, tmp_path):
    output = tmp_path / "pair"
    run_pair(client([candidates(), candidates()]), study, "holdout", output)
    old = summaries(output)[0]
    with pytest.raises(ValueError, match="one v2"):
        compare_pair(study, "holdout", [old, old])


def test_pending_review_and_small_holdout_block_registration(study):
    m = study["manifest"]
    acceptance = {**m["acceptance"], "min_holdout_channels": 2}
    with pytest.raises(ValueError, match="minimum size"):
        protocol(m["development"], m["holdout"], m["selection"], acceptance)
    criteria = copy.deepcopy(m["holdout"])
    criteria["criteria"]["rows"][0].update(decision="pending", reviewer="", evidence="", reason="")
    criteria["hash"] = digest(criteria["criteria"])
    with pytest.raises(ValueError, match="pending"):
        protocol(m["development"], criteria, m["selection"], m["acceptance"])


def test_cli_requires_paid_transfer_opt_in_before_loading_anything(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["run", "--protocol", "missing", "--split", "holdout", "--output", str(tmp_path / "run")])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_select_cli_publishes_pending_review_without_overwrite(tmp_path):
    for name, data in {"pool": [{"channel_id": "a", "text": "노래 커버", "stratum": "music"}],
                       "dev": [], "examples": [], "quotas": {"music": 1}}.items():
        write_new(tmp_path / (name + ".json"), data)
    output = tmp_path / "prepared"
    args = ["select", "--candidates", str(tmp_path / "pool.json"), "--development-inputs", str(tmp_path / "dev.json"),
            "--examples", str(tmp_path / "examples.json"), "--quotas", str(tmp_path / "quotas.json"),
            "--seed", "seed", "--output", str(output)]
    assert main(args) == 0
    assert (output / "holdout/human-review.private.csv").exists()
    assert len(read(output / "holdout/inputs.private.json")) == 1
    before = (output / "selection.private.json").read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert (output / "selection.private.json").read_bytes() == before


def test_historical_cohorts_preserve_overlap_and_missing_provenance(tmp_path):
    taxonomy = read("data/creator-subtopics.json")
    inputs = [{"channel_id": cid, "text": "노래 커버"} for cid in ("a", "b", "c")]
    pack = prepare(inputs, taxonomy)
    old = tmp_path / "old"
    from src.detail_tagging_batch import run_batch
    summary = run_batch(client([candidates(["MUSIC"]), candidates(["MUSIC"]), candidates()]), inputs,
                        taxonomy, "model", old, prompt_version=PROMPT_VERSION)
    # Historical c failed; a succeeded only after the first snapshot.
    index = {r["channel_id"]: "revisions/" + r["revision"] + ".json" for r in summary["rows"][:2]}
    base = {"model": "model", "policy": summary["conditions"]["policy_version"],
            "criteria_hash": digest(taxonomy), "source_manifest_hash": "historical-manifest"}
    first = {**base, "completed": 1, "failures": [{"channel_id": "a"}, {"channel_id": "c"}]}
    final = {**base, "completed": 2, "failures": [{"channel_id": "c"}]}
    paths = [tmp_path / (name + ".json") for name in ("first", "final", "index")]
    for path, value in zip(paths, (first, final, index)):
        write_new(path, value)
    newer = run(tmp_path / "v3", pack, [candidates(), candidates(["MUSIC"]), candidates()],
                "grounded-detail-candidates-v3")
    # Historical timeout was never recorded: preserve that unknown in both sources.
    raw = read(newer)
    del raw["conditions"]["timeout_seconds"]
    newer.write_text(json.dumps(raw), encoding="utf-8")
    for attempt in (newer.parent / "attempts").rglob("*.json"):
        raw = read(attempt)
        del raw["conditions"]["timeout_seconds"]
        attempt.write_text(json.dumps(raw), encoding="utf-8")
    report = historical_cohorts(inputs, taxonomy, *paths, old / "revisions", newer)
    assert report["cohorts"] == {"v3_unclassified_transition": ["a"], "baseline_final_failure": ["c"],
                                 "baseline_retry_success": ["a"]}
    assert len(report["overlap"]["a"]) == 2
    assert report["historical_only"]
    assert "timeout_seconds" not in report
    final["failures"] = [{"channel_id": "unknown"}]
    paths[1].write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot differs"):
        historical_cohorts(inputs, taxonomy, *paths, old / "revisions", newer)


def test_holds_block_adoption_even_when_improvement_gate_passes(study, tmp_path):
    m = copy.deepcopy(study["manifest"])
    criteria = m["holdout"]
    row = next(r for r in criteria["criteria"]["rows"] if r["code"] == "MUSIC_INSTRUMENT")
    row.update(decision="hold", reason="미합의")
    criteria["hash"] = digest(criteria["criteria"])
    value = protocol(m["development"], criteria, m["selection"], m["acceptance"])
    output = tmp_path / "held"
    run_pair(client([candidates(), candidates(["MUSIC"], ["MUSIC_VOCAL"])]), value, "holdout", output)
    report = compare_pair(value, "holdout", summaries(output))
    assert report["checks"]["topic_missing_reduction"]
    assert not report["checks"]["complete_human_review"]
    assert report["decision"] == "retain_v2"


def test_model_failure_blocks_adoption_and_preserves_failed_usage(study, tmp_path):
    bad = {"parents": [{"code": "MUSIC", "evidence": "없는 인용"}], "topics": []}
    output = tmp_path / "failure"
    run_pair(client([bad, bad, bad, candidates(["MUSIC"], ["MUSIC_VOCAL"])]), study, "holdout", output)
    report = compare_pair(study, "holdout", summaries(output))
    assert not report["checks"]["no_model_failures"]
    assert report["decision"] == "retain_v2"
    assert report["usage"][0]["attempts"] == 3
    assert report["usage"][0]["tokens"]["total_tokens"] == 30
    assert report["runs"][0]["all"]["classification_denominator"] == 0


def test_team_decision_revalidates_both_splits_and_invalidates_reused_holdout(study, tmp_path):
    for split in ("development", "holdout"):
        run_pair(client([candidates(), candidates(["MUSIC"], ["MUSIC_VOCAL"])]), study, split, tmp_path / split)
    dev, test = summaries(tmp_path / "development"), summaries(tmp_path / "holdout")
    record = record_decision(study, dev, test, "adopt_v4", "team", "사전 기준 확인")
    assert record["manifest"]["choice"] == "adopt_v4"
    with pytest.raises(ValueError, match="untouched holdout"):
        record_decision(study, dev, test, "adopt_v4", "team", "수정 후", True)
    retry = record_decision(study, dev, test, "additional_experiment", "team", "프롬프트 수정", True)
    assert retry["manifest"]["holdout_status"].startswith("reclassified")
    with pytest.raises(ValueError, match="reviewer"):
        record_decision(study, dev, test, "retain_v2", "", "")


def test_report_requires_exact_human_hash_even_if_legacy_evaluate_allows_missing(study, tmp_path):
    output = tmp_path / "pair"
    run_pair(client([candidates(), candidates()]), study, "holdout", output)
    paths = summaries(output)
    raw = read(paths[0])
    del raw["conditions"]["human_criteria_hash"]
    paths[0].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="human criteria"):
        compare_pair(study, "holdout", paths)


def test_cli_reports_safe_protocol_failure_before_client_creation(study, tmp_path, capsys, monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: pytest.fail("No provider client expected"))
    changed = copy.deepcopy(study)
    changed["manifest"]["prompt_hashes"][V4_PROMPT_VERSION] = "changed"
    changed["hash"] = digest(changed["manifest"])
    path = tmp_path / "protocol.json"
    write_new(path, changed)
    output = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        main(["run", "--protocol", str(path), "--split", "holdout", "--output", str(output),
              "--approved-paid-transfer"])
    assert error.value.code == 2
    assert "Protocol/settings/prompt changed; preregister a new evaluation" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("exception,hint", [(ValueError, "schemas"), (KeyError, "schemas"),
                                           (TypeError, "schemas"), (OSError, "permissions"),
                                           (RuntimeError, "private attempt records")])
def test_cli_dependency_errors_give_guidance_without_private_details(tmp_path, capsys, monkeypatch, exception, hint):
    import src.detail_tagging_holdout as holdout
    def fail_read(path):
        raise exception("PRIVATE_SOURCE_SENTINEL")
    monkeypatch.setattr(holdout, "read", fail_read)
    output = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        main(["register", "--development", "private", "--holdout", "private", "--selection", "private",
              "--acceptance", "private", "--output", str(output)])
    message = capsys.readouterr().err
    assert error.value.code == 2 and hint in message
    assert "PRIVATE_SOURCE_SENTINEL" not in message
    assert not output.exists()


def test_cli_shortage_message_does_not_echo_private_strata(tmp_path, capsys):
    for name, value in {"pool": [{"channel_id": "a", "text": "PRIVATE_SOURCE_SENTINEL", "stratum": "PRIVATE_STRATUM_SENTINEL"}],
                        "dev": [], "examples": [], "quotas": {"PRIVATE_STRATUM_SENTINEL": 2}}.items():
        write_new(tmp_path / (name + ".json"), value)
    output = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        main(["select", "--candidates", str(tmp_path / "pool.json"), "--development-inputs", str(tmp_path / "dev.json"),
              "--examples", str(tmp_path / "examples.json"), "--quotas", str(tmp_path / "quotas.json"),
              "--seed", "seed", "--output", str(output)])
    message = capsys.readouterr().err
    assert error.value.code == 2 and "Insufficient independent candidates by stratum" in message
    assert "PRIVATE_SOURCE_SENTINEL" not in message and "PRIVATE_STRATUM_SENTINEL" not in message
    assert not output.exists()


def test_public_attempt_history_validator_rejects_missing_records(tmp_path):
    from src.detail_tagging_human_eval import validate_attempt_history
    with pytest.raises(ValueError, match="Missing or invalid attempt history"):
        validate_attempt_history(tmp_path, {"channel_id": "a", "attempts": 1}, "text", {}, {"max_attempts": 3})
