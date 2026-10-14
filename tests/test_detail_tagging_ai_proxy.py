import copy

import pytest

from src.detail_tagging import PROMPT_VERSION, V4_PROMPT_VERSION, digest
from src.detail_tagging_batch import read, run_batch
from src.detail_tagging_human_eval import (evaluate, freeze, main, prepare,
                                          validated_criteria)
from tests.test_detail_tagging_human_eval import candidates, client, judged, write_csv


@pytest.fixture
def ai_criteria(tmp_path):
    pack = prepare([{"channel_id": "a", "text": "노래 커버"}], read("data/creator-subtopics.json"))
    rows = judged(pack)
    for row in rows:
        row["reviewer"] = "AI:Codex (user-authorized proxy)"
    path = tmp_path / "ai.csv"
    write_csv(path, rows)
    return freeze(pack, path, "ai_proxy")


def test_ai_freeze_requires_explicit_mode_and_remains_unusable_as_human(tmp_path, ai_criteria):
    pack = ai_criteria["criteria"]["pack"]
    assert ai_criteria["criteria"]["origin"] == "ai_proxy"
    assert validated_criteria(ai_criteria, "ai_proxy")[0] == pack
    with pytest.raises(ValueError, match="criteria changed"):
        validated_criteria(ai_criteria)
    with pytest.raises(ValueError, match="explicit AI proxy"):
        freeze(pack, tmp_path / "ai.csv")


def test_ai_mode_requires_ai_reviewer_labels(tmp_path, ai_criteria):
    pack = ai_criteria["criteria"]["pack"]
    rows = judged(pack)
    path = tmp_path / "unlabeled.csv"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="AI: labels"):
        freeze(pack, path, "ai_proxy")


def test_origin_relabeling_does_not_turn_ai_into_human(ai_criteria):
    forged = copy.deepcopy(ai_criteria)
    forged["criteria"]["origin"] = "human_declared"
    forged["hash"] = digest(forged["criteria"])
    with pytest.raises(ValueError, match="explicit AI proxy"):
        validated_criteria(forged)


def test_ai_pair_evaluates_semantics_without_human_claims_or_adoption(tmp_path, ai_criteria):
    pack = ai_criteria["criteria"]["pack"]["manifest"]
    summaries = []
    for version, payload in [(PROMPT_VERSION, candidates()), (V4_PROMPT_VERSION, candidates(["MUSIC"], ["MUSIC_VOCAL"]))]:
        output = tmp_path / version
        report = run_batch(client([payload]), pack["inputs"], pack["taxonomy"], "model", output,
                           prompt_version=version, ai_proxy_criteria=ai_criteria)
        assert report["human_review"] == "not_performed"
        assert report["conditions"]["reference_origin"] == "ai_proxy"
        assert report["conditions"]["reference_criteria_hash"] == ai_criteria["hash"]
        assert "human_criteria_hash" not in report["conditions"]
        summaries.append(next(output.glob("summary-*.json")))
    result = evaluate(ai_criteria, summaries, "ai_proxy")
    assert result["human_review"] == "not_performed"
    assert result["reference_origin"] == "ai_proxy"
    assert result["reference_review"] == "review_complete"
    assert "human_criteria_hash" not in result
    assert result["adoption"] == "v2_retained_ai_proxy_diagnostic_only"
    assert result["runs"][0]["all"]["topics"]["missing"] == 1
    assert result["runs"][1]["all"]["topics"]["missing"] == 0
    assert "human_positive_tags" not in result["runs"][1]["all"]["topics"]
    assert result["runs"][1]["all"]["topics"]["reference_positive_tags"] == 1
    with pytest.raises(ValueError):
        evaluate(ai_criteria, summaries)


@pytest.mark.parametrize("fault", ["both", "human_argument", "input", "pending"])
def test_invalid_ai_batch_rejected_before_calls(tmp_path, ai_criteria, fault):
    pack = ai_criteria["criteria"]["pack"]["manifest"]
    inputs = copy.deepcopy(pack["inputs"])
    kwargs = {"ai_proxy_criteria": ai_criteria}
    if fault == "both":
        kwargs["human_criteria"] = ai_criteria
    elif fault == "human_argument":
        kwargs = {"human_criteria": ai_criteria}
    elif fault == "input":
        inputs[0]["text"] = "changed"
    else:
        changed = copy.deepcopy(ai_criteria)
        changed["criteria"]["rows"][0].update(decision="pending", reviewer="", evidence="", reason="")
        changed["hash"] = digest(changed["criteria"])
        kwargs["ai_proxy_criteria"] = changed
    output = tmp_path / "unused"
    with pytest.raises(ValueError):
        run_batch(client([]), inputs, pack["taxonomy"], "model", output, **kwargs)
    assert not output.exists()


def test_resume_cannot_mix_ai_criteria_or_drop_mode(tmp_path, ai_criteria):
    pack = ai_criteria["criteria"]["pack"]["manifest"]
    output = tmp_path / "run"
    run_batch(client([candidates()]), pack["inputs"], pack["taxonomy"], "model", output,
              ai_proxy_criteria=ai_criteria)
    changed = copy.deepcopy(ai_criteria)
    changed["criteria"]["rows"][0]["reason"] = "updated"
    changed["hash"] = digest(changed["criteria"])
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), pack["inputs"], pack["taxonomy"], "model", output, ai_proxy_criteria=changed)
    with pytest.raises(ValueError, match="conditions mismatch"):
        run_batch(client([]), pack["inputs"], pack["taxonomy"], "model", output)


def test_unbound_historical_summary_cannot_be_ai_evaluation(tmp_path, ai_criteria):
    pack = ai_criteria["criteria"]["pack"]["manifest"]
    output = tmp_path / "run"
    run_batch(client([candidates()]), pack["inputs"], pack["taxonomy"], "model", output)
    with pytest.raises(ValueError, match="AI proxy run criteria differ"):
        evaluate(ai_criteria, [next(output.glob("summary-*.json"))], "ai_proxy")


def test_cli_ai_freeze_requires_flag_and_does_not_overwrite(tmp_path, ai_criteria):
    from src.detail_tagging_batch import write_new
    pack_path = tmp_path / "pack.json"
    write_new(pack_path, ai_criteria["criteria"]["pack"])
    output = tmp_path / "locked.json"
    args = ["freeze", "--pack", str(pack_path), "--reviews", str(tmp_path / "ai.csv"), "--output", str(output)]
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()
    assert main(args + ["--ai-proxy"]) == 0
    assert read(output)["criteria"]["origin"] == "ai_proxy"
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args + ["--ai-proxy"])
    assert output.read_bytes() == before
