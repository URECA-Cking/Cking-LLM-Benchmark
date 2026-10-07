import json
import pytest
from src.recommendation.application import allocate_sequence, bind_application, resolve_sequence, validate_sequence


def test_ledger_is_target_scoped_and_never_reuses_reserved_number(tmp_path):
    ledger = tmp_path / "applications.json"
    assert allocate_sequence(ledger, "a") == 1
    assert allocate_sequence(ledger, "a") == 2
    assert allocate_sequence(ledger, "b") == 1
    assert allocate_sequence(ledger, "a") == 3


@pytest.mark.parametrize("value", [True, 0, -1, 1.2, "1", 9223372036854775808])
def test_invalid_sequence(value):
    with pytest.raises(ValueError):
        validate_sequence(value)


def test_legacy_apply_state_resets_without_invalidating_generation():
    record = {"generationStatus": "success", "payloadSha256": "hash", "applyStatus": "success"}
    checkpoint = {"applyTarget": "a", "creators": {"1": record}}
    bind_application(checkpoint, "a", 3, "creators")
    assert record == {"generationStatus": "success", "payloadSha256": "hash", "applyStatus": "pending"}
    record["applyStatus"] = "success"
    bind_application(checkpoint, "a", 3, "creators")
    assert record["applyStatus"] == "success"


def test_corrupt_or_exhausted_ledger_is_not_reset(tmp_path):
    ledger = tmp_path / "applications.json"
    for value in [True, -1, 9223372036854775807]:
        ledger.write_text(json.dumps({"a": value}))
        with pytest.raises(ValueError):
            allocate_sequence(ledger, "a")


@pytest.mark.parametrize("state", [None, {}, {"a": 1}])
def test_resume_rejects_missing_or_rolled_back_ledger(tmp_path, state):
    ledger = tmp_path / "applications.json"
    if state is not None:
        ledger.write_text(json.dumps(state))
    checkpoint = {"applyTarget": "a", "applicationSequence": 2}
    with pytest.raises(ValueError, match="ledger"):
        resolve_sequence(ledger, "a", checkpoint)
    assert (json.loads(ledger.read_text()) if ledger.exists() else None) == state


def test_resolution_requires_path_and_preserves_reserved_sequence(tmp_path):
    with pytest.raises(ValueError, match="sequence_ledger"):
        resolve_sequence(None, "a", {})
    ledger = tmp_path / "applications.json"
    assert allocate_sequence(ledger, "a") == 1
    assert allocate_sequence(ledger, "a") == 2
    checkpoint = {"applyTarget": "a", "applicationSequence": 1}
    assert resolve_sequence(ledger, "a", checkpoint) == 1
    assert resolve_sequence(ledger, "a", checkpoint, explicit=2) == 2
    assert resolve_sequence(ledger, "b", checkpoint) == 1
    assert json.loads(ledger.read_text()) == {"a": 2, "b": 1}


@pytest.mark.parametrize("error", [None, "invalid", {"stage": "generation"}, {"stage": "apply"}])
def test_rebinding_tolerates_non_object_records_and_errors(error):
    record = {"applyStatus": "success", "error": error}
    checkpoint = {"creators": {"1": record, "2": None, "3": "invalid"}}
    bind_application(checkpoint, "a", 1, "creators")
    assert record["applyStatus"] == "pending"
    if isinstance(error, dict) and error.get("stage") == "apply":
        assert "error" not in record
    else:
        assert record["error"] == error
