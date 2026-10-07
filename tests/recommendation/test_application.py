import json
import pytest
from src.recommendation.application import allocate_sequence, bind_application, validate_sequence


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
