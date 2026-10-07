from __future__ import annotations

import io
import json

import pytest

from src.recommendation.e2e import scenarios


@pytest.mark.parametrize("response", ["", "{invalid\n", "[]\n"])
def test_control_rejects_eof_and_invalid_response(monkeypatch, response):
    monkeypatch.setattr(scenarios.sys, "stdin", io.StringIO(response))
    monkeypatch.setattr(scenarios.sys, "stdout", io.StringIO())
    with pytest.raises(scenarios.ControlChannelFailure):
        scenarios.control("snapshot")


def test_control_broken_pipe_is_environment_failure(monkeypatch):
    class ClosedOutput:
        def write(self, value):
            raise BrokenPipeError("private transport detail")
    monkeypatch.setattr(scenarios.sys, "stdout", ClosedOutput())
    with pytest.raises(scenarios.ControlChannelFailure, match="control_channel_write_failed"):
        scenarios.control("snapshot")


@pytest.mark.parametrize("failure", ["eof", "invalid", "write", "assertion", "initial_eof"])
def test_scenario_failure_report_classification(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("CKING_E2E_OUTPUT", str(tmp_path))
    monkeypatch.setenv("CKING_E2E_BE_ROOT", str(tmp_path))
    monkeypatch.setattr(scenarios, "git_sha", lambda path: "a" * 40)
    response = "{invalid\n" if failure == "invalid" else ""
    config = "" if failure == "initial_eof" else json.dumps({"taxonomyHash": "test"}) + "\n"
    monkeypatch.setattr(scenarios.sys, "stdin", io.StringIO(config + response))
    monkeypatch.setattr(scenarios.sys, "stdout", io.StringIO())
    class Suite:
        checks = [{"name": "previous_check", "status": "passed"}]
        def __init__(self, config, output):
            pass
        def batch_lifecycle(self):
            if failure == "assertion":
                raise AssertionError("private fixture detail")
            if failure == "write":
                class ClosedOutput:
                    def write(self, value):
                        raise BrokenPipeError("private transport detail")
                monkeypatch.setattr(scenarios.sys, "stdout", ClosedOutput())
            scenarios.control("snapshot")
    monkeypatch.setattr(scenarios, "Suite", Suite)
    assert scenarios.main() == (1 if failure == "assertion" else 2)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["completed"] is True
    assert report["status"] == ("verification_failure" if failure == "assertion" else "environment_failure")
    assert report["failureLocation"]
    if failure != "assertion":
        assert report["failedPhase"] == "control_channel"
        assert report["environmentError"].startswith("control_channel_")
    if failure != "initial_eof":
        assert report["checks"] == Suite.checks
        assert report["failedCheck"] == "batch_lifecycle"
    assert "private" not in json.dumps(report)
