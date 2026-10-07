from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from src.recommendation.e2e import runner


def test_environment_does_not_inherit_model_secrets_or_database_settings(monkeypatch):
    for name in ("OPENAI_API_KEY", "DEEPINFRA_API_KEY", "SPRING_DATASOURCE_URL", "SPRING_PROFILES_ACTIVE",
                 "CKING_RECOMMENDATION_API_KEY", "JWT_SECRET", "JAVA_TOOL_OPTIONS", "GRADLE_OPTS"):
        monkeypatch.setenv(name, "must-not-leak")
    assert "must-not-leak" not in runner.clean_environment().values()


def test_failed_command_never_exposes_tool_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b"private-key", b"raw-bio"))
    with pytest.raises(runner.EnvironmentFailure) as caught:
        runner.command(["tool"])
    assert str(caught.value) == "command_failed"


def test_container_binding_must_be_loopback(monkeypatch):
    monkeypatch.setattr(runner, "command", lambda *a, **k: "0.0.0.0:3306")
    with pytest.raises(runner.EnvironmentFailure):
        runner.container_port("owned", 3306, {})


def test_remote_docker_is_rejected_before_any_container_is_created(monkeypatch):
    monkeypatch.setattr(runner, "command", lambda *a, **k: "ssh://production.example")
    with pytest.raises(runner.EnvironmentFailure):
        runner.require_local_docker({})
    with pytest.raises(runner.EnvironmentFailure):
        runner.require_local_docker({"DOCKER_HOST": "tcp://remote.example:2376"})


@pytest.mark.parametrize("stage", ["prerequisites", "containers", "be_build_or_start", "verification", "success", "cleanup"])
def test_exit_classification_and_owned_cleanup(tmp_path, monkeypatch, stage, capsys):
    be = tmp_path / "be"
    be.mkdir()
    (be / "gradlew").touch()
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(runner, "git_sha", lambda path: "a" * 40)
    monkeypatch.setattr(runner, "wait_healthy", lambda *a, **k: None)
    def execute(args, **kwargs):
        calls.append(args)
        if args[:3] == ["docker", "context", "inspect"]:
            return "npipe:////./pipe/docker_engine"
        if args[:2] == ["docker", "version"] and stage == "prerequisites":
            raise runner.EnvironmentFailure("unavailable")
        if args[:2] == ["docker", "run"] and stage == "containers":
            raise runner.EnvironmentFailure("unavailable")
        if args[:2] == ["docker", "port"]:
            return "127.0.0.1:40000"
        if "recommendationE2e" in args:
            if stage == "be_build_or_start":
                raise runner.EnvironmentFailure("unavailable")
            report_path = Path(kwargs["env"]["CKING_E2E_OUTPUT"]) / "report.json"
            report_path.write_text(json.dumps({"status": "verification_failure" if stage == "verification" else "passed",
                                               "checks": [], "externalModelCalls": 0}), encoding="utf-8")
            if stage == "verification":
                raise runner.EnvironmentFailure("failed")
        if args[:2] == ["docker", "rm"] and stage == "cleanup":
            raise runner.EnvironmentFailure("cleanup failed")
        return "ok"
    monkeypatch.setattr(runner, "command", execute)
    code = runner.main(["--be-dir", str(be), "--output-dir", str(output)])
    report = json.loads(next(output.rglob("report.json")).read_text(encoding="utf-8"))
    assert code == (0 if stage == "success" else 1 if stage == "verification" else 2)
    assert report["cleanupVerified"] == (stage != "cleanup")
    created = [args[args.index("--name") + 1] for args in calls if args[:2] == ["docker", "run"]]
    deleted = [args[-1] for args in calls if args[:2] == ["docker", "rm"]]
    assert deleted == list(reversed(created))
    assert all(name.startswith("cking-e2e-") for name in deleted)
    assert "must-not-leak" not in capsys.readouterr().out
