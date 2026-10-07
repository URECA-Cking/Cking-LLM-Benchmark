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
    class FailedProcess:
        returncode = 1
        pid = 123
        def communicate(self, **kwargs):
            return b"private-key", b"raw-bio"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FailedProcess())
    monkeypatch.setattr(runner, "stop_process_tree", lambda process: None)
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


@pytest.mark.parametrize("stage", ["prerequisites", "containers", "be_build_or_start", "verification", "success", "cleanup", "running_timeout", "passed_crash", "corrupt", "audit", "channel"])
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
            status = "verification_failure" if stage in {"verification", "audit"} else "running" if stage == "running_timeout" else "passed"
            report = {"status": status, "checks": ["first_check"], "completed": stage != "running_timeout"}
            if stage in {"verification", "audit"}:
                report.update(failedCheck="be_final_fixture_audit" if stage == "audit" else "batch_lifecycle",
                              failureLocation=[{"file": "scenarios.py", "line": 1}])
            if stage == "channel":
                report.update(status="environment_failure", failedPhase="control_channel",
                              environmentError="control_channel_eof")
            report_path.write_text("{broken" if stage == "corrupt" else json.dumps(report), encoding="utf-8")
            if stage in {"verification", "audit", "passed_crash", "channel"}:
                raise runner.EnvironmentFailure("command_failed")
            if stage == "running_timeout":
                raise runner.EnvironmentFailure("command_unavailable_or_timeout")
        if args[:2] == ["docker", "rm"] and stage == "cleanup":
            raise runner.EnvironmentFailure("cleanup failed")
        return "ok"
    monkeypatch.setattr(runner, "command", execute)
    code = runner.main(["--be-dir", str(be), "--output-dir", str(output)])
    report = json.loads(next(output.rglob("report.json")).read_text(encoding="utf-8"))
    assert code == (0 if stage == "success" else 1 if stage in {"verification", "audit"} else 2)
    assert report["cleanupVerified"] == (stage != "cleanup")
    created = [args[args.index("--name") + 1] for args in calls if args[:2] == ["docker", "run"]]
    deleted = [args[-1] for args in calls if args[:2] == ["docker", "rm"]]
    assert deleted == list(reversed(created))
    assert all(name.startswith("cking-e2e-") for name in deleted)
    assert "must-not-leak" not in capsys.readouterr().out

    if stage in {"prerequisites", "containers", "be_build_or_start"}:
        assert "failedCheck" not in report
        assert "failureLocation" not in report
    if stage == "channel":
        assert report["status"] == "environment_failure"
        assert report["failedPhase"] == "control_channel"
        assert report["environmentError"] == "control_channel_eof"
    if stage in {"verification", "audit"}:
        assert report["failedCheck"]
        assert report["failureLocation"]
    if stage in {"running_timeout", "passed_crash"}:
        assert report["status"] == "environment_failure"
        assert report["checks"] == ["first_check"]
    if stage == "corrupt":
        assert report["failedCheck"] == "report_read"
        assert report["failureLocation"] == "report.json"
        assert next(output.rglob("report.invalid.json")).read_text() == "{broken"


def test_timeout_stops_tree_and_reaps_process(monkeypatch):
    events = []
    class Process:
        returncode = -9
        pid = 123
        def communicate(self, timeout=None):
            events.append(("communicate", timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired("tool", timeout)
            return b"secret", b"secret"
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(runner, "stop_process_tree", lambda p: events.append(("stop_tree", p.pid)))
    with pytest.raises(runner.EnvironmentFailure, match="command_unavailable_or_timeout"):
        runner.command(["tool"], timeout=1)
    assert events == [("communicate", 1), ("stop_tree", 123), ("communicate", None)]


def test_posix_cleanup_targets_process_group(monkeypatch):
    import types
    calls = []
    monkeypatch.setattr(runner, "signal", types.SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(runner, "os", types.SimpleNamespace(name="posix", killpg=lambda *args: calls.append(args)))
    runner.stop_process_tree(types.SimpleNamespace(pid=123))
    assert calls == [(123, runner.signal.SIGKILL)]


def test_windows_cleanup_targets_descendants(monkeypatch):
    import types
    calls = []
    monkeypatch.setattr(runner, "os", types.SimpleNamespace(name="nt"))
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: (calls.append(args) or subprocess.CompletedProcess(args, 0)))
    runner.stop_process_tree(types.SimpleNamespace(pid=123, poll=lambda: None))
    assert calls == [["taskkill", "/PID", "123", "/T", "/F"]]


@pytest.mark.skipif(runner.os.name != "nt", reason="Windows process tree integration")
def test_windows_timeout_kills_real_child(tmp_path):
    import ctypes
    import sys
    pid_file = tmp_path / "child.pid"
    script = ("import subprocess,sys,time,pathlib; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
              "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
    with pytest.raises(runner.EnvironmentFailure, match="command_unavailable_or_timeout"):
        runner.command([sys.executable, "-c", script, str(pid_file)], timeout=2)
    child_pid = int(pid_file.read_text())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, child_pid)
    if handle:
        try:
            code = ctypes.c_uint32()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            assert code.value != 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)


@pytest.mark.parametrize("content", ["{", "[]", '{"status": {}, "checks": []}',
                                     '{"status": "passed", "checks": null}'])
def test_invalid_report_is_environment_failure(tmp_path, content):
    path = tmp_path / "report.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(runner.EnvironmentFailure, match="report_invalid_or_missing"):
        runner.read_report(path)
