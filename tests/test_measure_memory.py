import json
import sys
from types import SimpleNamespace

import pytest

from src import measure_memory as mm
from src import pipeline


def test_peak_rss_mb_converts_platform_units(monkeypatch) -> None:
    monkeypatch.setattr(mm.resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=2 * 1024 * 1024))

    monkeypatch.setattr(mm.sys, "platform", "darwin")
    assert mm.peak_rss_mb() == pytest.approx(2.0)  # macOS는 바이트
    monkeypatch.setattr(mm.sys, "platform", "linux")
    assert mm.peak_rss_mb() == pytest.approx(2048.0)  # Linux는 KB


def test_cmd_memory_records_each_model_and_preserves_existing_entries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "LOCAL_EMBEDDING_MODELS", {"m1": {}, "m2": {}})
    (tmp_path / "e4_embedding.json").write_text(json.dumps({"m1": {"dim": 1024}}), encoding="utf-8")
    calls = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd[3:])
        payload = {"baseline_mb": 400.0, "peak_after_load_mb": 900.0, "peak_after_run_mb": 2000.0}
        return SimpleNamespace(returncode=0, stderr="", stdout="경고 로그\n" + json.dumps(payload))

    monkeypatch.setattr("subprocess.run", fake_run)

    pipeline.cmd_memory(SimpleNamespace(device="cpu"))

    saved = json.loads((tmp_path / "e4_embedding.json").read_text(encoding="utf-8"))
    assert calls == [[n, "--device", "cpu"] for n in ("m1", "m2", "reranker", "m5")]  # 모델마다 새 프로세스, M5는 임베딩+리랭커 함께
    assert saved["m1"]["dim"] == 1024 and saved["m1"]["memory"]["cpu"]["peak_after_run_mb"] == 2000.0
    assert "bge-reranker-v2-m3" in saved and "M5_bge-m3+reranker" in saved and saved["m2"]["memory"]["cpu"]["baseline_mb"] == 400.0


def test_cmd_memory_keeps_other_device_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "LOCAL_EMBEDDING_MODELS", {"m1": {}})
    (tmp_path / "e4_embedding.json").write_text(json.dumps({"m1": {"memory": {"cpu": {"peak_after_run_mb": 1.0}}}}), encoding="utf-8")
    payload = {"baseline_mb": 1.0, "peak_after_load_mb": 2.0, "peak_after_run_mb": 3.0}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(payload)))

    pipeline.cmd_memory(SimpleNamespace(device="auto"))

    memory = json.loads((tmp_path / "e4_embedding.json").read_text(encoding="utf-8"))["m1"]["memory"]
    assert memory["cpu"]["peak_after_run_mb"] == 1.0 and memory["auto"]["peak_after_run_mb"] == 3.0


def test_cmd_memory_saves_finished_models_and_shows_failure_output(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "LOCAL_EMBEDDING_MODELS", {"m1": {}, "m2": {}})
    payload = {"baseline_mb": 1.0, "peak_after_load_mb": 2.0, "peak_after_run_mb": 3.0}

    def fake_run(cmd, **_kwargs):
        if cmd[3] == "m2":
            return SimpleNamespace(returncode=1, stderr="OSError: 모델 파일을 찾을 수 없음", stdout="")
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(payload))

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="모델 파일을 찾을 수 없음"):
        pipeline.cmd_memory(SimpleNamespace(device="cpu"))

    saved = json.loads((tmp_path / "e4_embedding.json").read_text(encoding="utf-8"))
    assert saved["m1"]["memory"]["cpu"]["peak_after_run_mb"] == 3.0 and "m2" not in saved  # 실패 전 결과는 남아 있다
