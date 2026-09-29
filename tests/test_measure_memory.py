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
        return SimpleNamespace(stdout="경고 로그\n" + json.dumps(payload))

    monkeypatch.setattr("subprocess.run", fake_run)

    pipeline.cmd_memory(SimpleNamespace(device="cpu"))

    saved = json.loads((tmp_path / "e4_embedding.json").read_text(encoding="utf-8"))
    assert calls == [["m1", "--device", "cpu"], ["m2", "--device", "cpu"], ["reranker", "--device", "cpu"]]  # 모델마다 새 프로세스
    assert saved["m1"]["dim"] == 1024 and saved["m1"]["memory"]["cpu"]["peak_after_run_mb"] == 2000.0
    assert "bge-reranker-v2-m3" in saved and saved["m2"]["memory"]["cpu"]["baseline_mb"] == 400.0


def test_cmd_memory_keeps_other_device_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "LOCAL_EMBEDDING_MODELS", {"m1": {}})
    (tmp_path / "e4_embedding.json").write_text(json.dumps({"m1": {"memory": {"cpu": {"peak_after_run_mb": 1.0}}}}), encoding="utf-8")
    payload = {"baseline_mb": 1.0, "peak_after_load_mb": 2.0, "peak_after_run_mb": 3.0}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(payload)))

    pipeline.cmd_memory(SimpleNamespace(device="auto"))

    memory = json.loads((tmp_path / "e4_embedding.json").read_text(encoding="utf-8"))["m1"]["memory"]
    assert memory["cpu"]["peak_after_run_mb"] == 1.0 and memory["auto"]["peak_after_run_mb"] == 3.0
