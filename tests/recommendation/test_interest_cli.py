from __future__ import annotations

import json

import pytest

from src.recommendation import interest_cli
from src.recommendation.cache import JsonModelCache
from src.recommendation.manifest import build_manifest, write_manifest
from src.recommendation.models import CreatorProfile
from tests.recommendation.test_interest import FakeEmbedding, vector
from tests.recommendation.test_interest_batch import FakeBackend


def arguments(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, build_manifest([CreatorProfile(1, "creator")]))
    return ["--manifest", str(manifest), "--output-dir", str(tmp_path / "batch"),
            "--cache", str(tmp_path / "cache.json")]


def test_defaults_and_dry_run_use_local_bge_without_backend_or_tagger(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    parsed = interest_cli.build_parser().parse_args(["dry-run", *args])
    assert parsed.top_n == 20 and parsed.evaluation_top_n == 10
    assert parsed.embedding_provider == "local"
    client = FakeEmbedding({"creator": vector(1)})
    monkeypatch.setattr(interest_cli, "create_interest_embedding_client", lambda provider: client)
    monkeypatch.setattr(interest_cli, "CkingBackendClient", lambda *a, **k: pytest.fail("BE 호출 금지"))
    assert interest_cli.main(["dry-run", *args]) == 0
    assert json.loads((tmp_path / "batch" / "summary.json").read_text(encoding="utf-8"))["interestCount"] == 17


def test_apply_uses_api_key_and_returns_failure_exit_code(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    backend = FakeBackend(wrong={"inputHash": "a" * 64})
    seen = []

    def create_backend(config, **kwargs):
        seen.append(kwargs)
        return backend

    monkeypatch.setenv("CKING_RECOMMENDATION_API_KEY", "fake-key")
    monkeypatch.delenv("CKING_ADMIN_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(interest_cli, "CkingBackendClient", create_backend)
    monkeypatch.setattr(interest_cli, "create_interest_embedding_client",
                        lambda provider: FakeEmbedding({"creator": vector(1)}))
    assert interest_cli.main(["apply", *args, "--be-base-url", "https://be.example"]) == 1
    assert seen == [{"access_token": None, "recommendation_api_key": "fake-key"}]
    assert len(backend.puts) == 17


def test_apply_without_credentials_stops_before_creating_models(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.delenv("CKING_RECOMMENDATION_API_KEY", raising=False)
    monkeypatch.delenv("CKING_ADMIN_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(interest_cli, "create_interest_embedding_client", lambda p: pytest.fail("모델 호출 금지"))
    with pytest.raises(RuntimeError, match="CKING_RECOMMENDATION_API_KEY"):
        interest_cli.main(["apply", *args, "--be-base-url", "https://be.example"])


@pytest.mark.parametrize("option,value", [
    ("--zero-shot-tau", "nan"), ("--m3-bonus", "-0.1"), ("--score-decimals", "-1"),
    ("--top-n", "0"), ("--evaluation-top-n", "21"), ("--taxonomy-version", "v0.1"),
    ("--taxonomy-hash", "a" * 64), ("--embedding-model-version", "wrong"),
])
def test_invalid_configuration_precedes_model_creation(tmp_path, monkeypatch, option, value):
    args = arguments(tmp_path)
    monkeypatch.setattr(interest_cli, "create_interest_embedding_client", lambda p: pytest.fail("모델 생성 금지"))
    with pytest.raises(ValueError):
        interest_cli.main(["dry-run", *args, option, value])
    assert not (tmp_path / "batch").exists()


def test_cache_input_or_payload_path_collision_rejected_without_overwrite(tmp_path):
    args = arguments(tmp_path)
    path = tmp_path / "manifest.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="경로"):
        interest_cli.main(["dry-run", *args, "--cache", str(path)])
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="경로"):
        interest_cli.main(["dry-run", *args, "--cache", str(tmp_path / "batch" / "summary.json")])


def test_model_cache_is_shared_with_existing_recommendation_generation(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    client = FakeEmbedding({"creator": vector(1)})
    cache = JsonModelCache(tmp_path / "cache.json")
    from src.recommendation.cache import text_hash

    cache.put_embedding(text_hash("creator"), "BAAI/bge-m3@deepinfra-v1", vector(1).tolist())
    monkeypatch.setattr(interest_cli, "create_interest_embedding_client", lambda provider: client)
    assert interest_cli.main(["dry-run", *args, "--embedding-provider", "deepinfra"]) == 0
    assert "creator" not in client.calls[0]
