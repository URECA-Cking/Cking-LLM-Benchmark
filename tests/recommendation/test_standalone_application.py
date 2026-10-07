"""단독 CLI와 Python API의 실행 번호·실패 집계 회귀 검증."""
import json

import pytest

from src.recommendation import batch_cli, interest_cli
from src.recommendation.batch import BatchPaths, RecommendationBatch
from src.recommendation.http import BackendRequestError
from src.recommendation.interest_batch import InterestBatchPaths, InterestRecommendationBatch
from src.recommendation.manifest import build_manifest, write_manifest
from src.recommendation.models import CreatorProfile
from tests.recommendation.test_batch import FakeRecommender, FakeBackend, config, manifest
from tests.recommendation.test_daily_cli import Backend, Tagger, INTRO, OTHER
from tests.recommendation.test_interest import FakeEmbedding, vector, recommender, categories


@pytest.fixture(params=["similar", "interest"])
def standalone(request, tmp_path, monkeypatch):
    backend = Backend()
    embedding = FakeEmbedding({INTRO: vector(1), OTHER: vector(0, 1)})
    tagger = Tagger()
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, build_manifest(
        [CreatorProfile(row["creatorId"], row["introText"]) for row in backend.profiles]))
    module = batch_cli if request.param == "similar" else interest_cli
    monkeypatch.setenv("CKING_RECOMMENDATION_API_KEY", "fake-key")
    monkeypatch.setattr(module, "CkingBackendClient", lambda *a, **k: backend)
    monkeypatch.setattr(batch_cli, "create_external_clients", lambda *a: (embedding, tagger))
    monkeypatch.setattr(interest_cli, "create_interest_embedding_client", lambda *a: embedding)
    output = tmp_path / "batch"
    args = ["apply", "--manifest", str(manifest_path), "--output-dir", str(output),
            "--cache", str(tmp_path / "cache.json"), "--be-base-url", "https://be.example"]
    return module, args, backend, embedding, tagger, output, tmp_path / "cache.applications.json"


def test_standalone_resume_preserves_sequence_and_cached_generation(standalone):
    module, args, backend, embedding, tagger, output, ledger = standalone
    path = ("/api/admin/creators/2/similar" if module is batch_cli
            else f"/api/admin/interests/{categories()[0].code}/recommendations")
    backend.fail_once.add(path)
    assert module.main(args) == 1
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    before = len(backend.puts), len(embedding.calls), len(tagger.calls), ledger.read_bytes()
    assert module.main(args) == 0
    assert len(backend.puts) == before[0] + 1
    assert (len(embedding.calls), len(tagger.calls), ledger.read_bytes()) == before[1:]
    assert {payload["applicationSequence"] for _, payload in backend.puts} == {checkpoint["applicationSequence"]}


def test_standalone_missing_ledger_stops_before_put_or_regeneration(standalone):
    module, args, backend, embedding, tagger, output, ledger = standalone
    assert module.main(args) == 0
    ledger.unlink()
    before = len(backend.puts), len(embedding.calls), len(tagger.calls)
    with pytest.raises(ValueError, match="ledger"):
        module.main(args)
    assert (len(backend.puts), len(embedding.calls), len(tagger.calls)) == before
    assert not ledger.exists()


@pytest.mark.parametrize("code", ["STALE_RECOMMENDATION_INPUT", "RECOMMENDATION_INPUT_CONFLICT"])
def test_standalone_409_counts_only_as_apply_failure(standalone, monkeypatch, code):
    module, args, backend, _, _, output, _ = standalone
    calls = []
    def conflict(path, payload):
        calls.append(path)
        raise BackendRequestError(code, status=409, code=code)
    monkeypatch.setattr(backend, "put", conflict)
    assert module.main(args) == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failureCount"] == len(calls) > 0
    assert summary["appliedNowCount"] == summary["appliedSuccessCount"] == summary["idempotentCount"] == 0
    assert all(failure["stage"] == "apply" for failure in summary["failures"])


@pytest.mark.parametrize("kind", ["similar", "interest"])
def test_python_apply_requires_explicit_shared_ledger(tmp_path, kind):
    if kind == "similar":
        backend = FakeBackend({})
        batch = RecommendationBatch(manifest(), FakeRecommender(), BatchPaths(tmp_path),
                                    config(), top_n=1, backend=backend)
    else:
        from tests.recommendation.test_interest_batch import FakeBackend as InterestBackend
        backend = InterestBackend()
        batch = InterestRecommendationBatch(recommender(tmp_path), InterestBatchPaths(tmp_path / "batch"),
                                            backend=backend)
    with pytest.raises(ValueError, match="sequence_ledger"):
        batch.run("apply")
    assert backend.puts == []


@pytest.mark.parametrize("error", [None, "invalid"])
def test_standalone_target_switch_handles_optional_error_fields(standalone, error):
    module, args, backend, embedding, tagger, output, ledger = standalone
    assert module.main(args) == 0
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    records = checkpoint["creators" if module is batch_cli else "interests"]
    for record in records.values():
        record["error"] = error
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    backend.target_identity = "https://another-be.example"
    before = len(embedding.calls), len(tagger.calls), len(backend.puts)
    if module is interest_cli:
        # 관심 분야는 로드 시 손상된 오류 필드를 명시적으로 거부한다.
        with pytest.raises(ValueError, match="상태"):
            module.main(args)
        assert len(backend.puts) == before[2]
    else:
        assert module.main(args) == 0
        assert len(backend.puts) == before[2] * 2
    assert (len(embedding.calls), len(tagger.calls)) == before[:2]


def test_fake_backend_new_sequence_same_content_is_new_generation():
    backend = Backend()
    payload = {"creatorId": 1, "inputHash": "a" * 64, "candidates": [], "applicationSequence": 1}
    assert backend.put("/api/admin/creators/1/similar", payload)["applied"] is True
    assert backend.put("/api/admin/creators/1/similar", payload)["applied"] is False
    assert backend.put("/api/admin/creators/1/similar", {**payload, "applicationSequence": 2})["applied"] is True
