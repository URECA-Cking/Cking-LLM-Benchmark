from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.recommendation.batch import _json_hash
from src.recommendation.interest import M2_METHOD, M3_METHOD
from src.recommendation.interest_batch import InterestBatchPaths, InterestRecommendationBatch
from src.recommendation.models import CreatorProfile
from tests.recommendation.test_interest import categories, recommender


class FakeBackend:
    def __init__(self, fail_once=(), *, target="https://be.example", idempotent=(), wrong=None):
        self.target_identity = target
        self.fail_once = set(fail_once)
        self.idempotent = set(idempotent)
        self.wrong = wrong or {}
        self.puts = []

    def put(self, path, payload):
        self.puts.append((path, payload))
        code = payload["interestCode"]
        if code in self.fail_once:
            self.fail_once.remove(code)
            raise RuntimeError("temporary failure key-secret")
        return {
            "taxonomyVersion": payload["taxonomyVersion"], "interestCode": code,
            "inputHash": payload["inputHash"], "candidateCount": len(payload["candidates"]),
            "applied": code not in self.idempotent, **self.wrong,
        }

    def get(self, path):
        raise AssertionError("관심 분야 공개 조회 API는 호출하지 않음")


def test_dry_run_saves_all_payloads_baselines_summary_and_config_without_be_writes(tmp_path):
    service = recommender(tmp_path)
    paths = InterestBatchPaths(tmp_path / "batch")
    backend = FakeBackend()
    summary = InterestRecommendationBatch(service, paths, backend=backend, sequence_ledger=tmp_path / "applications.json").run("dry-run")
    assert summary["interestCount"] == summary["nonEmptyGenerationCount"] == 17
    assert summary["eligibleCreatorCount"] == 3
    assert summary["generationConfig"]["zeroShotTau"] == .4296248555
    assert summary["failureCount"] == 0
    assert backend.puts == []
    for row in categories():
        for method in (M2_METHOD, M3_METHOD):
            payload = json.loads(paths.payload(row.code, method).read_text(encoding="utf-8"))
            assert payload == service.recommend(row.code, method).to_backend_payload()
    assert paths.checkpoint.exists() and paths.summary.exists()


def test_partial_apply_retry_skips_successes_and_reuses_generation_and_cache(tmp_path):
    code = categories()[3].code
    paths = InterestBatchPaths(tmp_path / "batch")
    backend = FakeBackend([code], idempotent=[categories()[0].code])
    first = InterestRecommendationBatch(recommender(tmp_path), paths, backend=backend,
                                        secret_values=("key-secret",), sequence_ledger=tmp_path / "applications.json").run("apply")
    assert first["failureCount"] == 1
    assert first["idempotentCount"] == 1
    assert first["appliedSuccessCount"] == 15
    assert "key-secret" not in paths.checkpoint.read_text(encoding="utf-8")
    assert "key-secret" not in paths.summary.read_text(encoding="utf-8")
    second_service = recommender(tmp_path)
    second = InterestRecommendationBatch(second_service, paths, backend=backend, sequence_ledger=tmp_path / "applications.json").run("apply")
    assert second["reusedGenerationCount"] == 17
    assert second["reusedApplyCount"] == 16
    assert second["appliedNowCount"] == 1
    assert second["failureCount"] == 0
    assert second_service.embedding_client.calls == []
    assert len(backend.puts) == 18
    assert backend.puts[-1][0] == f"/api/admin/interests/{code}/recommendations"
    assert all(payload["method"] == M3_METHOD for _, payload in backend.puts)


def test_target_change_reapplies_all_existing_payloads(tmp_path):
    paths = InterestBatchPaths(tmp_path / "batch")
    first = FakeBackend()
    InterestRecommendationBatch(recommender(tmp_path), paths, backend=first, sequence_ledger=tmp_path / "applications.json").run("apply")
    second = FakeBackend(target="https://another-be.example")
    service = recommender(tmp_path)
    summary = InterestRecommendationBatch(service, paths, backend=second, sequence_ledger=tmp_path / "applications.json").run("apply")
    assert len(second.puts) == 17
    assert summary["reusedGenerationCount"] == 17
    assert summary["applyTarget"] == "https://another-be.example"
    assert summary["reusedApplyCount"] == 0
    assert service.embedding_client.calls == []


def test_empty_generations_are_applied_and_distinguished_from_failure(tmp_path):
    service = recommender(tmp_path, [CreatorProfile(1, "")])
    backend = FakeBackend()
    summary = InterestRecommendationBatch(service, InterestBatchPaths(tmp_path / "batch"),
                                          backend=backend, sequence_ledger=tmp_path / "applications.json").run("apply")
    assert summary["emptyGenerationCount"] == 17
    assert summary["generationFailureCount"] == summary["failureCount"] == 0
    assert summary["appliedSuccessCount"] == 17
    assert summary["evaluation"]["pairCount"] == 0
    assert all(payload["candidates"] == [] for _, payload in backend.puts)
    assert service.embedding_client.calls == []


def test_failed_generation_is_retried_and_other_interests_complete(tmp_path, monkeypatch):
    paths = InterestBatchPaths(tmp_path / "batch")
    service = recommender(tmp_path)
    code = categories()[0].code
    original = service.recommend

    def broken(interest_code, method=M3_METHOD):
        if interest_code == code:
            raise RuntimeError("generation failed")
        return original(interest_code, method)

    monkeypatch.setattr(service, "recommend", broken)
    first = InterestRecommendationBatch(service, paths).run("dry-run")
    assert first["generationFailureCount"] == 1
    assert first["nonEmptyGenerationCount"] == 16
    assert first["evaluation"]["status"] == "pending"
    assert not paths.evaluation_dir.exists()
    second = InterestRecommendationBatch(recommender(tmp_path), paths).run("dry-run")
    assert second["generatedNowCount"] == 1
    assert second["reusedGenerationCount"] == 16
    assert second["evaluation"]["status"] == "success"


@pytest.mark.parametrize("wrong", [
    {"taxonomyVersion": "v0.1"}, {"interestCode": "UNKNOWN"}, {"inputHash": "a" * 64},
    {"candidateCount": 20}, {"candidateCount": True}, {"applied": 1},
    {"taxonomyHash": "a" * 64},
])
def test_inconsistent_backend_response_is_recorded_as_failed(tmp_path, wrong):
    summary = InterestRecommendationBatch(recommender(tmp_path), InterestBatchPaths(tmp_path / "batch"),
                                          backend=FakeBackend(wrong=wrong), sequence_ledger=tmp_path / "applications.json").run("apply")
    assert summary["failureCount"] == 17
    assert all(row["stage"] == "apply" for row in summary["failures"])


@pytest.mark.parametrize("change", [{"m3_bonus": .2}, {"zero_shot_tau": .5}, {"top_n": 10}])
def test_changed_settings_refuse_checkpoint_before_calls(tmp_path, change):
    paths = InterestBatchPaths(tmp_path / "batch")
    service = recommender(tmp_path)
    InterestRecommendationBatch(service, paths).run("dry-run")
    other = recommender(tmp_path, config=replace(service.config, **change))
    backend = FakeBackend()
    with pytest.raises(ValueError, match="생성 설정"):
        InterestRecommendationBatch(other, paths, backend=backend, sequence_ledger=tmp_path / "applications.json").run("apply")
    assert backend.puts == other.embedding_client.calls == []


def test_changed_manifest_refuses_old_generations(tmp_path):
    paths = InterestBatchPaths(tmp_path / "batch")
    InterestRecommendationBatch(recommender(tmp_path), paths).run("dry-run")
    changed = recommender(tmp_path, [CreatorProfile(1, "changed")])
    with pytest.raises(ValueError, match="manifest"):
        InterestRecommendationBatch(changed, paths).run("dry-run")


def test_corrupt_or_wrong_metadata_payload_is_regenerated_and_reapplied(tmp_path):
    paths = InterestBatchPaths(tmp_path / "batch")
    backend = FakeBackend()
    InterestRecommendationBatch(recommender(tmp_path), paths, backend=backend, sequence_ledger=tmp_path / "applications.json").run("apply")
    code = categories()[0].code
    payload = json.loads(paths.payload(code).read_text(encoding="utf-8"))
    payload["taxonomyHash"] = "a" * 64
    paths.payload(code).write_text(json.dumps(payload), encoding="utf-8")
    # 파일 해시만 맞춰도 세대 계약 위반은 검출한다.
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["interests"][code]["payloadHashes"][M3_METHOD] = _json_hash(payload)
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary = InterestRecommendationBatch(recommender(tmp_path), paths, backend=backend, sequence_ledger=tmp_path / "applications.json").run("apply")
    assert summary["generatedNowCount"] == summary["appliedNowCount"] == 1
    assert backend.puts[-1][1]["taxonomyHash"] != "a" * 64


def test_corrupt_baseline_is_rebuilt_and_wrong_evaluation_contract_is_preserved(tmp_path):
    paths = InterestBatchPaths(tmp_path / "batch")
    InterestRecommendationBatch(recommender(tmp_path), paths).run("dry-run")
    paths.payload(categories()[0].code, M2_METHOD).write_text("{}", encoding="utf-8")
    judge = paths.evaluation_dir / "judge-input.json"
    judge.write_text("{}", encoding="utf-8")
    summary = InterestRecommendationBatch(recommender(tmp_path), paths).run("dry-run")
    assert summary["generatedNowCount"] == 1
    assert summary["evaluation"]["status"] == "failed"
    assert summary["failureCount"] == 1
    assert judge.read_text(encoding="utf-8") == "{}"


def test_apply_requires_backend_and_invalid_mode_is_rejected(tmp_path):
    batch = InterestRecommendationBatch(recommender(tmp_path), InterestBatchPaths(tmp_path / "batch"))
    with pytest.raises(ValueError, match="BE"):
        batch.run("apply")
    with pytest.raises(ValueError, match="mode"):
        batch.run("unknown")


def test_invalid_shared_embeddings_fail_once_then_report_all_fields(tmp_path):
    from tests.recommendation.test_interest import FakeEmbedding, vector

    client = FakeEmbedding({"invalid": vector(0)})
    service = recommender(tmp_path, [CreatorProfile(1, "invalid")], client=client)
    paths = InterestBatchPaths(tmp_path / "batch")
    summary = InterestRecommendationBatch(service, paths).run("dry-run")
    assert summary["generationFailureCount"] == summary["failureCount"] == 17
    assert summary["emptyGenerationCount"] == 0
    assert len(client.calls) == 1
    assert summary["evaluation"]["status"] == "pending"


def test_failed_checkpoint_without_error_is_rejected_before_applying(tmp_path):
    paths = InterestBatchPaths(tmp_path / "batch")
    InterestRecommendationBatch(recommender(tmp_path), paths).run("dry-run")
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    checkpoint["interests"][categories()[0].code]["applyStatus"] = "failed"
    paths.checkpoint.write_text(json.dumps(checkpoint), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="상태"):
        InterestRecommendationBatch(recommender(tmp_path), paths, backend=backend, sequence_ledger=tmp_path / "applications.json").run("apply")
    assert backend.puts == []
