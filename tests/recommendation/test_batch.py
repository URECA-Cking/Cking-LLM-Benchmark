from __future__ import annotations

import json

import pytest

from src.recommendation.batch import BatchPaths, RecommendationBatch
from src.recommendation.manifest import build_manifest
from src.recommendation.models import CreatorProfile, RecommendationCandidate, RecommendationResult
from src.recommendation.service import RecommendationConfig


INPUT_HASH = "a" * 64


class FakeRecommender:
    def __init__(self, fail_once: set[int] | None = None) -> None:
        self.calls: list[int] = []
        self.fail_once = fail_once or set()

    def recommend(self, seed, candidates, top_n=None):
        self.calls.append(seed.creator_id)
        if seed.creator_id in self.fail_once:
            self.fail_once.remove(seed.creator_id)
            raise RuntimeError(f"generation failed for {seed.creator_id}")
        candidates_result = ()
        if seed.text:
            similar = next(candidate for candidate in candidates if candidate.creator_id != seed.creator_id and candidate.text)
            candidates_result = (
                RecommendationCandidate(
                    seed.creator_id,
                    similar.creator_id,
                    0.75,
                    1,
                    "M2",
                    "model-v1",
                    INPUT_HASH,
                ),
            )
        return RecommendationResult(seed.creator_id, "M2", "model-v1", INPUT_HASH, candidates_result)


class FakeBackend:
    def __init__(self, responses: dict[int, object], target_identity: str = "https://be.example") -> None:
        self.responses = responses
        self.target_identity = target_identity
        self.puts: list[tuple[str, dict[str, object]]] = []

    def get(self, path: str) -> dict[str, object]:
        raise AssertionError("batch 실행 중 목록 GET을 호출하면 안 됩니다.")

    def put(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.puts.append((path, payload))
        creator_id = payload["creatorId"]
        response = self.responses[creator_id]
        if isinstance(response, Exception):
            raise response
        return response


def config() -> RecommendationConfig:
    return RecommendationConfig(
        top_n=1,
        embedding_model_version="model-v1",
        tag_model_version="tag-v1",
        tag_prompt="prompt",
        allowed_tags=frozenset({"A"}),
    )


def manifest():
    return build_manifest(
        [CreatorProfile(1, "first introduction"), CreatorProfile(2, "second introduction"), CreatorProfile(3, "")]
    )


def test_dry_run_writes_payload_summary_and_never_calls_backend(tmp_path) -> None:
    recommender = FakeRecommender()
    backend = FakeBackend({})
    paths = BatchPaths(tmp_path)
    summary = RecommendationBatch(
        manifest(), recommender, paths, config(), top_n=1, backend=backend,
        sequence_ledger=tmp_path / "applications.json",
    ).run("dry-run")

    assert recommender.calls == [1, 2, 3]
    assert backend.puts == []
    assert summary["nonEmptyGenerationCount"] == 2
    assert summary["emptyGenerationCount"] == 1
    assert summary["failureCount"] == 0
    assert json.loads(paths.payload(1).read_text(encoding="utf-8"))["creatorId"] == 1
    assert paths.checkpoint.exists() and paths.summary.exists()


def test_restart_reuses_successful_payload_and_retries_only_failed_seed(tmp_path) -> None:
    first = FakeRecommender({2})
    first_summary = RecommendationBatch(manifest(), first, BatchPaths(tmp_path), config(), top_n=1).run("dry-run")
    assert first_summary["failureCount"] == 1
    assert first.calls == [1, 2, 3]

    second = FakeRecommender()
    second_summary = RecommendationBatch(manifest(), second, BatchPaths(tmp_path), config(), top_n=1).run("dry-run")

    assert second.calls == [2]
    assert second_summary["generatedNowCount"] == 1
    assert second_summary["reusedGenerationCount"] == 2
    assert second_summary["failureCount"] == 0


def test_apply_distinguishes_success_idempotent_empty_and_partial_failure(tmp_path) -> None:
    backend = FakeBackend(
        {
            1: {"creatorId": 1, "inputHash": INPUT_HASH, "candidateCount": 1, "applied": True},
            2: {"creatorId": 2, "inputHash": INPUT_HASH, "candidateCount": 1, "applied": False},
            3: RuntimeError("temporary apply failure"),
        }
    )
    paths = BatchPaths(tmp_path)
    summary = RecommendationBatch(
        manifest(), FakeRecommender(), paths, config(), top_n=1, backend=backend,
        sequence_ledger=tmp_path / "applications.json",
    ).run("apply")

    assert [path for path, _ in backend.puts] == [
        "/api/admin/creators/1/similar",
        "/api/admin/creators/2/similar",
        "/api/admin/creators/3/similar",
    ]
    assert backend.puts[2][1]["candidates"] == []
    assert summary["appliedSuccessCount"] == 1
    assert summary["idempotentCount"] == 1
    assert summary["emptyGenerationCount"] == 1
    assert summary["failureCount"] == 1
    assert summary["failures"][0]["creatorId"] == 3
    assert summary["failures"][0]["stage"] == "apply"


def test_apply_resume_skips_already_applied_and_retries_failed_apply_without_regeneration(tmp_path) -> None:
    first_backend = FakeBackend(
        {
            1: {"creatorId": 1, "inputHash": INPUT_HASH, "candidateCount": 1, "applied": True},
            2: RuntimeError("once"),
            3: {"creatorId": 3, "inputHash": INPUT_HASH, "candidateCount": 0, "applied": True},
        }
    )
    paths = BatchPaths(tmp_path)
    RecommendationBatch(
        manifest(), FakeRecommender(), paths, config(), top_n=1, backend=first_backend,
        sequence_ledger=tmp_path / "applications.json",
    ).run("apply")

    recommender = FakeRecommender()
    second_backend = FakeBackend(
        {2: {"creatorId": 2, "inputHash": INPUT_HASH, "candidateCount": 1, "applied": True}}
    )
    summary = RecommendationBatch(
        manifest(), recommender, paths, config(), top_n=1, backend=second_backend,
        sequence_ledger=tmp_path / "applications.json",
    ).run("apply")

    assert recommender.calls == []
    assert [path for path, _ in second_backend.puts] == ["/api/admin/creators/2/similar"]
    assert summary["reusedGenerationCount"] == 3
    assert summary["reusedApplyCount"] == 2
    assert summary["appliedSuccessCount"] == 3
    assert summary["failureCount"] == 0


def test_apply_target_change_reuses_generation_and_reapplies_every_payload(tmp_path) -> None:
    responses = {
        1: {"creatorId": 1, "inputHash": INPUT_HASH, "candidateCount": 1, "applied": True},
        2: {"creatorId": 2, "inputHash": INPUT_HASH, "candidateCount": 1, "applied": True},
        3: {"creatorId": 3, "inputHash": INPUT_HASH, "candidateCount": 0, "applied": True},
    }
    paths = BatchPaths(tmp_path)
    RecommendationBatch(
        manifest(),
        FakeRecommender(),
        paths,
        config(),
        top_n=1,
        backend=FakeBackend(responses, "https://dev.example"),
        sequence_ledger=tmp_path / "applications.json",
    ).run("apply")

    recommender = FakeRecommender()
    production = FakeBackend(responses, "https://prod.example")
    summary = RecommendationBatch(
        manifest(), recommender, paths, config(), top_n=1, backend=production,
        sequence_ledger=tmp_path / "applications.json",
    ).run("apply")

    assert recommender.calls == []
    assert [path for path, _ in production.puts] == [
        "/api/admin/creators/1/similar",
        "/api/admin/creators/2/similar",
        "/api/admin/creators/3/similar",
    ]
    assert summary["reusedGenerationCount"] == 3
    assert summary["reusedApplyCount"] == 0
    assert summary["appliedSuccessCount"] == 3
    assert summary["applyTarget"] == "https://prod.example"
    assert json.loads(paths.checkpoint.read_text(encoding="utf-8"))["applyTarget"] == "https://prod.example"


def test_error_summary_redacts_environment_secrets(tmp_path) -> None:
    secret = "admin-super-secret"
    backend = FakeBackend({1: RuntimeError(f"bad {secret}"), 2: RuntimeError("bad"), 3: RuntimeError("bad")})
    summary = RecommendationBatch(
        manifest(), FakeRecommender(), BatchPaths(tmp_path), config(), top_n=1,
        backend=backend, secret_values=(secret,),
        sequence_ledger=tmp_path / "applications.json",
    ).run("apply")

    assert secret not in json.dumps(summary)
    assert "[REDACTED]" in json.dumps(summary)


def test_changed_generation_config_rejects_existing_checkpoint(tmp_path) -> None:
    RecommendationBatch(manifest(), FakeRecommender(), BatchPaths(tmp_path), config(), top_n=1).run("dry-run")

    with pytest.raises(ValueError, match="생성 설정"):
        RecommendationBatch(manifest(), FakeRecommender(), BatchPaths(tmp_path), config(), top_n=2).run("dry-run")


def test_tampered_payload_is_regenerated_from_checkpoint(tmp_path) -> None:
    paths = BatchPaths(tmp_path)
    RecommendationBatch(manifest(), FakeRecommender(), paths, config(), top_n=1).run("dry-run")
    paths.payload(2).write_text("{}", encoding="utf-8")
    recommender = FakeRecommender()

    summary = RecommendationBatch(manifest(), recommender, paths, config(), top_n=1).run("dry-run")

    assert recommender.calls == [2]
    assert summary["generatedNowCount"] == 1
