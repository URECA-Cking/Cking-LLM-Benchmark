from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.clients.openai_tagger import TagResult
from src.recommendation import daily_cli, batch_cli, interest_cli
from src.recommendation import daily
from src.recommendation.batch import _json_hash
from src.recommendation.daily import DailyPaths
from src.recommendation.interest import M2_METHOD, M3_METHOD
from src.recommendation.locking import batch_locks, BatchAlreadyRunning
from tests.recommendation.test_interest import FakeEmbedding, categories, vector


INTRO = "매일 운동과 건강한 생활을 소개하는 크리에이터"
OTHER = "다양한 요리와 레시피를 소개하는 크리에이터"


class Backend:
    target_identity = "https://be.example"

    def __init__(self):
        self.profiles = [{"creatorId": index, "introText": INTRO} for index in range(1, 26)]
        self.gets, self.puts = [], []
        self.fail_once = set()

    def get(self, path):
        self.gets.append(path)
        query = dict(part.split("=") for part in path.split("?")[1].split("&"))
        page, size = int(query["page"]), int(query["size"])
        count = len(self.profiles)
        return {"items": self.profiles[page * size:(page + 1) * size], "page": page, "size": size,
                "totalElements": count, "totalPages": (count + size - 1) // size,
                "hasNext": (page + 1) * size < count}

    def put(self, path, payload):
        self.puts.append((path, payload))
        if path in self.fail_once:
            self.fail_once.remove(path)
            raise RuntimeError("fake-key fake-openai fake-deepinfra fake-jwt")
        return {key: payload[key] for key in ("creatorId", "interestCode", "taxonomyVersion", "inputHash")
                if key in payload} | {"candidateCount": len(payload["candidates"]), "applied": True}


class Tagger:
    def __init__(self):
        self.calls = []
        self.fail_once = set()

    def tag(self, text):
        self.calls.append(text)
        if text in self.fail_once:
            self.fail_once.remove(text)
            raise RuntimeError("fake-key fake-openai")
        return TagResult((categories()[0].code,), 0, 0)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    backend, tagger = Backend(), Tagger()
    embedding = FakeEmbedding({INTRO: vector(1), OTHER: vector(0, 1)})
    credentials, models = [], []

    def client(config, **kwargs):
        credentials.append(kwargs)
        return backend

    def create_embedding(provider):
        models.append(provider)
        return embedding

    monkeypatch.setattr(daily_cli, "CkingBackendClient", client)
    monkeypatch.setattr(daily_cli, "create_interest_embedding_client", create_embedding)
    monkeypatch.setattr(daily_cli, "LazyOpenAITagger", lambda *args: tagger)
    for name, value in (("CKING_RECOMMENDATION_API_KEY", "fake-key"), ("OPENAI_API_KEY", "fake-openai"),
                        ("DEEPINFRA_API_KEY", "fake-deepinfra"), ("CKING_ADMIN_ACCESS_TOKEN", "fake-jwt")):
        monkeypatch.setenv(name, value)
    args = ["--be-base-url", "https://be.example", "--output-dir", str(tmp_path / "daily"),
            "--cache", str(tmp_path / "cache.json")]
    return backend, embedding, tagger, credentials, models, args, DailyPaths(tmp_path / "daily")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_first_apply_top20_then_same_manifest_skips_models_and_puts(setup):
    backend, embedding, tagger, credentials, models, args, paths = setup
    assert daily_cli.main(["apply", *args, "--page-size", "10"]) == 0
    assert len(backend.gets) == 3 and len(backend.puts) == 25 + 17
    assert all(len(payload["candidates"]) == 20 for _, payload in backend.puts)
    assert all(payload["method"] == M2_METHOD for _, payload in backend.puts if "interestCode" in payload)
    assert credentials == [{"recommendation_api_key": "fake-key"}]
    assert paths.completed.exists() and not paths.active.exists()
    embedding_count, tagging_count = len(embedding.calls), len(tagger.calls)
    # 목록 순서·페이지 크기 변경은 내용 hash를 바꾸지 않는다.
    backend.profiles.reverse()
    assert daily_cli.main(["apply", *args]) == 0
    assert read(paths.summary)["status"] == "skipped"
    assert len(backend.puts) == 42 and models == ["deepinfra"]
    assert (len(embedding.calls), len(tagger.calls)) == (embedding_count, tagging_count)


def test_changed_intro_recomputes_all_rankings_and_only_calls_models_for_changed_text(setup):
    backend, embedding, tagger, _, _, args, paths = setup
    assert daily_cli.main(["apply", *args]) == 0
    original = read(paths.completed)
    embedding.calls.clear()
    tagger.calls.clear()
    backend.profiles[0]["introText"] = OTHER
    assert daily_cli.main(["apply", *args]) == 0
    assert len(backend.puts) == 84
    assert read(paths.completed)["manifestHash"] != original["manifestHash"]
    assert embedding.calls == [[OTHER]] and tagger.calls == [OTHER]
    assert len(list((paths.output_dir / "runs").iterdir())) == 2


@pytest.mark.parametrize("path", ["/api/admin/creators/2/similar",
                                 f"/api/admin/interests/{categories()[0].code}/recommendations"])
def test_partial_apply_resume_reuses_successful_seed_and_interest_checkpoints(setup, path):
    backend, embedding, tagger, _, _, args, paths = setup
    backend.fail_once.add(path)
    assert daily_cli.main(["apply", *args]) == 1
    assert not paths.completed.exists() and paths.active.exists()
    assert read(paths.summary)["status"] == "partial"
    before = (len(embedding.calls), len(tagger.calls))
    assert daily_cli.main(["apply", *args]) == 0
    assert len(backend.puts) == 43 and backend.puts[-1][0] == path
    assert (len(embedding.calls), len(tagger.calls)) == before
    assert paths.completed.exists() and not paths.active.exists()
    for file in paths.output_dir.rglob("*.json"):
        content = file.read_text(encoding="utf-8")
        assert all(secret not in content for secret in ("fake-key", "fake-openai", "fake-deepinfra", "fake-jwt"))


def test_generation_failure_resumes_with_successful_model_cache(setup):
    backend, embedding, tagger, _, _, args, paths = setup
    backend.profiles[1]["introText"] = OTHER
    tagger.fail_once.add(OTHER)
    assert daily_cli.main(["apply", *args]) == 1
    assert not paths.completed.exists()
    before = len(embedding.calls)
    assert daily_cli.main(["apply", *args]) == 0
    assert len(embedding.calls) == before  # 관심 분야 단계가 소개 임베딩을 이미 저장했다.
    assert paths.completed.exists()


def test_switch_from_incomplete_hash_preserves_outputs_and_returns_to_old_hash_with_full_reapply(setup):
    backend, _, _, _, _, args, paths = setup
    assert daily_cli.main(["apply", *args]) == 0
    completed = paths.completed.read_bytes()
    backend.profiles[0]["introText"] = OTHER
    backend.fail_once.add("/api/admin/creators/1/similar")
    assert daily_cli.main(["apply", *args]) == 1
    assert paths.completed.read_bytes() == completed
    partial_hash = read(paths.active)["manifestHash"]
    backend.profiles[0]["introText"] = INTRO
    assert daily_cli.main(["apply", *args]) == 0
    assert read(paths.summary)["status"] == "completed"
    assert read(paths.summary)["supersededRun"]["manifestHash"] == partial_hash
    assert len(backend.puts) == 126  # 이전 완료 세대라도 부분 적용을 덮어쓴다.
    assert len(list((paths.output_dir / "runs").iterdir())) == 2


def test_dry_run_never_sets_complete_or_active_and_apply_reuses_generated_results(setup):
    backend, embedding, tagger, _, _, args, paths = setup
    assert daily_cli.main(["dry-run", *args]) == 0
    assert not backend.puts and not paths.completed.exists() and not paths.active.exists()
    before = (len(embedding.calls), len(tagger.calls))
    assert daily_cli.main(["apply", *args]) == 0
    assert len(backend.puts) == 42 and paths.completed.exists()
    assert (len(embedding.calls), len(tagger.calls)) == before


def test_backend_target_change_forces_reapply_without_model_calls(setup):
    backend, embedding, _, _, _, args, paths = setup
    assert daily_cli.main(["apply", *args]) == 0
    embedding.calls.clear()
    backend.target_identity = "https://new-be.example"
    assert daily_cli.main(["apply", *args]) == 0
    assert len(backend.puts) == 84 and not embedding.calls
    assert read(paths.completed)["applyTarget"] == backend.target_identity


def test_configuration_change_invalidates_skip_and_reuses_embedding_cache(setup):
    backend, embedding, tagger, _, _, args, paths = setup
    assert daily_cli.main(["apply", *args]) == 0
    old = read(paths.completed)
    embedding.calls.clear()
    tagger.calls.clear()
    assert daily_cli.main(["apply", *args, "--tag-prompt-version", "new-prompt-v2"]) == 0
    assert len(backend.puts) == 84 and not embedding.calls and tagger.calls == [INTRO]
    assert read(paths.completed)["generationConfigHash"] != old["generationConfigHash"]


def test_api_key_required_even_when_admin_jwt_present_before_models_and_http(setup, monkeypatch):
    backend, _, _, _, models, args, paths = setup
    monkeypatch.delenv("CKING_RECOMMENDATION_API_KEY")
    with pytest.raises(RuntimeError, match="CKING_RECOMMENDATION_API_KEY"):
        daily_cli.main(["apply", *args])
    assert not backend.gets and not backend.puts and not models and not paths.output_dir.exists()


def test_shared_cache_lock_rejects_daily_and_standalone_before_http_or_models(setup):
    backend, _, _, _, models, args, paths = setup
    cache = Path(args[-1])
    with batch_locks(paths.output_dir, cache):
        with pytest.raises(BatchAlreadyRunning):
            daily_cli.main(["apply", *args])
        with pytest.raises(BatchAlreadyRunning):
            batch_cli.main(["dry-run", "--output-dir", str(paths.output_dir / "other"), "--cache", str(cache)])
        with pytest.raises(BatchAlreadyRunning):
            interest_cli.main(["dry-run", "--output-dir", str(paths.output_dir / "other"), "--cache", str(cache)])
    assert not backend.gets and not models
    assert daily_cli.main(["apply", *args]) == 0


def decision_report(tmp_path, **changes):
    selected = {"method": M3_METHOD, "tau": .5, "bonus": .2, "maxTags": 2,
                "modelVersion": "BAAI/bge-m3@deepinfra-v1", "taxonomyVersion": "v0.2",
                "taxonomyHash": daily_cli.taxonomy_hash(categories()), "decisionVersion": "interest-decision-v1",
                **changes}
    report = {"decision": {"provisional": False, "selectedConfig": selected,
                           "selectedConfigHash": _json_hash(selected)}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_quality_report_applies_selected_method_and_settings(setup, tmp_path):
    backend, _, _, _, _, args, paths = setup
    path = decision_report(tmp_path)
    assert daily_cli.main(["apply", *args, "--decision-report", str(path)]) == 0
    assert all(payload["method"] == M3_METHOD for _, payload in backend.puts if "interestCode" in payload)
    summary_path = paths.output_dir / read(paths.summary)["runDirectory"] / "interests" / "summary.json"
    assert read(summary_path)["generationConfig"]["m3Bonus"] == .2
    assert read(summary_path)["generationConfig"]["zeroShotTau"] == .5


@pytest.mark.parametrize("change", [{"method": "INVALID"}, {"taxonomyHash": "a" * 64},
                                    {"method": M2_METHOD, "bonus": .1}, {"modelVersion": "BAAI/bge-m3@local-v1"},
                                    {"tau": float("nan")}, {"maxTags": True}])
def test_invalid_quality_settings_fail_before_http(setup, tmp_path, change):
    backend, _, _, _, models, args, _ = setup
    path = decision_report(tmp_path, **change)
    with pytest.raises(ValueError):
        daily_cli.main(["apply", *args, "--decision-report", str(path)])
    assert not backend.gets and not models


def test_quality_report_hash_or_provisional_m3_rejected(setup, tmp_path):
    backend, _, _, _, _, args, _ = setup
    for change in ({"selectedConfigHash": "a" * 64}, {"provisional": True}):
        path = decision_report(tmp_path)
        payload = read(path)
        payload["decision"].update(change)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            daily_cli.main(["apply", *args, "--decision-report", str(path)])
    assert not backend.gets


def test_empty_manifest_applies_all_empty_interest_generations_without_model_calls(setup):
    backend, embedding, tagger, _, _, args, paths = setup
    backend.profiles = []
    assert daily_cli.main(["apply", *args]) == 0
    assert len(backend.puts) == 17 and all(not payload["candidates"] for _, payload in backend.puts)
    assert not embedding.calls and not tagger.calls and paths.completed.exists()


def test_complete_marker_write_interruption_resumes_without_reapplying_successes(setup, monkeypatch):
    backend, _, _, _, _, args, paths = setup
    original = daily._write_json_atomic

    def fail_marker(path, payload):
        if path == paths.completed:
            raise OSError("simulated interruption")
        return original(path, payload)

    monkeypatch.setattr(daily, "_write_json_atomic", fail_marker)
    with pytest.raises(OSError):
        daily_cli.main(["apply", *args])
    assert not paths.completed.exists() and paths.active.exists()
    monkeypatch.setattr(daily, "_write_json_atomic", original)
    assert daily_cli.main(["apply", *args]) == 0
    assert len(backend.puts) == 42 and paths.completed.exists()


def test_factory_failure_keeps_incomplete_state_and_other_stage_can_finish(setup, monkeypatch):
    backend, _, _, _, _, args, paths = setup
    original = daily_cli.SimilarCreatorRecommender
    monkeypatch.setattr(daily_cli, "SimilarCreatorRecommender",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("fake-key fake-openai")))
    assert daily_cli.main(["apply", *args]) == 1
    assert len(backend.puts) == 17 and not paths.completed.exists()
    assert read(paths.summary)["stages"]["similar"]["errorType"] == "RuntimeError"
    assert "fake-key" not in paths.summary.read_text(encoding="utf-8")
    monkeypatch.setattr(daily_cli, "SimilarCreatorRecommender", original)
    assert daily_cli.main(["apply", *args]) == 0
    assert len(backend.puts) == 42


@pytest.mark.parametrize("module", [batch_cli, interest_cli])
def test_standalone_apply_rejects_jwt_only_environment_before_models(module, tmp_path, monkeypatch):
    from src.recommendation.manifest import build_manifest, write_manifest
    path = tmp_path / "manifest.json"
    write_manifest(path, build_manifest([]))
    monkeypatch.delenv("CKING_RECOMMENDATION_API_KEY", raising=False)
    monkeypatch.setenv("CKING_ADMIN_ACCESS_TOKEN", "fake-jwt")
    with pytest.raises(RuntimeError, match="CKING_RECOMMENDATION_API_KEY"):
        module.main(["apply", "--manifest", str(path), "--be-base-url", "https://be",
                     "--output-dir", str(tmp_path / "standalone"), "--cache", str(tmp_path / "cache.json")])
