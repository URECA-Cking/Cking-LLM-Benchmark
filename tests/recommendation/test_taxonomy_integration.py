from dataclasses import replace
import json

import pytest

from src.recommendation.batch import BatchPaths, RecommendationBatch, generation_config_hash
from src.recommendation.batch_cli import _recommendation_config, build_parser as batch_parser
from src.recommendation.cli import build_parser as service_parser
from src.recommendation.cli import main as service_main
from src.recommendation.service import RecommendationConfig
from src.recommendation.models import CreatorProfile
from src.recommendation.taxonomy import DEFAULT_CATEGORIES_CSV, load_service_categories, taxonomy_hash
from tests.recommendation.test_service import FakeEmbeddingClient, FakeTaggingClient, build_service, config
from tests.recommendation.test_batch import FakeRecommender, manifest


def test_service_and_batch_defaults_use_the_same_v02_taxonomy():
    single = service_parser().parse_args(["--input", "input.json", "--output", "output.json"])
    for args in [single, batch_parser().parse_args(["dry-run"]), batch_parser().parse_args(["apply", "--be-base-url", "https://be.example"])]:
        assert args.categories == DEFAULT_CATEGORIES_CSV
        assert args.taxonomy_version == "v0.2"
        assert args.tag_prompt_version == "creator-category-v2"
        settings, codes, prompt = _recommendation_config(args)
        assert len(codes) == 17
        assert settings.taxonomy_hash == taxonomy_hash(load_service_categories(args.categories))
        assert "LIFETIP(생활정보·제품 소개)" in prompt


@pytest.mark.parametrize("changes", [
    {"taxonomy_version": "taxonomy-v2"},
    {"taxonomy_hash": "a" * 64},
    {"tag_prompt_version": "prompt-v2"},
    {"tag_prompt": "새 프롬프트"},
])
@pytest.mark.parametrize("seed_text", ["홈트 운동", "짧음", ""])
def test_taxonomy_and_prompt_changes_invalidate_cache_hash_and_checkpoint(tmp_path, changes, seed_text):
    seed = CreatorProfile(10, seed_text)
    candidate = CreatorProfile(20, "근력 운동")
    embedder = FakeEmbeddingClient({seed.text: [1, 0], candidate.text: [1, 0]})
    tagger = FakeTaggingClient({seed.text: ("FITNESS",), candidate.text: ("FITNESS",)})
    settings = config()
    first = build_service(tmp_path, embedder, tagger, settings).recommend(seed, [candidate])
    tag_calls = len(tagger.calls)
    embed_calls = len(embedder.calls)
    changed = replace(settings, **changes)
    second = build_service(tmp_path, embedder, tagger, changed).recommend(seed, [candidate])
    assert first.input_hash != second.input_hash
    assert settings.tag_cache_version != changed.tag_cache_version
    assert generation_config_hash(settings, 5) != generation_config_hash(changed, 5)
    if first.method == "M4":
        assert len(tagger.calls) == tag_calls * 2
    assert len(embedder.calls) == embed_calls


@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "g" * 64])
def test_config_rejects_invalid_taxonomy_hash(digest):
    with pytest.raises(ValueError, match="taxonomy_hash"):
        config(taxonomy_hash=digest)


def test_taxonomy_content_change_rejects_previous_checkpoint(tmp_path):
    settings = config()
    paths = BatchPaths(tmp_path)
    RecommendationBatch(manifest(), FakeRecommender(), paths, settings, top_n=1).run("dry-run")
    with pytest.raises(ValueError, match="설정"):
        RecommendationBatch(
            manifest(), FakeRecommender(), paths, replace(settings, taxonomy_hash="b" * 64), top_n=1
        ).run("dry-run")


def test_single_cli_binds_the_loaded_taxonomy_hash_before_generation(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"creatorId": 1, "introduction": "", "candidateCreators": []}), encoding="utf-8")
    categories = tmp_path / "categories.csv"
    categories.write_text("code,name,description\nFOOD,요리,베이킹\n", encoding="utf-8")
    captured = []

    def capture_config(**kwargs):
        settings = RecommendationConfig(**kwargs)
        captured.append(settings)
        return settings

    def no_model_calls(*_args, **_kwargs):
        class UnusedClient:
            def embed(self, *_args):
                raise AssertionError("빈 세대에서 모델을 호출했습니다.")

            def tag(self, *_args):
                raise AssertionError("빈 세대에서 모델을 호출했습니다.")

        return UnusedClient(), UnusedClient()

    monkeypatch.setattr("src.recommendation.cli.RecommendationConfig", capture_config)
    monkeypatch.setattr("src.recommendation.cli.create_external_clients", no_model_calls)
    assert service_main([
        "--input", str(request), "--output", str(tmp_path / "output.json"),
        "--cache", str(tmp_path / "cache.json"), "--categories", str(categories),
    ]) == 0
    assert captured[0].taxonomy_hash == taxonomy_hash(load_service_categories(categories))
    assert captured[0].allowed_tags == frozenset({"FOOD"})
