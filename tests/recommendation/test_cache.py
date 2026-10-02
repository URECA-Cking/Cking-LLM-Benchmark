import json

from src.recommendation.cache import JsonModelCache, text_hash


def test_json_cache_persists_empty_tags_and_embeddings(tmp_path) -> None:
    path = tmp_path / "model-cache.json"
    cache = JsonModelCache(path)
    input_hash = text_hash("소개")

    cache.put_embedding(input_hash, "embed-v1", [1.0, 0.0])
    cache.put_tags(input_hash, "tag-v1", "prompt-v1", ())

    reloaded = JsonModelCache(path)
    assert reloaded.get_embedding(input_hash, "embed-v1") == [1.0, 0.0]
    assert reloaded.get_tags(input_hash, "tag-v1", "prompt-v1") == ()
    assert reloaded.get_embedding(input_hash, "embed-v2") is None
    assert reloaded.get_tags(input_hash, "tag-v1", "prompt-v2") is None
    assert json.loads(path.read_text(encoding="utf-8"))["schemaVersion"] == 1


def test_unknown_cache_schema_starts_fresh(tmp_path) -> None:
    path = tmp_path / "model-cache.json"
    path.write_text('{"schemaVersion":999,"embeddings":{},"tags":{}}', encoding="utf-8")

    cache = JsonModelCache(path)

    assert cache.get_embedding(text_hash("소개"), "embed-v1") is None
