import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import pipeline
from src.api_parity import compare_decisions, compare_embeddings
from src.clients import OpenAIEmbeddingClient
from src.clients.base import normalize_rows


def _random_unit(rows: int, dim: int, seed: int) -> np.ndarray:
    return normalize_rows(np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32))


def _compare(local_c, api_c, local_k, api_k):
    return compare_embeddings(local_c, api_c, local_k, api_k, k=5, max_abs_diff_limit=0.01, min_top_k_overlap=0.95)


def test_identical_vectors_are_equivalent() -> None:
    creators, categories = _random_unit(20, 16, 1), _random_unit(4, 16, 2)

    report = _compare(creators, creators.copy(), categories, categories.copy())

    assert report["similarity_equivalent"] is True
    assert report["creator_similarity_max_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    assert report["top5_overlap_mean"] == 1.0 and report["top1_agreement"] == 1.0
    assert report["self_cosine_min"] == pytest.approx(1.0, abs=1e-6)


def test_tiny_noise_stays_equivalent_but_large_noise_does_not() -> None:
    creators, categories = _random_unit(30, 32, 3), _random_unit(5, 32, 4)
    rng = np.random.default_rng(5)

    tiny = normalize_rows(creators + rng.normal(scale=1e-4, size=creators.shape).astype(np.float32))
    assert _compare(creators, tiny, categories, categories)["similarity_equivalent"] is True

    large = normalize_rows(creators + rng.normal(scale=0.5, size=creators.shape).astype(np.float32))
    report = _compare(creators, large, categories, categories)
    assert report["similarity_equivalent"] is False
    assert report["creator_similarity_max_abs_diff"] > 0.01


def test_category_similarity_drift_alone_breaks_equivalence() -> None:
    creators, categories = _random_unit(20, 16, 6), _random_unit(4, 16, 7)
    shifted = normalize_rows(categories + 0.3)  # 크리에이터 벡터는 그대로, 카테고리만 달라짐 → zero-shot 태깅이 흔들린다

    report = _compare(creators, creators.copy(), categories, shifted)

    assert report["top5_overlap_mean"] == 1.0
    assert report["category_similarity_max_abs_diff"] > 0.01
    assert report["similarity_equivalent"] is False


def _decisions(local_c, api_c, local_k, api_k, tau, cutoff):
    n, codes = len(local_c), ["A", "B", "C"]
    return compare_decisions(
        [f"C{i}" for i in range(n)], local_c, api_c, local_k, api_k, codes, [frozenset({"A"})] * n,
        tau=tau, max_tags=3, bonuses={"M3": 0.1, "M4": 0.2}, cutoffs={m: cutoff for m in ("M1", "M2", "M3", "M4")}, k=5,
    )


def test_identical_vectors_make_identical_decisions() -> None:
    creators, categories = _random_unit(10, 8, 12), _random_unit(3, 8, 13)

    report = _decisions(creators, creators.copy(), categories, categories.copy(), tau=0.1, cutoff=0.1)

    assert report["decisions_identical"] is True
    assert report["zero_shot_tag_mismatch"]["count"] == 0
    assert all(v["count"] == 0 for v in report["candidates_after_cutoff_mismatch"].values())


def test_tag_flip_at_tau_is_caught_even_though_similarity_is_equivalent() -> None:
    creators, categories = _random_unit(10, 8, 14), _random_unit(3, 8, 15)
    top = int(np.argmax(categories @ creators[0]))
    tau = float(categories[top] @ creators[0])  # tau가 어떤 크리에이터의 실제 점수와 같다 → 그 크리에이터는 경계에 있다
    api_creators = creators.copy()
    api_creators[0] = normalize_rows((creators[0] - 1e-3 * categories[top])[None, :])[0]  # 유사도는 0.001만 낮아진다

    similarity = _compare(creators, api_creators, categories, categories)
    decisions = _decisions(creators, api_creators, categories, categories, tau=tau, cutoff=-1.0)

    assert similarity["similarity_equivalent"] is True
    assert decisions["zero_shot_tag_mismatch"] == {"count": 1, "creators": ["C0"]}
    assert decisions["decisions_identical"] is False


def test_candidate_dropped_at_cutoff_is_caught() -> None:
    creators, categories = _random_unit(10, 8, 16), _random_unit(3, 8, 17)
    scores = creators @ creators[0]
    neighbor = int(np.argsort(-np.where(np.arange(10) == 0, -np.inf, scores))[2])  # 로컬에서 3위 후보
    cutoff = float(scores[neighbor])  # 컷오프가 그 후보의 점수와 같다 → 로컬은 통과, API는 미세하게 낮아져 탈락
    api_creators = creators.copy()
    api_creators[neighbor] = normalize_rows((creators[neighbor] - 1e-3 * creators[0])[None, :])[0]

    report = _decisions(creators, api_creators, categories, categories, tau=0.1, cutoff=cutoff)

    assert "C0" in report["candidates_after_cutoff_mismatch"]["M2"]["creators"]
    assert report["decisions_identical"] is False


def test_rank_swap_among_candidates_counts_as_a_different_result() -> None:
    creators, categories = _random_unit(10, 8, 20), _random_unit(3, 8, 21)
    scores = creators @ creators[0]
    order = [int(j) for j in np.argsort(-np.where(np.arange(10) == 0, -np.inf, scores))]
    first, second = order[0], order[1]
    api_creators = creators.copy()
    # 1위 후보의 C0과의 코사인이 2위보다 살짝 낮아지도록 C0 방향으로 다시 세운다(둘 다 상위 5 안에 남고 서로의 순서만 바뀐다)
    target = float(scores[second]) - 1e-4
    orthogonal = creators[first] - float(scores[first]) * creators[0]
    orthogonal = orthogonal / np.linalg.norm(orthogonal)
    api_creators[first] = (target * creators[0] + np.sqrt(1 - target**2) * orthogonal).astype(np.float32)

    report = _decisions(creators, api_creators, categories, categories, tau=0.1, cutoff=-1.0)

    assert "C0" in report["candidates_after_cutoff_mismatch"]["M2"]["creators"]
    assert report["decisions_identical"] is False


def test_client_sends_configured_model_name() -> None:
    calls = []

    def create(model, input):
        calls.append(model)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=7), data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])

    fake = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    client = OpenAIEmbeddingClient(client=fake, model="BAAI/bge-m3", dim=2)

    vectors = client.embed(["a", "b"])

    assert calls == ["BAAI/bge-m3"] and client.name == "BAAI/bge-m3" and client.last_input_tokens == 7
    assert np.allclose(vectors, [[0.6, 0.8], [0.6, 0.8]])  # 정규화됨


def _write_local_cache(cache_dir, creators, categories, creator_vectors, category_vectors) -> None:
    pipeline._save_vectors(cache_dir / "creators_bge-m3.npz", [c.id for c in creators], creator_vectors)
    pipeline._save_vectors(cache_dir / "categories_bge-m3.npz", [c.code for c in categories], category_vectors)
    input_hash = pipeline._content_hash(
        pipeline._embedding_model_identity("bge-m3"),
        *[c.input_text() for c in creators],
        *[c.description for c in categories],
    )
    (cache_dir / "embed_bge-m3.input_hash").write_text(input_hash, encoding="utf-8")


def _write_params_and_llm_tags(dir_, creators) -> None:
    per_embedding = {"tau": 0.0, "bonus_m3": 0.1, "bonus_m4": 0.2, "bonus_r2": 0.2, **{f"cutoff_m{i}": 0.0 for i in range(1, 5)}}
    (dir_ / "selected_params.json").write_text(
        json.dumps({"llm_model": "m", "per_embedding": {"bge-m3": per_embedding}}), encoding="utf-8"
    )
    (dir_ / "llm_tags_m_run0.json").write_text(json.dumps({c.id: ["FOOD"] for c in creators}), encoding="utf-8")


class _FakeApiClient:
    """로컬 캐시와 같은 벡터를 돌려주는 API 대역이다. 호출 순서대로 크리에이터 → 카테고리 벡터를 준다."""

    name = "BAAI/bge-m3"

    def __init__(self, *outputs: np.ndarray) -> None:
        self._outputs = list(outputs)
        self.last_input_tokens = None

    def embed(self, texts):
        self.last_input_tokens = 100
        return self._outputs.pop(0)


def test_cmd_api_parity_uses_cache_and_writes_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    creators, categories = pipeline.load_creators(), pipeline.load_categories()
    creator_vectors, category_vectors = _random_unit(len(creators), 8, 8), _random_unit(len(categories), 8, 9)
    _write_local_cache(tmp_path, creators, categories, creator_vectors, category_vectors)
    _write_params_and_llm_tags(tmp_path, creators)

    pipeline.cmd_api_parity(SimpleNamespace(), api_client=_FakeApiClient(creator_vectors, category_vectors))

    saved = json.loads((tmp_path / "api_parity.json").read_text(encoding="utf-8"))
    assert saved["similarity_equivalent"] is True and saved["creator_count"] == len(creators)
    assert saved["input_tokens"] == 200 and saved["cost_usd"] == pytest.approx(200 / 1_000_000 * 0.01)


def test_cmd_api_parity_stops_when_local_cache_is_stale(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    creators, categories = pipeline.load_creators(), pipeline.load_categories()
    _write_local_cache(tmp_path, creators, categories, _random_unit(len(creators), 8, 10), _random_unit(len(categories), 8, 11))
    (tmp_path / "embed_bge-m3.input_hash").write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="embed"):
        pipeline.cmd_api_parity(SimpleNamespace(), api_client=_FakeApiClient())

    assert not (tmp_path / "api_parity.json").exists()  # API를 부르기 전에 멈춘다


def test_api_client_requires_deepinfra_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPINFRA_API_KEY"):
        pipeline._api_bge_m3_client()


def test_cmd_api_parity_stops_before_api_when_selected_params_are_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    creators, categories = pipeline.load_creators(), pipeline.load_categories()
    _write_local_cache(tmp_path, creators, categories, _random_unit(len(creators), 8, 18), _random_unit(len(categories), 8, 19))

    with pytest.raises(RuntimeError, match="select-params"):
        pipeline.cmd_api_parity(SimpleNamespace(), api_client=_FakeApiClient())


def _prepare_api_select_params(tmp_path, monkeypatch, seed: int):
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    creators, categories = pipeline.load_creators(), pipeline.load_categories()
    creator_vectors, category_vectors = _random_unit(len(creators), 8, seed), _random_unit(len(categories), 8, seed + 1)
    _write_local_cache(tmp_path, creators, categories, creator_vectors, category_vectors)
    _write_params_and_llm_tags(tmp_path, creators)
    return creators, categories, creator_vectors, category_vectors


def test_api_select_params_caches_api_vectors_and_keeps_selected_params(tmp_path, monkeypatch) -> None:
    creators, categories, creator_vectors, category_vectors = _prepare_api_select_params(tmp_path, monkeypatch, 22)
    before = (tmp_path / "selected_params.json").read_text(encoding="utf-8")

    pipeline.cmd_api_select_params(SimpleNamespace(force=False), api_client=_FakeApiClient(creator_vectors, category_vectors))

    saved = json.loads((tmp_path / "api_selected_params.json").read_text(encoding="utf-8"))
    assert set(saved) >= {"local_params", "api_params", "test_zero_shot", "local_params_on_local_vs_api_params_on_api"}
    assert {"tau", "bonus_m3", "bonus_m4", "cutoff_m2"} <= set(saved["api_params"])
    assert saved["input_tokens_this_run"] == 200
    assert (tmp_path / "creators_bge-m3-api.npz").exists() and (tmp_path / "embed_bge-m3-api.input_hash").exists()
    assert (tmp_path / "selected_params.json").read_text(encoding="utf-8") == before  # 로컬 선택값은 그대로


def test_api_select_params_reuses_cached_vectors_unless_forced(tmp_path, monkeypatch) -> None:
    creators, categories, creator_vectors, category_vectors = _prepare_api_select_params(tmp_path, monkeypatch, 24)
    pipeline.cmd_api_select_params(SimpleNamespace(force=False), api_client=_FakeApiClient(creator_vectors, category_vectors))

    # 캐시가 있으면 API를 부르지 않는다: 호출되면 출력이 없어 IndexError로 드러난다
    pipeline.cmd_api_select_params(SimpleNamespace(force=False), api_client=_FakeApiClient())
    assert json.loads((tmp_path / "api_selected_params.json").read_text(encoding="utf-8"))["input_tokens_this_run"] == 0

    pipeline.cmd_api_select_params(SimpleNamespace(force=True), api_client=_FakeApiClient(creator_vectors, category_vectors))
    assert json.loads((tmp_path / "api_selected_params.json").read_text(encoding="utf-8"))["input_tokens_this_run"] == 200


def test_api_select_params_stops_before_api_when_selected_params_are_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    creators, categories = pipeline.load_creators(), pipeline.load_categories()
    _write_local_cache(tmp_path, creators, categories, _random_unit(len(creators), 8, 26), _random_unit(len(categories), 8, 27))

    with pytest.raises(RuntimeError, match="select-params"):
        pipeline.cmd_api_select_params(SimpleNamespace(force=False), api_client=_FakeApiClient())
