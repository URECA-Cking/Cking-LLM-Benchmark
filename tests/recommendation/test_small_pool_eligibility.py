"""고정 모델로 후보 선별 계약을 검증한다. 실제 BGE/GPT의 의미 품질 평가가 아니다."""
import pytest

from src.recommendation.cache import text_hash
from src.recommendation.eligibility import validate
from src.recommendation.interest import M2_METHOD, M3_METHOD
from src.recommendation.models import CreatorProfile
from tests.recommendation.test_interest import categories, recommender, vector
from tests.recommendation.test_service import FakeEmbeddingClient, FakeTaggingClient, build_service


@pytest.mark.parametrize("size", [0, 1, 2, 5, 19, 20, 21, 540])
@pytest.mark.parametrize("method", [M2_METHOD, M3_METHOD])
def test_only_qualified_candidates_before_top_n_without_padding(tmp_path, size, method):
    profiles = [CreatorProfile(i + 1, "relevant" if i < 2 else "other") for i in range(size)]
    service = recommender(tmp_path, profiles, {"relevant": vector(1), "other": vector(0, 1)})
    payload = service.recommend(categories()[0].code, method).to_backend_payload()
    assert [c["creatorId"] for c in payload["candidates"]] == list(range(1, min(size, 2) + 1))
    assert all(p["tags"] == [categories()[0].code] for p in payload["eligibility"]["profiles"])
    assert service.recommend(categories()[2].code, method).candidates == ()


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_zero_shot_tag_count_is_an_upper_bound_not_three_required(tmp_path, count):
    values = [1] * count if count else [-1] * 17
    service = recommender(tmp_path, [CreatorProfile(1, "sample")], {"sample": vector(*values)})
    found = [r.code for r in categories() if service.recommend(r.code).candidates]
    assert len(found) == count


def test_unrelated_and_unclassified_llm_tags_are_not_promoted_by_high_cosine(tmp_path):
    seed = CreatorProfile(1, "홈트 운동 채널")
    candidates = [CreatorProfile(2, "요리 채널"), CreatorProfile(3, "분류 근거 없음"), CreatorProfile(4, "근력 운동 채널"), CreatorProfile(5, " ")]
    vectors = {p.text: [1, 0] for p in [seed, *candidates] if p.text}
    tags = {seed.text: ("FITNESS",), candidates[0].text: ("FOOD",), candidates[1].text: (), candidates[2].text: ("FITNESS",)}
    result = build_service(tmp_path, FakeEmbeddingClient(vectors), FakeTaggingClient(tags)).recommend(seed, candidates)
    assert [c.similar_creator_id for c in result.candidates] == [4]
    assert result.method == "M4"


def test_invalid_four_tag_cache_is_regenerated_not_truncated_or_padded(tmp_path):
    seed, other = CreatorProfile(1, "홈트 운동 채널"), CreatorProfile(2, "근력 운동 채널")
    model = FakeTaggingClient({seed.text: ("FITNESS",), other.text: ("FITNESS",)})
    service = build_service(tmp_path, FakeEmbeddingClient({seed.text:[1, 0], other.text:[1, 0]}), model)
    service.cache.put_tag_records([( text_hash(seed.text), service.config.tag_model_version, service.config.tag_cache_version, ("FITNESS",)*4)])
    assert len(service.recommend(seed, [other]).candidates) == 1
    assert seed.text in model.calls


def test_cached_proof_cannot_assign_requested_tag_to_unrelated_creator(tmp_path):
    service = recommender(tmp_path)
    payload = service.recommend(categories()[0].code).to_backend_payload()
    payload['eligibility']['profiles'][0]['tags'] = [categories()[2].code]
    with pytest.raises(ValueError):
        validate(payload['eligibility'], {p.creator_id:p for p in service.manifest.creators},
                 [c['creatorId'] for c in payload['candidates']], interest=categories()[0].code,
                 allowed=[r.code for r in categories()])
