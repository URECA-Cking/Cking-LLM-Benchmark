"""의미 분류는 LLM만, 임베딩은 순위만 담당하는 회귀 검증."""
from dataclasses import replace

import pytest

from src.recommendation.models import CreatorProfile
from tests.recommendation.test_interest import FakeTagger, categories, recommender, vector


def test_topn_is_applied_after_llm_filter_even_with_unrelated_maximum_cosine(tmp_path):
    tagger = FakeTagger({'cook': ['FOOD'], 'runner': ['FITNESS'], 'unknown': []})
    service = recommender(tmp_path,
        [CreatorProfile(1, 'runner'), CreatorProfile(2, 'cook'), CreatorProfile(3, 'unknown'), CreatorProfile(4, '')],
        {'runner': vector(0, 1), 'cook': vector(.8, .6), 'unknown': vector(0, 1)}, tagger=tagger)
    service.config = replace(service.config, top_n=20)
    payload = service.recommend('FOOD').to_backend_payload()
    assert [(c['creatorId'], c['score']) for c in payload['candidates']] == [(2, .6)]
    assert payload['eligibility']['profiles'][0]['tags'] == ['FOOD']
    assert '' not in tagger.calls
    assert service.recommend('BEAUTY').candidates == ()
    assert len(tagger.calls) == 3


def test_llm_tags_are_reused_across_all_interests_and_reruns(tmp_path):
    tagger = FakeTagger({'creator': ['FOOD', 'HOBBY']})
    first = recommender(tmp_path, [CreatorProfile(1, 'creator')], {'creator': vector(1)}, tagger=tagger)
    for c in categories():
        result = first.recommend(c.code)
        assert bool(result.candidates) == (c.code in {'FOOD', 'HOBBY'})
    assert tagger.calls == ['creator']
    second_tagger = FakeTagger()
    second = recommender(tmp_path, [CreatorProfile(1, 'creator')], client=first.embedding_client, tagger=second_tagger)
    assert second.recommend('FOOD').to_backend_payload() == first.recommend('FOOD').to_backend_payload()
    assert second_tagger.calls == []


@pytest.mark.parametrize('tags', [['UNKNOWN'], ['FOOD', 'FOOD'], ['FOOD', 'GAME', 'TECH', 'FITNESS']])
def test_invalid_llm_tags_fail_closed_without_embedding_fallback(tmp_path, tags):
    service = recommender(tmp_path, [CreatorProfile(1, 'creator')], {'creator': vector(1)},
                          tagger=FakeTagger({'creator': tags}))
    with pytest.raises(ValueError):
        service.recommend('FITNESS')
    assert service.embedding_client.calls == []


def test_tagger_outage_does_not_fall_back_or_repeat_for_17_interests(tmp_path):
    class Broken:
        calls = 0

        def tag(self, text):
            self.calls += 1
            raise RuntimeError('provider unavailable')

    tagger = Broken()
    service = recommender(tmp_path, tagger=tagger)
    for c in categories():
        with pytest.raises(RuntimeError, match='provider unavailable'):
            service.recommend(c.code)
    assert tagger.calls == 1
    assert service.embedding_client.calls == []


@pytest.mark.parametrize('change', [
    {'tag_model_version': 'another-model'}, {'tag_prompt_version': 'next'}, {'tag_prompt': 'changed prompt'},
])
def test_tagging_contract_change_invalidates_generation_and_tag_cache(tmp_path, change):
    old = recommender(tmp_path)
    new = recommender(tmp_path, config=replace(old.config, **change))
    assert old.input_hash('FOOD') != new.input_hash('FOOD')
    assert old._classifier.config.tag_cache_version != new._classifier.config.tag_cache_version or (
        old.config.tag_model_version != new.config.tag_model_version)
