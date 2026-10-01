"""Boundary and fallback behavior for subtopic experiments."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from src.subtopic_tags import load_taxonomy, validate_llm_topics, assign_zero_topics, hierarchical_bonus


@pytest.fixture
def taxonomy():
    return load_taxonomy(Path(__file__).resolve().parents[1]/'data/creator-subtopics.json', ['FITNESS','FOOD','GAME','BEAUTY','FASHION','TRAVEL','MUSIC','PET','TECH','KNOWLEDGE','ENTERTAIN','HUMOR','SPORTS','NEWS','LIFE','HOBBY','LIFETIP'])


def test_axes_are_disjoint_and_all_topics_defined(taxonomy):
    assert len(taxonomy['subtopics'])==85
    assert len(taxonomy['formats'])==10
    assert all(not t['code'].startswith('FORMAT_') for t in taxonomy['subtopics'])


def test_evidence_and_parent_restrictions(taxonomy):
    value=[{'code':'FITNESS_RUNNING','evidence':'달리기'}]
    assert validate_llm_topics(value,'달리기 훈련',{'FITNESS'},taxonomy)=={'FITNESS_RUNNING'}
    for text,parents in [('헬스',{'FITNESS'}),('달리기',{'SPORTS'})]:
        with pytest.raises(ValueError):validate_llm_topics(value,text,parents,taxonomy)
    with pytest.raises(ValueError):validate_llm_topics([{'code':'FORMAT_VLOG','evidence':'브이로그'}],'브이로그',{'LIFE'},taxonomy)
    assert validate_llm_topics([],'이름',{'LIFE'},taxonomy)==frozenset()


def test_missing_subtopics_fallback_and_mismatch_weaken(taxonomy):
    base=.2; parent={'FITNESS'}; running={'FITNESS_RUNNING'}; yoga={'FITNESS_YOGA'}
    assert hierarchical_bonus(parent,parent,set(),running,taxonomy,base)==base
    assert hierarchical_bonus(parent,parent,running,running,taxonomy,base)==pytest.approx(.3)
    assert hierarchical_bonus(parent,parent,running,yoga,taxonomy,base)==pytest.approx(.1)
    assert hierarchical_bonus(parent,{'FOOD'},set(),set(),taxonomy,base)==0


def test_unrelated_topic_rejected(taxonomy):
    with pytest.raises(ValueError):hierarchical_bonus({'FITNESS'},{'FITNESS'},{'FOOD_BAKING'},set(),taxonomy,.2)


def test_zero_topics_gated_by_parent_and_threshold(taxonomy):
    vectors=np.zeros((85,2))
    bycode={t['code']:i for i,t in enumerate(taxonomy['subtopics'])}
    vectors[bycode['FITNESS_RUNNING']]=[.6,0]
    vectors[bycode['FOOD_BAKING']]=[.9,0]
    assert assign_zero_topics(np.array([1.,0.]),vectors,taxonomy,{'FITNESS'},.5)=={'FITNESS_RUNNING'}
    assert assign_zero_topics(np.array([1.,0.]),vectors,taxonomy,{'FITNESS'},.7)==frozenset()


def test_bad_taxonomy_rejected(taxonomy,tmp_path):
    bad=deepcopy(taxonomy);bad['subtopics'][0]['exclude']=[]
    path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):load_taxonomy(path,taxonomy['parents'])


def test_invalid_evidence_retried_and_logged(taxonomy,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import openai
    from src.subtopic_tags import tag_llm
    from src.real_eval import Channel
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        quote='E999' if len(calls)==1 else 'E0'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'topics':[{'code':'FITNESS_RUNNING','evidence':quote}]})))],usage=SimpleNamespace(prompt_tokens=100,completion_tokens=10))
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:client)
    args=SimpleNamespace(output_dir=tmp_path)
    data=SimpleNamespace(pool=[Channel('a','name','달리기 훈련')])
    tag_llm(args,taxonomy,data,{'a':frozenset({'FITNESS'})},{'llm_calls':1,'tagger_model':'fake'})
    record=json.loads((tmp_path/'subtopics_llm.json').read_text())['tags']['a']
    assert record['attempts']==2
    assert record['input_tokens']==200
    assert record['topics'][0]['evidence']=='달리기 훈련'
    assert len((tmp_path/'validation_failures.jsonl').read_text().splitlines())==1


def test_rate_limit_is_retried_without_saving_a_missing_tag(taxonomy,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import openai
    import src.subtopic_tags as module
    from src.real_eval import Channel
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            error=RuntimeError('rate limit');error.status_code=429;raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'topics':[{'code':'FITNESS_RUNNING','evidence':'E0'}]})))],usage=SimpleNamespace(prompt_tokens=100,completion_tokens=10))
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    sleeps=[];monkeypatch.setattr(module.time,'sleep',lambda seconds:sleeps.append(seconds))
    module.tag_llm(SimpleNamespace(output_dir=tmp_path),taxonomy,SimpleNamespace(pool=[Channel('a','name','달리기')]),{'a':frozenset({'FITNESS'})},{'llm_calls':1,'tagger_model':'fake'})
    record=json.loads((tmp_path/'subtopics_llm.json').read_text())['tags']['a']
    assert len(calls)==2 and sleeps==[5]
    assert record['topics']==[{'code':'FITNESS_RUNNING','evidence':'달리기'}]
    assert record['input_tokens']==100
