import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.detail_tagging import digest, classify
from src.recommendation.cache import JsonModelCache
from src.recommendation.http import BackendRequestError, BackendClientConfig
from src.recommendation.saved_tags import (METHOD, ROOT, SavedTagBackend, generate, recommend, run, validate_snapshot)
from src.recommendation.saved_tags import recommend_for_member


def snapshot(count=2):
    result = {'contractVersion': 'saved-tags-v1', 'taxonomyVersion': 'v1', 'taxonomyHash': 'a'*64,
              'generationConfigHash': 'b'*64,
              'tags': [{'type': kind, 'code': code, 'name': code, 'description': text} for kind, code, text in
                       [('PARENT','FOOD','food'), ('PARENT','GAME','game'), ('TOPIC','FOOD_BAKING','baking'), ('TOPIC','FOOD_EATING','eating')]],
              'items': [{'creatorId': i+1, 'introduction': f'intro{i}', 'inputRevision': 1, 'confirmationId': i+1,
                         'parents': ['FOOD'], 'topics': ['FOOD_BAKING']} for i in range(count)]}
    result['snapshotHash'] = digest(result)
    result['lastApplicationSequence'] = 0
    return result


class Model:
    dim = 2
    def __init__(self): self.calls = []
    def embed(self, texts):
        self.calls.extend(texts)
        return np.array([[0,1] if t == 'game' else [1,0] for t in texts])


def payload(tmp_path, value=None):
    value = value or snapshot()
    return generate(value, Model(), JsonModelCache(tmp_path/'cache.json'), 'fake-v1')


def test_exact_tags_before_top_n_and_no_filling(tmp_path):
    value = snapshot()
    result = payload(tmp_path, value)
    assert len(recommend(value, result, topics=['FOOD_BAKING'], top_n=20)) == 2
    assert recommend(value, result, parents=['GAME']) == []
    assert recommend(value, result, topics=['FOOD_EATING']) == []
    assert recommend(value, result) == []
    assert all(row['code'] != 'GAME' for row in result['scores'])


def test_or_intersection_and_deduplication(tmp_path):
    value = snapshot()
    result = recommend(value, payload(tmp_path), parents=['FOOD','GAME'], topics=['FOOD_BAKING'])
    assert [row['creatorId'] for row in result] == [1,2]
    assert [t['code'] for t in result[0]['matchedTags']] == ['FOOD','FOOD_BAKING']


def test_member_interest_never_mixes_follow_or_popularity_and_excludes_before_limit(tmp_path):
    value = snapshot(25)
    result = payload(tmp_path, value)
    assert recommend_for_member(value, result, source='INTEREST', followed_ids=[1]) == []
    assert recommend_for_member(value, result, source='INTEREST', interest_codes=['UNKNOWN']) == []
    assert recommend_for_member(value, result, source='INTEREST', interest_codes=['GAME']) == []
    rows = recommend_for_member(value, result, source='INTEREST', interest_codes=['FOOD'], followed_ids=list(range(1, 24)), self_id=24)
    assert [r['creatorId'] for r in rows] == [25]
    assert rows[0]['interestCodes'] == ['FOOD'] and rows[0]['seedCreatorIds'] == []


def test_follow_all_seeds_and_same_parent_sibling_are_related_without_inventing_tags(tmp_path):
    value = snapshot(5)
    value['items'][1].update(parents=['GAME'], topics=[])
    value['items'][2].update(topics=['FOOD_EATING'])
    value['items'][3].update(parents=['GAME'], topics=[])
    value['items'][4].update(parents=[], topics=[])
    value['items'].pop()  # 미확인/태그 없는 후보는 BE snapshot부터 제외한다.
    value['snapshotHash'] = digest({k:v for k,v in value.items() if k not in {'snapshotHash','lastApplicationSequence'}})
    result = payload(tmp_path, value)
    rows = recommend_for_member(value, result, source='FOLLOW', followed_ids=[1,2,999])
    assert [r['creatorId'] for r in rows] == [3,4]
    assert [r['seedCreatorIds'] for r in rows] == [[1],[2]]
    assert rows[0]['interestCodes'] == []
    assert [t['code'] for t in rows[0]['matchedTags']] == ['FOOD']
    assert recommend_for_member(value, result, source='FOLLOW', followed_ids=[999]) == []
    assert recommend_for_member(value, result, source='FOLLOW', followed_ids=[1,2,3,4]) == []


def test_full_eligible_pool_survives_global_twenty_limit(tmp_path):
    value = snapshot(25)
    result = payload(tmp_path, value)
    assert len(result['scores']) == 50  # 모든 적격 tag/Creator 점수를 적재한다.
    assert any(r['creatorId'] == 25 for r in result['scores'])
    assert len(recommend(value, result, parents=['FOOD'], top_n=20)) == 20


def test_empty_generation_does_not_call_model(tmp_path):
    model = Model()
    result = generate(snapshot(0), model, JsonModelCache(tmp_path/'cache.json'),'fake-v1')
    assert result['scores'] == [] and model.calls == []


def test_cache_and_changed_tags_without_intro_invalidate_identity(tmp_path):
    value = snapshot(); model = Model(); cache = JsonModelCache(tmp_path/'cache.json')
    first = generate(value, model, cache,'fake-v1'); count = len(model.calls)
    assert generate(value, model, cache,'fake-v1') == first
    assert len(model.calls) == count
    value['items'][0]['topics'] = []
    identity = {k:v for k,v in value.items() if k not in {'snapshotHash','lastApplicationSequence'}}
    value['snapshotHash'] = digest(identity)
    changed = generate(value, model, cache,'fake-v1')
    assert len(changed['scores']) == len(first['scores'])-1
    assert changed['snapshotHash'] != first['snapshotHash']
    assert len(model.calls) == count


@pytest.mark.parametrize('mutate', [
    lambda s:s['items'][0].update(introduction=' '),
    lambda s:s['items'][0].update(parents=['GAME','GAME']),
    lambda s:s['items'][0].update(topics=['OTHER']),
    lambda s:s['items'][0].update(confirmationId=0),
    lambda s:s['items'].append(s['items'][0]),
])
def test_bad_input_rejected_before_model(tmp_path,mutate):
    value=snapshot(); mutate(value)
    value['snapshotHash']=digest({k:v for k,v in value.items() if k not in {'snapshotHash','lastApplicationSequence'}})
    model=Model()
    with pytest.raises(ValueError): generate(value,model,JsonModelCache(tmp_path/'cache.json'),'fake-v1')
    assert not model.calls


class Backend:
    target_identity='https://example.test'
    def __init__(self): self.value=snapshot(); self.writes=[]; self.sequence=0; self.lose=False; self.reject=False
    def get(self,path):
        assert path==ROOT+'/inputs'
        return {**copy.deepcopy(self.value),'lastApplicationSequence':self.sequence}
    def put(self,path,payload):
        assert path==ROOT+'/rankings'
        self.writes.append(payload)
        if self.reject: raise BackendRequestError('conflict',status=409)
        self.sequence=payload['applicationSequence']
        if self.lose:
            self.lose=False
            raise BackendRequestError('lost response')
        return {'applied':True,'applicationSequence':self.sequence,'snapshotHash':payload['snapshotHash']}


def test_durable_payload_lost_receipt_resume_and_skip(tmp_path):
    backend=Backend(); backend.lose=True; model=Model()
    args=(backend,tmp_path/'state',tmp_path/'cache.json',model,'fake-v1')
    with pytest.raises(BackendRequestError): run(*args,apply=True)
    calls=len(model.calls)
    assert run(*args,apply=True)['status']=='completed'
    assert backend.writes[0]==backend.writes[1]
    assert len(model.calls)==calls
    assert run(*args,apply=True)['status']=='skipped'
    assert len(backend.writes)==2


def test_409_is_not_success_and_a_to_b_to_a_uses_new_sequence(tmp_path):
    backend=Backend(); args=(backend,tmp_path/'state',tmp_path/'cache.json',Model(),'fake-v1')
    run(*args,apply=True)
    old=copy.deepcopy(backend.value)
    backend.value=snapshot(1); run(*args,apply=True)
    backend.value=old; run(*args,apply=True)
    assert [r['applicationSequence'] for r in backend.writes]==[1,2,3]
    backend.reject=True; backend.value=snapshot(0)
    with pytest.raises(BackendRequestError): run(*args,apply=True)
    state=json.loads((tmp_path/'state/current.json').read_text())
    assert state['fenced'] and not state['applied']


def test_model_failure_does_not_apply_empty_result(tmp_path):
    backend=Backend()
    class Broken(Model):
        def embed(self,texts): raise RuntimeError('provider')
    with pytest.raises(RuntimeError): run(backend,tmp_path/'state',tmp_path/'cache.json',Broken(),'fake-v1',apply=True)
    assert not backend.writes


def test_dry_run_never_puts(tmp_path):
    backend=Backend()
    assert run(backend,tmp_path/'state',tmp_path/'cache.json',Model(),'fake-v1')['status']=='generated'
    assert not backend.writes and (tmp_path/'state/dry-run-payload.json').exists()


def test_dedicated_transport_and_exact_path_allowlist():
    with pytest.raises(Exception): SavedTagBackend(BackendClientConfig('http://example.test'),'token')
    backend=SavedTagBackend(BackendClientConfig('https://example.test'),'token')
    with pytest.raises(ValueError): backend.get('/api/creators')
    with pytest.raises(ValueError): backend.put('/api/admin/creators/1/similar',{})


@pytest.mark.parametrize('count',[0,1,2])
def test_classification_is_maximum_not_exactly_three(count):
    codes=['FOOD','GAME'][:count]
    taxonomy={'parents':['FOOD','GAME'],'subtopics':[]}
    candidates={'parents':[{'code':c,'evidence':'intro','core_evidence':'','repetition_evidence':[]} for c in codes], 'topics':[]}
    assert len(classify('intro',candidates,taxonomy)['parents'])==count
