"""Frozen evidence and complete-sample reporting behavior."""
import json
from types import SimpleNamespace
import pytest
from src import hierarchy_judge as module
from src.flow_eval import digest
from src.judge import pair_text_hash


@pytest.fixture
def prepared(tmp_path):
    flow=tmp_path/'flow';base=tmp_path/'base';out=tmp_path/'out'
    for path in (flow,base,out):path.mkdir()
    case={'id':'q','route':'creators','group':'regular','seeds':['seed'],'tags':[]}
    plan={'cases':[case],'seed':42,'top_k':5}
    frozen={'plan':plan,'hash':digest(plan)}
    baseline={'base_hash':frozen['hash'],'case_ids':['q'],'model':'fake','creator_prompt':'p','other_prompt':'p','temperature':0}
    candidates={'q':{'M2':[{'id':'c','score':.9}],'H3':[{'id':'d','score':.8}]}}
    pairs={f'q::{cid}':{'input':'q','candidate':cid,'route':'creators','hash':pair_text_hash('q',cid)} for cid in ('c','d')}
    hierarchy={'baseline_hash':digest(baseline),'candidate_hash':digest(candidates),'pair_hashes':{k:v['hash'] for k,v in pairs.items()},'evaluation_inputs':1}
    for path,value in [(flow/'plan.json',frozen),(base/'evaluation_plan.json',baseline),(out/'hierarchy_plan.json',hierarchy),(out/'candidates.json',candidates),(out/'evaluation_pairs.json',pairs),(out/'reused_judgments.json',{'q::c':{'score':1,'hash':pairs['q::c']['hash'],'source':'old'}})]:path.write_text(json.dumps(value))
    return SimpleNamespace(flow_dir=flow,baseline_dir=base,output_dir=out,judge=False)


def test_partial_judgments_do_not_generate_biased_group_results(prepared):
    module.execute(prepared)
    summary=json.loads((prepared.output_dir/'summary.json').read_text())
    assert summary['pending_pairs']==1 and summary['groups']=={}


def test_changed_evidence_rejected(prepared):
    p=prepared.output_dir/'evaluation_pairs.json';value=json.loads(p.read_text());value['q::d']['candidate']='changed';p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='evidence changed'):module.execute(prepared)


def test_complete_scores_resume_without_duplicate_calls(prepared,monkeypatch):
    calls=[]
    monkeypatch.setattr(module,'OpenAI',lambda **kwargs:object())
    def factory(**kwargs):
        def judge(q,c):calls.append((q,c));return SimpleNamespace(score=1,input_tokens=5,output_tokens=2)
        return SimpleNamespace(judge=judge)
    monkeypatch.setattr(module,'OpenAIJudge',factory)
    prepared.judge=True;module.execute(prepared);module.execute(prepared)
    assert calls==[('q','d')]
    summary=json.loads((prepared.output_dir/'summary.json').read_text())
    assert summary['status']=='llm_complete_human_unverified'
    assert summary['groups']['creators/regular']['p_at5']=={'M2':.2,'H3':.2}
