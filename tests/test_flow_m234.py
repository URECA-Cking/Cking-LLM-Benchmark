"""Method separation and per-seed aggregation checks."""
import numpy as np
import pytest
from src.flow_m234 import rankings, reusable
from src.config import OPENAI_JUDGE_MODEL
from src.flow_experiment import PROMPT
from src.judge import pair_text_hash


def test_tag_input_is_shared_but_candidate_tags_are_method_specific():
    ids=['a','b']; vectors=np.array([[.9,0.],[.85,0.]])
    zero={'a':set(),'b':{'X'}}; llm={'a':{'X'},'b':set()}
    result=rankings({'route':'tags','seeds':[],'tags':['X']},ids,vectors,np.eye(2),['X','Y'],zero,llm,{'bonus_m3':.1,'bonus_m4':.2},1)
    assert result['M2'][0]['id']=='a'
    assert result['M3'][0]['id']=='b'
    assert result['M4'][0]['id']=='a'


def test_bonus_and_cosine_must_come_from_same_favorite():
    ids=['x','y','a','b']
    vectors=np.array([[1.,0.],[0.,1.],[.9,.1],[.8,.15]])
    tags={'x':{'X'},'y':{'Y'},'a':{'Y'},'b':{'X'}}
    case={'route':'favorites','seeds':['x','y'],'tags':[]}
    result=rankings(case,ids,vectors,np.eye(2),['X','Y'],tags,tags,{'bonus_m3':.1,'bonus_m4':.2},2)
    assert result['M2'][0]['id']=='a'
    assert result['M4'][0]['id']=='b'
    assert result['M4'][0]['score']==pytest.approx(1.)
    assert not {'x','y'} & {v['id'] for selected in result.values() for v in selected}


def test_unknown_and_duplicate_seeds_rejected():
    ids=['x']; tags={'x':set()}
    for seeds in (['missing'],['x','x']):
        with pytest.raises(ValueError):
            rankings({'route':'favorites','seeds':seeds,'tags':[]},ids,np.ones((1,1)),np.ones((1,1)),['X'],tags,tags,{'bonus_m3':.1,'bonus_m4':.2},5)


def test_old_pilot_score_requires_matching_text_and_prompt():
    key='tag-X::a'; h=pair_text_hash('query','candidate')
    pilot={key:{'score':1,'hash':h}}
    contract={'model':OPENAI_JUDGE_MODEL,'prompt':PROMPT,'temperature':0,'pair_text_hashes':{key:h}}
    assert reusable(key,'query','candidate',None,{},pilot,contract)['score']==1
    assert reusable(key,'changed','candidate',None,{},pilot,contract) is None
    contract['prompt']='changed'
    assert reusable(key,'query','candidate',None,{},pilot,contract) is None
