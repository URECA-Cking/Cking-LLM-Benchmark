"""Vectorized hierarchical scoring agrees with the scalar fallback contract."""
import numpy as np
from src.hierarchy_eval import bonus_vector
from src.subtopic_tags import hierarchical_bonus


def test_vectorized_bonus_matches_scalar():
    taxonomy={'subtopics':[{'code':'X_A','parent':'X'},{'code':'X_B','parent':'X'},{'code':'Y_A','parent':'Y'}]}
    qp={'X','Y'};qt={'X_A'}
    cp=[{'X'},{'X'},{'X'},{'Y'},set()];ct=[{'X_A'},{'X_B'},set(),{'Y_A'},set()]
    expected=[hierarchical_bonus(qp,p,qt,t,taxonomy,.2) for p,t in zip(cp,ct)]
    assert np.allclose(bonus_vector(qp,qt,cp,ct,taxonomy,.2,.5,1),expected)


def test_parent_only_selection_preserves_original_bonus():
    from src.hierarchy_eval import hierarchical_rank
    from src.flow_m234 import rankings
    taxonomy={'subtopics':[{'code':'X_A','parent':'X'}]}
    ids=['a','b'];vectors=np.array([[.9,0.],[.8,0.]])
    parents={'a':set(),'b':{'X'}};topics={'a':set(),'b':{'X_A'}}
    case={'route':'tags','tags':['X'],'seeds':[]}
    expected=rankings(case,ids,vectors,np.eye(2),['X','Y'],parents,parents,{'bonus_m3':.2,'bonus_m4':.2},2)['M3']
    actual=hierarchical_rank(case,ids,vectors,np.eye(2),['X','Y'],parents,topics,taxonomy,.2,.5,1.,2)
    assert actual==expected
