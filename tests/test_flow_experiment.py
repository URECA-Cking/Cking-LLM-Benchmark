"""Pilot selection and honest incomplete-score handling."""
import pytest

from src.flow_experiment import precision, paired_interval, select_pilot, score_cases


def test_precision_requires_complete_binary_scores_and_penalizes_empty_slots():
    selected = [{'id': 'c'}]
    assert precision(selected, 'q', {'q::c': {'score': 1}}, 5) == .2
    assert precision([], 'q', {}, 5) == 0
    with pytest.raises(ValueError, match='Incomplete'):
        precision(selected, 'q', {}, 5)
    with pytest.raises(ValueError, match='Invalid'):
        precision(selected, 'q', {'q::c': {'score': '1'}}, 5)


def test_paired_interval_uses_matching_inputs_and_is_repeatable():
    result = paired_interval([1, .8], [.8, .6], 42)
    assert result['difference'] == pytest.approx(.2)
    assert result == paired_interval([1, .8], [.8, .6], 42)
    assert result['interval95'] == pytest.approx([.2, .2])


def test_select_pilot_keeps_all_singles_and_favorites_and_freezes_pairs():
    plan = {'seed': 42, 'cases': [
        {'id': 't1', 'group': 'single_tag', 'route': 'tags'},
        {'id': 't2', 'group': 'single_tag', 'route': 'tags'},
        *({'id': f'p{i}', 'group': 'two_tags', 'route': 'tags'} for i in range(5)),
        {'id': 'f', 'group': 'mixed_fields', 'route': 'favorites'},
        {'id': 'c', 'group': 'regular', 'route': 'creators'},
    ]}
    chosen = select_pilot(plan)
    assert len(chosen) == 5
    assert {'t1', 't2', 'f'} <= {c['id'] for c in chosen}
    assert 'c' not in {c['id'] for c in chosen}
    assert chosen == select_pilot(plan)


def test_score_groups_separately_and_shared_pairs_reused():
    cases = [{'id': 'q', 'route': 'tags', 'group': 'single_tag'}]
    candidates = {'q': {'left': [{'id': 'c'}], 'right': [{'id': 'c'}]}}
    result = score_cases(cases, candidates, {'q::c': {'score': 1}}, 5, 42)['tags/single_tag']
    assert result['p_at5'] == {'left': .2, 'right': .2}
    assert result['paired']['left minus right']['difference'] == 0


def test_unequal_paired_inputs_rejected():
    with pytest.raises(ValueError, match='equal lengths'):
        paired_interval([1, 0], [0], 42)
