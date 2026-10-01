"""Behavioral checks for frozen recommendation-flow experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from src import flow_eval as flow
from src.real_eval import Channel, RealData


@pytest.fixture
def source():
    """Build orthogonal interests and several non-seed candidates."""
    data = RealData([Channel(cid, cid, cid) for cid in ['a', 'b', 'c', 'd', 'e', 'f', 'g']],
                    {'a': 'X', 'b': 'X', 'c': 'X', 'd': 'X', 'e': 'Y', 'f': 'Y', 'g': 'Y'},
                    [('X', 'X', 'X'), ('Y', 'Y', 'Y')])
    vectors = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    categories = np.eye(2)
    tags = {cid: frozenset({code}) for cid, code in data.gold.items()}
    params = {'dev_ids': ['a'], 'test_ids': ['b', 'c', 'd', 'e', 'f', 'g'],
              'queries': {'regular': ['b', 'e'], 'short': []}, 'bonus_m4': .2}
    return data, vectors, categories, tags, params


def rank(source, case):
    """Rank on the fixture's shared pool."""
    data, vectors, categories, tags, _ = source
    return flow.rank_candidates(case, [c.id for c in data.pool], vectors, categories, data.codes, tags, .2, 5, 42)


def test_plan_is_repeatable_and_never_uses_dev_seeds(source):
    data, _, _, _, params = source
    plan = flow.make_plan(data, params, 'snapshot')
    assert plan == flow.make_plan(data, params, 'snapshot')
    assert all('a' not in case['seeds'] for case in plan['cases'])
    assert plan['queries'] == params['queries']
    assert plan['history_source'] == 'synthetic_not_user_history'
    assert {case['group'] for case in plan['cases']} >= {'single_tag', 'two_tags', 'same_field', 'mixed_fields'}


def test_all_favorites_excluded_and_distinct_interests_preserved(source):
    case = {'id': 'mixed', 'route': 'favorites', 'seeds': ['b', 'e'], 'tags': []}
    result = rank(source, case)
    assert all(not {'b', 'e'} & {c['id'] for c in selected} for selected in result.values())
    assert {source[3][c['id']] for c in result['rank_fusion']} == {frozenset({'X'}), frozenset({'Y'})}
    assert result == rank(source, case)


def test_filtered_tags_do_not_fill_with_nonmatching_candidates(source):
    result = rank(source, {'id': 'tag', 'route': 'tags', 'seeds': [], 'tags': ['Y']})
    assert len(result['tag_filtered_embedding']) == 3
    assert {c['id'] for c in result['tag_overlap']} == {'e', 'f', 'g'}


@pytest.mark.parametrize('seeds', [['missing'], ['b', 'b']])
def test_bad_seed_input_rejected(source, seeds):
    with pytest.raises(ValueError):
        rank(source, {'id': 'bad', 'route': 'favorites', 'seeds': seeds, 'tags': []})


def test_source_hash_changes_with_nonquery_vectors_and_labels(source):
    data, vectors, categories, tags, params = source
    original = flow.source_snapshot(*source)
    changed = vectors.copy()
    changed[-1] = [1., 0.]
    assert original != flow.source_snapshot(data, changed, categories, tags, params)
    data.gold['g'] = 'X'
    assert original != flow.source_snapshot(*source)


def test_stale_plan_rejected_and_completed_sheet_protected(source, tmp_path, monkeypatch):
    monkeypatch.setattr(flow, 'load_source', lambda *_: source)
    args = argparse.Namespace(step='prepare', data_dir=Path('.'), source_dir=Path('.'), output_dir=tmp_path)
    flow.run(args)
    args.step = 'candidates'
    flow.run(args)
    sheet = tmp_path / 'human_sheet.csv'
    original = sheet.read_bytes()
    with pytest.raises(FileExistsError):
        flow.run(args)
    assert sheet.read_bytes() == original
    saved = json.loads((tmp_path / 'plan.json').read_text())
    saved['plan']['top_k'] = 1
    (tmp_path / 'plan.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='changed'):
        flow.run(args)


def test_empty_filtered_list_and_zero_centroid_are_valid(source):
    source[3].update({cid: frozenset() for cid in source[3]})
    result = rank(source, {'id': 'tag', 'route': 'tags', 'seeds': [], 'tags': ['Y']})
    assert result['tag_overlap'] == []
    assert result['tag_filtered_embedding'] == []
    assert np.array_equal(flow.normalized(np.zeros(2)), np.zeros(2))
