import json
from pathlib import Path

import pytest

from src.detail_tagging import automatic_record, digest
from src.recommendation.content_worker import ContentWorker, ROOT, input_hash, normalized_intro, fingerprint
from src.recommendation.http import BackendRequestError
from src.recommendation.locking import file_lock, BatchAlreadyRunning
from src.recommendation.subtopic_taxonomy import subtopic_hash


@pytest.fixture
def taxonomy():
    return json.loads(Path('data/creator-subtopics.json').read_text(encoding='utf-8'))


class Backend:
    target_identity = 'https://test.example'

    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
        self.model = "mock-model"
        self.rows = [self.row('노래 커버', 1)]
        self.sent, self.acks = [], []
        self.error = None

    @staticmethod
    def row(text, revision):
        return dict(creatorId=1, introduction=text, inputRevision=revision, inputHash=input_hash(text))

    def get(self, path):
        return dict(contractVersion='content-tags-v1', taxonomyApproved=True,
                    taxonomyHash=subtopic_hash(self.taxonomy), generationConfigHash=digest(fingerprint(self.taxonomy,self.model)), items=self.rows, nextCursor=None)

    def put(self, path, payload):
        if path.endswith('/reconcile'):
            return {'completedCount': len(self.rows)}
        if path.endswith('/claim'):
            return dict(jobId='job-1', leaseToken='lease', generation=1)
        if path.endswith('/ack'):
            self.acks.append(payload)
            return {}
        self.sent.append(payload)
        if self.error:
            raise self.error
        return dict(automaticRevision=payload['automaticRevision'], status='idempotent')


def setup_worker(tmp_path, taxonomy, **kwargs):
    backend = Backend(taxonomy)
    calls = []

    def model(cid, text, tax, model):
        calls.append(text)
        return automatic_record(cid, text, {'parents': [], 'topics': []}, tax, model)

    worker = ContentWorker(backend, tmp_path, taxonomy, 'mock-model', model, lambda: 0,
                           approved=True, max_calls=10, **kwargs)
    return worker, backend, calls


def test_reuse_and_introduction_change(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    assert worker.run()['failureCount'] == 0
    worker.run()
    assert len(calls) == len(be.sent) == 1
    be.rows = [be.row('다른 소개', 2)]
    worker.run()
    assert len(calls) == 2
    assert be.sent[0]['automaticRevision'] != be.sent[1]['automaticRevision']
    assert be.sent[0]['creatorConfirmed'] is False


def test_model_change_and_empty(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    worker.run()
    other = ContentWorker(be, tmp_path, taxonomy, 'new-model', worker.classify_fn, lambda: 0,
                          approved=True, max_calls=10)
    be.model = "new-model"
    other.run()
    assert len(calls) == 2
    be.rows = [be.row(None, 2)]
    assert other.run()['failureCount'] == 0
    assert len(calls) == 2
    assert be.sent[-1]['record']['result']['unclassified']


def test_network_retry_reuses_immutable_payload(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.error = BackendRequestError('private secret', status=503)
    assert worker.run('event')['failureCount'] == 1
    original = be.sent[0]
    assert be.acks[-1]['status'] == 'retry'
    be.error = None
    assert worker.run('event')['failureCount'] == 0
    assert be.sent[-1] == original
    assert len(calls) == 1
    assert be.acks[-1]['generation'] == 1
    assert 'private secret' not in (tmp_path / 'summary.json').read_text()


def test_conflict_never_replays_old_payload(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.error = BackendRequestError('stale', status=409)
    worker.run()
    worker.run()
    assert len(be.sent) == 1
    be.rows = [be.row('새 소개', 2)]
    be.error = None
    assert worker.run()['failureCount'] == 0
    assert len(calls) == 2


def test_provider_failure_budget_and_independent_stages(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    def fail(*args):
        calls.append('failure')
        raise RuntimeError('sensitive input')
    worker.classify_fn = fail
    worker.max_calls = 1
    first = worker.run()
    assert first['stages']['recommendations']['status'] == 'success'
    worker.run()
    assert calls == ['failure']
    assert not be.sent


def test_recommendation_failure_not_hidden(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    worker.recommend = lambda: 1
    assert worker.run('event')['failureCount'] == 1
    assert be.acks[-1]['status'] == 'retry'
    worker.recommend = lambda: 0
    assert worker.run('event')['failureCount'] == 0
    assert len(calls) == 1


def test_lock_does_not_claim(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    with file_lock(tmp_path / '.worker.lock'):
        with pytest.raises(BatchAlreadyRunning):
            worker.run('event')
    assert not be.acks and not calls


def test_approval_and_contract_fail_closed(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    worker.approved = False
    assert worker.run()['failureCount'] == 1
    assert not calls
    be.rows[0]['inputHash'] = 'bad'
    assert worker.run()['failureCount'] == 1
    assert not calls


def test_shared_normalization_fixture():
    fixture = json.loads(Path('fixtures/content-tags-v1.json').read_text(encoding='utf-8'))
    for row in fixture['normalization']:
        assert normalized_intro(row['raw']) == row['normalized']
        assert input_hash(row['raw']) == row['sha256']

def test_restart_after_remote_commit_reuses_payload(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    worker.run()
    next((tmp_path / 'records').glob('*/receipt.json')).unlink()
    original = be.sent[0]
    restarted = ContentWorker(be, tmp_path, taxonomy, 'mock-model', worker.classify_fn, lambda: 0,
                              approved=True, max_calls=10)
    restarted.run()
    assert be.sent[-1] == original
    assert len(calls) == 1


def test_edit_during_run_keeps_claimed_generation(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    def edit():
        be.rows = [be.row('수정 중 변경', 2)]
        return 0
    worker.recommend = edit
    worker.run('event')
    assert be.acks[-1]['generation'] == 1
    worker.recommend = lambda: 0
    worker.run('event')
    assert len(calls) == 2
    assert be.sent[-1]['inputRevision'] == 2


def test_content_http_allowlist():
    from src.recommendation.content_worker import ContentBackend
    from src.recommendation.http import BackendClientConfig
    client = ContentBackend(BackendClientConfig('https://example.com'), access_token='scoped')
    with pytest.raises(ValueError):
        client.put('/api/admin/creators/1', {})
    with pytest.raises(ValueError):
        client.get(ROOT + '/inputs?cursor=../secret')


def test_backend_config_mismatch_stops_before_provider(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.model = 'unapproved-model'
    result = worker.run()
    assert result['stages']['tags']['status'] == 'failed'
    assert not calls and not be.sent
    assert not list(tmp_path.glob('budget-*.json'))
