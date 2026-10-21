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
        self.sent, self.acks, self.reconciles = [], [], []
        self.error = None

    @staticmethod
    def row(text, revision):
        return dict(creatorId=1, introduction=text, inputRevision=revision, inputHash=input_hash(text))

    def get(self, path):
        return dict(contractVersion='content-tags-v1', taxonomyApproved=True,
                    taxonomyHash=subtopic_hash(self.taxonomy), generationConfigHash=digest(fingerprint(self.taxonomy,self.model)), items=self.rows, nextCursor=None)

    def put(self, path, payload):
        if path.endswith('/reconcile'):
            self.reconciles.append(payload)
            return {'completedCount': len(self.rows)}
        if path.endswith('/claim'):
            return dict(jobId='1', leaseToken='lease', generation=1)
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
        canonical = json.dumps(row['normalized'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
        assert canonical == row['canonicalJson']
        import hashlib
        assert hashlib.sha256(canonical.encode('utf-8')).hexdigest() == row['recordInputHash']
    for invalid in fixture['invalidUnicode']:
        with pytest.raises(ValueError):
            input_hash(invalid)

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


def test_permanent_failure_isolated_from_claimed_creator_and_daily_reconcile(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.rows.append({**be.row('private refused text', 1), 'creatorId': 2})
    original = worker.classify_fn
    def model(cid, *args):
        if cid == '2':
            calls.append('refusal')
            raise ValueError('private refusal contents token=secret')
        return original(cid, *args)
    worker.classify_fn = model
    for _ in range(4):
        result = worker.run('event')
        assert result['failureCount'] == 1  # alert remains nonzero for the failing row
        assert be.acks[-1]['status'] == 'completed'
    result = worker.run('daily')
    assert result['failureCount'] == 1
    assert [r['creatorId'] for r in be.reconciles[-1]['inputs']] == [1]
    failed = next(r for r in result['stages']['tags']['rows'] if r['creatorId'] == 2)
    assert failed['code'] == 'FINAL_PROVIDER_FAILURE'
    assert (tmp_path / 'records' / failed['recordKey'] / 'attempts.json').exists()
    assert calls.count('refusal') == 3
    assert 'private' not in (tmp_path / 'summary.json').read_text()
    assert 'secret' not in (tmp_path / 'summary.json').read_text()


@pytest.mark.parametrize('status', [401, 403])
def test_authorization_stops_remaining_rows(tmp_path, taxonomy, status):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.rows.append({**be.row('second', 1), 'creatorId': 2})
    be.error = BackendRequestError('do not disclose', status=status)
    result = worker.run('event')
    assert len(calls) == len(be.sent) == 1
    assert be.acks[-1]['status'] == 'retry'
    assert result['stages']['tags']['rows'][0]['code'] == 'BACKEND_AUTHORIZATION_FAILED'
    assert not be.reconciles


def test_paging_and_duplicate_or_cyclic_cursor_rejected(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    get = be.get
    paths = []
    def page(path):
        paths.append(path)
        return {**get(path), 'items': [be.rows[0]] if len(paths) == 1 else [{**be.row('next', 1), 'creatorId': 2}],
                'nextCursor': 'page_2' if len(paths) == 1 else None}
    be.get = page
    assert [r['creatorId'] for r in worker.inputs()] == [1, 2]
    assert paths == [ROOT + '/inputs', ROOT + '/inputs?cursor=page_2']
    be.get = lambda path: {**get(path), 'items': [], 'nextCursor': 'repeat'}
    with pytest.raises(RuntimeError, match='INVALID_CURSOR'):
        worker.inputs()
    be.get = lambda path: {**get(path), 'nextCursor': 'next'}
    with pytest.raises(RuntimeError, match='INVALID_CREATOR_INPUT'):
        worker.inputs()


def test_summary_failure_codes_and_no_success_marker_on_failure(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    worker.approved = False
    result = worker.run()
    assert result['stages']['tags']['rows'][0]['code'] == 'EXTERNAL_TRANSFER_NOT_APPROVED'
    assert not (tmp_path / 'last-success.json').exists()
    worker.approved = True
    worker.max_calls = 0
    assert worker.run()['stages']['tags']['rows'][0]['code'] == 'DAILY_BUDGET_EXHAUSTED'
    be.model = 'wrong'
    assert worker.run()['stages']['tags']['code'] == 'CONFIGURATION_MISMATCH'
    assert not calls


def test_input_and_ack_conflicts_are_refresh_not_outage(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.error = BackendRequestError('stale secret', status=409)
    result = worker.run()
    assert result['failureCount'] == 0 and result['status'] == 'refresh_required'
    assert not be.reconciles
    be.error = None
    be.rows = [be.row('new revision', 2)]
    put = be.put
    def conflict_ack(path, payload):
        if path.endswith('/ack'):
            raise BackendRequestError('superseded lease', status=409)
        return put(path, payload)
    be.put = conflict_ack
    result = worker.run('event')
    assert result['failureCount'] == 0 and result['coordination']['status'] == 'refresh_required'
    assert json.loads((tmp_path / 'last-success.json').read_text())['source'] == 'event'


def test_idle_poll_updates_deadman_marker(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.put = lambda *args: {}
    assert worker.run('event')['status'] == 'idle'
    assert json.loads((tmp_path / 'last-success.json').read_text())['status'] == 'idle'
    assert not calls


@pytest.mark.parametrize('ack_fails', [False, True])
def test_real_heartbeat_thread_renews_and_stops_before_return(tmp_path, taxonomy, ack_fails):
    import threading
    worker, be, calls = setup_worker(tmp_path, taxonomy, heartbeat_interval=0.01)
    renewed = threading.Event()
    put = be.put
    renewals = []
    def backend(path, payload):
        if path.endswith('/renew'):
            renewals.append(payload.copy())
            renewed.set()
            if ack_fails:
                raise BackendRequestError('expired private', status=409)
            return {}
        return put(path, payload)
    be.put = backend
    worker.recommend = lambda: 0 if renewed.wait(2) else 1
    assert worker.run('event')['failureCount'] == 0
    assert renewals[0] == {'leaseToken': 'lease', 'generation': 1}
    assert worker._active_job['jobId'] == '1'
    # No sleeping process is required: renew is invoked from an actual background thread.
    assert not any(t.name.startswith('content-heartbeat') for t in threading.enumerate())


def test_policy_hash_ignores_comments_and_tracks_each_policy_function(monkeypatch):
    import src.recommendation.content_worker as module
    original = module.inspect.getsource
    baseline = module.policy_hash()
    monkeypatch.setattr(module.inspect, 'getsource', lambda f: '# unrelated comment\n' + original(f))
    assert module.policy_hash() == baseline
    for target in (module.classify, module._rank, module.validate_llm_topics):
        monkeypatch.setattr(module.inspect, 'getsource', lambda f: original(f) + ('\n    policy_changed = True\n' if f is target else ''))
        assert module.policy_hash() != baseline


@pytest.mark.parametrize('url', ['https://api.example.com', 'http://127.0.0.1:8080', 'http://[::1]:8080', 'http://localhost:8080'])
def test_secure_backend_accepts_https_and_loopback(url):
    from src.recommendation.content_worker import ContentBackend
    from src.recommendation.http import BackendClientConfig
    ContentBackend(BackendClientConfig(url), access_token='test')


@pytest.mark.parametrize('url', ['http://api.example.com', 'http://192.168.0.1', 'http://localhost.example.com', 'http://127.0.0.1.example.com'])
def test_plaintext_remote_backend_rejected_before_credentials_sent(url):
    from src.recommendation.content_worker import ContentBackend
    from src.recommendation.http import BackendClientConfig
    with pytest.raises(RuntimeError, match='HTTPS_REQUIRED'):
        ContentBackend(BackendClientConfig(url), access_token='secret')


@pytest.fixture
def cli(monkeypatch, tmp_path, taxonomy):
    from types import SimpleNamespace
    import openai
    import src.recommendation.content_worker as module
    import src.recommendation.daily_cli as daily
    monkeypatch.setenv('CKING_CONTENT_TAG_ACCESS_TOKEN', 'private-token')
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(module, 'load_taxonomy', lambda *args: taxonomy)
    monkeypatch.setattr(module, 'load_service_categories', lambda *args: [])
    args = ['--state', str(tmp_path), '--be-base-url', 'https://backend.example',
            '--approve-external-transfer', '--max-daily-calls', '10']
    return module, daily, args


@pytest.mark.parametrize('separator', [[], ['--']])
def test_main_remainder_and_pinned_operational_arguments(cli, monkeypatch, separator):
    module, daily, args = cli
    calls = []
    monkeypatch.setattr(daily, 'main', lambda argv: calls.append(argv) or 0)
    def run(self, source):
        self.recommend()
        return {'failureCount': 0}
    monkeypatch.setattr(module.ContentWorker, 'run', run)
    assert module.main(args + separator + ['apply', '--limit', '3']) == 0
    assert calls[0][:3] == ['apply', '--limit', '3']
    assert '--output-dir' in calls[0] and '--cache' in calls[0] and '--be-base-url' in calls[0]


def test_main_opt_in_runs_strict_ranking_with_same_state_and_dedicated_token(cli, monkeypatch):
    module, daily, args = cli
    import src.recommendation.saved_tags as saved
    import src.recommendation.interest_runtime as runtime
    calls=[]
    monkeypatch.setattr(daily, 'main', lambda argv: calls.append('personalized') or 0)
    monkeypatch.setattr(saved, 'SavedTagBackend', lambda config,token: ('dedicated',token))
    monkeypatch.setattr(runtime, 'create_interest_embedding_client', lambda provider: ('model',provider))
    def rank(backend,state,cache,model,version,apply):
        assert backend==('dedicated','private-token')
        assert state.name=='saved-tags' and cache.name=='models.json'
        assert model==('model','deepinfra') and apply
        calls.append('strict')
    monkeypatch.setattr(saved,'run',rank)
    def run(self,source):
        assert self.recommend()==0
        self.saved_tag_recommend()
        return {'failureCount':0}
    monkeypatch.setattr(module.ContentWorker,'run',run)
    assert module.main(args+['--saved-tag-recommendations','--','apply'])==0
    assert calls==['personalized','strict']


@pytest.mark.parametrize('option', ['--output-dir=x', '--cache=x', '--be-base-url=x', '--cache'])
def test_main_forbidden_arguments_fail_before_worker(cli, monkeypatch, capsys, option):
    module, daily, args = cli
    monkeypatch.setattr(module.ContentWorker, 'run', lambda *args: pytest.fail('worker must not start'))
    assert module.main(args + ['--', 'apply', option]) == 1
    assert 'FORBIDDEN_DAILY_ARGUMENTS' in capsys.readouterr().out


@pytest.mark.parametrize('budget', ['0', '-1'])
def test_main_nonpositive_budget(cli, capsys, budget):
    module, daily, args = cli
    args[-1] = budget
    assert module.main(args + ['--', 'apply']) == 1
    assert 'INVALID_BUDGET' in capsys.readouterr().out


def test_main_lock_conflict_maps_to_75(cli, monkeypatch):
    module, daily, args = cli
    def locked(*args):
        raise BatchAlreadyRunning('locked')
    monkeypatch.setattr(module.ContentWorker, 'run', locked)
    assert module.main(args + ['--', 'apply']) == 75


def test_main_requires_approval_and_apply(cli, capsys):
    module, daily, args = cli
    args.remove('--approve-external-transfer')
    assert module.main(args + ['--', 'apply']) == 1
    assert 'EXTERNAL_TRANSFER_NOT_APPROVED' in capsys.readouterr().out
    args.insert(0, '--approve-external-transfer')
    assert module.main(args + ['--', 'dry-run']) == 1
    assert 'INVALID_DAILY_ARGUMENTS' in capsys.readouterr().out


def test_main_external_exception_message_is_never_printed(cli, monkeypatch, capsys):
    module, daily, args = cli
    def fail(*args):
        raise RuntimeError('private profile and token=secret')
    monkeypatch.setattr(module.ContentWorker, 'run', fail)
    assert module.main(args + ['--', 'apply']) == 1
    output = capsys.readouterr().out
    assert 'WORKER_STARTUP_FAILED' in output
    assert 'private' not in output and 'secret' not in output


def test_main_coordination_diagnostic_keeps_nonzero_exit(cli, monkeypatch, capsys):
    module, daily, args = cli
    monkeypatch.setattr(module.ContentWorker, 'run', lambda *args: {'failureCount': 1, 'coordination': {'code': 'BACKEND_REQUEST_FAILED'}})
    assert module.main(args + ['--', 'apply']) == 1
    assert 'BACKEND_REQUEST_FAILED' in capsys.readouterr().out


def test_persistent_conflict_escalates_across_event_daily_and_freezes_success_marker(tmp_path, taxonomy):
    now = [1000.0]
    worker, be, calls = setup_worker(tmp_path, taxonomy, clock=lambda: now[0])
    be.error = BackendRequestError('private conflict payload', status=409)
    marker = None
    for i, source in enumerate(['event', 'daily'] * 3):
        result = worker.run(source)
        if i < 2:
            assert result['failureCount'] == 0
            marker = (tmp_path / 'last-success.json').read_bytes()
        else:
            assert result['failureCount'] == 1
            assert result['stages']['tags']['rows'][0]['code'] == 'PERSISTENT_INPUT_CONFLICT'
            assert (tmp_path / 'last-success.json').read_bytes() == marker
        now[0] += 1
    assert len(be.sent) == len(calls) == 1
    conflict = json.loads(next((tmp_path / 'records').glob('*/conflict.json')).read_text())
    assert conflict['observations'] == 6 and conflict['escalated'] is True
    assert 'private' not in (tmp_path / 'summary.json').read_text()
    be.error = None
    be.rows = [be.row('new introduction', 2)]
    assert worker.run()['failureCount'] == 0
    assert len(be.sent) == len(calls) == 2
    assert (tmp_path / 'last-success.json').read_bytes() != marker


def test_conflict_counter_survives_restart(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.error = BackendRequestError('conflict', status=409)
    assert worker.run()['failureCount'] == 0
    for expected in [0, 1]:
        restarted = ContentWorker(be, tmp_path, taxonomy, 'mock-model', worker.classify_fn, lambda: 0, approved=True, max_calls=10)
        assert restarted.run()['failureCount'] == expected
    assert len(be.sent) == len(calls) == 1


def test_conflict_age_escalates_without_observation_limit(tmp_path, taxonomy):
    now = [0.0]
    worker, be, calls = setup_worker(tmp_path, taxonomy, clock=lambda: now[0], max_conflict_observations=100)
    be.error = BackendRequestError('conflict', status=409)
    assert worker.run()['failureCount'] == 0
    now[0] = 299
    assert worker.run()['failureCount'] == 0
    now[0] = 300
    assert worker.run()['failureCount'] == 1
    assert len(be.sent) == 1


def test_legacy_conflict_flag_starts_observations_without_inventing_history(tmp_path, taxonomy):
    now = [0.0]
    worker, be, calls = setup_worker(tmp_path, taxonomy, clock=lambda: now[0])
    be.error = BackendRequestError('conflict', status=409)
    worker.run()
    path = next((tmp_path / 'records').glob('*/conflict.json'))
    path.write_text(json.dumps({'refreshRequired': True}))
    now[0] = 10000
    assert worker.run()['failureCount'] == 0
    assert json.loads(path.read_text())['firstObservedAt'] == 10000
    assert json.loads(path.read_text())['observations'] == 1


def test_persistent_conflict_does_not_block_other_creator_ack(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.rows.append({**be.row('healthy', 1), 'creatorId': 2})
    put = be.put
    def backend(path, payload):
        if path.endswith('/claim'):
            return dict(jobId='2', leaseToken='lease', generation=1)
        if path == ROOT + '/creators/1/automatic':
            be.sent.append(payload)
            raise BackendRequestError('conflict', status=409)
        return put(path, payload)
    be.put = backend
    for expected in [0, 0, 1]:
        assert worker.run('event')['failureCount'] == expected
        assert be.acks[-1]['status'] == 'completed'


def test_main_persistent_conflict_returns_one(cli, monkeypatch, capsys):
    module, daily, args = cli
    monkeypatch.setattr(module.ContentWorker, 'run', lambda *args: {'failureCount': 1, 'stages': {'tags': {'rows': [{'code': 'PERSISTENT_INPUT_CONFLICT'}]}}})
    assert module.main(args + ['--', 'apply']) == 1
    assert 'PERSISTENT_INPUT_CONFLICT' in capsys.readouterr().out


def test_operator_conflict_recovery_replays_saved_payload_without_model_call(tmp_path, taxonomy):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    be.error = BackendRequestError('conflict', status=409)
    for _ in range(3):
        result = worker.run()
    assert result['failureCount'] == 1
    payload = be.sent[0]
    flag = next((tmp_path / 'records').glob('*/conflict.json'))
    # Operator stops workers, fixes/verifies BE, and archives only the conflict flag.
    be.error = None
    flag.rename(flag.with_name('conflict.resolved.json'))
    assert worker.run()['failureCount'] == 0
    assert be.sent == [payload, payload]
    assert len(calls) == 1
    assert flag.with_name('conflict.resolved.json').exists()
    assert flag.with_name('receipt.json').exists()


@pytest.mark.parametrize('source', ['daily', 'event'])
@pytest.mark.parametrize('error', [BackendRequestError('private token', status=404), RuntimeError('private token')])
def test_saved_tag_failure_preserves_personalized_coordination(tmp_path, taxonomy, source, error):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    def fail():
        print('private token')
        raise error
    worker.saved_tag_recommend = fail
    result = worker.run(source)
    assert result['failureCount'] == 1 and result['status'] == 'partial'
    assert result['stages']['recommendations']['status'] == 'success'
    assert result['stages']['savedTags']['status'] == 'failed'
    assert result['stages']['savedTags']['code'] == (
        'BACKEND_REQUEST_FAILED' if isinstance(error, BackendRequestError) else 'SAVED_TAG_RECOMMENDATION_FAILED')
    if source == 'event':
        assert be.acks[-1]['status'] == 'completed'
    else:
        assert len(be.reconciles) == 1
    assert not (tmp_path / 'last-success.json').exists()
    assert 'private' not in (tmp_path / 'summary.json').read_text()
    worker.saved_tag_recommend = lambda: None
    assert worker.run(source)['failureCount'] == 0


@pytest.mark.parametrize('provider,override', [('local', None), ('deepinfra', 'custom-v2')])
def test_main_saved_tags_uses_daily_embedding_settings(cli, monkeypatch, provider, override):
    module, daily, args = cli
    import src.recommendation.saved_tags as saved
    import src.recommendation.interest_runtime as runtime
    monkeypatch.setattr(daily, 'main', lambda argv: 0)
    monkeypatch.setattr(saved, 'SavedTagBackend', lambda *args: None)
    monkeypatch.setattr(runtime, 'create_interest_embedding_client', lambda value: value)
    received = []
    monkeypatch.setattr(saved, 'run', lambda backend, state, cache, model, version, **kw: received.append((model, version)))
    def run(self, source):
        self.saved_tag_recommend()
        return {'failureCount': 0}
    monkeypatch.setattr(module.ContentWorker, 'run', run)
    options = ['--embedding-provider', provider]
    if override:
        options += ['--embedding-model-version', override]
    assert module.main(args + ['--saved-tag-recommendations', '--', 'apply'] + options) == 0
    assert received == [(provider, override or f'BAAI/bge-m3@{provider}-v1')]


@pytest.mark.parametrize('source', ['daily', 'event'])
def test_saved_tag_success_does_not_mask_personalized_failure(tmp_path, taxonomy, source):
    worker, be, calls = setup_worker(tmp_path, taxonomy)
    worker.recommend = lambda: 1
    worker.saved_tag_recommend = lambda: None
    result = worker.run(source)
    assert result['failureCount'] == 1
    assert result['stages']['savedTags']['status'] == 'success'
    assert result['stages']['recommendations']['status'] == 'failed'
    assert not be.reconciles
    if source == 'event':
        assert be.acks[-1]['status'] == 'retry'
