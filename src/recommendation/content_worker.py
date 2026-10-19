"""Durable v2 delivery worker for the BE #654 contract (v1).

BE owns transactional outbox, latest-input coalescing, leases and stale-write fencing.
All entrypoints share one persistent worker directory; daily_cli retains its own locks.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import ipaddress
from urllib.parse import urlsplit
import io
from contextlib import redirect_stdout
import json
import os
import re
import time
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from src.detail_tagging import (PROMPT_VERSION, POLICY_VERSION, automatic_record,
                                build_prompt, candidate_schema, classify, _rank, classify_with_model, digest)
from src.detail_tagging_batch import write_new
from src.recommendation.http import BackendClientConfig, BackendRequestError, CkingBackendClient
from src.recommendation.locking import file_lock, BatchAlreadyRunning
from src.recommendation.storage import _write_json_atomic
from src.recommendation.subtopic_taxonomy import subtopic_hash
from src.subtopic_tags import load_taxonomy, validate_llm_topics
from src.recommendation.taxonomy import load_service_categories, DEFAULT_CATEGORIES_CSV

from src.recommendation.policy_fingerprint import normalized_policy_tokens

ROOT = '/api/internal/content-tags'


def normalized_intro(value):
    if value is None:
        value = ''
    if not isinstance(value, str):
        raise ValueError('Invalid introduction')
    # No trim or case folding. Java fixture must implement the same operations.
    return unicodedata.normalize('NFC', value.replace('\r\n', '\n').replace('\r', '\n'))


def input_hash(value):
    return hashlib.sha256(normalized_intro(value).encode('utf-8')).hexdigest()


class ContentBackend(CkingBackendClient):
    """Dedicated credential and exact path allowlist; no admin-key expansion."""
    def __init__(self, config, *args, **kwargs):
        require_secure_backend(config.base_url)
        super().__init__(config, *args, **kwargs)

    def get(self, path):
        if path != ROOT + '/inputs' and not re.fullmatch(ROOT + r'/inputs\?cursor=[A-Za-z0-9_-]+', path):
            raise ValueError('Unsupported content input path')
        return self._request('GET', path, None, authenticated=True)

    def put(self, path, payload):
        if not re.fullmatch(ROOT + r'/(jobs/(claim|reconcile|[A-Za-z0-9_-]+/(ack|renew|retry))|creators/[1-9][0-9]*/automatic)', path):
            raise ValueError('Unsupported content write path')
        return self._request('PUT', path, payload, authenticated=True)


class WorkerFailure(RuntimeError):
    """Only application-owned constant codes may enter summaries or stdout."""
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def diagnostic(error, fallback):
    if isinstance(error, WorkerFailure):
        return error.code
    if isinstance(error, BackendRequestError):
        if error.status == 409:
            return 'INPUT_REFRESH_REQUIRED'
        if error.status in {401, 403}:
            return 'BACKEND_AUTHORIZATION_FAILED'
        return 'BACKEND_REQUEST_FAILED'
    return fallback


def policy_hash():
    functions = [normalized_policy_tokens(inspect.getsource(function))
                 for function in (classify, _rank, validate_llm_topics)]
    return digest({'format': 'policy-tokens-v1', 'version': POLICY_VERSION, 'functions': functions})


def require_secure_backend(url):
    parsed = urlsplit(url)
    if parsed.scheme == 'https':
        return
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == 'localhost'
    if parsed.scheme != 'http' or not loopback:
        raise WorkerFailure('HTTPS_REQUIRED')


def fingerprint(taxonomy, model):
    return {'contractVersion': 'content-tags-v1', 'modelVersion': model,
            'promptVersion': PROMPT_VERSION, 'promptHash': digest(build_prompt(taxonomy)),
            'schemaHash': digest(candidate_schema(taxonomy)), 'policyVersion': POLICY_VERSION,
            'policyHash': policy_hash(),
            'taxonomyVersion': taxonomy['version'], 'taxonomyHash': subtopic_hash(taxonomy),
            'criteriaHash': digest(taxonomy)}


class ContentWorker:
    def __init__(self, backend, directory, taxonomy, model, classify_fn, recommend,
                 *, approved=False, max_calls=0, max_attempts=3, clock=time.time, heartbeat_interval=60):
        if type(max_calls) is not int or max_calls < 0 or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('Invalid budget')
        self.backend, self.directory = backend, Path(directory)
        self.taxonomy, self.model = taxonomy, model
        self.classify_fn, self.recommend = classify_fn, recommend
        self.approved, self.max_calls, self.max_attempts = approved, max_calls, max_attempts
        if heartbeat_interval <= 0:
            raise ValueError('Invalid heartbeat interval')
        self.heartbeat_interval = heartbeat_interval
        self.clock = clock
        self.identity = fingerprint(taxonomy, model)

    def inputs(self):
        rows, seen, cursor = [], set(), None
        cursors = set()
        while True:
            page = self.backend.get(ROOT + '/inputs' + (f'?cursor={cursor}' if cursor else ''))
            if page.get('contractVersion') != 'content-tags-v1':
                raise WorkerFailure('BACKEND_CONTRACT_MISMATCH')
            if (page.get('taxonomyHash') != self.identity['taxonomyHash']
                    or page.get('taxonomyApproved') is not True):
                raise WorkerFailure('TAXONOMY_NOT_APPROVED')
            if page.get('generationConfigHash') != digest(self.identity):
                raise WorkerFailure('CONFIGURATION_MISMATCH')
            for row in page['items']:
                cid = row['creatorId']
                if type(cid) is not int or cid < 1 or cid in seen:
                    raise WorkerFailure('INVALID_CREATOR_INPUT')
                if type(row['inputRevision']) is not int or row['inputRevision'] < 1:
                    raise WorkerFailure('INVALID_INPUT_REVISION')
                if input_hash(row['introduction']) != row['inputHash']:
                    raise WorkerFailure('INPUT_HASH_MISMATCH')
                seen.add(cid)
                rows.append(row)
            next_cursor = page.get('nextCursor')
            if next_cursor is None:
                return rows
            if not isinstance(next_cursor, str) or not re.fullmatch('[A-Za-z0-9_-]+', next_cursor) or next_cursor in cursors:
                raise WorkerFailure('INVALID_CURSOR')
            cursors.add(next_cursor)
            cursor = next_cursor

    def deliver(self, row):
        text = normalized_intro(row['introduction'])
        key = digest([self.backend.target_identity, row['creatorId'], row['inputRevision'],
                      row['inputHash'], self.identity])
        folder = self.directory / 'records' / key
        result_path = folder / 'payload.json'
        if not result_path.exists():
            if text.strip():
                if not self.approved:
                    raise WorkerFailure('EXTERNAL_TRANSFER_NOT_APPROVED')
                attempts = folder / 'attempts.json'
                count = json.loads(attempts.read_text())['count'] if attempts.exists() else 0
                if count >= self.max_attempts:
                    raise WorkerFailure('FINAL_PROVIDER_FAILURE')
                budget_path = self.directory / ('budget-' + datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat() + '.json')
                budget = json.loads(budget_path.read_text())['calls'] if budget_path.exists() else 0
                if budget >= self.max_calls:
                    raise WorkerFailure('DAILY_BUDGET_EXHAUSTED')
                # Reserve before invocation: crash must not reset paid-call budget.
                _write_json_atomic(budget_path, {'calls': budget + 1})
                _write_json_atomic(attempts, {'count': count + 1})
                try:
                    record = self.classify_fn(str(row['creatorId']), text, self.taxonomy, self.model)
                except Exception:
                    raise WorkerFailure('PROVIDER_FAILURE') from None
            else:
                record = automatic_record(str(row['creatorId']), text, {'parents': [], 'topics': []},
                                          self.taxonomy, self.model)
            payload = {'creatorId': row['creatorId'], 'inputHash': row['inputHash'],
                       'inputRevision': row['inputRevision'], 'automaticRevision': key,
                       'idempotencyKey': key, 'provenance': self.identity,
                       'createdAt': datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
                       'origin': 'automatic', 'creatorConfirmed': False, 'record': record}
            write_new(result_path, payload)
        payload = json.loads(result_path.read_text(encoding='utf-8'))
        receipt = folder / 'receipt.json'
        if receipt.exists():
            return 'reused'
        try:
            response = self.backend.put(f'{ROOT}/creators/{row["creatorId"]}/automatic', payload)
        except BackendRequestError as error:
            if error.status == 409:
                # Never replay a fenced payload; next cycle recollects latest input.
                _write_json_atomic(folder / 'conflict.json', {'refreshRequired': True})
            raise
        if response.get('automaticRevision') != key or response.get('status') not in {'applied', 'idempotent'}:
            raise WorkerFailure('INVALID_DELIVERY_RECEIPT')
        _write_json_atomic(receipt, response)
        return response['status']

    def run(self, source='daily'):
        with file_lock(self.directory / '.worker.lock'):
            stopped = threading.Event()
            self._active_job = None
            def heartbeat():
                while not stopped.wait(self.heartbeat_interval):
                    job = self._active_job
                    if job:
                        try:
                            self.backend.put(f'{ROOT}/jobs/{job["jobId"]}/renew', {
                                'leaseToken': job['leaseToken'], 'generation': job['generation']})
                        except Exception:
                            # Expired/superseded generation cannot be acknowledged; BE retains it.
                            pass
            thread = threading.Thread(target=heartbeat, daemon=True, name='content-heartbeat')
            thread.start()
            try:
                return self._run(source)
            finally:
                stopped.set()
                thread.join(timeout=15)

    def _run(self, source):
        started = self.clock()
        job = self.backend.put(ROOT + '/jobs/claim', {'leaseSeconds': 900}) if source == 'event' else None
        if source == 'event' and not job.get('jobId'):
            result = {'status': 'idle', 'failureCount': 0, 'source': source,
                      'startedAt': started, 'completedAt': self.clock()}
            self._save_summary(result)
            return result
        if job and (not re.fullmatch('[1-9][0-9]*', str(job.get('jobId', '')))
                    or not isinstance(job.get('leaseToken'), str)
                    or type(job.get('generation')) is not int or job['generation'] < 1):
            raise WorkerFailure('INVALID_LEASE')
        self._active_job = job
        rows, outcomes, stages = [], {}, {}
        tag_started = self.clock()
        try:
            rows = self.inputs()
            for row in rows:
                key = digest([self.backend.target_identity, row['creatorId'], row['inputRevision'], row['inputHash'], self.identity])
                outcome = {'creatorId': row['creatorId'], 'recordKey': key, 'status': 'success'}
                try:
                    if (self.directory / 'records' / key / 'conflict.json').exists():
                        raise WorkerFailure('INPUT_REFRESH_REQUIRED')
                    self.deliver(row)
                except Exception as error:
                    code = diagnostic(error, 'TAG_EXECUTION_FAILED')
                    outcome.update(status='refresh_required' if code == 'INPUT_REFRESH_REQUIRED' else 'failed', code=code)
                    outcomes[row['creatorId']] = outcome
                    if isinstance(error, BackendRequestError) and error.status in {401, 403}:
                        break
                outcomes[row['creatorId']] = outcome
            failures = sum(r['status'] == 'failed' for r in outcomes.values())
            refresh = any(r['status'] == 'refresh_required' for r in outcomes.values())
            stages['tags'] = {'status': 'failed' if failures else 'success',
                              'failureCount': failures, 'refreshRequired': refresh,
                              'rows': list(outcomes.values())}
        except Exception as error:
            stages['tags'] = {'status': 'failed', 'failureCount': 1,
                              'code': diagnostic(error, 'INPUT_COLLECTION_FAILED'), 'rows': []}
        stages['tags']['durationSeconds'] = self.clock() - tag_started
        rec_started = self.clock()
        try:
            with redirect_stdout(io.StringIO()):
                code = self.recommend()
            stages['recommendations'] = {'status': 'success' if code == 0 else 'failed', 'exitCode': code}
            if code:
                stages['recommendations']['code'] = 'RECOMMENDATION_FAILED'
        except Exception:
            stages['recommendations'] = {'status': 'failed', 'code': 'RECOMMENDATION_FAILED'}
        stages['recommendations']['durationSeconds'] = self.clock() - rec_started
        failed = sum(s['status'] != 'success' for s in stages.values())
        refresh = stages['tags'].get('refreshRequired', False)
        summary = {'status': 'partial' if failed else 'refresh_required' if refresh else 'completed',
                   'failureCount': failed, 'source': source, 'startedAt': started,
                   'completedAt': self.clock(), 'stages': stages}
        try:
            rec_ok = stages['recommendations']['status'] == 'success'
            if rec_ok and source != 'event':
                successful = sorted((row for row in rows if outcomes.get(row['creatorId'], {}).get('status') == 'success'), key=lambda row: row['creatorId'])
                if successful:
                    self.backend.put(ROOT + '/jobs/reconcile', {'inputs': [
                        {key: row[key] for key in ('creatorId', 'inputRevision', 'inputHash')} for row in successful]})
            if job:
                # An unrelated permanent failure must not consume this Creator's retry budget.
                claimed_ok = outcomes.get(int(job['jobId']), {}).get('status') == 'success'
                current = any(row['creatorId'] == int(job['jobId']) and row['inputRevision'] == job['generation'] for row in rows)
                self.backend.put(f'{ROOT}/jobs/{job["jobId"]}/ack', {
                    'leaseToken': job['leaseToken'], 'status': 'completed' if claimed_ok and current and rec_ok else 'retry',
                    'generation': job['generation'], 'summary': summary})
        except Exception as error:
            code = diagnostic(error, 'COORDINATION_FAILED')
            if code == 'INPUT_REFRESH_REQUIRED':
                summary['coordination'] = {'status': 'refresh_required', 'code': code}
                if not failed:
                    summary['status'] = 'refresh_required'
            else:
                summary.update(status='partial', failureCount=failed + 1,
                               coordination={'status': 'failed', 'code': code})
        summary['completedAt'] = self.clock()
        self._save_summary(summary)
        return summary

    def _save_summary(self, summary):
        _write_json_atomic(self.directory / 'summary.json', summary)
        if not summary['failureCount']:
            # Includes successful idle polls, so operators can distinguish a stopped timer.
            _write_json_atomic(self.directory / 'last-success.json', {
                'completedAt': summary['completedAt'], 'source': summary['source'], 'status': summary['status']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--be-base-url', required=True)
    parser.add_argument('--taxonomy', type=Path, default=Path('data/creator-subtopics.json'))
    parser.add_argument('--model', default='gpt-5.4-nano-2026-03-17')
    parser.add_argument('--source', choices=['daily', 'event', 'manual'], default='daily')
    parser.add_argument('--approve-external-transfer', action='store_true')
    parser.add_argument('--max-daily-calls', type=int, default=0)
    parser.add_argument('daily_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        from openai import OpenAI
        from src.recommendation.daily_cli import main as daily
        token = os.environ['CKING_CONTENT_TAG_ACCESS_TOKEN']
        backend = ContentBackend(BackendClientConfig(args.be_base_url), access_token=token)
        taxonomy = load_taxonomy(args.taxonomy, [c.code for c in load_service_categories(DEFAULT_CATEGORIES_CSV)])
        if not args.approve_external_transfer:
            raise WorkerFailure('EXTERNAL_TRANSFER_NOT_APPROVED')
        if args.max_daily_calls <= 0:
            raise WorkerFailure('INVALID_BUDGET')
        daily_args = args.daily_args[1:] if args.daily_args[:1] == ['--'] else args.daily_args
        if daily_args[:1] != ['apply']:
            raise WorkerFailure('INVALID_DAILY_ARGUMENTS')
        # Pin all operational state beneath a single persistent root.
        if any(v.split('=')[0] in {'--output-dir', '--cache', '--be-base-url'} for v in daily_args):
            raise WorkerFailure('FORBIDDEN_DAILY_ARGUMENTS')
        client = OpenAI(timeout=60, max_retries=0)
        worker = ContentWorker(backend, args.state, taxonomy, args.model,
            lambda cid, text, tax, model: classify_with_model(client, cid, text, tax, model),
            lambda: daily(daily_args + ['--output-dir', str(args.state / 'recommendations'),
                '--cache', str(args.state / 'models.json'), '--be-base-url', args.be_base_url]),
            approved=True, max_calls=args.max_daily_calls)
        result = worker.run(args.source)
        if result['failureCount']:
            codes = {row.get('code') for stage in result.get('stages', {}).values() for row in stage.get('rows', []) if row.get('code')}
            codes.update(stage['code'] for stage in result.get('stages', {}).values() if 'code' in stage)
            if result.get('coordination', {}).get('code'):
                codes.add(result['coordination']['code'])
            codes = codes or {'WORKER_PARTIAL_FAILURE'}
            print('content-worker: partial failure (' + ','.join(sorted(codes)) + '); inspect protected summary.json')
        return 1 if result['failureCount'] else 0
    except BatchAlreadyRunning:
        # No claim occurred; durable BE job remains available.
        return 75
    except Exception as error:
        code = diagnostic(error, 'WORKER_STARTUP_FAILED')
        print('content-worker: failed (' + code + ')')
        _write_json_atomic(args.state / 'summary.json', {'status': 'failed', 'failureCount': 1,
            'source': args.source, 'completedAt': time.time(), 'code': code})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
