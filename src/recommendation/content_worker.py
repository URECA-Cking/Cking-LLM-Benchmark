"""Durable v2 delivery worker for the proposed BE #654 contract (v1).

BE owns transactional outbox, latest-input coalescing, leases and stale-write fencing.
All entrypoints share one persistent worker directory; daily_cli retains its own locks.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
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
                                build_prompt, candidate_schema, classify, classify_with_model, digest)
from src.detail_tagging_batch import write_new
from src.recommendation.http import BackendClientConfig, BackendRequestError, CkingBackendClient
from src.recommendation.locking import file_lock, BatchAlreadyRunning
from src.recommendation.storage import _write_json_atomic
from src.recommendation.subtopic_taxonomy import subtopic_hash
from src.subtopic_tags import load_taxonomy, validate_llm_topics
from src.recommendation.taxonomy import load_service_categories, DEFAULT_CATEGORIES_CSV

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
    def get(self, path):
        if path != ROOT + '/inputs' and not re.fullmatch(ROOT + r'/inputs\?cursor=[A-Za-z0-9_-]+', path):
            raise ValueError('Unsupported content input path')
        return self._request('GET', path, None, authenticated=True)

    def put(self, path, payload):
        if not re.fullmatch(ROOT + r'/(jobs/(claim|reconcile|[A-Za-z0-9_-]+/(ack|renew|retry))|creators/[1-9][0-9]*/automatic)', path):
            raise ValueError('Unsupported content write path')
        return self._request('PUT', path, payload, authenticated=True)


def fingerprint(taxonomy, model):
    return {'contractVersion': 'content-tags-v1', 'modelVersion': model,
            'promptVersion': PROMPT_VERSION, 'promptHash': digest(build_prompt(taxonomy)),
            'schemaHash': digest(candidate_schema(taxonomy)), 'policyVersion': POLICY_VERSION,
            'policyHash': digest([Path(inspect.getfile(f)).read_text(encoding='utf-8')
                                  for f in (classify, validate_llm_topics)]),
            'taxonomyVersion': taxonomy['version'], 'taxonomyHash': subtopic_hash(taxonomy),
            'criteriaHash': digest(taxonomy)}


class ContentWorker:
    def __init__(self, backend, directory, taxonomy, model, classify_fn, recommend,
                 *, approved=False, max_calls=0, max_attempts=3, clock=time.time):
        if type(max_calls) is not int or max_calls < 0 or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('Invalid budget')
        self.backend, self.directory = backend, Path(directory)
        self.taxonomy, self.model = taxonomy, model
        self.classify_fn, self.recommend = classify_fn, recommend
        self.approved, self.max_calls, self.max_attempts = approved, max_calls, max_attempts
        self.clock = clock
        self.identity = fingerprint(taxonomy, model)

    def inputs(self):
        rows, seen, cursor = [], set(), None
        cursors = set()
        while True:
            page = self.backend.get(ROOT + '/inputs' + (f'?cursor={cursor}' if cursor else ''))
            if page.get('contractVersion') != 'content-tags-v1':
                raise ValueError('BE contract is not enabled')
            if (page.get('taxonomyHash') != self.identity['taxonomyHash']
                    or page.get('taxonomyApproved') is not True):
                raise ValueError('BE taxonomy approval/hash mismatch')
            if page.get('generationConfigHash') != digest(self.identity):
                raise ValueError('BE generation configuration mismatch')
            for row in page['items']:
                cid = row['creatorId']
                if type(cid) is not int or cid < 1 or cid in seen:
                    raise ValueError('Invalid/duplicate creator')
                if type(row['inputRevision']) is not int or row['inputRevision'] < 1:
                    raise ValueError('Invalid input revision')
                if input_hash(row['introduction']) != row['inputHash']:
                    raise ValueError('Introduction hash mismatch')
                seen.add(cid)
                rows.append(row)
            next_cursor = page.get('nextCursor')
            if next_cursor is None:
                return rows
            if not isinstance(next_cursor, str) or not re.fullmatch('[A-Za-z0-9_-]+', next_cursor) or next_cursor in cursors:
                raise ValueError('Invalid cursor')
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
                    raise RuntimeError('External transfer approval required')
                attempts = folder / 'attempts.json'
                count = json.loads(attempts.read_text())['count'] if attempts.exists() else 0
                if count >= self.max_attempts:
                    raise RuntimeError('Final provider failure; manual retry required')
                budget_path = self.directory / ('budget-' + datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat() + '.json')
                budget = json.loads(budget_path.read_text())['calls'] if budget_path.exists() else 0
                if budget >= self.max_calls:
                    raise RuntimeError('Daily call budget exhausted')
                # Reserve before invocation: crash must not reset paid-call budget.
                _write_json_atomic(budget_path, {'calls': budget + 1})
                _write_json_atomic(attempts, {'count': count + 1})
                record = self.classify_fn(str(row['creatorId']), text, self.taxonomy, self.model)
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
            raise ValueError('Invalid delivery receipt')
        _write_json_atomic(receipt, response)
        return response['status']

    def run(self, source='daily'):
        with file_lock(self.directory / '.worker.lock'):
            stopped = threading.Event()
            self._active_job = None
            def heartbeat():
                while not stopped.wait(60):
                    job = self._active_job
                    if job:
                        try:
                            self.backend.put(f'{ROOT}/jobs/{job["jobId"]}/renew', {
                                'leaseToken': job['leaseToken'], 'generation': job['generation']})
                        except Exception:
                            # Expired/superseded generation cannot be acknowledged; BE retains it.
                            pass
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                return self._run(source)
            finally:
                stopped.set()
                thread.join(timeout=15)

    def _run(self, source):
        started = self.clock()
        # Event claim stays pending in BE until both independent stages succeed.
        job = self.backend.put(ROOT + '/jobs/claim', {'leaseSeconds': 900}) if source == 'event' else None
        if source == 'event' and not job.get('jobId'):
            return {'status': 'idle', 'failureCount': 0}
        if job and (not re.fullmatch('[A-Za-z0-9_-]+', str(job.get('jobId', '')))
                    or not isinstance(job.get('leaseToken'), str)):
            raise ValueError('Invalid lease')
        self._active_job = job
        stages = {}
        try:
            rows = self.inputs()
            failures, refresh = 0, False
            for row in rows:
                try:
                    folder = self.directory / 'records' / digest([self.backend.target_identity, row['creatorId'],
                        row['inputRevision'], row['inputHash'], self.identity])
                    if (folder / 'conflict.json').exists():
                        raise RuntimeError('Fenced input requires new revision')
                    self.deliver(row)
                except Exception as error:
                    failures += 1
                    refresh |= isinstance(error, BackendRequestError) and error.status == 409
                    if isinstance(error, BackendRequestError) and error.status in {401, 403}:
                        break
            stages['tags'] = {'status': 'failed' if failures else 'success',
                              'failureCount': failures, 'refreshRequired': refresh}
        except Exception as error:
            stages['tags'] = {'status': 'failed', 'failureCount': 1, 'errorType': type(error).__name__}
        try:
            with redirect_stdout(io.StringIO()):
                code = self.recommend()
            stages['recommendations'] = {'status': 'success' if code == 0 else 'failed', 'exitCode': code}
        except Exception as error:
            stages['recommendations'] = {'status': 'failed', 'errorType': type(error).__name__}
        failed = sum(s['status'] != 'success' for s in stages.values())
        summary = {'status': 'partial' if failed else 'completed', 'failureCount': failed,
                   'source': source, 'startedAt': started, 'completedAt': self.clock(), 'stages': stages}
        try:
            if not failed and source != 'event':
                self.backend.put(ROOT + '/jobs/reconcile', {'inputs': [
                    {key: row[key] for key in ('creatorId', 'inputRevision', 'inputHash')} for row in rows]})
            if job:
                # BE atomically acks only claimed generation; newer edits stay queued.
                self.backend.put(f'{ROOT}/jobs/{job["jobId"]}/ack', {
                    'leaseToken': job['leaseToken'], 'status': 'retry' if failed else 'completed',
                    'generation': job['generation'], 'summary': summary})
        except Exception as error:
            summary.update(status='partial', failureCount=failed + 1,
                           coordination={'status': 'failed', 'errorType': type(error).__name__})
        _write_json_atomic(self.directory / 'summary.json', summary)
        return summary


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
            raise ValueError('Recommendation and tagging external transfer must be approved')
        if args.max_daily_calls <= 0:
            raise ValueError('Positive tagging budget required')
        daily_args = args.daily_args[1:] if args.daily_args[:1] == ['--'] else args.daily_args
        if daily_args[:1] != ['apply']:
            raise ValueError('Worker requires daily apply arguments')
        # Pin all operational state beneath a single persistent root.
        if any(v.split('=')[0] in {'--output-dir', '--cache', '--be-base-url'} for v in daily_args):
            raise ValueError('Worker controls shared output, cache and backend')
        client = OpenAI(timeout=60, max_retries=0)
        worker = ContentWorker(backend, args.state, taxonomy, args.model,
            lambda cid, text, tax, model: classify_with_model(client, cid, text, tax, model),
            lambda: daily(daily_args + ['--output-dir', str(args.state / 'recommendations'),
                '--cache', str(args.state / 'models.json'), '--be-base-url', args.be_base_url]),
            approved=True, max_calls=args.max_daily_calls)
        result = worker.run(args.source)
        if result['failureCount']:
            print('content-worker: partial failure; inspect protected summary.json')
        return 1 if result['failureCount'] else 0
    except BatchAlreadyRunning:
        # No claim occurred; durable BE job remains available.
        return 75
    except Exception as error:
        print('content-worker: failed (' + type(error).__name__ + ')')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
