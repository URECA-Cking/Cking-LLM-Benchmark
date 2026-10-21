"""BE가 발급한 공개 확인 태그 snapshot만 사용한다. zero-shot으로 태그를 보충하지 않는다."""
from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
from src.detail_tagging import digest
from src.recommendation.cache import JsonModelCache, text_hash
from src.recommendation.embeddings import validated_embeddings, EMBEDDING_CONTRACT_VERSION
from src.recommendation.http import BackendClientConfig, CkingBackendClient, BackendRequestError
from src.recommendation.content_worker import require_secure_backend, normalized_intro
from src.recommendation.locking import batch_locks
from src.recommendation.storage import _write_json_atomic

METHOD = 'SAVED_TAG_M2_V1'
ROOT = '/api/internal/content-tags/saved-recommendations'
MAX_SEQUENCE = 2**63 - 1


class SavedTagBackend(CkingBackendClient):
    def __init__(self, config, token, **kwargs):
        require_secure_backend(config.base_url)
        if not token or not token.strip():
            raise ValueError('CONTENT_TAG_TOKEN_REQUIRED')
        super().__init__(config, access_token=token, **kwargs)

    def get(self, path):
        if path != ROOT + '/inputs':
            raise ValueError('UNSUPPORTED_SAVED_TAG_PATH')
        return self._request('GET', path, None, authenticated=True)

    def put(self, path, payload):
        if path != ROOT + '/rankings':
            raise ValueError('UNSUPPORTED_SAVED_TAG_PATH')
        return self._request('PUT', path, payload, authenticated=True)


def positive(value):
    if type(value) is not int or not 0 < value <= MAX_SEQUENCE:
        raise ValueError('INVALID_POSITIVE_INTEGER')
    return value


def validate_snapshot(snapshot):
    required = {'contractVersion', 'taxonomyVersion', 'taxonomyHash', 'generationConfigHash', 'tags', 'items',
                'snapshotHash', 'lastApplicationSequence'}
    if not isinstance(snapshot, dict) or set(snapshot) != required or snapshot['contractVersion'] != 'saved-tags-v1':
        raise ValueError('INVALID_SAVED_TAG_SNAPSHOT')
    for field in ('taxonomyHash', 'generationConfigHash', 'snapshotHash'):
        if not isinstance(snapshot[field], str) or not re.fullmatch('[0-9a-f]{64}', snapshot[field]):
            raise ValueError('INVALID_SNAPSHOT_HASH')
    if not isinstance(snapshot['taxonomyVersion'], str) or not snapshot['taxonomyVersion'].strip():
        raise ValueError('INVALID_TAXONOMY_VERSION')
    sequence = snapshot['lastApplicationSequence']
    if type(sequence) is not int or not 0 <= sequence < MAX_SEQUENCE:
        raise ValueError('INVALID_APPLICATION_SEQUENCE')
    identity = {k: v for k, v in snapshot.items() if k not in {'snapshotHash', 'lastApplicationSequence'}}
    if digest(identity) != snapshot['snapshotHash']:
        raise ValueError('SNAPSHOT_HASH_MISMATCH')
    if not isinstance(snapshot['tags'], list) or not isinstance(snapshot['items'], list):
        raise ValueError('INVALID_SNAPSHOT_ARRAY')
    catalog = {}
    for tag in snapshot['tags']:
        if (not isinstance(tag, dict) or set(tag) != {'type', 'code', 'name', 'description'}
                or tag['type'] not in {'PARENT', 'TOPIC'}
                or any(not isinstance(tag[k], str) or not tag[k].strip() for k in ('code', 'name', 'description'))
                or not re.fullmatch('[A-Z][A-Z0-9_]*', tag['code'])):
            raise ValueError('INVALID_TAG_DEFINITION')
        key = (tag['type'], tag['code'])
        if key in catalog:
            raise ValueError('DUPLICATE_TAG')
        catalog[key] = tag
    previous = 0
    for item in snapshot['items']:
        if not isinstance(item, dict) or set(item) != {'creatorId', 'introduction', 'inputRevision', 'confirmationId', 'parents', 'topics'}:
            raise ValueError('INVALID_CONFIRMED_CANDIDATE')
        cid = positive(item['creatorId'])
        positive(item['inputRevision']); positive(item['confirmationId'])
        if cid <= previous or not isinstance(item['introduction'], str) or not item['introduction'].strip():
            raise ValueError('INVALID_CANDIDATE_ORDER_OR_INTRO')
        if normalized_intro(item['introduction']) != item['introduction']:
            raise ValueError('NON_CANONICAL_INTRODUCTION')
        previous = cid
        for field, kind, limit in [('parents', 'PARENT', 3), ('topics', 'TOPIC', 9)]:
            codes = item[field]
            if (not isinstance(codes, list) or any(not isinstance(c, str) for c in codes)
                    or codes != sorted(set(codes)) or len(codes) > limit
                    or any((kind, c) not in catalog for c in codes)):
                raise ValueError('INVALID_CONFIRMED_TAGS')
        if not item['parents'] and not item['topics']:
            raise ValueError('EMPTY_CONFIRMED_TAGS')
    return catalog


def generate(snapshot, embedder, cache, model_version):
    catalog = validate_snapshot(snapshot)
    if not isinstance(model_version, str) or not model_version.strip() or len(model_version) > 255:
        raise ValueError('INVALID_MODEL_VERSION')
    # 전체 적격 후보를 먼저 선택한다. 어떤 태그에도 속하지 않은 소개는 임베딩하지 않는다.
    pairs = []
    texts = []
    for item in snapshot['items']:
        for field, kind in [('parents', 'PARENT'), ('topics', 'TOPIC')]:
            for code in item[field]:
                tag = catalog[kind, code]
                pairs.append((tag, item))
                texts.extend([tag['description'], item['introduction']])
    vectors = {}
    missing = []
    for text in dict.fromkeys(texts):
        cached = cache.get_embedding(text_hash(text), model_version)
        if cached is not None:
            try:
                vectors[text] = validated_embeddings([cached], 1, embedder.dim)[1][0]
                continue
            except (ValueError, TypeError, OverflowError):
                pass
        missing.append(text)
    for start in range(0, len(missing), 96):
        batch = missing[start:start+96]
        raw, unit = validated_embeddings(embedder.embed(batch), len(batch), embedder.dim)
        cache.put_embeddings([(text_hash(t), model_version, v.tolist()) for t, v in zip(batch, raw)])
        vectors.update(zip(batch, unit))
    scores = []
    for tag, item in pairs:
        score = round(float(np.clip(vectors[tag['description']] @ vectors[item['introduction']], -1, 1)), 8)
        scores.append({'type': tag['type'], 'code': tag['code'], 'creatorId': item['creatorId'], 'score': score})
    scores.sort(key=lambda r: (r['type'], r['code'], r['creatorId']))
    # BE가 조회 시 회원별 제외/선택 태그 OR를 적용한 뒤 Top-N을 자를 수 있도록 점수는 모두 적재한다.
    return {'snapshotHash': snapshot['snapshotHash'], 'method': METHOD, 'modelVersion': model_version, 'scores': scores}


def recommend(snapshot, payload, parents=(), topics=(), top_n=20):
    """오프라인 검증용: 실태그 OR → 점수 정렬 → Top-N. 후보 보충은 없다."""
    catalog = validate_snapshot(snapshot)
    if type(top_n) is not int or not 1 <= top_n <= 20:
        raise ValueError('INVALID_TOP_N')
    selected = {('PARENT', c) for c in parents} | {('TOPIC', c) for c in topics}
    if not selected <= catalog.keys() or payload['snapshotHash'] != snapshot['snapshotHash']:
        raise ValueError('SELECTION_OR_SNAPSHOT_MISMATCH')
    scores, matched = {}, {}
    for row in payload['scores']:
        key = (row['type'], row['code'])
        if key in selected:
            cid = row['creatorId']
            scores[cid] = max(scores.get(cid, -2), row['score'])
            matched.setdefault(cid, []).append(catalog[key])
    return [{'creatorId': cid, 'score': scores[cid], 'matchedTags': sorted(matched[cid], key=lambda t: (t['type'], t['code']))}
            for cid in sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_n]]


def run(backend, directory: Path, cache_path: Path, embedder, model_version, *, apply=False):
    with batch_locks(directory, cache_path):
        snapshot = backend.get(ROOT + '/inputs')
        validate_snapshot(snapshot)
        identity = digest({'snapshotHash': snapshot['snapshotHash'], 'modelVersion': model_version,
                           'method': METHOD, 'embeddingContract': EMBEDDING_CONTRACT_VERSION,
                           'target': backend.target_identity})
        # 원문 snapshot은 서버에서만 읽는다. 로컬 상태에는 지문·점수만 보존한다.
        state_path = directory / 'current.json'
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        if state and state.get('identity') == identity and state.get('sequence') == snapshot['lastApplicationSequence'] and state.get('applied'):
            return {'status': 'skipped', 'count': len(state['payload']['scores'])}
        same = state and state.get('identity') == identity and not state.get('fenced')
        if same and state.get('payload'):
            payload = state['payload']
        else:
            payload = generate(snapshot, embedder, JsonModelCache(cache_path), model_version)
        if not apply:
            _write_json_atomic(directory / 'dry-run-payload.json', payload)
            return {'status': 'generated', 'count': len(payload['scores']), 'payload': payload}
        # 응답 유실이면 같은 번호로 재전송한다. 서버보다 과거인 번호는 새 적용 의도로 재발급한다.
        if not same or state.get('sequence', 0) < snapshot['lastApplicationSequence']:
            sequence = max(snapshot['lastApplicationSequence'], state.get('sequence', 0) if state else 0) + 1
            positive(sequence)
            state = {'identity': identity, 'sequence': sequence, 'payload': payload, 'applied': False}
            _write_json_atomic(state_path, state)
        wire = {**payload, 'applicationSequence': state['sequence']}
        try:
            receipt = backend.put(ROOT + '/rankings', wire)
        except BackendRequestError as error:
            if error.status == 409:
                state['fenced'] = True
                _write_json_atomic(state_path, state)
            raise
        if (receipt.get('applicationSequence') != state['sequence'] or receipt.get('snapshotHash') != payload['snapshotHash']
                or type(receipt.get('applied')) is not bool):
            raise ValueError('INVALID_RANKING_RECEIPT')
        state['applied'] = True
        _write_json_atomic(state_path, state)
        return {'status': 'completed', 'count': len(payload['scores'])}


def main(argv=None):
    import argparse
    import os
    from src.recommendation.interest_runtime import create_interest_embedding_client
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['dry-run', 'apply'])
    parser.add_argument('--be-base-url', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/saved-tags'))
    parser.add_argument('--cache', type=Path, default=Path('results/recommendation/model-cache.json'))
    parser.add_argument('--approve-external-transfer', action='store_true')
    args = parser.parse_args(argv)
    try:
        if not args.approve_external_transfer:
            raise ValueError('EXTERNAL_TRANSFER_NOT_APPROVED')
        backend = SavedTagBackend(BackendClientConfig(args.be_base_url), os.environ.get('CKING_CONTENT_TAG_ACCESS_TOKEN'))
        result = run(backend, args.output_dir, args.cache, create_interest_embedding_client('deepinfra'),
                     'BAAI/bge-m3@deepinfra-v1', apply=args.mode == 'apply')
        print(f"saved-tags: {result['status']} count={result['count']}")
        return 0
    except Exception as error:
        # 모델/HTTP 예외 원문은 소개글·키를 포함할 수 있다.
        print(f'saved-tags: failed ({type(error).__name__})')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
