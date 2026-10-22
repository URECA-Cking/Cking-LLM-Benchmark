"""실제 BE HTTP/DB와 합성 모델로 11명 후보의 무보충 계약을 검증한다."""
from __future__ import annotations

import os
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
from src.clients.openai_tagger import TagResult
from src.recommendation.cache import JsonModelCache
from src.recommendation.http import CkingBackendClient
from src.recommendation.interest import InterestRecommender
from src.recommendation.manifest import fetch_manifest
from src.recommendation.service import SimilarCreatorRecommender, RecommendationConfig
from src.recommendation.storage import _write_json_atomic
from src.recommendation.taxonomy import DEFAULT_CATEGORIES_CSV, load_service_categories
from src.recommendation.e2e.scenarios import receive_control, control, ObservedBackend
from src.recommendation.http import BackendClientConfig


class SyntheticModels:
    name = 'BAAI/bge-m3'
    dim = 17

    def __init__(self, categories):
        self.categories = categories
        self.descriptions = {row.description: i for i, row in enumerate(categories)}
        self.indices = {row.code: i for i, row in enumerate(categories)}

    def tag(self, text):
        return TagResult(('FOOD',) if '요리' in text else ('FITNESS',) if '운동' in text else (), 0, 0)

    def embed(self, texts):
        rows = []
        for text in texts:
            tags = self.tag(text).tags
            i = self.descriptions.get(text, self.indices[tags[0]] if tags else None)
            row = np.full(17, -1.) if i is None else np.eye(17)[i]
            rows.append(row)
        return np.asarray(rows)


def main():
    output = Path(os.environ['CKING_E2E_OUTPUT'])
    report = {'schemaVersion': 1, 'status': 'running', 'completed': False, 'checks': [],
              'externalConnectionAttempts': 0, 'modelEvidence': 'synthetic_contract_only'}
    original_connect = socket.socket.connect

    def local(sock, address):
        if address[0] not in ('127.0.0.1', '::1', 'localhost'):
            report['externalConnectionAttempts'] += 1
            raise AssertionError('External connection forbidden')
        return original_connect(sock, address)

    def passed(name):
        report['checks'].append({'name': name, 'status': 'passed'})
        _write_json_atomic(output / 'report.json', report)

    try:
        config = receive_control()
        ids = {int(k): v for k, v in config['creatorIds'].items()}
        control('smallPoolSetup')
        categories = load_service_categories(DEFAULT_CATEGORIES_CSV)
        models = SyntheticModels(categories)
        cache = JsonModelCache(output / 'small-pool-cache.json')
        backend = ObservedBackend(BackendClientConfig(config['baseUrl']), config['apiKey'])
        user = CkingBackendClient(backend.config, access_token=config['userToken'])
        with patch.object(socket.socket, 'connect', local):
            manifest = fetch_manifest(backend)
            assert len(manifest.creators) == 11
            similar = SimilarCreatorRecommender(models, models, cache, RecommendationConfig(
                allowed_tags=frozenset(c.code for c in categories)))
            interest = InterestRecommender(manifest, categories, models, cache, tagging_client=models)
            seed = next(p for p in manifest.creators if p.creator_id == ids[2])
            similar_payload = similar.recommend(seed, manifest.creators).to_backend_payload()
            assert [c['similarCreatorId'] for c in similar_payload['candidates']] == [ids[3], ids[4]]
            backend.put(f'/api/admin/creators/{ids[2]}/similar', {**similar_payload, 'applicationSequence': 1})
            for category in categories:
                payload = interest.recommend(category.code).to_backend_payload()
                result = backend.put(f'/api/admin/interests/{category.code}/recommendations', {**payload, 'applicationSequence': 1})
                assert result['candidateCount'] == len(payload['candidates'])
            passed('python_proof_accepted_by_real_be_for_all_17_interests')

            def recommended():
                return user._request('GET', '/api/me/creator-recommendations?size=20', None, authenticated=True)

            def creators():
                return [i['creatorId'] for i in recommended()['items']]

            def public_similar():
                return [i['similarCreatorId'] for i in backend.get(f'/api/creators/{ids[2]}/similar?size=20')['candidates']]

            assert creators() == [ids[3], ids[4]]
            assert public_similar() == [ids[3], ids[4]]
            passed('small_pool_two_related_only_and_no_unclassified_or_blank_candidates')
            # 분야 편집 자체도 실제 회원 JWT API를 사용한다.
            user._request('PUT', '/api/me/interests', {'taxonomyVersion': 'v0.2', 'interestCodes': ['FITNESS']}, authenticated=True)
            assert creators() == [ids[5]]
            passed('interest_edit_excludes_unrelated_follow_candidates')
            user._request('PUT', '/api/me/interests', {'taxonomyVersion': 'v0.2', 'interestCodes': ['FOOD']}, authenticated=True)
            control('profile', logical=3, intro='   ')
            assert creators() == [ids[4]] and public_similar() == [ids[4]]
            control('profile', logical=4, intro='운동 근력 채널로 변경')
            assert creators() == [] and public_similar() == []
            passed('current_profile_hash_removes_blank_and_changed_profiles_without_padding')
            # 동일 실행 재전송은 멱등이어도 현재 소개 검사 때문에 과거 후보가 살아나지 않는다.
            assert backend.put(f'/api/admin/creators/{ids[2]}/similar', {**similar_payload, 'applicationSequence': 1})['applied'] is False
            assert public_similar() == []
            passed('idempotent_retry_does_not_restore_stale_candidates')
            # 새 적재 중 소개가 바뀌어도 묶음은 저장하고 오래된 후보는 조회에서만 제외한다.
            assert backend.put(f'/api/admin/creators/{ids[2]}/similar', {**similar_payload, 'applicationSequence': 2})['applied'] is True
            assert public_similar() == []
            passed('changed_profile_does_not_fail_entire_new_generation_and_stays_hidden')
            old = {k:v for k,v in similar_payload.items() if k != 'eligibility'}
            backend.put(f'/api/admin/creators/{ids[2]}/similar', {**old, 'applicationSequence': 3})
            assert public_similar() == [] and creators() == []
            passed('legacy_generation_without_classification_proof_is_not_exposed')
        report.update(status='passed', completed=True)
        return 0
    except Exception as error:
        import traceback
        report.update(status='verification_failure', completed=True, errorType=type(error).__name__,
                      failureLocation=[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(error.__traceback__)])
        return 1
    finally:
        _write_json_atomic(output / 'report.json', report)


if __name__ == '__main__':
    raise SystemExit(main())
