"""Java 테스트가 시작한 실제 BE에 Python의 운영 배치/HTTP 클라이언트로 접근한다."""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import sys
import urllib.error
import urllib.parse
from unittest.mock import patch

# src.config의 .env 자동 로드까지 차단한다. 테스트에 운영 키가 들어올 경로를 닫는다.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import numpy as np
from src.clients.openai_tagger import TagResult
from src.recommendation.batch import BatchPaths, RecommendationBatch
from src.recommendation.cache import JsonModelCache
from src.recommendation.daily import DailyPaths, DailyRecommendationBatch
from src.recommendation.http import BackendClientConfig, BackendRequestError, CkingBackendClient, _open_without_redirect
from src.recommendation.interest import InterestRecommendationConfig, InterestRecommender, M2_METHOD
from src.recommendation.interest_batch import InterestBatchPaths, InterestRecommendationBatch
from src.recommendation.manifest import fetch_manifest
from src.recommendation.service import RecommendationConfig, SimilarCreatorRecommender
from src.recommendation.storage import _write_json_atomic
from src.recommendation.taxonomy import DEFAULT_CATEGORIES_CSV, load_service_categories, taxonomy_hash
from src.recommendation.e2e.runner import ROOT, git_sha


def control(op, **values):
    print(json.dumps({"op": op, **values}), flush=True)
    return json.loads(sys.stdin.readline())


class FixedModels:
    """제공자 SDK 없이 텍스트 해시별 고정 벡터·고정 태그를 반환한다."""
    name = "cking-e2e-fixed"
    dim = 8

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return np.asarray([[value + 1 for value in hashlib.sha256(text.encode()).digest()[:8]]
                           for text in texts], dtype=np.float64)

    def tag(self, text):
        self.calls += 1
        return TagResult(("FOOD",), 0, 0)


class ObservedBackend(CkingBackendClient):
    """실제 HTTP 전후에 응답 유실/전송 실패만 주입한다. BE 응답을 만들어내지 않는다."""
    def __init__(self, config, key):
        super().__init__(config, recommendation_api_key=key, opener=self.transport)
        self.blocked = set()
        self.lose_once = set()
        self.receipts = []
        self.requests = []

    def transport(self, request, *, timeout):
        path = urllib.parse.urlsplit(request.full_url).path
        if request.method == "PUT" and path in self.blocked:
            raise urllib.error.URLError("injected transport outage")
        response = _open_without_redirect(request, timeout=timeout)
        if request.method == "PUT" and path in self.lose_once:
            self.lose_once.remove(path)
            # BE는 이미 커밋하고 실제 200을 보냈지만 Python은 응답을 받지 못한 상황.
            with response:
                response.read()
            raise http.client.IncompleteRead(b"")
        return response

    def put(self, path, payload):
        self.requests.append((path, copy.deepcopy(payload)))
        response = super().put(path, payload)
        self.receipts.append((path, copy.deepcopy(payload), response))
        return response


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Suite:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.ids = {int(key): value for key, value in config["creatorIds"].items()}
        self.categories = load_service_categories(DEFAULT_CATEGORIES_CSV)
        self.backend = ObservedBackend(BackendClientConfig(config["baseUrl"], max_retries=1,
                                                          retry_backoff_seconds=0), config["apiKey"])
        self.models = FixedModels()
        self.checks = []
        self.cache_path = output / "model-cache.json"
        cache = JsonModelCache(self.cache_path)
        similar_config = RecommendationConfig(top_n=20, embedding_model_version="BAAI/bge-m3@e2e-fixed-v1",
                                               allowed_tags=frozenset(row.code for row in self.categories))
        interest_config = InterestRecommendationConfig(top_n=20, embedding_model_version="BAAI/bge-m3@e2e-fixed-v1")
        def similar(manifest, directory):
            recommender = SimilarCreatorRecommender(self.models, self.models, cache, similar_config)
            return RecommendationBatch(manifest, recommender, BatchPaths(directory), similar_config,
                                       top_n=20, backend=self.backend)
        def interests(manifest, directory):
            recommender = InterestRecommender(manifest, self.categories, self.models, cache, interest_config)
            return InterestRecommendationBatch(recommender, InterestBatchPaths(directory), backend=self.backend,
                                               selected_method=M2_METHOD)
        self.paths = DailyPaths(output / "daily")
        self.daily = DailyRecommendationBatch(self.backend, self.paths, self.cache_path,
                                              {"model": "e2e-fixed-v1"}, similar, interests, page_size=3)

    def verify_active(self, receipts):
        snapshot = control("snapshot")
        for path, payload, response in receipts:
            similar = "creatorId" in payload
            rows = snapshot["similar" if similar else "interests"]
            key = "creator_id" if similar else "interest_code"
            identity = payload["creatorId" if similar else "interestCode"]
            row = next(row for row in rows if row[key] == identity)
            assert row["application_sequence"] == payload["applicationSequence"]
            assert row["input_hash"] == payload["inputHash"]
            assert row["candidate_count"] == len(payload["candidates"])
            assert row["generation_id"] == response["generationId"]
            candidate_rows = snapshot["similarCandidates" if similar else "interestCandidates"]
            actual = [candidate for candidate in candidate_rows if candidate[key] == identity]
            id_key = "similar_creator_id" if similar else "creator_id"
            payload_id = "similarCreatorId" if similar else "creatorId"
            assert [(candidate[id_key], candidate["rank_no"], candidate["score"]) for candidate in actual] == [
                (candidate[payload_id], candidate["rank"], candidate["score"]) for candidate in payload["candidates"]]
        return snapshot

    def passed(self, name):
        self.checks.append({"name": name, "status": "passed"})

    def batch_lifecycle(self):
        assert taxonomy_hash(self.categories) == self.config["taxonomyHash"]
        paged = fetch_manifest(self.backend, page_size=3)
        assert len(paged.creators) == len(self.ids)
        assert paged.manifest_hash == fetch_manifest(self.backend, page_size=100).manifest_hash
        self.passed("paginated_fixed_manifest")
        similar_path = f"/api/admin/creators/{self.ids[2]}/similar"
        interest_path = "/api/admin/interests/FOOD/recommendations"
        self.backend.lose_once.add(similar_path)
        assert self.daily.run("apply")["status"] == "completed"
        original = list(self.backend.receipts)
        assert len(original) == len(self.ids) + 17
        assert any(path == similar_path and result["applied"] is False for path, _, result in original)
        snapshot_a = self.verify_active(original)
        self.passed("initial_apply_and_lost_response_retry")
        before_models, before_puts = self.models.calls, len(self.backend.requests)
        assert self.daily.run("apply")["status"] == "skipped"
        assert self.models.calls == before_models and len(self.backend.requests) == before_puts
        for path, payload, result in original:
            retried = self.backend.put(path, payload)
            assert retried["applied"] is False and retried["generationId"] == result["generationId"]
        assert control("snapshot") == snapshot_a
        self.passed("same_run_skip_and_idempotent_reapply")
        for path, payload, _ in (original[0], next(row for row in original if "interestCode" in row[1])):
            conflict = {**payload, "modelVersion": "changed-model"}
            conflict["candidates"] = [{**candidate, "modelVersion": "changed-model"} for candidate in payload["candidates"]]
            self.reject(self.backend, path, conflict, 409, "RECOMMENDATION_INPUT_CONFLICT")
        assert control("snapshot") == snapshot_a
        self.passed("same_sequence_payload_conflict")

        # B 부분 적용 후 같은 실행으로 실패한 두 대상만 재개한다.
        control("intro", changed=True)
        self.backend.blocked.update((similar_path, interest_path))
        before = len(self.backend.requests)
        partial = self.daily.run("apply")
        assert partial["status"] == "partial" and partial["failureCount"] == 2
        b_requests = self.backend.requests[before:]
        assert read(self.paths.completed)["manifestHash"] == paged.manifest_hash
        mixed = control("snapshot")
        assert len({row["application_sequence"] for row in mixed["similar"] + mixed["interests"]}) == 2
        self.backend.blocked.clear()
        before, model_calls = len(self.backend.requests), self.models.calls
        resumed = self.daily.run("apply")
        assert resumed["status"] == "completed" and resumed["applicationSequence"] == partial["applicationSequence"]
        assert len(self.backend.requests) == before + 2 and self.models.calls == model_calls
        self.verify_active(self.backend.receipts[-2:])
        self.passed("partial_failure_resume_only_failed_targets")
        # A -> B -> A에서 같은 내용이 새 번호로 적용되어야 한다.
        control("intro", changed=False)
        start = len(self.backend.receipts)
        reverted = self.daily.run("apply")
        assert reverted["status"] == "completed" and reverted["manifestHash"] == paged.manifest_hash
        assert reverted["applicationSequence"] > resumed["applicationSequence"]
        self.verify_active(self.backend.receipts[start:])
        for old, new in zip(original, self.backend.receipts[start:]):
            assert old[1]["inputHash"] == new[1]["inputHash"]
        self.passed("a_b_a_reversion")

        # B 부분 적용 중 A로 다시 전환한다. 전송된 적 없는 지연 B도 409여야 한다.
        control("intro", changed=True)
        self.backend.blocked.update((similar_path, interest_path))
        start = len(self.backend.requests)
        assert self.daily.run("apply")["status"] == "partial"
        delayed = self.backend.requests[start:]
        control("intro", changed=False)
        self.backend.blocked.clear()
        start = len(self.backend.receipts)
        assert self.daily.run("apply")["status"] == "completed"
        self.verify_active(self.backend.receipts[start:])
        snapshot_reverted = control("snapshot")
        for requests in (b_requests, delayed):
            for path, payload in requests:
                self.reject(self.backend, path, payload, 409, "STALE_RECOMMENDATION_INPUT")
        assert control("snapshot") == snapshot_reverted
        self.verify_active(self.backend.receipts[start:])
        self.passed("partial_b_a_reversion_and_delayed_b_rejection")

    def empty_generation(self):
        for identity in self.ids.values():
            assert self.backend.put(f"/api/admin/creators/{identity}/similar", self.payload(50, identity, []))["applied"] is True
        for category in self.categories:
            assert self.backend.put(f"/api/admin/interests/{category.code}/recommendations",
                                    self.payload(50, category.code, [], interest=True))["applied"] is True
        snapshot = control("snapshot")
        assert all(row["candidate_count"] == 0 and row["application_sequence"] == 50
                   for row in snapshot["similar"] + snapshot["interests"])
        assert snapshot["similarCandidates"] == [] and snapshot["interestCandidates"] == []
        assert self.backend.get(f"/api/creators/{self.ids[2]}/similar")["candidates"] == []
        self.passed("empty_generation_http_activation_and_db_state")

    @staticmethod
    def reject(client, path, payload, status, error_code=None):
        try:
            client.put(path, payload)
        except BackendRequestError as error:
            assert error.status == status
            if error_code:
                assert error.code == error_code
        else:
            raise AssertionError("Expected HTTP rejection")

    def payload(self, sequence, identity, candidates, *, interest=False):
        input_hash = hashlib.sha256(f"fixture:{sequence}:{identity}".encode()).hexdigest()
        method = "INTEREST_M2_V1" if interest else "M2"
        common = {"method": method, "modelVersion": "e2e-fixed-v1", "inputHash": input_hash}
        payload = {**common, "applicationSequence": sequence,
                   "candidates": [{**common, "rank": rank, "score": round(1 - index / 100, 8),
                                   **({"interestCode": identity, "creatorId": creator} if interest else
                                      {"creatorId": identity, "similarCreatorId": creator})}
                                  for index, (creator, rank) in enumerate(candidates)]}
        payload.update({"interestCode": identity, "taxonomyVersion": "v0.2",
                        "taxonomyHash": self.config["taxonomyHash"]} if interest else {"creatorId": identity})
        return payload

    def personalization(self):
        fixture = read(ROOT / "fixtures/hybrid_personalized_v1.json")
        # 정책 fixture는 API의 Top-100/연속 rank보다 넓다. 희소·int 최대 rank는 DB fixture로 준비한다.
        for case_index, case in enumerate(fixture["cases"]):
            data = copy.deepcopy(case["input"])
            expected = copy.deepcopy(case["expected"])
            def mapped(value):
                return self.ids[value]
            data["memberCreatorId"] = mapped(data["memberCreatorId"]) if data["memberCreatorId"] else None
            data["followedCreatorIds"] = sorted({mapped(value) for value in data["followedCreatorIds"]})
            data["selectedInterestCodes"] = sorted(set(data["selectedInterestCodes"]))
            for space in data["creatorSpaces"]:
                space["creatorId"] = mapped(space["creatorId"])
            for source in data["followSources"]:
                source["seedCreatorId"] = mapped(source["seedCreatorId"])
            for source in data["interestSources"] + data["followSources"]:
                if source["activeGeneration"]:
                    for candidate in source["activeGeneration"]["candidates"]:
                        candidate["creatorId"] = mapped(candidate["creatorId"])
            sequence = 100 + case_index
            # 빈 세대도 실제 적재 API로 활성화하고 DB에서 빈 후보를 확인한다.
            for identity in self.ids.values():
                payload = self.payload(sequence, identity, [])
                response = self.backend.put(f"/api/admin/creators/{identity}/similar", payload)
                assert response["candidateCount"] == 0
            for category in self.categories:
                payload = self.payload(sequence, category.code, [], interest=True)
                response = self.backend.put(f"/api/admin/interests/{category.code}/recommendations", payload)
                assert response["candidateCount"] == 0
            snapshot = control("snapshot")
            assert all(row["candidate_count"] == 0 for row in snapshot["similar"] + snapshot["interests"])
            control("personalization", interests=data["selectedInterestCodes"],
                    followed=data["followedCreatorIds"], spaces=data["creatorSpaces"])
            control("fixtureCandidates", input=data)
            client = CkingBackendClient(self.backend.config, access_token=self.config["userToken"])
            response = client._request("GET", "/api/me/creator-recommendations?size=20", None, authenticated=True)
            actual = [{key: item[key] for key in ("creatorId", "aggregateScore", "interestCodes", "seedCreatorIds")}
                      for item in response["items"]]
            for item in expected["items"]:
                item["creatorId"] = mapped(item["creatorId"])
                item["seedCreatorIds"] = [mapped(value) for value in item["seedCreatorIds"]]
            if expected["items"]:
                assert response["policyVersion"] == expected["policyVersion"]
                from decimal import Decimal
                for item in actual:
                    item["aggregateScore"] = format(Decimal(str(item["aggregateScore"])), ".8f")
                assert actual == expected["items"]
                assert len(actual) < 20  # 후보 부족 시 인기순을 섞어 채우지 않는다.
            else:
                # 공용 fixture의 빈 개인화 결과는 최신 BE 계약에서 인기순으로 대체된다.
                assert response["policyVersion"] == "POPULAR_FALLBACK_V1"
                excluded = set(data["followedCreatorIds"]) | {self.ids[1]}
                eligible = sorted(space["creatorId"] for space in data["creatorSpaces"]
                                  if space["hasCreatorSpace"] and space["creatorId"] not in excluded)
                assert [item["creatorId"] for item in actual] == eligible
                assert all(item["aggregateScore"] == 0 and item["interestCodes"] == []
                           and item["seedCreatorIds"] == [] for item in actual)
            self.passed("shared_fixture:" + case["id"])

    def popular_fallback(self):
        # 앞선 fixture의 seed 후보가 Space 복원으로 다시 유효해지지 않게 빈 세대로 교체한다.
        assert self.backend.put(f"/api/admin/creators/{self.ids[2]}/similar",
                                self.payload(900, self.ids[2], []))["candidateCount"] == 0
        control("personalization", interests=[], followed=[self.ids[2]],
                spaces=[{"creatorId": identity, "hasCreatorSpace": True} for identity in self.ids.values()])
        control("popularity")
        client = CkingBackendClient(self.backend.config, access_token=self.config["userToken"])
        response = client._request("GET", "/api/me/creator-recommendations?size=3", None, authenticated=True)
        assert response["policyVersion"] == "POPULAR_FALLBACK_V1"
        assert [item["creatorId"] for item in response["items"]] == [self.ids[101], self.ids[102], self.ids[3]]
        assert [item["aggregateScore"] for item in response["items"]] == [2, 1, 0]
        assert all(item["interestCodes"] == [] and item["seedCreatorIds"] == [] for item in response["items"])
        self.passed("popular_fallback_follower_counts_order_and_size")

    def authentication(self):
        sequence = 1000
        for interest in (False, True):
            identity = "FOOD" if interest else self.ids[2]
            path = (f"/api/admin/interests/{identity}/recommendations" if interest
                    else f"/api/admin/creators/{identity}/similar")
            payload = self.payload(sequence, identity, [], interest=interest)
            assert self.backend.put(path, payload)["applied"] is True
            before = control("snapshot")
            self.reject(CkingBackendClient(self.backend.config, recommendation_api_key="invalid-e2e-key"), path, payload, 401)
            # 운영 클라이언트는 의도적으로 헤더를 혼합하지 않으므로 이 보안 시나리오만 transport에 JWT를 추가한다.
            def mixed(request, *, timeout):
                request.add_header("Authorization", f"Bearer {self.config['adminToken']}")
                return _open_without_redirect(request, timeout=timeout)
            self.reject(CkingBackendClient(self.backend.config, recommendation_api_key="invalid-e2e-key", opener=mixed),
                        path, payload, 401)
            user = CkingBackendClient(self.backend.config, access_token=self.config["userToken"])
            self.reject(user, path, payload, 403)
            admin = CkingBackendClient(self.backend.config, access_token=self.config["adminToken"])
            assert admin.put(path, payload)["applied"] is False
            assert control("snapshot") == before
            self.passed("interest_authentication" if interest else "similar_authentication")
        control("revokeAdmin")
        before = control("snapshot")
        for interest in (False, True):
            identity = "FOOD" if interest else self.ids[2]
            path = f"/api/admin/interests/{identity}/recommendations" if interest else f"/api/admin/creators/{identity}/similar"
            self.reject(CkingBackendClient(self.backend.config, access_token=self.config["adminToken"]), path,
                        self.payload(sequence + 1, identity, [], interest=interest), 403)
        assert control("snapshot") == before
        self.passed("issued_admin_jwt_rejected_after_role_revocation")


def main():
    config = json.loads(sys.stdin.readline())
    output = Path(os.environ["CKING_E2E_OUTPUT"])
    report_path = output / "report.json"
    report = {"schemaVersion": 1, "status": "verification_failure", "checks": [],
              "llmCommitSha": git_sha(ROOT), "beCommitSha": git_sha(Path(os.environ["CKING_E2E_BE_ROOT"])),
              "taxonomyHash": config["taxonomyHash"], "externalModelCalls": 0, "externalConnectionAttempts": 0}
    original_connect = socket.socket.connect
    _write_json_atomic(report_path, report)
    def loopback_only(sock, address):
        if address[0] not in ("127.0.0.1", "::1", "localhost"):
            report["externalConnectionAttempts"] += 1
            raise AssertionError("External network prohibited")
        return original_connect(sock, address)
    try:
        with patch.object(socket.socket, "connect", loopback_only):
            suite = Suite(config, output)
            report["checks"] = suite.checks
            for name in ("batch_lifecycle", "empty_generation", "personalization", "popular_fallback", "authentication"):
                report["failedCheck"] = name
                _write_json_atomic(report_path, report)
                getattr(suite, name)()
            assert report["externalConnectionAttempts"] == 0
        report.update(status="passed", sharedFixtureCases=18, syntheticModelInputs=suite.models.calls)
        report.pop("failedCheck", None)
        return 0
    except Exception as error:
        # 메시지·HTTP 응답·SQL·토큰·소개 원문은 보고하지 않는다.
        report["errorType"] = type(error).__name__
        import traceback
        report["failureLocation"] = [{"file": Path(frame.filename).name, "line": frame.lineno}
                                     for frame in traceback.extract_tb(error.__traceback__)]
        return 1
    finally:
        _write_json_atomic(report_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
