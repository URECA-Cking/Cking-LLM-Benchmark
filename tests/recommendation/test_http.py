from __future__ import annotations

import http.client
import io
import json
import traceback
import urllib.error

import pytest

from src.recommendation.http import BackendClientConfig, BackendRequestError, CkingBackendClient


class Response:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.body = json.dumps(payload).encode()
        self.status = status

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class IncompleteResponse(Response):
    def read(self) -> bytes:
        raise http.client.IncompleteRead(self.body[:5], len(self.body) - 5)


def test_recommendation_api_key_authenticates_put_without_jwt_and_is_not_sent_for_get():
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return Response({"data": {"applied": True}})

    client = CkingBackendClient(
        BackendClientConfig("https://be"), access_token="jwt-secret",
        recommendation_api_key="key-secret", opener=opener,
    )
    client.put("/api/admin/interests/EDU/recommendations", {"interestCode": "EDU", "candidates": []})
    client.get("/api/creators")
    headers = dict((key.lower(), value) for key, value in requests[0].header_items())
    assert headers["x-cking-recommendation-key"] == "key-secret"
    assert "authorization" not in headers
    public_headers = dict((key.lower(), value) for key, value in requests[1].header_items())
    assert "x-cking-recommendation-key" not in public_headers and "authorization" not in public_headers


@pytest.mark.parametrize("api_key", ["", " ", "secret\nvalue", "secret\rvalue"])
def test_invalid_recommendation_api_key_rejected(api_key):
    with pytest.raises(ValueError, match="API Key"):
        CkingBackendClient(BackendClientConfig("https://be"), recommendation_api_key=api_key)


def test_get_unwraps_api_response_and_retries_transient_status() -> None:
    calls = []
    sleeps = []
    outcomes: list[object] = [
        urllib.error.HTTPError("https://be/api/creators", 503, "busy", {}, io.BytesIO()),
        Response({"code": "SUCCESS", "data": {"items": []}}),
    ]

    def opener(request: object, timeout: float) -> object:
        calls.append((request, timeout))
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    client = CkingBackendClient(
        BackendClientConfig("https://be", timeout_seconds=3, max_retries=1, retry_backoff_seconds=0.25),
        opener=opener,
        sleep=sleeps.append,
    )

    assert client.get("/api/creators") == {"items": []}
    assert len(calls) == 2
    assert calls[0][1] == 3
    assert sleeps == [0.25]


def test_incomplete_response_body_is_retried_then_succeeds() -> None:
    calls = []
    sleeps = []
    outcomes = [
        IncompleteResponse({"data": {"items": []}}),
        Response({"data": {"items": []}}),
    ]

    def opener(request: object, timeout: float) -> object:
        calls.append((request, timeout))
        return outcomes.pop(0)

    client = CkingBackendClient(
        BackendClientConfig("https://be", max_retries=2, retry_backoff_seconds=0.25),
        opener=opener,
        sleep=sleeps.append,
    )

    assert client.get("/api/creators") == {"items": []}
    assert len(calls) == 2
    assert sleeps == [0.25]


def test_backend_target_identity_normalizes_equivalent_base_urls() -> None:
    first = CkingBackendClient(BackendClientConfig(" HTTPS://BE.Example:443/api/ "))
    second = CkingBackendClient(BackendClientConfig("https://be.example/api"))

    assert first.target_identity == second.target_identity == "https://be.example/api"


def test_non_transient_http_error_is_not_retried_or_leaked() -> None:
    calls = 0

    def opener(request: object, timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("https://be/api", 400, "token-secret", {}, io.BytesIO(b"token-secret"))

    client = CkingBackendClient(
        BackendClientConfig("https://be", max_retries=3),
        access_token="token-secret",
        opener=opener,
    )
    with pytest.raises(BackendRequestError, match="HTTP 400") as raised:
        client.put("/api/admin/creators/1/similar", {"creatorId": 1})

    assert calls == 1
    assert raised.value.transient is False
    assert "token-secret" not in str(raised.value)


def test_apply_requires_access_token_before_http_call() -> None:
    client = CkingBackendClient(BackendClientConfig("https://be"), opener=lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="CKING_ADMIN_ACCESS_TOKEN"):
        client.put("/api/admin/creators/1/similar", {"creatorId": 1})


@pytest.mark.parametrize("path", ["/api/admin/creators/1/similar", "/api/admin/interests/EDU/recommendations"])
@pytest.mark.parametrize("status,retries", [(401, 1), (403, 1), (302, 1), (503, 3)])
def test_key_http_status_retries_and_full_traceback_redaction(path, status, retries):
    calls = []
    sleeps = []

    def opener(request, timeout):
        calls.append(request)
        headers = dict((key.lower(), value) for key, value in request.header_items())
        assert headers["x-cking-recommendation-key"] == "key-secret"
        assert "authorization" not in headers
        raise urllib.error.HTTPError("https://be/api", status, "key-secret", {}, io.BytesIO(b"key-secret"))

    client = CkingBackendClient(BackendClientConfig("https://be", max_retries=2),
                                recommendation_api_key="key-secret", access_token="jwt-secret",
                                opener=opener, sleep=sleeps.append)
    with pytest.raises(BackendRequestError) as raised:
        client.put(path, {})
    assert len(calls) == retries and len(sleeps) == retries - 1
    assert str(raised.value) == str(raised.value).rstrip()
    assert raised.value.status == status and raised.value.transient == (status == 503)
    assert "key-secret" not in "".join(traceback.format_exception(raised.value))


def test_timeout_retries_and_masks_exception_chain():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise TimeoutError("key-secret")

    client = CkingBackendClient(BackendClientConfig("https://be", max_retries=1),
                                recommendation_api_key="key-secret", opener=opener, sleep=lambda _: None)
    with pytest.raises(BackendRequestError) as raised:
        client.put("/api/admin/creators/1/similar", {})
    assert len(calls) == 2 and raised.value.transient and raised.value.status is None
    assert "key-secret" not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize("path", ["/api/admin/members", "/api/admin/creators/0/similar",
                                 "/api/admin/creators/1/similar?key=x", "https://other/api"])
def test_key_cannot_be_sent_to_other_paths(path):
    client = CkingBackendClient(BackendClientConfig("https://be"), recommendation_api_key="key-secret",
                                opener=lambda *_a, **_k: pytest.fail("HTTP 호출 금지"))
    with pytest.raises(ValueError, match="적재 경로"):
        client.put(path, {})


def test_unexpected_transport_error_is_sanitized_without_retry():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise ValueError("key-secret")

    client = CkingBackendClient(BackendClientConfig("https://be", max_retries=3),
                                recommendation_api_key="key-secret", opener=opener)
    with pytest.raises(BackendRequestError) as raised:
        client.put("/api/admin/creators/1/similar", {})
    assert len(calls) == 1 and not raised.value.transient
    assert "key-secret" not in "".join(traceback.format_exception(raised.value))


def test_backend_error_message_and_nested_cause_are_redacted():
    def opener(request, timeout):
        try:
            raise ValueError("key-secret jwt-secret")
        except ValueError as error:
            raise BackendRequestError("key-secret jwt-secret", status=401) from error

    client = CkingBackendClient(BackendClientConfig("https://be"), access_token="jwt-secret",
                                recommendation_api_key="key-secret", opener=opener)
    with pytest.raises(BackendRequestError) as raised:
        client.put("/api/admin/creators/1/similar", {})
    rendered = "".join(traceback.format_exception(raised.value))
    assert "key-secret" not in rendered and "jwt-secret" not in rendered
    assert raised.value.status == 401


@pytest.mark.parametrize("code", ["STALE_RECOMMENDATION_INPUT", "RECOMMENDATION_INPUT_CONFLICT", "UNKNOWN"])
def test_409_classifies_only_known_codes_without_retry_or_body_leak(code):
    calls, sleeps = [], []
    def opener(request, timeout):
        calls.append(request)
        body = json.dumps({"code": code, "message": "private profile key-secret"}).encode()
        raise urllib.error.HTTPError("https://be", 409, "conflict", {}, io.BytesIO(body))
    client = CkingBackendClient(BackendClientConfig("https://be"),
                               recommendation_api_key="key-secret", opener=opener, sleep=sleeps.append)
    with pytest.raises(BackendRequestError) as caught:
        client.put("/api/admin/creators/1/similar", {})
    assert caught.value.status == 409 and caught.value.transient is False
    assert caught.value.code == (None if code == "UNKNOWN" else code)
    assert len(calls) == 1 and not sleeps
    assert "private profile" not in str(caught.value) and "key-secret" not in str(caught.value)
