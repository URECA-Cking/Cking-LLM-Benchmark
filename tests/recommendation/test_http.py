from __future__ import annotations

import io
import json
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
