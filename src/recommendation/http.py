"""Cking-BE JSON API를 제한 재시도와 함께 호출한다."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class JsonBackend(Protocol):
    """manifest 조회와 추천 적재가 의존하는 BE 인터페이스다."""

    def get(self, path: str) -> dict[str, object]:
        ...

    def put(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        ...


class BackendRequestError(RuntimeError):
    """BE 요청이 재시도 뒤에도 성공하지 못했음을 나타낸다."""

    def __init__(self, message: str, *, status: int | None = None, transient: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.transient = transient


@dataclass(frozen=True)
class BackendClientConfig:
    base_url: str
    timeout_seconds: float = 10.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("BE base URL은 비어 있을 수 없습니다.")
        if self.timeout_seconds <= 0:
            raise ValueError("요청 timeout은 양수여야 합니다.")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("재시도 횟수는 0 이상의 정수여야 합니다.")
        if self.retry_backoff_seconds < 0:
            raise ValueError("재시도 대기 시간은 0 이상이어야 합니다.")


class CkingBackendClient:
    """표준 라이브러리만 사용해 Cking-BE의 JSON 응답 봉투를 읽는다."""

    def __init__(
        self,
        config: BackendClientConfig,
        access_token: str | None = None,
        *,
        opener: Callable[..., object] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._access_token = access_token
        self._opener = opener
        self._sleep = sleep

    def get(self, path: str) -> dict[str, object]:
        return self._request("GET", path, None, authenticated=False)

    def put(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        if not self._access_token:
            raise RuntimeError("apply에는 CKING_ADMIN_ACCESS_TOKEN이 필요합니다.")
        return self._request("PUT", path, payload, authenticated=True)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        *,
        authenticated: bool,
    ) -> dict[str, object]:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if authenticated:
            headers["Authorization"] = f"Bearer {self._access_token}"

        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with self._opener(request, timeout=self.config.timeout_seconds) as response:  # type: ignore[attr-defined]
                    decoded = self._decode_json(response.read(), status=getattr(response, "status", None))
                    return self._unwrap(decoded)
            except urllib.error.HTTPError as error:
                transient = error.code in TRANSIENT_HTTP_STATUSES
                if transient and attempt < self.config.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise BackendRequestError(
                    f"BE {method} 요청 실패: HTTP {error.code}",
                    status=error.code,
                    transient=transient,
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt < self.config.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise BackendRequestError(
                    f"BE {method} 요청 실패: {type(error).__name__}",
                    transient=True,
                ) from error

        raise AssertionError("도달할 수 없는 재시도 상태입니다.")

    def _wait_before_retry(self, attempt: int) -> None:
        self._sleep(self.config.retry_backoff_seconds * (2**attempt))

    @staticmethod
    def _decode_json(raw: bytes, *, status: int | None) -> dict[str, object]:
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendRequestError(
                "BE 응답이 UTF-8 JSON 객체가 아닙니다.",
                status=status,
            ) from error
        if not isinstance(decoded, dict):
            raise BackendRequestError("BE 응답 JSON 최상위 값은 객체여야 합니다.", status=status)
        return decoded

    @staticmethod
    def _unwrap(payload: dict[str, object]) -> dict[str, object]:
        """공통 ApiResponse의 data를 읽되 테스트용 평문 객체도 허용한다."""
        if "data" not in payload:
            return payload
        data = payload["data"]
        if not isinstance(data, dict):
            raise BackendRequestError("BE 응답의 data는 객체여야 합니다.")
        return data
