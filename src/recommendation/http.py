"""Cking-BE JSON API를 제한 재시도와 함께 호출한다."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class JsonBackend(Protocol):
    """manifest 조회와 추천 적재가 의존하는 BE 인터페이스다."""

    @property
    def target_identity(self) -> str:
        ...

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
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
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

    @property
    def target_identity(self) -> str:
        """체크포인트의 적용 상태를 구분하는 정규화된 BE 대상 주소다."""
        return self.config.base_url

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
            except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, OSError) as error:
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


def normalize_base_url(base_url: str) -> str:
    """동일한 HTTP(S) BE 주소 표기를 체크포인트 식별자 하나로 정규화한다."""
    raw = base_url.strip()
    if not raw:
        raise ValueError("BE base URL은 비어 있을 수 없습니다.")
    try:
        parsed = urllib.parse.urlsplit(raw)
        port = parsed.port
    except ValueError as error:
        raise ValueError("BE base URL 형식이 올바르지 않습니다.") from error
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("BE base URL은 http 또는 https 절대 URL이어야 합니다.")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("BE base URL에는 인증 정보, query, fragment를 포함할 수 없습니다.")

    hostname = parsed.hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    authority = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((scheme, authority, path, "", ""))
