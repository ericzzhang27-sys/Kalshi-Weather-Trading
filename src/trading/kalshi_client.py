from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import time
from typing import Any
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class KalshiClientError(RuntimeError):
    """Base error for Kalshi client failures."""


@dataclass(frozen=True)
class KalshiAPIError(KalshiClientError):
    method: str
    path: str
    status_code: int
    response_text: str

    def __str__(self) -> str:
        safe_text = self.response_text[:500]
        return (
            f"Kalshi API {self.method} {self.path} failed with "
            f"HTTP {self.status_code}: {safe_text}"
        )


def sign_request_text(private_key: RSAPrivateKey, text: str) -> str:
    signature = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def signing_path_from_url(base_url: str, path: str) -> str:
    request_path = _ensure_leading_slash(path)
    parsed = urlparse(base_url.rstrip("/") + request_path)
    return parsed.path.split("?", 1)[0]


def create_kalshi_signature(
    private_key: RSAPrivateKey,
    timestamp_ms: str,
    method: str,
    signing_path: str,
) -> str:
    path_without_query = signing_path.split("?", 1)[0]
    message = f"{timestamp_ms}{method.upper()}{path_without_query}"
    return sign_request_text(private_key, message)


class KalshiClient:
    def __init__(
        self,
        base_url: str,
        api_key_id: str | None = None,
        private_key: RSAPrivateKey | None = None,
        session: requests.Session | None = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        timestamp_ms_fn: Callable[[], str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_id = api_key_id
        self.private_key = private_key
        self.session = session or requests.Session()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.timestamp_ms_fn = timestamp_ms_fn or _current_timestamp_ms

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key_id and self.private_key)

    def auth_headers(self, method: str, path: str) -> dict[str, str]:
        if not self.api_key_id or self.private_key is None:
            raise KalshiClientError("Authenticated Kalshi request requires API key ID and private key")
        timestamp = self.timestamp_ms_fn()
        signing_path = signing_path_from_url(self.base_url, path)
        signature = create_kalshi_signature(
            self.private_key,
            timestamp_ms=timestamp,
            method=method,
            signing_path=signing_path,
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        auth: bool = False,
    ) -> dict[str, Any]:
        return self.request("GET", path, params=params, auth=auth)

    def post(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        return self.request("POST", path, json_payload=payload, auth=auth)

    def delete(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        return self.request("DELETE", path, params=params, auth=auth)

    def request(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        json_payload: Mapping[str, Any] | None = None,
        auth: bool = False,
    ) -> dict[str, Any]:
        method_upper = method.upper()
        request_path = _ensure_leading_slash(path)
        url = self.base_url + request_path

        for attempt in range(self.max_retries + 1):
            headers: dict[str, str] = {}
            if auth:
                headers.update(self.auth_headers(method_upper, request_path))
            response = self.session.request(
                method_upper,
                url,
                params=dict(params or {}),
                json=dict(json_payload or {}) if json_payload is not None else None,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if response.status_code not in TRANSIENT_STATUS_CODES or attempt >= self.max_retries:
                break
            time.sleep(self.retry_backoff_seconds * (2**attempt))

        if response.status_code >= 400:
            raise KalshiAPIError(
                method=method_upper,
                path=request_path,
                status_code=response.status_code,
                response_text=response.text,
            )
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise KalshiClientError(
                f"Kalshi API {method_upper} {request_path} returned non-JSON response"
            ) from exc
        if not isinstance(payload, dict):
            raise KalshiClientError(
                f"Kalshi API {method_upper} {request_path} returned unexpected JSON payload"
            )
        return payload

    def iter_markets(
        self,
        status: str | None = "open",
        series_ticker: str | None = None,
        limit: int = 1000,
        max_pages: int = 5,
        auth: bool = False,
    ) -> Iterable[dict[str, Any]]:
        params: dict[str, Any] = {"limit": int(limit)}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker

        cursor: str | None = None
        for _ in range(int(max_pages)):
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self.get("/markets", params=page_params, auth=auth)
            markets = payload.get("markets", [])
            if not isinstance(markets, list):
                raise KalshiClientError("Kalshi /markets response missing markets list")
            for market in markets:
                if isinstance(market, dict):
                    yield market
            cursor_value = str(payload.get("cursor", "") or "")
            if not cursor_value:
                break
            cursor = cursor_value


def _ensure_leading_slash(path: str) -> str:
    stripped = str(path).strip()
    if not stripped:
        raise ValueError("Kalshi API path cannot be empty")
    if not stripped.startswith("/"):
        stripped = "/" + stripped
    return stripped


def _current_timestamp_ms() -> str:
    return str(int(time.time() * 1000))
