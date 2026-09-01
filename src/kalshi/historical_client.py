from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from src.trading.kalshi_client import KalshiClient, KalshiAPIError, KalshiClientError, TRANSIENT_STATUS_CODES

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
CITY_SERIES_CONFIG: dict[str, list[str]] = {
    "NYC": ["KXHIGHNY", "HIGHNY"],  # HIGHNY = legacy, KXHIGHNY = current
    "CHI": ["KXHIGHCHI", "HIGHCHI"],
    "MIA": ["KXHIGHMIA", "HIGHMIA"],
    "LAX": ["KXHIGHLAX", "HIGHLAX"],
    "SFO": ["KXHIGHTSFO", "HIGHTSFO"],
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _sleep_backoff(attempt: int, base: float, retry_after: float | None = None) -> None:
    if retry_after is not None and retry_after > 0:
        time.sleep(retry_after)
        return
    time.sleep(base * (2**attempt))


def _parse_retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class HistoricalCutoff:
    market_settled_ts: str | None
    trades_created_ts: str | None
    orders_updated_ts: str | None
    market_positions_last_updated_ts: str | None = None


class KalshiHistoricalClient(KalshiClient):
    """Extended Kalshi client with historical endpoints, pagination, and retry logic."""

    def get_historical_cutoff(self) -> dict[str, Any]:
        """GET /historical/cutoff"""
        return self.get("/historical/cutoff", auth=False)

    def iter_historical_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        tickers: str | None = None,
        limit: int = 1000,
        max_pages: int = 100,
        auth: bool = False,
    ) -> Iterable[dict[str, Any]]:
        """Yield historical markets with cursor pagination."""
        params: dict[str, Any] = {"limit": int(limit)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if tickers:
            params["tickers"] = tickers
        cursor: str | None = None
        for _ in range(int(max_pages)):
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._request_with_retry("GET", "/historical/markets", params=page_params, auth=auth)
            markets = payload.get("markets", [])
            if not isinstance(markets, list):
                raise KalshiClientError("Historical /markets missing markets list")
            for m in markets:
                if isinstance(m, dict):
                    yield m
            cursor = str(payload.get("cursor", "") or "")
            if not cursor:
                break

    def iter_events(
        self,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int = 1000,
        max_pages: int = 100,
        auth: bool = False,
        with_nested_markets: bool = False,
    ) -> Iterable[dict[str, Any]]:
        params: dict[str, Any] = {"limit": int(limit)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        cursor: str | None = None
        for _ in range(int(max_pages)):
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._request_with_retry("GET", "/events", params=page_params, auth=auth)
            events = payload.get("events", [])
            if not isinstance(events, list):
                raise KalshiClientError("Events response missing events list")
            for e in events:
                if isinstance(e, dict):
                    yield e
            cursor = str(payload.get("cursor", "") or "")
            if not cursor:
                break

    def get_historical_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        auth: bool = False,
    ) -> dict[str, Any]:
        """GET /historical/markets/{ticker}/candlesticks  (archived)"""
        return self._request_with_retry(
            "GET",
            f"/historical/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
            auth=auth,
        )

    def get_live_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        auth: bool = False,
    ) -> dict[str, Any]:
        """GET /series/{series}/markets/{ticker}/candlesticks  (live)"""
        return self._request_with_retry(
            "GET",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
            auth=auth,
        )

    def get_candlesticks_auto(
        self,
        ticker: str,
        series_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        cutoff_ts: datetime | None = None,
        auth: bool = False,
    ) -> dict[str, Any]:
        """Try historical then live, or route by cutoff."""
        # If we have a cutoff we can route: if settlement before cutoff -> historical
        # Otherwise try historical first, fall back to live on 404
        try:
            return self.get_historical_candlesticks(ticker, start_ts, end_ts, period_interval, auth=auth)
        except KalshiAPIError as exc:
            if exc.status_code == 404:
                logger.info("Historical candlesticks 404 for %s, trying live endpoint", ticker)
                return self.get_live_candlesticks(series_ticker, ticker, start_ts, end_ts, period_interval, auth=auth)
            raise

    def iter_historical_trades(
        self,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 1000,
        max_pages: int = 100,
        auth: bool = False,
    ) -> Iterable[dict[str, Any]]:
        params: dict[str, Any] = {"limit": int(limit)}
        if ticker:
            params["ticker"] = ticker
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        if max_ts is not None:
            params["max_ts"] = int(max_ts)
        cursor: str | None = None
        for _ in range(int(max_pages)):
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._request_with_retry("GET", "/historical/trades", params=page_params, auth=auth)
            trades = payload.get("trades", [])
            if not isinstance(trades, list):
                raise KalshiClientError("Historical trades missing trades list")
            for t in trades:
                if isinstance(t, dict):
                    yield t
            cursor = str(payload.get("cursor", "") or "")
            if not cursor:
                break

    def iter_live_trades(
        self,
        ticker: str | None = None,
        limit: int = 1000,
        max_pages: int = 20,
        auth: bool = False,
    ) -> Iterable[dict[str, Any]]:
        params: dict[str, Any] = {"limit": int(limit)}
        if ticker:
            params["ticker"] = ticker
        cursor: str | None = None
        for _ in range(int(max_pages)):
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            payload = self._request_with_retry("GET", "/markets/trades", params=page_params, auth=auth)
            trades = payload.get("trades", [])
            if not isinstance(trades, list):
                raise KalshiClientError("Live trades missing trades list")
            for t in trades:
                if isinstance(t, dict):
                    yield t
            cursor = str(payload.get("cursor", "") or "")
            if not cursor:
                break

    # --------------------------------------------------------
    # Robust request with rate-limit & exponential backoff
    # --------------------------------------------------------
    def _request_with_retry(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        json_payload: Mapping[str, Any] | None = None,
        auth: bool = False,
    ) -> dict[str, Any]:
        method_upper = method.upper()
        request_path = path if path.startswith("/") else "/" + path
        url = self.base_url + request_path
        last_exc: Exception | None = None
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
            if response.status_code == 429:
                retry_after = _parse_retry_after(response)
                logger.warning("Rate limited on %s %s (429), attempt %d/%d", method_upper, request_path, attempt, self.max_retries)
                if attempt >= self.max_retries:
                    raise KalshiAPIError(method_upper, request_path, response.status_code, response.text)
                _sleep_backoff(attempt, self.retry_backoff_seconds, retry_after)
                continue
            if response.status_code in TRANSIENT_STATUS_CODES:
                logger.warning("Transient error %d on %s %s attempt %d/%d", response.status_code, method_upper, request_path, attempt, self.max_retries)
                if attempt >= self.max_retries:
                    raise KalshiAPIError(method_upper, request_path, response.status_code, response.text)
                _sleep_backoff(attempt, self.retry_backoff_seconds)
                continue
            if response.status_code >= 400:
                raise KalshiAPIError(method_upper, request_path, response.status_code, response.text)
            if not response.content:
                return {}
            try:
                payload = response.json()
            except ValueError as exc:
                raise KalshiClientError(f"Non-JSON response {method_upper} {request_path}") from exc
            if not isinstance(payload, dict):
                raise KalshiClientError(f"Unexpected JSON payload {method_upper} {request_path}")
            return payload
        raise KalshiClientError(f"Failed after {self.max_retries} retries: {last_exc}")

    # --------------------------------------------------------
    # Discovery helpers
    # --------------------------------------------------------
    def discover_all_weather_events(
        self,
        cities: Iterable[str] | None = None,
        status: str | None = None,
        auth: bool = False,
    ) -> list[dict[str, Any]]:
        """Discover events for all configured cities automatically."""
        cities = list(cities or ["NYC"])
        all_events: list[dict[str, Any]] = []
        seen: set[str] = set()
        for city in cities:
            series_list = CITY_SERIES_CONFIG.get(city, [f"KXHIGH{city}"])
            for series in series_list:
                try:
                    for ev in self.iter_events(series_ticker=series, status=status, auth=auth):
                        ticker = str(ev.get("event_ticker") or ev.get("ticker") or "")
                        if ticker and ticker not in seen:
                            ev["_discovered_series"] = series
                            ev["_city"] = city
                            all_events.append(ev)
                            seen.add(ticker)
                except KalshiAPIError as exc:
                    logger.warning("Failed to fetch events for series %s: %s", series, exc)
        return all_events


def create_historical_client_from_env(
    base_url: str | None = None,
    timeout_seconds: float = 15.0,
    max_retries: int = 5,
    retry_backoff_seconds: float = 1.0,
) -> KalshiHistoricalClient:
    """Create client without auth (public historical data)."""
    from src.trading.config import load_trading_config

    try:
        config = load_trading_config()
        kalshi_cfg = config.kalshi
        resolved_base = base_url or kalshi_cfg.production_base_url
        timeout = kalshi_cfg.request_timeout_seconds
        retries = kalshi_cfg.max_retries
        backoff = kalshi_cfg.retry_backoff_seconds
    except Exception:
        resolved_base = base_url or "https://api.elections.kalshi.com/trade-api/v2"
        timeout = timeout_seconds
        retries = max_retries
        backoff = retry_backoff_seconds

    # Try to load credentials if available, but don't require them for public data
    api_key_id = None
    private_key = None
    try:
        from src.trading.secrets import load_kalshi_credentials

        creds = load_kalshi_credentials()
        api_key_id = creds.api_key_id
        private_key = creds.private_key
    except Exception:
        pass

    return KalshiHistoricalClient(
        base_url=resolved_base,
        api_key_id=api_key_id,
        private_key=private_key,
        timeout_seconds=timeout,
        max_retries=retries,
        retry_backoff_seconds=backoff,
    )
