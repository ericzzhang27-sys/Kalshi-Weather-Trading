from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.trading.config import MarketSettings, TradingConfig
from src.trading.kalshi_client import KalshiClient


MARKET_DISCOVERY_COLUMNS = [
    "fetched_at",
    "eligible",
    "rejection_reason",
    "location",
    "market_type",
    "ticker",
    "event_ticker",
    "status",
    "title",
    "subtitle",
    "yes_sub_title",
    "no_sub_title",
    "close_time",
    "expiration_time",
    "expected_expiration_time",
    "latest_expiration_time",
    "settlement_timer_seconds",
    "last_price_dollars",
    "previous_price_dollars",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "no_bid_dollars",
    "no_ask_dollars",
    "volume_fp",
    "volume_24h_fp",
    "open_interest_fp",
    "liquidity_dollars",
    "rules_primary",
    "rules_secondary",
    "price_level_structure",
    "floor_strike",
    "cap_strike",
    "functional_strike",
    "raw_market_json",
]


@dataclass(frozen=True)
class MarketDiscoverySettings:
    location: str
    market_type: str
    status: str
    tradable_statuses: tuple[str, ...]
    min_minutes_to_close: int
    page_limit: int
    max_pages: int
    series_tickers: tuple[str, ...]
    location_terms: tuple[str, ...]
    weather_terms: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveredMarket:
    fetched_at: str
    eligible: bool
    rejection_reason: str
    location: str
    market_type: str
    ticker: str
    event_ticker: str
    status: str
    title: str
    subtitle: str
    yes_sub_title: str
    no_sub_title: str
    close_time: str
    expiration_time: str
    expected_expiration_time: str
    latest_expiration_time: str
    settlement_timer_seconds: Any
    last_price_dollars: Any
    previous_price_dollars: Any
    yes_bid_dollars: Any
    yes_ask_dollars: Any
    no_bid_dollars: Any
    no_ask_dollars: Any
    volume_fp: Any
    volume_24h_fp: Any
    open_interest_fp: Any
    liquidity_dollars: Any
    rules_primary: str
    rules_secondary: str
    price_level_structure: str
    floor_strike: Any
    cap_strike: Any
    functional_strike: str
    raw_market_json: str


def settings_from_config(
    config: TradingConfig,
    location: str | None = None,
    market_type: str | None = None,
    series_tickers: Iterable[str] | None = None,
) -> MarketDiscoverySettings:
    market_settings = config.markets
    selected_location = location or market_settings.default_location
    selected_market_type = market_type or market_settings.target_market_type
    if selected_location not in market_settings.supported_locations:
        raise ValueError(f"Unsupported market discovery location: {selected_location}")
    if selected_market_type not in market_settings.supported_market_types:
        raise ValueError(f"Unsupported market type: {selected_market_type}")

    configured_series = market_settings.series_tickers.get(selected_location, ())
    selected_series = tuple(series_tickers) if series_tickers is not None else configured_series
    location_terms = market_settings.location_terms.get(selected_location, (selected_location,))
    return MarketDiscoverySettings(
        location=selected_location,
        market_type=selected_market_type,
        status=market_settings.status,
        tradable_statuses=tuple(status.lower() for status in market_settings.tradable_statuses),
        min_minutes_to_close=market_settings.min_minutes_to_close,
        page_limit=market_settings.page_limit,
        max_pages=market_settings.max_pages,
        series_tickers=tuple(item for item in selected_series if item),
        location_terms=tuple(location_terms),
        weather_terms=tuple(market_settings.weather_terms),
    )


def discover_weather_markets(
    client: KalshiClient,
    settings: MarketDiscoverySettings,
    fetched_at: datetime | None = None,
    auth: bool = False,
) -> list[DiscoveredMarket]:
    fetched_at_dt = fetched_at or datetime.now(timezone.utc)
    raw_markets = _fetch_raw_markets(client, settings, auth=auth)
    discovered: list[DiscoveredMarket] = []
    seen_tickers: set[str] = set()

    for market in raw_markets:
        ticker = _string_field(market, "ticker")
        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        if not _matches_terms(market, settings.location_terms):
            continue
        if not _matches_terms(market, settings.weather_terms):
            continue
        eligible, reason = _eligibility(market, settings, fetched_at_dt)
        discovered.append(
            _normalize_market(
                market=market,
                settings=settings,
                fetched_at=fetched_at_dt,
                eligible=eligible,
                rejection_reason=reason,
            )
        )
    return discovered


def save_market_discovery_snapshot(
    markets: list[DiscoveredMarket],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(market) for market in markets])
    if frame.empty:
        frame = pd.DataFrame(columns=MARKET_DISCOVERY_COLUMNS)
    frame = frame.reindex(columns=MARKET_DISCOVERY_COLUMNS)
    frame.to_csv(path, index=False)


def save_raw_market_payload(
    markets: list[DiscoveredMarket],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = []
    for market in markets:
        try:
            raw.append(json.loads(market.raw_market_json))
        except json.JSONDecodeError:
            raw.append({"raw_market_json": market.raw_market_json})
    path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")


def _fetch_raw_markets(
    client: KalshiClient,
    settings: MarketDiscoverySettings,
    auth: bool = False,
) -> list[dict[str, Any]]:
    if settings.series_tickers:
        markets: list[dict[str, Any]] = []
        for series_ticker in settings.series_tickers:
            markets.extend(
                client.iter_markets(
                    status=settings.status,
                    series_ticker=series_ticker,
                    limit=settings.page_limit,
                    max_pages=settings.max_pages,
                    auth=auth,
                )
            )
        return markets
    return list(
        client.iter_markets(
            status=settings.status,
            limit=settings.page_limit,
            max_pages=settings.max_pages,
            auth=auth,
        )
    )


def _normalize_market(
    market: dict[str, Any],
    settings: MarketDiscoverySettings,
    fetched_at: datetime,
    eligible: bool,
    rejection_reason: str,
) -> DiscoveredMarket:
    return DiscoveredMarket(
        fetched_at=fetched_at.isoformat(),
        eligible=eligible,
        rejection_reason=rejection_reason,
        location=settings.location,
        market_type=settings.market_type,
        ticker=_string_field(market, "ticker"),
        event_ticker=_string_field(market, "event_ticker"),
        status=_string_field(market, "status"),
        title=_string_field(market, "title"),
        subtitle=_string_field(market, "subtitle"),
        yes_sub_title=_string_field(market, "yes_sub_title"),
        no_sub_title=_string_field(market, "no_sub_title"),
        close_time=_string_field(market, "close_time"),
        expiration_time=_string_field(market, "expiration_time"),
        expected_expiration_time=_string_field(market, "expected_expiration_time"),
        latest_expiration_time=_string_field(market, "latest_expiration_time"),
        settlement_timer_seconds=market.get("settlement_timer_seconds"),
        last_price_dollars=market.get("last_price_dollars"),
        previous_price_dollars=market.get("previous_price_dollars"),
        yes_bid_dollars=market.get("yes_bid_dollars"),
        yes_ask_dollars=market.get("yes_ask_dollars"),
        no_bid_dollars=market.get("no_bid_dollars"),
        no_ask_dollars=market.get("no_ask_dollars"),
        volume_fp=market.get("volume_fp"),
        volume_24h_fp=market.get("volume_24h_fp"),
        open_interest_fp=market.get("open_interest_fp"),
        liquidity_dollars=market.get("liquidity_dollars"),
        rules_primary=_string_field(market, "rules_primary"),
        rules_secondary=_string_field(market, "rules_secondary"),
        price_level_structure=_string_field(market, "price_level_structure"),
        floor_strike=market.get("floor_strike"),
        cap_strike=market.get("cap_strike"),
        functional_strike=_string_field(market, "functional_strike"),
        raw_market_json=json.dumps(market, sort_keys=True),
    )


def _eligibility(
    market: dict[str, Any],
    settings: MarketDiscoverySettings,
    now: datetime,
) -> tuple[bool, str]:
    status = _string_field(market, "status").lower()
    if status and status not in settings.tradable_statuses:
        return False, f"status_is_{status}"

    close_time_text = _string_field(market, "close_time")
    if not close_time_text:
        return False, "missing_close_time"
    close_time = _parse_iso_datetime(close_time_text)
    if close_time is None:
        return False, "invalid_close_time"
    minutes_to_close = (close_time - now).total_seconds() / 60.0
    if minutes_to_close <= settings.min_minutes_to_close:
        return False, "too_close_to_close"
    return True, ""


def _matches_terms(market: dict[str, Any], terms: tuple[str, ...]) -> bool:
    if not terms:
        return True
    haystack = _market_search_text(market).lower()
    return any(term.lower() in haystack for term in terms if term)


def _market_search_text(market: dict[str, Any]) -> str:
    fields = [
        "ticker",
        "event_ticker",
        "title",
        "subtitle",
        "yes_sub_title",
        "no_sub_title",
        "rules_primary",
        "rules_secondary",
        "functional_strike",
    ]
    return " ".join(_string_field(market, field) for field in fields)


def _string_field(market: dict[str, Any], field: str) -> str:
    value = market.get(field, "")
    if value is None:
        return ""
    return str(value)


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
