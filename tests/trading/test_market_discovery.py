from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.trading.market_discovery import (
    MarketDiscoverySettings,
    discover_weather_markets,
    save_market_discovery_snapshot,
)


class FakeKalshiClient:
    def __init__(self, markets):
        self.markets = markets

    def iter_markets(self, **kwargs):
        yield from self.markets


def _settings() -> MarketDiscoverySettings:
    return MarketDiscoverySettings(
        location="NYC",
        market_type="daily_high_temperature",
        status="open",
        tradable_statuses=("open", "active"),
        min_minutes_to_close=30,
        page_limit=1000,
        max_pages=1,
        series_tickers=(),
        location_terms=("NYC", "New York"),
        weather_terms=("temperature", "weather"),
    )


def test_discovery_filters_for_location_weather_and_close_time() -> None:
    client = FakeKalshiClient(
        [
            {
                "ticker": "KXHIGHNY-26JUN02-B75",
                "event_ticker": "KXHIGHNY-26JUN02",
                "status": "active",
                "title": "NYC high temperature on June 2",
                "subtitle": "Temperature bucket",
                "close_time": "2026-06-02T23:00:00Z",
            },
            {
                "ticker": "UNRELATED",
                "status": "open",
                "title": "Will a sports team win?",
                "close_time": "2026-06-02T23:00:00Z",
            },
            {
                "ticker": "CLOSING",
                "status": "open",
                "title": "NYC high temperature on June 2",
                "close_time": "2026-06-02T12:10:00Z",
            },
        ]
    )

    markets = discover_weather_markets(
        client,
        _settings(),
        fetched_at=datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert [market.ticker for market in markets] == [
        "KXHIGHNY-26JUN02-B75",
        "CLOSING",
    ]
    assert markets[0].eligible is True
    assert markets[1].eligible is False
    assert markets[1].rejection_reason == "too_close_to_close"


def test_save_market_discovery_snapshot_writes_headers_for_empty_result(tmp_path: Path) -> None:
    output_path = tmp_path / "market_discovery_snapshot.csv"

    save_market_discovery_snapshot([], output_path)

    frame = pd.read_csv(output_path)
    assert "ticker" in frame.columns
    assert frame.empty
