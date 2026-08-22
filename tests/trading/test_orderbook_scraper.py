from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.trading.config import parse_trading_config
from src.trading.orderbook_scraper import (
    LEVELS_STORAGE_COLUMNS,
    SUMMARY_STORAGE_COLUMNS,
    OrderbookScraperError,
    parse_orderbook_scraper_settings,
    run_scrape_cycle,
    seconds_until_next_cycle,
)


FETCHED_AT = datetime(2026, 6, 3, 15, 0, tzinfo=timezone.utc)
TICKER = "KXHIGHNY-26JUN03-T72"

GOOD_BOOK = {
    "orderbook_fp": {
        "yes_dollars": [[0.40, 10], [0.35, 5]],
        "no_dollars": [[0.55, 7], [0.60, 2]],
    }
}

CROSSED_BOOK = {
    "orderbook_fp": {
        "yes_dollars": [[0.70, 5]],
        "no_dollars": [[0.60, 5]],
    }
}


class FakeKalshiClient:
    """Minimal stand-in for KalshiClient covering discovery + orderbooks."""

    def __init__(self, markets: list[dict], orderbooks: dict[str, dict]) -> None:
        self.markets = markets
        self.orderbooks = orderbooks
        self.orderbook_calls: list[str] = []

    def get(self, path, params=None, auth=False):
        if path == "/markets":
            return {"markets": self.markets, "cursor": ""}
        if path.startswith("/markets/") and path.endswith("/orderbook"):
            ticker = path.split("/")[2]
            self.orderbook_calls.append(ticker)
            return self.orderbooks[ticker]
        raise AssertionError(f"Unexpected path in fake client: {path}")

    def iter_markets(self, **kwargs):
        yield from self.markets


def _market(ticker: str = TICKER) -> dict:
    return {
        "ticker": ticker,
        "event_ticker": "KXHIGHNY-26JUN03",
        "status": "open",
        "title": "NYC daily high temperature on June 3",
        "subtitle": "NYC high temperature",
        "yes_sub_title": "High temperature at or above 72F",
        "no_sub_title": "High temperature below 72F",
        "close_time": "2026-06-03T20:00:00Z",
        "expiration_time": "2026-06-04T14:00:00Z",
    }


def _trading_config():
    return parse_trading_config(
        {
            "mode": "shadow",
            "kalshi": {"env": "demo"},
            "markets": {
                "series_tickers": {"NYC": ["KXHIGHNY"]},
                "location_terms": {"NYC": ["NYC"]},
                "weather_terms": ["temperature"],
            },
        }
    )


def _settings(tmp_path: Path, **overrides):
    raw = {
        "storage_dir": str(tmp_path / "store"),
        "request_pause_seconds": 0.0,
    }
    raw.update(overrides)
    return parse_orderbook_scraper_settings(raw)


def test_run_scrape_cycle_writes_append_only_store(tmp_path: Path) -> None:
    client = FakeKalshiClient([_market()], {TICKER: GOOD_BOOK})
    settings = _settings(tmp_path)

    result = run_scrape_cycle(
        client, settings, _trading_config(), now=FETCHED_AT
    )

    assert result.status == "OK"
    assert result.n_markets_scraped == 1
    assert client.orderbook_calls == [TICKER]

    levels_path = tmp_path / "store" / "orderbook_levels_202606.csv"
    summary_path = tmp_path / "store" / "orderbook_summary_202606.csv"
    assert result.levels_path == levels_path
    assert result.summary_path == summary_path

    levels = pd.read_csv(levels_path)
    assert list(levels.columns) == LEVELS_STORAGE_COLUMNS
    assert len(levels) > 0
    assert set(levels["event_ticker"]) == {"KXHIGHNY-26JUN03"}
    assert set(levels["eligible"]) == {True}

    summary = pd.read_csv(summary_path)
    assert list(summary.columns) == SUMMARY_STORAGE_COLUMNS
    assert len(summary) == 1
    assert summary.iloc[0]["orderbook_status"] == "OK"

    # Raw payloads stored for audit.
    raw_files = list((tmp_path / "store" / "raw").rglob("*.jsonl"))
    assert len(raw_files) == 1
    records = [json.loads(line) for line in raw_files[0].read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["ticker"] == TICKER
    assert records[0]["payload"] == GOOD_BOOK

    # Scrape log and watermark updated.
    log = pd.read_csv(tmp_path / "store" / "scrape_log.csv")
    assert len(log) == 1
    assert log.iloc[0]["status"] == "OK"
    watermark = json.loads(
        (tmp_path / "store" / "state" / "last_fetch_by_ticker.json").read_text()
    )
    assert watermark[TICKER] == FETCHED_AT.isoformat()


def test_second_cycle_appends_without_duplicates(tmp_path: Path) -> None:
    client = FakeKalshiClient([_market()], {TICKER: GOOD_BOOK})
    settings = _settings(tmp_path)

    first = run_scrape_cycle(client, settings, _trading_config(), now=FETCHED_AT)
    second_time = datetime(2026, 6, 3, 16, 0, tzinfo=timezone.utc)
    second = run_scrape_cycle(client, settings, _trading_config(), now=second_time)

    assert first.status == "OK"
    assert second.status == "OK"

    levels = pd.read_csv(second.levels_path)
    fetched_times = sorted(levels["fetched_at"].unique())
    assert len(fetched_times) == 2
    log = pd.read_csv(tmp_path / "store" / "scrape_log.csv")
    assert len(log) == 2


def test_failed_validation_quarantines_and_skips_canonical_store(tmp_path: Path) -> None:
    client = FakeKalshiClient([_market()], {TICKER: CROSSED_BOOK})
    settings = _settings(tmp_path)

    result = run_scrape_cycle(client, settings, _trading_config(), now=FETCHED_AT)

    assert result.status == "FAIL"
    checks = {violation.check for violation in result.report.violations}
    assert "crossed_book" in checks

    # Canonical partitions must not exist.
    assert not list((tmp_path / "store").glob("orderbook_levels_*.csv"))
    assert not list((tmp_path / "store").glob("orderbook_summary_*.csv"))

    # Quarantine holds the rejected snapshot.
    quarantine_dirs = list((tmp_path / "store" / "quarantine").iterdir())
    assert len(quarantine_dirs) == 1
    quarantined_levels = quarantine_dirs[0] / "orderbook_levels.csv"
    assert quarantined_levels.exists()

    log = pd.read_csv(tmp_path / "store" / "scrape_log.csv")
    assert log.iloc[0]["status"] == "FAIL"


def test_timestamp_regression_fails_closed_and_quarantines(tmp_path: Path) -> None:
    client = FakeKalshiClient([_market()], {TICKER: GOOD_BOOK})
    settings = _settings(tmp_path)

    first = run_scrape_cycle(client, settings, _trading_config(), now=FETCHED_AT)
    earlier = datetime(2026, 6, 3, 14, 0, tzinfo=timezone.utc)
    second = run_scrape_cycle(client, settings, _trading_config(), now=earlier)

    assert first.status == "OK"
    assert second.status == "FAIL"
    checks = {violation.check for violation in second.report.violations}
    assert "timestamp_regression" in checks

    # Only the first (valid) cycle's rows are in the canonical store.
    levels = pd.read_csv(first.levels_path)
    assert set(levels["fetched_at"]) == {FETCHED_AT.isoformat()}
    assert second.levels_path is None


def test_no_matching_markets_warns_and_writes_header_only_store(tmp_path: Path) -> None:
    client = FakeKalshiClient([], {})
    settings = _settings(tmp_path)

    result = run_scrape_cycle(client, settings, _trading_config(), now=FETCHED_AT)

    assert result.status == "WARN"
    assert result.n_markets_scraped == 0
    levels = pd.read_csv(result.levels_path)
    assert levels.empty
    assert list(levels.columns) == LEVELS_STORAGE_COLUMNS
    log = pd.read_csv(tmp_path / "store" / "scrape_log.csv")
    assert log.iloc[0]["status"] == "WARN"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    client = FakeKalshiClient([_market()], {TICKER: GOOD_BOOK})
    settings = _settings(tmp_path)

    result = run_scrape_cycle(
        client, settings, _trading_config(), now=FETCHED_AT, dry_run=True
    )

    assert result.status == "OK"
    assert result.levels_path is None
    assert not (tmp_path / "store").exists()


def test_parse_orderbook_scraper_settings_defaults() -> None:
    settings = parse_orderbook_scraper_settings({})
    assert settings.interval_minutes == 60
    assert settings.align_to_hour is True
    assert settings.orderbook_depth == 20
    assert settings.max_markets_per_cycle == 100


@pytest.mark.parametrize(
    "raw,field",
    [
        ({"interval_minutes": 0}, "interval_minutes"),
        ({"orderbook_depth": -1}, "orderbook_depth"),
        ({"max_markets_per_cycle": 0}, "max_markets_per_cycle"),
        ({"request_pause_seconds": -0.5}, "request_pause_seconds"),
    ],
)
def test_parse_orderbook_scraper_settings_rejects_invalid(raw, field) -> None:
    with pytest.raises(OrderbookScraperError):
        parse_orderbook_scraper_settings(raw)


def test_seconds_until_next_cycle_aligns_to_hour_boundary() -> None:
    settings = parse_orderbook_scraper_settings({"align_to_hour": True})
    now = datetime(2026, 6, 3, 12, 0, 30, tzinfo=timezone.utc)
    seconds = seconds_until_next_cycle(settings, now=now)
    # Next boundary is 13:00:00 (3570s away) plus a 2-second buffer.
    assert 3568 <= seconds <= 3573


def test_seconds_until_next_cycle_unaligned_uses_interval() -> None:
    settings = parse_orderbook_scraper_settings({"align_to_hour": False})
    now = datetime(2026, 6, 3, 12, 0, 30, tzinfo=timezone.utc)
    assert seconds_until_next_cycle(settings, now=now) == 3600.0