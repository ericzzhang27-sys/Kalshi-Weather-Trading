"""Hourly Kalshi orderbook scraper for NYC daily-high temperature markets.

Systematically discovers open KXHIGHNY markets, downloads full order books,
validates data quality (fail closed), and appends snapshots to an
append-only store designed for backtesting:

    <storage_dir>/
        orderbook_levels_YYYYMM.csv     normalized book levels (append-only)
        orderbook_summary_YYYYMM.csv    per-ticker per-cycle summaries
        raw/YYYY/MM/DD/<cycle_id>.jsonl full API payloads (audit trail)
        scrape_log.csv                  one row per cycle
        state/last_fetch_by_ticker.json continuity watermark
        quarantine/<cycle_id>/          rejected cycles (FAIL status)
        latest_quality_report.md        human-readable report

Cycles whose validation fails are quarantined and never appended to the
canonical CSV partitions, keeping the backtesting store clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml

from src.trading.config import TradingConfig, load_trading_config
from src.trading.market_discovery import (
    DiscoveredMarket,
    discover_weather_markets,
    settings_from_config,
)
from src.trading.orderbook import (
    ORDERBOOK_COLUMNS,
    ORDERBOOK_SUMMARY_COLUMNS,
    normalize_orderbook,
    summarize_orderbook,
)
from src.trading.orderbook_quality import (
    QualityReport,
    QualityViolation,
    build_quality_report,
    check_timestamp_continuity,
    latest_fetched_at_by_ticker,
    validate_orderbook_levels,
    validate_orderbook_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRAPER_CONFIG_PATH = REPO_ROOT / "config" / "orderbook_scraper_config.yaml"

MARKET_META_COLUMNS = ["event_ticker", "close_time", "market_status", "eligible"]
LEVELS_STORAGE_COLUMNS = ORDERBOOK_COLUMNS + MARKET_META_COLUMNS
SUMMARY_STORAGE_COLUMNS = ORDERBOOK_SUMMARY_COLUMNS + MARKET_META_COLUMNS

SCRAPER_LOG_COLUMNS = [
    "cycle_id",
    "fetched_at",
    "status",
    "n_markets_matched",
    "n_markets_scraped",
    "n_level_rows",
    "n_summary_rows",
    "n_violations",
    "n_failures",
    "n_warnings",
    "violation_summary",
    "levels_path",
    "summary_path",
    "raw_path",
]


class OrderbookScraperError(RuntimeError):
    """Raised when scraper configuration or a cycle is invalid."""


@dataclass(frozen=True)
class OrderbookScraperSettings:
    storage_dir: Path = REPO_ROOT / "data" / "raw" / "kalshi_orderbooks"
    interval_minutes: int = 60
    align_to_hour: bool = True
    orderbook_depth: int = 20
    request_pause_seconds: float = 0.25
    include_ineligible_markets: bool = False
    max_markets_per_cycle: int = 100


@dataclass(frozen=True)
class ScrapeCycleResult:
    cycle_id: str
    fetched_at: str
    status: str
    n_markets_matched: int
    n_markets_scraped: int
    n_level_rows: int
    n_summary_rows: int
    report: QualityReport
    levels_path: Path | None
    summary_path: Path | None
    raw_path: Path | None


def load_orderbook_scraper_settings(
    path: str | Path = DEFAULT_SCRAPER_CONFIG_PATH,
) -> OrderbookScraperSettings:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Orderbook scraper config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise OrderbookScraperError("Orderbook scraper config must be a YAML mapping")
    return parse_orderbook_scraper_settings(raw)


def parse_orderbook_scraper_settings(raw: dict[str, Any]) -> OrderbookScraperSettings:
    storage_dir_value = raw.get("storage_dir", "data/raw/kalshi_orderbooks")
    storage_dir = Path(str(storage_dir_value))
    if not storage_dir.is_absolute():
        storage_dir = REPO_ROOT / storage_dir

    interval_minutes = int(raw.get("interval_minutes", 60))
    if interval_minutes < 1:
        raise OrderbookScraperError("interval_minutes must be positive")
    depth = int(raw.get("orderbook_depth", 20))
    if depth < 1:
        raise OrderbookScraperError("orderbook_depth must be positive")
    pause = float(raw.get("request_pause_seconds", 0.25))
    if pause < 0.0:
        raise OrderbookScraperError("request_pause_seconds must be nonnegative")
    max_markets = int(raw.get("max_markets_per_cycle", 100))
    if max_markets < 1:
        raise OrderbookScraperError("max_markets_per_cycle must be positive")

    return OrderbookScraperSettings(
        storage_dir=storage_dir,
        interval_minutes=interval_minutes,
        align_to_hour=bool(raw.get("align_to_hour", True)),
        orderbook_depth=depth,
        request_pause_seconds=pause,
        include_ineligible_markets=bool(raw.get("include_ineligible_markets", False)),
        max_markets_per_cycle=max_markets,
    )


def run_scrape_cycle(
    client: Any,
    scraper_settings: OrderbookScraperSettings,
    trading_config: TradingConfig,
    *,
    now: datetime | None = None,
    auth: bool = False,
    dry_run: bool = False,
) -> ScrapeCycleResult:
    """Run one discovery -> fetch -> validate -> store cycle."""
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    cycle_id = now_dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fetched_at_iso = now_dt.astimezone(timezone.utc).isoformat()

    discovery_settings = settings_from_config(trading_config)
    markets = discover_weather_markets(client, discovery_settings, fetched_at=now_dt, auth=auth)

    selected = [market for market in markets if market.eligible]
    if scraper_settings.include_ineligible_markets:
        eligible_tickers = {market.ticker for market in selected}
        selected.extend(
            market for market in markets if market.ticker not in eligible_tickers
        )
    selected = selected[: scraper_settings.max_markets_per_cycle]

    levels_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    raw_records: list[dict[str, Any]] = []

    for index, market in enumerate(selected):
        if index > 0 and scraper_settings.request_pause_seconds > 0:
            time.sleep(scraper_settings.request_pause_seconds)
        payload = client.get(
            f"/markets/{market.ticker}/orderbook",
            params={"depth": int(scraper_settings.orderbook_depth)},
            auth=auth,
        )
        levels = normalize_orderbook(market.ticker, payload, fetched_at=now_dt)
        summary = summarize_orderbook(levels, market.ticker, fetched_at=now_dt)
        levels_frames.append(_attach_market_meta(levels, market))
        summary_frames.append(_attach_market_meta(summary, market))
        raw_records.append(
            {
                "fetched_at": fetched_at_iso,
                "cycle_id": cycle_id,
                "ticker": market.ticker,
                "event_ticker": market.event_ticker,
                "depth": int(scraper_settings.orderbook_depth),
                "payload": payload,
            }
        )

    levels_df = _combine(levels_frames, LEVELS_STORAGE_COLUMNS)
    summary_df = _combine(summary_frames, SUMMARY_STORAGE_COLUMNS)

    violations: list[QualityViolation] = []
    violations.extend(validate_orderbook_levels(levels_df))
    violations.extend(validate_orderbook_summary(summary_df))

    previous_watermark = {} if dry_run else _load_watermark(scraper_settings.storage_dir)
    violations.extend(check_timestamp_continuity(previous_watermark, levels_df))
    report = build_quality_report(violations)

    result = ScrapeCycleResult(
        cycle_id=cycle_id,
        fetched_at=fetched_at_iso,
        status=report.status,
        n_markets_matched=len(markets),
        n_markets_scraped=len(selected),
        n_level_rows=len(levels_df),
        n_summary_rows=len(summary_df),
        report=report,
        levels_path=None,
        summary_path=None,
        raw_path=None,
    )

    if dry_run:
        return result

    storage_dir = scraper_settings.storage_dir
    if not report.ok:
        quarantine_dir = storage_dir / "quarantine" / cycle_id
        _write_raw_records(raw_records, quarantine_dir / f"{cycle_id}.jsonl")
        if not levels_df.empty:
            levels_df.to_csv(quarantine_dir / "orderbook_levels.csv", index=False)
        if not summary_df.empty:
            summary_df.to_csv(quarantine_dir / "orderbook_summary.csv", index=False)
        _write_scrape_log_row(storage_dir, result, paths=(None, None, None))
        _write_quality_report(storage_dir, result)
        return result

    month_partition = now_dt.astimezone(timezone.utc).strftime("%Y%m")
    levels_path = storage_dir / f"orderbook_levels_{month_partition}.csv"
    summary_path = storage_dir / f"orderbook_summary_{month_partition}.csv"
    raw_path = storage_dir / "raw" / now_dt.strftime("%Y/%m/%d") / f"{cycle_id}.jsonl"

    _append_csv(levels_path, levels_df, LEVELS_STORAGE_COLUMNS)
    _append_csv(summary_path, summary_df, SUMMARY_STORAGE_COLUMNS)
    _write_raw_records(raw_records, raw_path)
    _update_watermark(storage_dir, latest_fetched_at_by_ticker(levels_df))
    _write_scrape_log_row(storage_dir, result, paths=(levels_path, summary_path, raw_path))
    _write_quality_report(storage_dir, result)

    return ScrapeCycleResult(
        cycle_id=result.cycle_id,
        fetched_at=result.fetched_at,
        status=result.status,
        n_markets_matched=result.n_markets_matched,
        n_markets_scraped=result.n_markets_scraped,
        n_level_rows=result.n_level_rows,
        n_summary_rows=result.n_summary_rows,
        report=result.report,
        levels_path=levels_path,
        summary_path=summary_path,
        raw_path=raw_path,
    )


def log_cycle_error(
    scraper_settings: OrderbookScraperSettings,
    cycle_id: str,
    error: Exception,
) -> None:
    """Record an unexpected cycle failure in the scrape log without crashing."""
    row = {
        "cycle_id": cycle_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "status": "ERROR",
        "n_markets_matched": 0,
        "n_markets_scraped": 0,
        "n_level_rows": 0,
        "n_summary_rows": 0,
        "n_violations": 0,
        "n_failures": 0,
        "n_warnings": 0,
        "violation_summary": f"{type(error).__name__}: {error}",
        "levels_path": "",
        "summary_path": "",
        "raw_path": "",
    }
    frame = pd.DataFrame([row], columns=SCRAPER_LOG_COLUMNS)
    path = scraper_settings.storage_dir / "scrape_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def seconds_until_next_cycle(
    settings: OrderbookScraperSettings,
    now: datetime | None = None,
) -> float:
    """Seconds to sleep before the next scrape cycle."""
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    if not settings.align_to_hour:
        return float(settings.interval_minutes * 60)
    interval_seconds = settings.interval_minutes * 60
    epoch = now_dt.timestamp()
    next_boundary = (int(epoch) // interval_seconds + 1) * interval_seconds
    return max(0.0, next_boundary - epoch) + 2.0  # small buffer past the boundary


def _attach_market_meta(frame: pd.DataFrame, market: DiscoveredMarket) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=LEVELS_STORAGE_COLUMNS if "price_dollars" in getattr(
            frame, "columns", []
        ) else SUMMARY_STORAGE_COLUMNS)
    enriched = frame.copy()
    enriched["event_ticker"] = market.event_ticker
    enriched["close_time"] = market.close_time
    enriched["market_status"] = market.status
    enriched["eligible"] = bool(market.eligible)
    return enriched


def _combine(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.reindex(columns=columns)


def _append_csv(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.reindex(columns=columns) if not frame.empty else pd.DataFrame(columns=columns)
    payload.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_raw_records(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _watermark_path(storage_dir: Path) -> Path:
    return storage_dir / "state" / "last_fetch_by_ticker.json"


def _load_watermark(storage_dir: Path) -> dict[str, str]:
    path = _watermark_path(storage_dir)
    if not path.exists():
        # Fall back to scanning existing monthly partitions so continuity is
        # still enforced after state loss.
        return _scan_watermark_from_partitions(storage_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {str(key): str(value) for key, value in data.items()} if isinstance(data, dict) else {}


def _scan_watermark_from_partitions(storage_dir: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    for path in sorted(storage_dir.glob("orderbook_levels_*.csv")):
        try:
            frame = pd.read_csv(path, usecols=["ticker", "fetched_at"])
        except (OSError, ValueError):
            continue
        for ticker, value in latest_fetched_at_by_ticker(frame).items():
            current = latest.get(ticker)
            if current is None or str(value) > current:
                latest[ticker] = str(value)
    return latest


def _update_watermark(storage_dir: Path, latest_by_ticker: dict[str, str]) -> None:
    if not latest_by_ticker:
        return
    path = _watermark_path(storage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_watermark(storage_dir)
    merged.update(latest_by_ticker)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")


def _write_scrape_log_row(
    storage_dir: Path,
    result: ScrapeCycleResult,
    paths: tuple[Path | None, Path | None, Path | None],
) -> None:
    levels_path, summary_path, raw_path = paths
    violation_counts: dict[str, int] = {}
    for violation in result.report.violations:
        violation_counts[violation.check] = violation_counts.get(violation.check, 0) + 1
    row = {
        "cycle_id": result.cycle_id,
        "fetched_at": result.fetched_at,
        "status": result.status,
        "n_markets_matched": result.n_markets_matched,
        "n_markets_scraped": result.n_markets_scraped,
        "n_level_rows": result.n_level_rows,
        "n_summary_rows": result.n_summary_rows,
        "n_violations": len(result.report.violations),
        "n_failures": result.report.fail_count,
        "n_warnings": result.report.warn_count,
        "violation_summary": json.dumps(violation_counts, sort_keys=True),
        "levels_path": str(levels_path) if levels_path else "",
        "summary_path": str(summary_path) if summary_path else "",
        "raw_path": str(raw_path) if raw_path else "",
    }
    frame = pd.DataFrame([row], columns=SCRAPER_LOG_COLUMNS)
    path = storage_dir / "scrape_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_quality_report(storage_dir: Path, result: ScrapeCycleResult) -> None:
    lines = [
        "# Orderbook Scraper Quality Report",
        "",
        f"- Cycle ID: `{result.cycle_id}`",
        f"- Fetched at: `{result.fetched_at}`",
        f"- Status: **{result.status}**",
        f"- Markets matched: {result.n_markets_matched}",
        f"- Markets scraped: {result.n_markets_scraped}",
        f"- Level rows: {result.n_level_rows}",
        f"- Summary rows: {result.n_summary_rows}",
        f"- Violations: {len(result.report.violations)} "
        f"({result.report.fail_count} FAIL / {result.report.warn_count} WARN)",
        "",
    ]
    if result.report.violations:
        lines.append("| Severity | Check | Detail |")
        lines.append("|---|---|---|")
        for violation in result.report.violations:
            detail = violation.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {violation.severity} | {violation.check} | {detail} |")
    else:
        lines.append("No violations detected.")
    path = storage_dir / "latest_quality_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_SCRAPER_CONFIG_PATH",
    "LEVELS_STORAGE_COLUMNS",
    "OrderbookScraperError",
    "OrderbookScraperSettings",
    "SCRAPER_LOG_COLUMNS",
    "ScrapeCycleResult",
    "SUMMARY_STORAGE_COLUMNS",
    "load_orderbook_scraper_settings",
    "log_cycle_error",
    "parse_orderbook_scraper_settings",
    "run_scrape_cycle",
    "seconds_until_next_cycle",
]