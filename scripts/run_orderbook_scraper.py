"""Run the Kalshi orderbook scraper for NYC daily-high temperature markets.

Single cycle (default):

    python scripts/run_orderbook_scraper.py

Hourly loop, aligned to the top of the hour:

    python scripts/run_orderbook_scraper.py --loop

Dry run (fetch + validate, no writes):

    python scripts/run_orderbook_scraper.py --dry-run

Public market data is used by default; pass --auth to sign requests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trading.config import DEFAULT_TRADING_CONFIG_PATH, load_trading_config  # noqa: E402
from src.trading.kalshi_client import KalshiClient  # noqa: E402
from src.trading.orderbook_scraper import (  # noqa: E402
    DEFAULT_SCRAPER_CONFIG_PATH,
    log_cycle_error,
    parse_orderbook_scraper_settings,
    run_scrape_cycle,
    seconds_until_next_cycle,
)
from src.trading.secrets import load_private_key, load_trading_secrets  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape full order books for NYC daily-high Kalshi markets on an "
            "hourly cadence, validate data quality, and store snapshots for "
            "backtesting."
        )
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously, scraping once per interval (default: single cycle).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="With --loop, stop after this many cycles (0 = run forever).",
    )
    parser.add_argument("--config-path", default=str(DEFAULT_SCRAPER_CONFIG_PATH))
    parser.add_argument("--trading-config-path", default=str(DEFAULT_TRADING_CONFIG_PATH))
    parser.add_argument("--depth", type=int, default=None, help="Override order book depth.")
    parser.add_argument(
        "--include-ineligible",
        action="store_true",
        help="Also scrape markets that fail trading eligibility filters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate but write nothing to disk.",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Use signed authenticated GET requests. Public market data is used by default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scraper_settings = _scraper_settings_from_args(args)
    trading_config = load_trading_config(args.trading_config_path)

    secrets = load_trading_secrets(require_private_key=args.auth)
    private_key = load_private_key(secrets.private_key_path) if args.auth else None
    client = KalshiClient(
        base_url=trading_config.kalshi.base_url,
        api_key_id=secrets.api_key_id,
        private_key=private_key,
        timeout_seconds=trading_config.kalshi.request_timeout_seconds,
        max_retries=trading_config.kalshi.max_retries,
        retry_backoff_seconds=trading_config.kalshi.retry_backoff_seconds,
    )

    if not args.loop:
        return _run_single_cycle(client, scraper_settings, trading_config, args)
    return _run_loop(client, scraper_settings, trading_config, args)


def _run_single_cycle(client, settings, trading_config, args) -> int:
    result = run_scrape_cycle(
        client,
        settings,
        trading_config,
        auth=args.auth,
        dry_run=args.dry_run,
    )
    _print_result(result, dry_run=args.dry_run)
    return 0 if result.status != "FAIL" else 1


def _run_loop(client, settings, trading_config, args) -> int:
    completed = 0
    print(
        f"Starting orderbook scrape loop: interval={settings.interval_minutes}m, "
        f"align_to_hour={settings.align_to_hour}, max_cycles="
        f"{'unlimited' if args.max_cycles <= 0 else args.max_cycles}"
    )
    while args.max_cycles <= 0 or completed < args.max_cycles:
        cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            result = run_scrape_cycle(
                client,
                settings,
                trading_config,
                auth=args.auth,
                dry_run=args.dry_run,
            )
            _print_result(result, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 - a scraper must survive overnight
            traceback.print_exc()
            log_cycle_error(settings, cycle_id, exc)
        completed += 1
        if args.max_cycles > 0 and completed >= args.max_cycles:
            break
        sleep_seconds = seconds_until_next_cycle(settings)
        next_run = datetime.now(timezone.utc).timestamp() + sleep_seconds
        next_run_iso = datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat()
        print(f"Next cycle at {next_run_iso} (sleeping {sleep_seconds:.0f}s)")
        time.sleep(sleep_seconds)
    return 0


def _scraper_settings_from_args(args: argparse.Namespace):
    raw_overrides: dict[str, object] = {}
    if args.depth is not None:
        raw_overrides["orderbook_depth"] = args.depth
    if args.include_ineligible:
        raw_overrides["include_ineligible_markets"] = True

    config_path = Path(args.config_path)
    if config_path.exists():
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            raw.update(raw_overrides)
            return parse_orderbook_scraper_settings(raw)
    # No config file present: build settings purely from defaults + overrides.
    base = {
        "storage_dir": "data/raw/kalshi_orderbooks",
        "interval_minutes": 60,
        "align_to_hour": True,
        "orderbook_depth": 20,
        "request_pause_seconds": 0.25,
        "include_ineligible_markets": False,
        "max_markets_per_cycle": 100,
    }
    base.update(raw_overrides)
    return parse_orderbook_scraper_settings(base)


def _print_result(result, dry_run: bool = False) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    print(
        f"{prefix}Cycle {result.cycle_id}: status={result.status}, "
        f"markets matched={result.n_markets_matched}, scraped={result.n_markets_scraped}, "
        f"level rows={result.n_level_rows}, violations={len(result.report.violations)} "
        f"({result.report.fail_count} FAIL / {result.report.warn_count} WARN)"
    )
    for violation in result.report.violations:
        print(f"  {violation.describe()}")
    if result.levels_path:
        print(f"  levels: {result.levels_path}")
    if result.summary_path:
        print(f"  summary: {result.summary_path}")
    if result.raw_path:
        print(f"  raw payloads: {result.raw_path}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))