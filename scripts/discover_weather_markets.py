from __future__ import annotations

import argparse
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trading.config import DEFAULT_TRADING_CONFIG_PATH, load_trading_config  # noqa: E402
from src.trading.kalshi_client import KalshiClient  # noqa: E402
from src.trading.market_discovery import (  # noqa: E402
    discover_weather_markets,
    save_market_discovery_snapshot,
    save_raw_market_payload,
    settings_from_config,
)
from src.trading.secrets import load_private_key, load_trading_secrets  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover read-only Kalshi weather markets and save an auditable snapshot."
    )
    parser.add_argument("--config-path", default=str(DEFAULT_TRADING_CONFIG_PATH))
    parser.add_argument("--location", default=None)
    parser.add_argument("--market-type", default=None)
    parser.add_argument(
        "--series-ticker",
        action="append",
        default=None,
        help="Optional Kalshi series ticker filter. Repeat for multiple series.",
    )
    parser.add_argument(
        "--kalshi-env",
        choices=["demo", "production"],
        default=None,
        help="Override config Kalshi environment.",
    )
    parser.add_argument("--base-url", default=None, help="Override Kalshi API base URL.")
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Use signed authenticated GET requests. Public market data is used by default.",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--raw-output-path", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_trading_config(args.config_path)
    settings = settings_from_config(
        config,
        location=args.location,
        market_type=args.market_type,
        series_tickers=args.series_ticker,
    )

    base_url = args.base_url or _base_url_for_args(config, args.kalshi_env)
    secrets = load_trading_secrets(require_private_key=args.auth)
    private_key = load_private_key(secrets.private_key_path) if args.auth else None
    client = KalshiClient(
        base_url=base_url,
        api_key_id=secrets.api_key_id,
        private_key=private_key,
        timeout_seconds=config.kalshi.request_timeout_seconds,
        max_retries=config.kalshi.max_retries,
        retry_backoff_seconds=config.kalshi.retry_backoff_seconds,
    )

    markets = discover_weather_markets(client, settings, auth=args.auth)
    output_path = Path(args.output_path) if args.output_path else config.outputs.market_discovery_snapshot_path
    raw_output_path = (
        Path(args.raw_output_path) if args.raw_output_path else config.outputs.market_discovery_raw_path
    )
    save_market_discovery_snapshot(markets, output_path)
    save_raw_market_payload(markets, raw_output_path)

    eligible_count = sum(1 for market in markets if market.eligible)
    print(
        "Saved Kalshi weather market discovery snapshot: "
        f"{len(markets)} matched markets, {eligible_count} eligible."
    )
    print(f"CSV: {output_path}")
    print(f"Raw JSON: {raw_output_path}")


def _base_url_for_args(config, env_override: str | None) -> str:
    if env_override is None:
        return config.kalshi.base_url
    if env_override == "demo":
        return config.kalshi.demo_base_url
    if env_override == "production":
        return config.kalshi.production_base_url
    raise ValueError(f"Unsupported Kalshi environment: {env_override}")


if __name__ == "__main__":
    main(sys.argv[1:])
