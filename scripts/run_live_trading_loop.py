from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trading.config import DEFAULT_TRADING_CONFIG_PATH, load_trading_config  # noqa: E402
from src.trading.kalshi_client import KalshiClient  # noqa: E402
from src.trading.live_loop import run_trading_cycle  # noqa: E402
from src.trading.secrets import load_private_key, load_trading_secrets  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Day 4 trading cycle through risk, intents, and paper broker."
    )
    parser.add_argument("--config-path", default=str(DEFAULT_TRADING_CONFIG_PATH))
    parser.add_argument("--event-ticker", default=None)
    parser.add_argument("--target-date", default=None, help="YYYY-MM-DD target date.")
    parser.add_argument(
        "--prediction-time",
        default=None,
        help="Optional local/aware ISO timestamp for reproducible scoring.",
    )
    parser.add_argument("--depth", type=int, default=20, help="Order-book depth.")
    parser.add_argument("--auth-market-data", action="store_true")
    parser.add_argument("--auth-orderbooks", action="store_true")
    parser.add_argument("--auth-portfolio", action="store_true")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Execute READY intents through the local paper broker even if config mode is shadow.",
    )
    parser.add_argument(
        "--no-paper",
        action="store_true",
        help="Do not execute the local paper broker even if config mode is paper.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.paper and args.no_paper:
        raise ValueError("--paper and --no-paper cannot both be supplied")

    config = load_trading_config(args.config_path)
    client = _client_for_args(config, args)
    paper_enabled = True if args.paper else False if args.no_paper else None
    result = run_trading_cycle(
        config,
        event_ticker=args.event_ticker,
        target_date=args.target_date,
        depth=args.depth,
        kalshi_client=client,
        prediction_time=datetime.fromisoformat(args.prediction_time) if args.prediction_time else None,
        auth_market_data=args.auth_market_data,
        auth_orderbooks=args.auth_orderbooks,
        auth_portfolio=args.auth_portfolio,
        paper_enabled=paper_enabled,
        write_outputs=True,
    )

    print("Completed Day 4 trading cycle.")
    print(f"Event: {result.dashboard_state.status.get('event_ticker', '')}")
    print(f"Dashboard status: {result.dashboard_state.status.get('dashboard_status', '')}")
    print(f"Edge rows: {len(result.dashboard_state.edge_table)}")
    print(f"Risk approved: {_count(result.risk_decisions, 'risk_status', 'APPROVED')}")
    print(f"Ready intents: {_count(result.order_intents, 'intent_status', 'READY')}")
    if result.paper_result is not None:
        print(
            "Paper broker: "
            f"{result.paper_result.filled_count} filled, "
            f"{result.paper_result.rejected_count} rejected."
        )
    else:
        print("Paper broker: skipped.")
    print(f"Risk decisions: {config.outputs.risk_decisions_path}")
    print(f"Order intents: {config.outputs.order_intents_path}")
    print(f"Cycle log: {config.outputs.trading_cycle_log_path}")


def _client_for_args(config, args: argparse.Namespace) -> KalshiClient | None:
    use_auth = args.auth_market_data or args.auth_orderbooks or args.auth_portfolio
    if not use_auth:
        return None
    secrets = load_trading_secrets(require_private_key=True)
    private_key = load_private_key(secrets.private_key_path)
    return KalshiClient(
        base_url=config.kalshi.base_url,
        api_key_id=secrets.api_key_id,
        private_key=private_key,
        timeout_seconds=config.kalshi.request_timeout_seconds,
        max_retries=config.kalshi.max_retries,
        retry_backoff_seconds=config.kalshi.retry_backoff_seconds,
    )


def _count(frame, column: str, value: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int((frame[column].astype(str) == value).sum())


if __name__ == "__main__":
    main(sys.argv[1:])
