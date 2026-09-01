from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trading.config import load_trading_config  # noqa: E402
from src.trading.orderbook_stream import (  # noqa: E402
    DEMO_WS_URL,
    PRODUCTION_WS_URL,
    OrderbookDepthStore,
    ShadowOrderbookStream,
)
from src.trading.secrets import load_private_key, load_trading_secrets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shadow-only Kalshi order-book depth recorder")
    parser.add_argument("tickers", nargs="+", help="Exact Kalshi market tickers")
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/live_trading/orderbook_stream.sqlite")
    args = parser.parse_args(argv)
    config = load_trading_config()
    if config.mode != "shadow" or config.trading_enabled or config.live_auto_enabled:
        raise RuntimeError("depth collection requires shadow mode with all live trading disabled")
    secrets = load_trading_secrets(require_private_key=True)
    private_key = load_private_key(secrets.private_key_path)
    ws_url = PRODUCTION_WS_URL if secrets.kalshi_env == "production" else DEMO_WS_URL
    stream = ShadowOrderbookStream(
        ws_url=ws_url,
        api_key_id=str(secrets.api_key_id),
        private_key=private_key,
        market_tickers=args.tickers,
        store=OrderbookDepthStore(args.output),
    )
    asyncio.run(stream.run(duration_seconds=args.duration_seconds))
    print(f"Shadow depth recorded to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
