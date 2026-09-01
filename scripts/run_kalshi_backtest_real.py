"""Compatibility entry point for the strict immutable real-data runner."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_kalshi_backtest import main as strict_main  # noqa: E402
from src.kalshi.normalize_markets import normalize_historical_markets  # noqa: E402


def main() -> int:
    processed = REPO_ROOT / "data/kalshi/processed"
    markets = pd.read_csv(processed / "historical_markets_processed.csv")
    candles = pd.read_csv(processed / "historical_candles_processed.csv")
    canonical = normalize_historical_markets(markets, candles)
    input_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    input_dir = REPO_ROOT / "outputs/backtest_inputs" / input_id
    input_dir.mkdir(parents=True, exist_ok=False)
    canonical_path = input_dir / "canonical_markets.csv"
    canonical.to_csv(canonical_path, index=False)
    probabilities = next(
        path for path in [
            REPO_ROOT / "outputs/ngboost_bucket_probabilities_calibrated.csv",
            REPO_ROOT / "outputs/final_bucket_probability_predictions.csv",
        ]
        if path.exists()
    )
    return strict_main([
        "--canonical-markets", str(canonical_path),
        "--probabilities", str(probabilities),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
