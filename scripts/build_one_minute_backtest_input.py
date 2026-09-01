from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.align_probabilities import align_probabilities_with_markets  # noqa: E402
from src.kalshi.normalize_markets import normalize_historical_files  # noqa: E402
from src.research.registry import sha256_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an immutable one-minute proxy backtest input")
    parser.add_argument("--markets", type=Path, default=REPO_ROOT / "data/kalshi/processed/historical_markets_processed.csv")
    parser.add_argument("--candles", type=Path, default=REPO_ROOT / "data/kalshi/processed/historical_candles_1m_processed.csv")
    parser.add_argument("--probabilities", type=Path, default=REPO_ROOT / "outputs/ngboost_bucket_probabilities_calibrated.csv")
    parser.add_argument("--canonical", type=Path, default=None, help="Reuse an already normalized immutable canonical Parquet file")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/backtest_inputs")
    args = parser.parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-1m-" + uuid.uuid4().hex[:8]
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    canonical_path = output_dir / "canonical_markets_1m.parquet"
    if args.canonical:
        canonical_path = args.canonical
        normalization = {
            "output_path": str(canonical_path.resolve()),
            "output_sha256": sha256_file(canonical_path),
            "reused_immutable_canonical": True,
        }
    else:
        normalization = normalize_historical_files(args.markets, args.candles, canonical_path)
    canonical = pd.read_parquet(canonical_path)
    probabilities = pd.read_parquet(args.probabilities) if args.probabilities.suffix.lower() == ".parquet" else pd.read_csv(args.probabilities)
    aligned = align_probabilities_with_markets(probabilities, canonical, city="NYC")
    if aligned.empty:
        raise ValueError("one-minute causal alignment produced no rows")
    aligned_path = output_dir / "aligned_rows_1m.parquet"
    aligned.to_parquet(aligned_path, index=False, compression="zstd")
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalization": normalization,
        "probabilities_path": str(args.probabilities.resolve()),
        "probabilities_sha256": sha256_file(args.probabilities),
        "aligned_path": str(aligned_path.resolve()),
        "aligned_sha256": sha256_file(aligned_path),
        "aligned_rows": int(len(aligned)),
        "event_days": int(pd.to_datetime(aligned["target_date"]).dt.date.nunique()),
        "evidence_label": "historical_proxy_input_one_minute",
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
