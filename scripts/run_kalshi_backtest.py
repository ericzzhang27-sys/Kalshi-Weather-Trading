from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.align_probabilities import align_probabilities_with_markets  # noqa: E402
from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.backtest.metrics import (  # noqa: E402
    ledger_metrics,
    season_aware_date_block_uncertainty,
    select_threshold_on_validation,
    untouched_test_rows,
)
from src.backtest.sizing import SizingConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal Kalshi historical backtest")
    parser.add_argument("--canonical-markets", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--city", default="NYC")
    parser.add_argument("--thresholds", default="0.02,0.03,0.05,0.075,0.10,0.15")
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--adverse-slippage-ticks", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/backtests")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output_dir = args.output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"Backtest run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    canonical = pd.read_csv(args.canonical_markets)
    probabilities = pd.read_csv(args.probabilities)
    aligned = align_probabilities_with_markets(probabilities, canonical, city=args.city)
    if aligned.empty:
        raise ValueError("Causal alignment produced no rows")
    sizing = SizingConfig(
        method="fixed_contracts", fixed_contracts=1,
        max_contracts_per_order=1, max_contracts_per_market=1,
    )
    base = BacktestConfig(
        execution_mode="next_candle_open",
        sizing=sizing,
        adverse_slippage_ticks=args.adverse_slippage_ticks,
    )
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    selected, selection_table = select_threshold_on_validation(
        aligned, thresholds, validation_start=args.validation_start,
        test_start=args.test_start, base_config=base,
    )
    final_config = BacktestConfig(
        threshold=selected,
        execution_mode=base.execution_mode,
        sizing=base.sizing,
        adverse_slippage_ticks=base.adverse_slippage_ticks,
    )
    test_ledger = run_backtest(
        untouched_test_rows(aligned, test_start=args.test_start), final_config
    )
    test_metrics = ledger_metrics(test_ledger)
    uncertainty = season_aware_date_block_uncertainty(test_ledger)

    aligned.to_csv(output_dir / "aligned_rows.csv", index=False)
    selection_table.to_csv(output_dir / "validation_threshold_selection.csv", index=False)
    test_ledger.to_csv(output_dir / "test_ledger.csv", index=False)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "test_uncertainty.json").write_text(
        json.dumps(uncertainty, indent=2, sort_keys=True), encoding="utf-8"
    )
    supported = bool(test_metrics["supports_profitability_claim"])
    manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "validated_executable_depth" if supported else
            "historical_proxy_validated" if not test_ledger.empty else
            "diagnostic_no_test_trades"
        ),
        "city": args.city,
        "inputs": {
            "canonical_markets": str(args.canonical_markets.resolve()),
            "canonical_markets_sha256": _sha256(args.canonical_markets),
            "probabilities": str(args.probabilities.resolve()),
            "probabilities_sha256": _sha256(args.probabilities),
            "model_bundle_sha256": _sha256(REPO_ROOT / "models/production_model_bundle.json"),
        },
        "split_policy": {
            "selection_split": "validation_only",
            "validation_start": args.validation_start,
            "test_start": args.test_start,
            "test_used_for_selection": False,
        },
        "execution": {
            "mode": "next_candle_open", "maximum_gap_minutes": 5,
            "adverse_slippage_ticks": args.adverse_slippage_ticks,
            "whole_contracts": True, "settlement_source": "kalshi_result",
            "historical_depth_required_for_profitability_claim": True,
            "historical_depth_available": supported,
            "evidence_label": "historical_proxy_validated",
        },
        "selected_threshold": selected,
        "test_metrics": test_metrics,
        "uncertainty": uncertainty,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Backtest run written to {output_dir}")
    print(f"Validation-selected threshold: {selected:.4f}; test trades: {len(test_ledger)}")
    print(
        "Test Sharpe/Sortino/Calmar: "
        f"{test_metrics['sharpe_ratio']}/{test_metrics['sortino_ratio']}/{test_metrics['calmar_ratio']}"
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
