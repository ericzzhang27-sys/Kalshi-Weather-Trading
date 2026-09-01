# Historical Kalshi Backtest — Real Data Only

End-to-end historical backtest over **real Kalshi hourly candles** (downloaded via the
authenticated API; the pipeline refuses synthetic data outright).

## Quick Start

```powershell
# From repo root Kalshi-Weather-Trading/
python scripts/run_kalshi_backtest.py --city NYC --strategy A --threshold 0.05
```

Requires `data/kalshi/processed/historical_markets_processed.csv` and
`historical_candles_processed.csv` (real API downloads with per-row raw JSON).
If missing, authenticate (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`) and run:

```powershell
python -m src.kalshi.download_weather_history --city NYC
```

## Hard Rules Enforced by Code

1. **No synthetic data.** `src/kalshi/synthetic.py` was removed. The runner verifies
   provenance (raw API JSON per row) and exits if absent.
2. **Settlements come from Kalshi's own `result` field** in the canonical markets —
   never from any prediction artifact. Prediction files have historically carried a
   corrupted `actual_high` column (wrong on 98.5% of dates pre-fix).
3. **Timestamp conventions** (the source of a past 4–5h lookahead bug):
   - Kalshi candle timestamps: **true UTC** (epoch from API)
   - Model `prediction_time`: **tz-naive America/New_York** — localized then converted
     in `align_probabilities._standardize_prob_df`; never parse it as UTC directly
   - `data/processed/hourly_clean.csv`: also tz-naive America/New_York
4. **Pre-settlement guard**: rows whose candle timestamp falls outside the target day
   (ET) are dropped at alignment and again rejected in the engine (defense in depth).
5. **Run manifest**: every run writes `run_manifest.json` (input SHA hashes, config,
   threshold sweep) into its output directory.

## Outputs

```
outputs/backtests/          # default real-data runs (manifest-stamped)
outputs/backtests_real/     # alternative runner (scripts/run_kalshi_backtest_real.py)
  trades.csv / trades.parquet   # transaction-level ledger
  summary.csv                   # base-threshold metrics
  strategy_comparison.csv       # thresholds 2%..15%
  oos_metrics.json              # chronological validation/test split
  robustness_checks.csv
  data_quality.json             # incl. post-settlement rows dropped
  breakdown_*.csv, calibration_*
  run_manifest.json             # input hashes + config for this exact run
```

## Verified Baseline (2026-08-25, NYC, strategy A, 1 contract, fees included)

| Threshold | Trades | Win rate | Net PnL | Max DD | PF |
|---|---|---|---|---|---|
| 2% | 734 | 18.0% | $19.44 | −$6.55 | 1.25 |
| 5% | 620 | 19.0% | $24.24 | −$5.07 | 1.38 |
| 15% | 380 | 15.8% | $11.31 | −$4.46 | 1.30 |

Real-data period Jun 2024 → May 2026. Thin but positive edge; treat as a baseline,
not a profitability claim.

## Regenerating Model Probabilities

After any retrain or data-pipeline rebuild, regenerate predictions before backtesting:

```powershell
python -m src.train_ngboost        # refresh params/predictions from modeling_rows_v1.csv
python scripts/calibrate_ngboost.py  # regenerate outputs/ngboost_bucket_probabilities_calibrated.csv
```

Then verify before running: the calibrated file's `actual_high` must match
`daily_clean.official_daily_high_f` for ~100% of dates.

## Visualization & Diagnostics (`src/backtest/viz/`)

```python
from src.backtest.viz import BacktestResult, create_backtest_report
result = BacktestResult.from_ledger(trades_df)
figures = create_backtest_report(result, output_path="outputs/backtests/report.html")
```

55 registered plots across performance/risk/trades/edge/execution/robustness;
optional inputs degrade gracefully. Comparison mode:
`create_comparison_report({"a": r1, "b": r2})`. Dashboard:
`streamlit run apps/backtest_dashboard.py`. Demo: `python -m src.backtest.viz.demo`
(only uses ledgers that exist).

## Key Implementation Notes

- **Fees:** `src/backtest/fees.py` — Kalshi taker fee `0.07·p·(1−p)` with ceiling rounding.
- **Execution:** entry at YES ask close (BUY_YES) or implied NO ask `1 − yes_bid`
  (BUY_NO); hold to settlement; missing/dead quotes skipped fail-closed.
- **Sizing:** fixed_contracts (default) / fixed_dollar / fractional Kelly (never default).
- **Trade policies:** `first_signal_only`, `one_position_per_market` (default),
  `one_trade_per_event`, `allow_reentry_after_exit`, `continuous_rebalancing`.
- **Data integrity:** `metrics.data_quality_report` + alignment-time leakage checks.
