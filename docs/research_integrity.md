# Research Integrity Contract

Production probability scoring is pinned by `models/production_model_bundle.json`.
The manifest fixes the model, ordered features, calibration policy, split dates,
source hashes, and package versions. Loose artifact paths are migration-only and
must still agree exactly with model metadata.
The authoritative frozen-artifact inventory is
`outputs/research/baseline/champion_manifest_v2.json`; earlier manifests are
retained as immutable audit history and are not the active scoring authority.

Run the strict repository audit with:

```bash
python scripts/audit_repository.py --strict
```

The command creates a unique directory under `outputs/repository_audit/` with a
JSON report, CSV ledger, Markdown report, and artifact-status index. P0/P1
findings produce a nonzero exit code.

Historical backtests use `scripts/run_kalshi_backtest.py`. The runner requires
canonical settled Kalshi candles and model probabilities, selects thresholds on
2024 validation rows only, and evaluates 2025+ once. Signals use candle closes;
execution requires the first later candle open within five minutes. Settlement
must be a resolved Kalshi `result`. Whole-contract cash and exposure limits are
enforced while capital remains locked until settlement.

Closed-loop challenger research uses `scripts/run_research_wave.py` and
`scripts/assess_research_loop.py`. Experiments are immutable, registered in
SQLite, and grouped/weighted by whole target-day. The current proxy runner uses
official one-minute Kalshi candlesticks, first-later quotes within five minutes,
strictly prior folds, cross-fitted stacking/calibration, one contract, current
fees, one-tick base slippage, and doubled-fee/two-tick stresses. Inactive outer
days are retained as zero-return days for Sharpe, Sortino, Calmar, bootstrap,
and DSR calculations.

The intraday hurdle component models the same-feed five-minute maximum, then
convolves it with a strictly prior-day empirical reconciliation distribution to
the official Daily Climate Report settlement target. This prevents the
same-feed final maximum from silently substituting for settlement truth.

Runs without executable historical order-book depth are diagnostic and cannot
support profitability claims. Legacy backtest folders are retained but are not
validated evidence.

Forward depth capture is implemented by `scripts/collect_orderbook_depth.py`.
It stores raw payload hashes and exact fixed-point ladders, detects sequence
gaps, and restores state from a snapshot. It refuses to run unless the trading
configuration remains shadow-only. Missing credentials block depth collection,
not the historical research loop.

Canonical weather rebuilds stage CSVs beside their destinations and atomically
promote them after validation. Missing KNYC/NDFD history blocks the rebuild; no
Open-Meteo substitute is permitted for training. Truncated upstream METAR rows
are omitted and listed in `outputs/data_audit/blocked_source_ranges.csv`.
