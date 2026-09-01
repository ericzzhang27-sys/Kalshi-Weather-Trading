# AGENTS.md G�� Agent Working Guide

Instructions and skills for AI agents (and new engineers) working in this repository. Read this before making changes.

## Mission

Maintain a two-layer system for NYC daily high-temperature Kalshi markets:

1. **Probability engine** (`src/` modeling modules): NGBoost distributional model of `forecast_error = actual_high - forecast_high`, priced into bucket probabilities.
2. **Trading foundation** (`src/trading/`): fail-closed live loop in shadow/paper mode. No live orders by default.

## Non-Negotiable Rules

1. **Never enable live trading casually.** Keep `mode: shadow`, `trading_enabled: false`, `live_auto_enabled: false` in `config/trading_config.yaml` unless the user explicitly requests a change. Never commit credentials, private keys, or `.env`.
2. **Leakage safety is absolute.** No feature may use information after `prediction_time`. Forbidden columns: `actual_high`, `forecast_error`, final daily highs, settlement/market results, future observations, post-`prediction_time` forecast updates, any target-derived column. Run leakage checks when touching features.
3. **Fail closed in trading code.** Missing data, stale timestamps, ambiguous bucket mappings, or failed validations must produce `NO_TRADE` / an error G�� never a guessed value or order.
4. **Trust config over filenames.** Model artifact filenames are historical; `config/model_config.yaml` and metadata JSONs are authoritative.
5. **Preserve the frozen feature contract.** The selected model expects exactly the columns in `outputs/final_feature_list.json`, in order. Do not add/remove/rename features without retraining and re-validating.
6. **Do not loosen risk limits** (exposure caps, kill switch, denylists) without explicit user instruction.
7. **Run tests after code changes**: `pytest` from the repo root.

## Environment

- Windows 11, cmd shell; repo root: `Kalshi-Weather-Trading/` inside workspace `c:\Weather Trading`.
- Python with dependencies in `requirements.txt` (pandas, numpy, scikit-learn, scipy, ngboost, matplotlib, plotly, streamlit, PyYAML, requests, cryptography) and `requirements-dev.txt` (notebooks/tests).
- All commands run from the repo root: `cd "c:\Weather Trading\Kalshi-Weather-Trading"` first if needed.

## Core Skills & Commands

### Data production pipeline
Rebuild all processed data end-to-end:

```bash
python scripts/run_data_pipeline.py
```

Stages (each fails closed; report written to `outputs/reports/data_pipeline_run.md`):

| Stage | Script | Produces |
|---|---|---|
| 1. Clean + audit raw data | `scripts/run_day6_data_verification.py` | `data/processed/hourly_clean.csv`, `daily_clean.csv`, `forecasts_clean.csv`, audit reports |
| 2. Build supervised targets | `scripts/build_day7_supervised_table.py` | `supervised_forecast_error_rows.csv`, target summaries |
| 3. Build timestamp-safe features | `scripts/build_features.py` | `modeling_rows_v1.csv`, feature/leakage reports |
| 4. Verify feature integrity | `scripts/verify_feature_integrity.py` | provenance/integrity checks |

NDFD archive maintenance (only when refreshing historical forecasts):

```bash
python scripts/build_ndfd_daily_high_archive.py
python scripts/build_ndfd_point_forecasts.py
python scripts/build_ndfd_csv_safe.py
```

### Modeling

```bash
python -m src.train_ngboost                      # train configured NGBoost model
python scripts/train_robust_laplace_baseline.py  # fixed robust baseline
python -m src.distribution_pricing               # distributions -> bucket probabilities
python scripts/evaluate_ngboost.py               # NLL/Brier/coverage metrics
python scripts/calibrate_ngboost.py              # calibration diagnostics
```

Model search / ablation utilities exist under `scripts/` (`search_ngboost_model_space.py`, `run_focused_nll_brier_search.py`, `run_refined_skew_search.py`, `run_ngboost_feature_ablation.py`, `compare_ngboost_distributions.py`) G�� use only for research branches, not to silently replace the production artifact.

### Trading loop

```bash
python scripts/discover_weather_markets.py       # read-only market discovery
python scripts/run_orderbook_scraper.py          # hourly orderbook scraper (see docs/orderbook_scraper.md)
python scripts/run_live_trading_loop.py          # full shadow/paper cycle
streamlit run apps/live_trading_dashboard.py     # monitoring dashboard
```

Cycle outputs land in `outputs/live_trading/` per the path map in `config/trading_config.yaml`.
### Backtest reporting

After every backtest run, ALWAYS report the risk-adjusted metric triple
alongside headline P&L/drawdown numbers:

- **Sharpe ratio** (daily P&L, annualized)
- **Sortino ratio** (downside-deviation based)
- **Calmar ratio** (annualized return / |max drawdown|)

Reference implementation: `src/backtest/viz/stats.py` `summary_stats`
(Sharpe/Sortino); Calmar = `annualized_return / abs(max_drawdown)`.

### Tests

```bash
pytest
```

## Key Files to Read Before Task Types

| Task type | Read first |
|---|---|
| Feature changes | `src/features.py`, `src/leakage_checks.py`, `outputs/final_feature_list.json`, `CONTEXT.md` gotchas |
| Model changes | `config/model_config.yaml`, `src/distributional_model.py`, `src/train_ngboost.py`, `outputs/best_model_notes.md` |
| Pricing changes | `src/distribution_pricing.py`, `src/bucket_schema.py`, `src/error_boundaries.py` |
| Trading changes | `docs/trading/LIVE_TRADING_IMPLEMENTATION_PLAN.md`, `config/trading_config.yaml`, relevant `src/trading/*.py` |
| Data source changes | `docs/data_sources.md`, `config/model_config.yaml` `data_sources:` section |

## Code Conventions

- Scripts are standalone CLIs with `if __name__ == "__main__":`; they insert the repo root into `sys.path` and import from `src/`.
- Modules under `src/` are importable libraries; keep side effects out of imports.
- Timestamps: local America/New_York; temperatures Fahrenheit.
- Bucket bounds: lower-open, upper-closed; open-ended tails supported.
- Probability outputs must be validated (finite, nonnegative, rows sum to 1).
- Write human-readable reports (Markdown) alongside machine-readable CSV/JSON outputs.
- Prefer small, verifiable edits; run the relevant script or tests to confirm behavior before declaring done.

## Verification Checklist Before Completing Any Task

- [ ] Leakage rules respected (no post-`prediction_time` information in features).
- [ ] Trading safety flags unchanged unless explicitly requested.
- [ ] Relevant scripts/tests run successfully.
- [ ] Backtest runs report Sharpe, Sortino, and Calmar ratios.
- [ ] New outputs documented (README tables, `outputs/README.md`) if user-facing.
- [ ] No secrets committed; `.gitignore` still covers `.env`, keys, runtime overrides.
