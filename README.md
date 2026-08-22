# Kalshi Weather Trading

A research and trading system for Kalshi-style weather bucket markets, focused on New York City daily high-temperature contracts (`KXHIGHNY`).

The repository has two layers:

1. **Probability engine (research layer).** Estimates calibrated probabilities for final-temperature buckets from an NGBoost distributional model of NWS forecast error.
2. **Live trading foundation (execution layer).** A controlled, fail-closed trading loop that discovers live Kalshi markets, maps contracts to model buckets, builds timestamp-safe live features, scores probabilities, fetches order books, computes edge after fees/spread/liquidity, applies risk limits, and routes orders to paper trading. Live order placement is disabled by default (`mode: shadow`, `trading_enabled: false`, `live_auto_enabled: false`).

This project does not guarantee profitability and makes no profitability claims.

## Project Goal

The central goal is to estimate the probability distribution of the NWS-reported final daily high temperature for Central Park (station `KNYC`, GHCND `USW00094728`), which matches Kalshi's settlement station.

Rather than predicting raw temperature from scratch, the model predicts the distribution of forecast error:

```text
forecast_error = actual_high - forecast_high
```

Interpretation:

- `forecast_error > 0`: the actual high was warmer than the forecast.
- `forecast_error < 0`: the actual high was cooler than the forecast.
- `forecast_error = 0`: the forecast exactly matched the actual high.

## Why Probability Buckets Need Calculus

Weather markets settle into mutually exclusive final-temperature buckets. The model must therefore produce a full probability distribution, not just a point forecast.

For a bucket from `a_i` to `b_i`, the probability is the area under the model's density curve:

```text
P(B_i) = F(b_i) - F(a_i)
```

The pricing layer converts final-temperature bucket boundaries into forecast-error boundaries (using the lower-open, upper-closed convention) and prices each bucket with CDF differences. Bucket probabilities are validated to be finite, nonnegative, and summing to 1 per row.

## Current Modeling Direction

- Primary model: NGBoost / distributional gradient boosting on `forecast_error`.
- Configured distribution: Normal (`config/model_config.yaml`); the selected validation candidate is `official_migration_depth3_subsample_15`.
- Required baseline: empirical historical forecast-error distribution (`src/empirical_error_model.py`).
- Bucket probabilities: derived from one coherent model-implied CDF.
- Evaluation priority: probability quality, calibration, interval coverage, and leakage safety.
- Live trading: shadow/paper mode only; no live execution, sizing, or automated capital at risk.

## Repository Layout

| Path | Purpose |
|---|---|
| `apps/` | Streamlit live-trading dashboard (`live_trading_dashboard.py`). |
| `config/` | Model configuration (`model_config.yaml`) and trading configuration (`trading_config.yaml`). |
| `data/raw/` | Raw observations, NWS/NDFD archives, and market snapshots. |
| `data/processed/` | Cleaned data, supervised target rows, and modeling tables. |
| `docs/project/` | Project spec and working context notes. |
| `docs/trading/` | Live trading implementation plan and trading docs. |
| `docs/presentation/` | Video presentation script, UML, and pseudocode deliverables. |
| `docs/specs/` | Long-form build specification. |
| `models/` | Trained model artifacts, feature lists, and calibration config. |
| `notebooks/` | Analysis, walkthrough, and presentation notebooks. |
| `outputs/` | Generated predictions, reports, diagnostics, figures, and live-trading cycle outputs. |
| `scripts/` | Reproducible command-line workflows (data pipeline, training, evaluation, live loop). |
| `src/` | Reusable project modules (`src/data/`, `src/trading/`). |
| `tests/` | Unit and regression tests. |

Conventional entry files such as `README.md`, `.gitignore`, and `requirements*.txt` stay at the repository root so standard tooling works normally.

## Main Source Modules

### Modeling layer (`src/`)

| Module | Role |
|---|---|
| `src/weather_data.py` | Load and standardize observed weather data (official NOAA/NWS daily TMAX, IEM/NWS ASOS hourly). |
| `src/forecast_data.py` | Load and standardize forecast data (timestamp-safe NWS/NDFD MaxT archive). |
| `src/target_builder.py` | Build daily forecast-error targets. |
| `src/supervised_table.py` | Expand daily targets into timestamped prediction rows. |
| `src/features.py` | Build timestamp-safe modeling features. |
| `src/leakage_checks.py` | Validate that no future data or target columns leak into model features. |
| `src/splits.py` | Create chronological train/validation/test splits. |
| `src/distributional_model.py` | Train and score distributional NGBoost models. |
| `src/train_ngboost.py` | Main NGBoost training workflow. |
| `src/distribution_pricing.py` | Convert model distributions into bucket probabilities. |
| `src/evaluation.py` | Compute NLL, Brier score, interval coverage, and related diagnostics. |
| `src/calibration.py` | Build calibration tables and figures. |
| `src/empirical_error_model.py` | Empirical historical forecast-error baseline. |
| `src/predict_distribution.py` | Final prediction interface (`load_probability_engine()`). |

### Trading layer (`src/trading/`)

| Module | Role |
|---|---|
| `config.py` | Load and validate `config/trading_config.yaml`. |
| `secrets.py` | Environment/key-path validation; fails closed without credentials. |
| `kalshi_client.py` | Kalshi REST client with RSA-PSS/SHA256 request signing, timeouts, retries. |
| `market_discovery.py` | Discover/filter live NYC daily-high weather markets. |
| `contract_mapping.py` | Parse Kalshi contracts into model bucket bounds; ambiguous contracts become `NO_TRADE`. |
| `live_weather.py` | Fetch timestamp-safe live observations and forecasts. |
| `live_features.py` | Build live feature rows matching the trained feature contract exactly. |
| `probability_signal.py` | Score live rows through the probability engine. |
| `orderbook.py` | Normalize YES/NO order books; infer executable asks; reject stale/wide books. |
| `edge.py` | Fair value vs executable price after fees, slippage, liquidity, and spread checks. |
| `risk.py` | Exposure limits, kill switch, denylists; runs before any order intent. |
| `portfolio.py` | Position/open-order reconciliation snapshots. |
| `order_intents.py` | Deterministic limit-order intents with idempotent `client_order_id`. |
| `paper_broker.py` | Hypothetical fills, paper positions, cash, and PnL. |
| `settlement_state.py` | Settlement-window gating (peak hours, verified highs, post-peak rules). |
| `dashboard_data.py` | Data assembly for the Streamlit dashboard. |
| `live_loop.py` | Full decision-cycle orchestration used by `scripts/run_live_trading_loop.py`. |

## Data Pipeline

The data pipeline is timestamp based. Each modeling row represents a prediction made at `prediction_time`.

Canonical sources:

- Actual daily high: official NOAA/NWS daily TMAX for Central Park.
- Intraday observed features: IEM/NWS ASOS station observations (Open-Meteo only as explicit emergency fallback).
- Forecast high baseline: timestamp-safe NWS/NDFD historical MaxT archive (issue-time filtered).
- Open-Meteo forecast history: legacy/auxiliary only; not the training forecast anchor.

High-level flow:

```text
NWS observations + NDFD forecasts
-> cleaned processed data            (scripts/run_day6_data_verification.py)
-> supervised forecast-error rows    (scripts/build_day7_supervised_table.py)
-> timestamp-safe feature table      (scripts/build_features.py)
-> feature integrity verification    (scripts/verify_feature_integrity.py)
-> chronological train/validation/test split
-> NGBoost distribution model        (python -m src.train_ngboost)
-> bucket probabilities              (python -m src.distribution_pricing)
-> evaluation and calibration outputs
```

Run the entire offline data production pipeline in one command:

```bash
python scripts/run_data_pipeline.py
```

This orchestrator chains verification, supervised-table construction, feature building, and integrity checks, fails closed on any stage error, and writes a run report to `outputs/reports/data_pipeline_run.md`.

Important processed files:

| File | Purpose |
|---|---|
| `data/processed/hourly_clean.csv` | Cleaned hourly observations. |
| `data/processed/hourly_forecasts_clean.csv` | Legacy/auxiliary hourly forecast data; ignored by the current NWS/NDFD feature contract unless replaced by an NWS-issued hourly source. |
| `data/processed/daily_clean.csv` | Cleaned daily observations. |
| `data/processed/forecasts_clean.csv` | Timestamp-safe NWS/NDFD daily MaxT forecast rows. |
| `data/processed/supervised_forecast_error_rows.csv` | Forecast-error target rows at prediction timestamps. |
| `data/processed/modeling_rows_v1.csv` | Final modeling rows with timestamp-safe features. |

## Feature Engineering

Features must be available at or before `prediction_time`.

Feature groups include:

- Calendar and clock features such as day of year, month, season, and hour.
- Observed weather features such as current temperature, dew point, cloud cover, wind, and precipitation.
- Forecast-relative features such as current temperature minus the timestamp-safe NDFD forecast baseline.
- Intraday path features such as max temperature so far, time since max so far, area under the temperature curve so far, and recent new-high counts.

The current NDFD archive is daily MaxT, not hourly forecast temperature. To preserve the 36-feature model contract, hourly forecast-relative columns are reproduced from the as-of-available NDFD daily-high forecast.

The selected final model feature list is stored in `outputs/final_feature_list.json`.

## Leakage Rules

The model cannot use information from after `prediction_time`.

Forbidden model features include:

- `actual_high`
- `forecast_error`
- final daily high fields
- settlement bucket or market result fields
- future observations
- forecast updates issued after prediction time
- any target-derived column

Leakage checks are implemented in `src/leakage_checks.py` and written by `scripts/build_features.py`.

## Model Training

The main training command is:

```bash
python -m src.train_ngboost
```

The configured training path reads:

- `config/model_config.yaml`
- `data/processed/modeling_rows_v1.csv`
- `outputs/final_feature_list.json`

It writes model artifacts under `models/` and distribution parameters/metrics under `outputs/`. Despite historical artifact names containing `normal_v0` or `laplace`, the active configuration is authoritative: `model_config.yaml` currently selects the Normal-distribution candidate `official_migration_depth3_subsample_15` (validation NLL 1.756, test NLL 1.687, test mean bucket Brier 0.117).

The default live probability engine artifact is `models/ngboost_laplace_current36_default.pkl`, loaded through `load_probability_engine()` in `src/predict_distribution.py`.

## Bucket Probability Conversion

```bash
python -m src.distribution_pricing
```

Reads distribution parameters from `outputs/ngboost_distribution_params_v0.csv` and writes bucket probability tables plus a validation report under `outputs/`.

## Evaluation and Calibration

```bash
python scripts/evaluate_ngboost.py
python scripts/calibrate_ngboost.py
```

Key outputs:

| Output | Purpose |
|---|---|
| `outputs/ngboost_evaluation_report.csv` | NLL, interval log loss, Brier score, coverage error, residual diagnostics. |
| `outputs/coverage_report.csv` | Prediction interval coverage by split. |
| `outputs/bucket_brier_scores.csv` | Bucket-level Brier scores. |
| `outputs/calibration_tables.csv` | Reliability/calibration tables. |
| `outputs/ngboost_calibration_report.csv` | Raw vs calibrated probability diagnostics. |
| `outputs/figures/` | PIT histograms, calibration plots, coverage plots, presentation visuals. |

## Live Trading Loop

The system currently runs in **shadow mode**: it performs every read-only step of the trading cycle and records outputs, but places no orders.

```bash
python scripts/run_live_trading_loop.py
```

Cycle sequence (every step fails closed):

1. Load config and check kill switch (`runtime/KILL_SWITCH_TRADING`).
2. Discover NYC daily-high markets (`KXHIGHNY`).
3. Parse contract buckets and map them to model buckets.
4. Fetch live weather (NWS station `KNYC`).
5. Build live feature rows matching the trained feature contract.
6. Generate model bucket probabilities.
7. Fetch and normalize order books.
8. Compute edge after fees, spread, slippage, and liquidity.
9. Reconcile portfolio state.
10. Run risk checks.
11. Route intents to the paper broker (or manual approval when enabled).

All cycle artifacts are written to `outputs/live_trading/` (see `config/trading_config.yaml` `outputs:` section for the full file map): market discovery snapshots, contract-bucket mapping, weather/feature freshness diagnostics, bucket probabilities, order-book snapshots, edge tables, risk decisions, order intents, paper orders/positions/PnL, and the trading cycle log.

Dashboard:

```bash
streamlit run apps/live_trading_dashboard.py
```

Safety posture:

- Modes: `shadow`, `paper`, `live_manual_approve`, `live_auto`. Defaults are `shadow` with `trading_enabled: false` and `live_auto_enabled: false`.
- Missing credentials, stale weather, failed mapping, failed probability validation, risk breaches, or an active kill switch all stop order generation.
- Automated live trading remains disabled until forward paper trading is reviewed over multiple weeks. See `docs/trading/LIVE_TRADING_IMPLEMENTATION_PLAN.md`.

## Presentation Deliverables

The video-ready notebook is `notebooks/project_walkthrough.ipynb`. Supporting materials live in `docs/presentation/` and `outputs/figures/`.

## Installation

```bash
pip install -r requirements.txt       # runtime dependencies
pip install -r requirements-dev.txt   # notebook and test dependencies
```

## Common Commands

Data production pipeline (all stages):

```bash
python scripts/run_data_pipeline.py
```

Individual stages:

```bash
python scripts/run_day6_data_verification.py     # clean + audit raw data
python scripts/build_day7_supervised_table.py    # forecast-error targets
python scripts/build_features.py                 # timestamp-safe features
python scripts/verify_feature_integrity.py       # provenance/integrity checks
```

Modeling:

```bash
python -m src.train_ngboost                      # train configured NGBoost model
python scripts/train_robust_laplace_baseline.py  # fixed robust baseline
python -m src.distribution_pricing               # distributions -> bucket probs
python scripts/evaluate_ngboost.py               # probability-quality metrics
python scripts/calibrate_ngboost.py              # calibration diagnostics
```

Trading:

```bash
python scripts/discover_weather_markets.py       # read-only market discovery
python scripts/run_live_trading_loop.py          # full shadow/paper cycle
streamlit run apps/live_trading_dashboard.py     # monitoring dashboard
```

Tests:

```bash
pytest
```

## Current Limitations

- Forecast highs come from historical NWS/NDFD MaxT rows with issue-time filtering; they are gridded NDFD forecasts rather than direct Kalshi-visible quote history.
- The NDFD archive supplies daily MaxT, not full hourly forecast-temperature paths, so hourly forecast-relative features use the timestamp-safe daily-high forecast as a fallback baseline.
- Hourly data may miss true intrahour daily highs.
- Settlement rules, station choice, endpoint conventions, and rounding can materially affect bucket mapping.
- Market prices are not yet integrated as timestamp-correct historical order-book data; backtesting uses forward paper trading instead.
- Apparent edge can disappear after spreads, fees, slippage, liquidity constraints, and fill assumptions.
- This project does not guarantee profitability.

## Documentation Map

- Agent working guide: `AGENTS.md`
- Project context handoff: `CONTEXT.md`
- Project spec: `docs/project/PROJECT_SPEC.md`
- Documentation folder map: `docs/README.md`
- Long-form build spec: `docs/specs/weather_probability_modeling_codex_build_spec_dgbm_ngboost.md`
- Live trading implementation plan: `docs/trading/LIVE_TRADING_IMPLEMENTATION_PLAN.md`
- Video presentation materials: `docs/presentation/`
- Outputs guide: `outputs/README.md`