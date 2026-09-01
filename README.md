# Kalshi Weather Trading

This repository builds a probability signal for Kalshi-style weather bucket markets, focused on New York City daily high-temperature contracts.

The project is not a live trading bot. It is a research and modeling layer that estimates calibrated probabilities for final-temperature buckets, evaluates those probabilities, and shows how model probabilities could later be compared with market-implied probabilities.

## Project Goal

The central goal is to estimate the probability distribution of the NWS-reported final daily high temperature.

Rather than predicting raw temperature from scratch, the model predicts the distribution of forecast error:

```text
forecast_error = actual_high - forecast_high
```

Interpretation:

- `forecast_error > 0`: the actual high was warmer than the forecast.
- `forecast_error < 0`: the actual high was cooler than the forecast.
- `forecast_error = 0`: the forecast exactly matched the actual high.

This framing uses the forecast high as a baseline and asks the model to learn the residual error around it.

## Why Probability Buckets Need Calculus

Weather markets settle into mutually exclusive final-temperature buckets. The model must therefore produce a full probability distribution, not just a point forecast.

For a bucket from `a_i` to `b_i`, the probability is the area under the model's probability density curve:

```text
P(B_i) = integral from a_i to b_i of f(t) dt
```

Using the cumulative distribution function:

```text
P(B_i) = F(b_i) - F(a_i)
```

The project converts final-temperature bucket boundaries into forecast-error boundaries, then prices each bucket with CDF differences.

## Current Modeling Direction

- Primary model: NGBoost / distributional gradient boosting on `forecast_error`.
- Current configured distribution: Normal, with sigma scaling from `config/model_config.yaml`.
- Frozen hurdle stage: LightGBM estimates whether the rounded KNYC five-minute maximum will increase again before day-end, and a shifted-Poisson NGBoost models the positive remaining increase. Their integer distribution is now convolved with a strictly prior-day station-to-Daily-Climate-Report reconciliation model before pricing Kalshi buckets.
- Required baseline: empirical historical forecast-error distribution.
- Bucket probabilities: derived from one coherent model-implied CDF.
- Evaluation priority: probability quality, calibration, interval coverage, and leakage safety.
- Trading research: coherent weather/market stacking, conservative one-minute proxy fills, quarter-Kelly-compatible risk interfaces, full cost stresses, immutable experiment reports, and shadow WebSocket depth capture. Order submission remains disabled and historical results are not executable-depth or profitability claims.

## Repository Layout

| Path | Purpose |
|---|---|
| `config/` | Model configuration, split settings, NGBoost parameters. |
| `data/raw/` | Raw observation data. |
| `data/forecasts/` | Raw or proxy forecast data. |
| `data/processed/` | Cleaned data, supervised target rows, and modeling tables. |
| `docs/project/` | Project spec and working context notes. |
| `docs/presentation/` | Video presentation script, UML, and pseudocode deliverables. |
| `docs/specs/` | Long-form build specification. |
| `models/` | Trained model artifacts and calibration config. |
| `notebooks/` | Analysis and presentation notebooks. |
| `outputs/` | Generated predictions, reports, diagnostics, and figures. |
| `scripts/` | Reproducible command-line workflows. |
| `src/` | Reusable project modules. |
| `tests/` | Unit and regression tests. |

Conventional entry files such as `README.md`, `.gitignore`, and `requirements*.txt` stay at the repository root so standard tooling works normally.

## Main Source Modules

| Module | Role |
|---|---|
| `src/weather_data.py` | Load and standardize observed weather data. |
| `src/forecast_data.py` | Load and standardize forecast data. |
| `src/target_builder.py` | Build daily forecast-error targets. |
| `src/supervised_table.py` | Expand daily targets into timestamped prediction rows. |
| `src/features.py` | Build timestamp-safe modeling features. |
| `src/leakage_checks.py` | Validate no future data or target columns leak into model features. |
| `src/splits.py` | Create chronological train, validation, and test splits. |
| `src/distributional_model.py` | Train and score distributional NGBoost models. |
| `src/train_ngboost.py` | Main NGBoost training workflow. |
| `src/distribution_pricing.py` | Convert model distributions into bucket probabilities. |
| `src/evaluation.py` | Compute NLL, Brier score, interval coverage, and related diagnostics. |
| `src/calibration.py` | Build calibration tables and figures. |

## Data Pipeline

The data pipeline is timestamp based. Each modeling row represents a prediction made at `prediction_time`.

Current canonical sources:

- Actual daily high: official NOAA/NWS daily TMAX for Central Park.
- Intraday observed features: IEM/NWS ASOS station observations.
- Forecast high baseline: timestamp-safe NWS/NDFD historical MaxT archive.
- Open-Meteo forecast history: retained only as legacy/auxiliary input and not used as the training forecast anchor.

High-level flow:

```text
NWS observations + NDFD forecasts
-> cleaned processed data
-> supervised forecast-error rows
-> timestamp-safe feature table
-> chronological train/validation/test split
-> NGBoost distribution model
-> bucket probabilities
-> evaluation and calibration outputs
```

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

The current NDFD archive is daily MaxT, not hourly forecast temperature. To preserve the restored 36-feature model contract, hourly forecast-relative columns are reproduced from the as-of-available NDFD daily-high forecast.

Key sequential features include:

```text
current_temp_minus_max_so_far
minutes_since_max_temp_so_far
hour_of_max_temp_so_far
max_so_far_minus_forecast_high
mean_temp_error_so_far
max_temp_error_so_far
num_new_highs_last_3h
temp_range_so_far
area_under_temp_curve_so_far
near_boundary_duration_so_far
```

The selected final model feature list is stored in:

```text
outputs/final_feature_list.json
```

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

It writes:

- `models/ngboost_normal_v0.pkl`
- `models/ngboost_normal_v0_features.json`
- `outputs/ngboost_distribution_params_v0.csv`
- `outputs/ngboost_nll_v0.json`

Despite the historical `normal_v0` artifact name, the current config and saved output may use the configured distribution type from `model_config.yaml`, currently Laplace.

## Bucket Probability Conversion

Run:

```bash
python -m src.distribution_pricing
```

This reads distribution parameters from:

```text
outputs/ngboost_distribution_params_v0.csv
```

and writes:

```text
outputs/ngboost_bucket_probs_v0.csv
outputs/ngboost_bucket_probs_validation_v0.csv
outputs/ngboost_bucket_probs_test_v0.csv
outputs/ngboost_bucket_prob_validation.md
```

The pricing layer validates that bucket probabilities are finite, nonnegative, and sum to 1 for each prediction row.

## Evaluation and Calibration

Run evaluation:

```bash
python scripts/evaluate_ngboost.py
```

Run calibration diagnostics:

```bash
python scripts/calibrate_ngboost.py
```

Optimize the already cross-fitted OOS trading signals for CAGR subject to the
hard drawdown constraint:

```bash
python scripts/optimize_final_strategy.py
```

The primary result remains one contract per trade. Any multi-contract result is
reported separately because historical Kalshi order-book depth is unavailable.

Test fixed-contract scaling without changing the one-contract strategy's signal
weights:

```bash
python scripts/optimize_constant_leverage.py
```

Key outputs:

| Output | Purpose |
|---|---|
| `outputs/ngboost_evaluation_report.csv` | NLL, interval log loss, Brier score, coverage error, and residual diagnostics. |
| `outputs/coverage_report.csv` | Prediction interval coverage by split. |
| `outputs/bucket_brier_scores.csv` | Bucket-level Brier scores. |
| `outputs/calibration_tables.csv` | Reliability/calibration tables. |
| `outputs/ngboost_calibration_report.csv` | Raw vs calibrated probability diagnostics. |
| `outputs/figures/` | PIT histograms, calibration plots, coverage plots, and presentation visuals. |

## Presentation Deliverables

The video-ready notebook is:

```text
notebooks/project_walkthrough.ipynb
```

Supporting presentation docs:

```text
docs/presentation/video_talking_points.md
docs/presentation/project_uml.md
docs/presentation/project_pseudocode.md
outputs/figures/project_architecture.png
outputs/figures/notebook_ngboost_trees_information_matrix.png
```

The notebook is designed to be run top-to-bottom and used as the visual basis for a project video.

## Installation

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install notebook and test dependencies:

```bash
pip install -r requirements-dev.txt
```

## Common Commands

Build timestamp-safe features:

```bash
python scripts/build_features.py
```

Verify feature provenance and model-safe feature integrity:

```bash
python scripts/verify_feature_integrity.py
```

Train the configured NGBoost model:

```bash
python -m src.train_ngboost
```

Train the fixed robust no-search baseline model:

```bash
python scripts/train_robust_laplace_baseline.py
```

Convert distributions to bucket probabilities:

```bash
python -m src.distribution_pricing
```

Evaluate probability quality:

```bash
python scripts/evaluate_ngboost.py
```

Run calibration diagnostics:

```bash
python scripts/calibrate_ngboost.py
```

Open the walkthrough notebook:

```bash
jupyter notebook notebooks/project_walkthrough.ipynb
```

Run tests:

```bash
pytest
```

## Current Limitations

- Forecast highs now come from historical NWS/NDFD MaxT rows with issue-time filtering, but they are still gridded NDFD forecasts rather than direct Kalshi-visible quote history.
- The current NDFD archive supplies daily MaxT, not full hourly forecast-temperature paths, so hourly forecast-relative features use the timestamp-safe daily-high forecast as a fallback baseline.
- Hourly data may miss true intrahour daily highs.
- Settlement rules, station choice, endpoint conventions, and rounding can materially affect bucket mapping.
- Market prices are not yet integrated as timestamp-correct order-book data.
- Apparent edge can disappear after spreads, fees, slippage, liquidity constraints, and fill assumptions.
- This project does not guarantee profitability.

## Documentation Map

- Project spec: `docs/project/PROJECT_SPEC.md`
- Project context and handoff notes: `docs/project/CONTEXT.md`
- Documentation folder map: `docs/README.md`
- Long-form build spec: `docs/specs/weather_probability_modeling_codex_build_spec_dgbm_ngboost.md`
- Video presentation materials: `docs/presentation/`
- Outputs guide: `outputs/README.md`
