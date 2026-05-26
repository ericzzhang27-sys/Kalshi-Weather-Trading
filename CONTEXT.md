# Weather Probability Modeling Project Context

Last updated: 2026-05-25

This file is the working handoff context for the 4-week weather probability modeling project. It summarizes what has been added through Days 1-8, what files matter, what assumptions are currently active, and what constraints future work must preserve.

## Core Thesis

This project builds a probability model for weather prediction-market temperature buckets.

The model is not trying to predict raw temperature from scratch. It models the distribution of forecast error:

```text
forecast_error = actual_high - forecast_high
```

Interpretation:

- `forecast_error > 0`: the actual official high was warmer than the forecast.
- `forecast_error < 0`: the actual official high was cooler than the forecast.
- `forecast_error = 0`: forecast and actual high matched.

The eventual market-pricing logic converts actual-temperature buckets into forecast-error intervals using:

```text
actual_high = forecast_high + forecast_error
```

The model should eventually output a coherent probability distribution across mutually exclusive temperature buckets.

## Timestamp Safety Rule

Every modeling row represents a prediction made at `prediction_time`.

For a row `X_{d,t}`, every feature must use only information available at or before `prediction_time`.

Do not introduce leakage from:

- final daily highs as features
- future hourly observations
- daily aggregate actuals as features
- forecast runs issued after `prediction_time`
- target-derived columns

The target `forecast_error` and audit field `actual_high` may remain in modeling tables for supervision and validation, but they must not be included in model feature columns.

## Current Data Scope

Current data is NYC-only and comes from Open-Meteo exports/proxies.

Cleaned date range:

- Daily actuals: 2022-01-01 to 2026-05-20
- Daily forecasts: 2022-01-01 to 2026-05-20
- Hourly actuals: 2022-01-01 00:00:00 to 2026-05-20 23:00:00
- Hourly forecasts: 2022-01-01 00:00:00 to 2026-05-20 23:00:00

Important caveats:

- Forecast data is `open_meteo_historical_forecast`, not confirmed official NWS archived forecast data.
- Forecast files do not include issue/run/reference timestamps.
- Forecast point-in-time availability cannot be fully verified until true archived forecast runs are added.
- Location/station was filled as `NYC` because row-level station identifiers were not present in the source CSVs.
- Temperature appears in degrees F in the raw Open-Meteo column names; no unit conversion was applied during cleaning.

## Days 1-5 Foundation

Days 1-5 established the project direction and early codebase structure.

Completed foundation:

- Project thesis and modeling objective defined in `PROJECT_SPEC.md`.
- Forecast-error framing chosen over raw-temperature prediction.
- Bucket-market probability logic defined.
- CDF-based probability conversion documented.
- Basic repo structure established.
- Open-Meteo raw actual and forecast CSVs added under `data/raw/` and `data/forecasts/`.
- Early utilities and demonstrations added for bucket intervals, normal CDF probabilities, and bucket probability conversion.
- Initial notebooks added for codebase and Day 5 function walkthroughs.

Key files from the foundation:

- `PROJECT_SPEC.md`
- `README.md`
- `weather_probability_modeling_codex_build_spec_dgbm_ngboost.md`
- `src/bucket_schema.py`
- `src/distribution_pricing.py`
- `src/error_boundaries.py`
- `src/main_demo.py`
- `src/visuals.py`
- `tests/distribution_pricing_tests.py`
- `notebooks/codebase_function_guide.ipynb`
- `notebooks/day5_function_guide.ipynb`

Core utilities added:

- `Bucket` dataclass for market bucket intervals.
- Bucket validation to ensure ordered, non-overlapping, exhaustive buckets.
- Actual-temperature bucket to forecast-error interval conversion.
- CDF boundary extraction from bucket intervals.
- Normal CDF and normal bucket-probability demo helpers.

Generated early outputs:

- `outputs/bucket_probability_demo.csv`
- `outputs/notebook_bucket_probability_demo.csv`
- normal CDF figures under `outputs/figures/`
- `outputs/data_inventory.csv`
- `outputs/notebook_data_inventory_demo.csv`

## Day 6: Data Verification And Cleaning

Day 6 verified raw downloaded data and wrote cleaned processed datasets.

Main code:

- `src/weather_data.py`
- `src/forecast_data.py`
- `src/data_audit.py`
- `scripts/run_day6_data_verification.py`

Main outputs:

- `data/processed/daily_clean.csv`
- `data/processed/hourly_clean.csv`
- `data/processed/forecasts_clean.csv`
- `data/processed/hourly_forecasts_clean.csv`
- `data/processed/modeling_base_preview.csv`
- `outputs/data_inventory.csv`
- `outputs/data_verification_report.md`

Cleaned rows:

- `daily_clean.csv`: 1,601 rows
- `forecasts_clean.csv`: 1,601 rows
- `hourly_clean.csv`: 38,424 rows
- `hourly_forecasts_clean.csv`: 38,424 rows

Day 6 identified:

- Actual high column: `temperature_2m_max`, standardized to `actual_high`.
- Forecast high column: `temperature_2m_max`, standardized to `forecast_high`.
- Hourly observed temperature: `temperature_2m`.
- Hourly forecast temperature: `temperature_2m`.
- Hourly valid timestamp: `timestamp`.
- Daily key: `date`, `location`.
- Hourly key: `timestamp`, `location`.

Missingness and warnings:

- `forecasts_clean.precipitation_probability_max` is 63.09 percent missing.
- `hourly_forecasts_clean.precipitation_probability` is 63.14 percent missing.
- Forecast issue/run timestamps are missing.
- Open-Meteo forecasts are proxies, not verified official NWS forecasts.

## Day 7: Supervised Forecast-Error Table

Day 7 created the supervised target table.

Main code:

- `src/target_builder.py`
- `src/supervised_table.py`
- `scripts/build_day7_supervised_table.py`
- `tests/test_day7_tables.py`

Main outputs:

- `data/processed/daily_forecast_error_targets.csv`
- `data/processed/supervised_forecast_error_rows.csv`
- `outputs/target_summary.csv`
- `notebooks/day7_function_guide.ipynb`

Prediction times:

```text
00:00
01:00
...
23:00
```

Day 7 row counts:

- Daily target rows: 1,601
- Supervised rows: 38,424
- Rows per date/location: 24

Target summary:

- Mean forecast error: -0.863
- Median forecast error: -0.9
- Standard deviation: 2.261
- Min forecast error: -15.4
- Max forecast error: 10.5

Day 7 table columns:

```text
date
location
prediction_time
prediction_timestamp
actual_high
forecast_high
forecast_error
forecast_source
```

Day 7 validation checks:

- One row per `date`, `location`, `prediction_time`.
- Required target columns present.
- Prediction timestamps parse.
- Forecast error equals `actual_high - forecast_high`.
- No target/audit columns marked as baseline features.

## Day 8: Timestamp-Safe Feature Engineering And Leakage Checks

Day 8 added the first full timestamp-safe modeling table and leakage report.

Main code:

- `src/features.py`
- `src/leakage_checks.py`
- `scripts/build_features.py`
- `tests/test_day8_features.py`

Main outputs:

- `data/processed/modeling_rows_v1.csv`
- `outputs/feature_columns.json`
- `outputs/leakage_check_report.md`
- `outputs/feature_missingness_report.csv`
- `outputs/modeling_rows_v1_preview.csv`
- `notebooks/day8_feature_engineering_and_leakage_checks.ipynb`

Day 8 build command:

```bash
python scripts/build_features.py
```

Day 8 test command:

```bash
python -m pytest
```

Current Day 8 build result:

- Rows: 38,424
- Columns in modeling table: 40
- Model feature columns: 29
- Critical rows dropped: 0
- Target date range: 2022-01-01 to 2026-05-20
- Prediction timestamp range: 2022-01-01 00:00:00 to 2026-05-20 23:00:00
- Rows per date/location: exactly 24
- Leakage status: `WARN`, with 0 failed checks

The `WARN` is expected because forecast issue/run/reference timestamps are absent from the source forecast data.

### Day 8 Feature Functions

`src/features.py` provides:

- `load_inputs(...)`
- `add_time_features(df)`
- `add_observed_weather_features(rows, hourly)`
- `add_forecast_relative_features(rows, hourly_forecasts)`
- `add_solar_time_features(df)`
- `add_forecast_update_features(rows, forecasts)`
- `handle_missing_features(df)`
- `build_feature_matrix(...)`
- `write_feature_columns(df, output_path)`

### Day 8 Time Features

Added:

- `day_of_year_sin`
- `day_of_year_cos`
- `hour_sin`
- `hour_cos`
- `month`
- `season`
- `forecast_horizon_hours`
- `minutes_until_typical_peak`

Definition:

- `forecast_horizon_hours` is hours from `prediction_time` until 3 PM local time on `target_date`.
- `minutes_until_typical_peak` uses the same 3 PM local peak assumption.

### Day 8 Observed Weather Features

Added from `hourly_clean.csv`:

- `current_temp`
- `dew_point`
- `cloud_cover_now`
- `wind_speed`
- `precipitation_now`
- `temp_minus_dew_point`
- `wind_dir_sin`
- `wind_dir_cos`
- `max_temp_so_far`
- `temp_change_60m`
- `temp_change_120m`
- `temp_change_180m`
- `temp_change_240m`
- `temp_change_300m`
- `temp_acceleration_60m`
- `temp_change_60m_minus_3h_avg_rate`

Audit metadata:

- `current_temp_source_time`
- `max_temp_so_far_source_time`

Timestamp-safe definitions:

- `current_temp` is the latest observed temperature at or before `prediction_time`.
- `max_temp_so_far` is the max observed temperature on `target_date` using only observations with `timestamp <= prediction_time`.
- `temp_change_60m`, `temp_change_120m`, `temp_change_180m`, `temp_change_240m`, and `temp_change_300m` use prior observations at safe lookback times.
- `temp_acceleration_60m = 2 * temp_change_60m - temp_change_120m`.
- `temp_change_60m_minus_3h_avg_rate = temp_change_60m - temp_change_180m / 3`.
- `temp_change_30m` is present in the table for documentation/missingness but excluded from features because hourly data cannot support true 30-minute precision.

### Day 8 Forecast-Relative Features

Added:

- `forecast_high`
- `forecast_temp_current_hour`
- `current_temp_minus_forecast_temp`
- `forecast_max_so_far`
- `max_so_far_minus_forecast_max_so_far`

Audit metadata:

- `forecast_temp_source_valid_time`
- `forecast_max_so_far_source_valid_time`

Important limitation:

- Forecast issue/run/reference timestamps are missing.
- Available forecast-valid-time audit columns are checked to ensure they are not after `prediction_time`.
- Future-window forecast features are skipped because future valid timestamps cannot be proven to come from a run issued at or before `prediction_time`.

Skipped Day 8 optional forecast features:

- `cloud_cover_next_3h`
- `precip_probability_next_3h`
- `recent_forecast_revision`
- `forecast_spread`
- `model_disagreement`

Reasons:

- No forecast issue/run/reference timestamp.
- No repeated forecast revisions or multiple forecast sources.
- Forecast precipitation probability is sparse.

### Day 8 Feature Columns

`outputs/feature_columns.json` currently includes these 29 model features:

```text
forecast_high
day_of_year_sin
day_of_year_cos
hour_sin
hour_cos
month
season
forecast_horizon_hours
current_temp
dew_point
cloud_cover_now
wind_speed
precipitation_now
temp_minus_dew_point
wind_dir_sin
wind_dir_cos
max_temp_so_far
temp_change_60m
temp_change_120m
temp_change_180m
temp_change_240m
temp_change_300m
temp_acceleration_60m
temp_change_60m_minus_3h_avg_rate
forecast_temp_current_hour
current_temp_minus_forecast_temp
forecast_max_so_far
max_so_far_minus_forecast_max_so_far
minutes_until_typical_peak
```

Excluded from features:

- identifiers and dates such as `location`, `date`, `target_date`, `prediction_time`
- target/audit fields such as `forecast_error`, `actual_high`
- source metadata timestamp columns
- all-null optional columns
- string/text columns
- any final-high leakage columns

### Day 8 Leakage Checks

`src/leakage_checks.py` provides:

- `run_leakage_checks(df, feature_columns)`
- `write_leakage_report(checks, output_path)`

Implemented checks:

- Target leakage check
- Future timestamp check
- Max-so-far sanity check
- Chronological validity check
- Feature reproducibility check

Current report:

- Overall status: `WARN`
- Target leakage check: `PASS`
- Future timestamp check: `WARN`
- Max-so-far sanity check: `PASS`
- Chronological validity check: `PASS`
- Feature reproducibility check: `PASS`

Manual validation found:

- `max_temp_so_far` decreases within day/location groups: 0
- observed source timestamps after `prediction_time`: 0
- forecast valid timestamps after `prediction_time`: 0
- `max_temp_so_far > actual_high + tolerance`: 0
- `forecast_error` in feature columns: false
- `actual_high` in feature columns: false
- all-null features in feature spec: none

## Current Repository Map

Important source modules:

- `src/bucket_schema.py`: bucket definitions and validation.
- `src/distribution_pricing.py`: normal CDF and bucket probability helpers.
- `src/error_boundaries.py`: actual-temperature bucket to error-boundary conversion.
- `src/weather_data.py`: raw actual weather loading and standardization.
- `src/forecast_data.py`: raw forecast loading and standardization.
- `src/data_audit.py`: CSV inventory and Day 6 verification report helpers.
- `src/target_builder.py`: daily forecast-error target construction.
- `src/supervised_table.py`: Day 7 prediction-time row expansion and validation.
- `src/features.py`: Day 8 timestamp-safe feature engineering.
- `src/leakage_checks.py`: Day 8 leakage checks.
- `src/visuals.py`: plotting helpers.
- `src/main_demo.py`: early demo entry point.

Important scripts:

- `scripts/inspect_day5_data.py`
- `scripts/run_day6_data_verification.py`
- `scripts/build_day7_supervised_table.py`
- `scripts/build_features.py`

Important notebooks:

- `notebooks/codebase_function_guide.ipynb`
- `notebooks/day5_function_guide.ipynb`
- `notebooks/day7_function_guide.ipynb`
- `notebooks/day8_feature_engineering_and_leakage_checks.ipynb`

Important tests:

- `tests/distribution_pricing_tests.py`
- `tests/test_day7_tables.py`
- `tests/test_day8_features.py`

## Reproducibility Commands

Run Day 6 cleaning and verification:

```bash
python scripts/run_day6_data_verification.py
```

Run Day 7 supervised target/table build:

```bash
python scripts/build_day7_supervised_table.py
```

Run Day 8 feature build:

```bash
python scripts/build_features.py
```

Run tests:

```bash
python -m pytest
```

Current test status:

```text
10 passed
```

## Known Risks Before Modeling

These must remain visible in future modeling work:

- Current forecast target is based on Open-Meteo historical forecast proxy data, not verified NWS archived forecast data.
- Forecast issue/run timestamps are missing, so true forecast as-of availability cannot be fully verified.
- Forecast high is treated as the Day 7 baseline known at prediction time, but this relies on the current data limitation.
- Future forecast-window features should not be used until forecast issue/run timestamps are available.
- Daily actual high, final high, and forecast error must never be included as model features.
- Missing forecast precipitation probability columns should not be treated as complete.
- NYC row-level station identifier is currently synthetic/fill value, not an official station ID.

## Next Likely Work

The original 4-week plan lists Day 9 as timestamp-safe forecast features. Because Day 8 already added the forecast-relative features that are safe with current data, the next useful step is likely one of:

- Add true archived forecast issue/run data, then rebuild next-3-hour forecast features safely.
- Build an empirical forecast-error baseline.
- Begin distributional modeling setup with NGBoost or an equivalent distributional GBM.
- Build cumulative classification or multiclass GBM benchmark targets from `modeling_rows_v1.csv`.

Any modeling step should use:

- rows from `data/processed/modeling_rows_v1.csv`
- features from `outputs/feature_columns.json`
- target `forecast_error`

Do not manually type feature columns in modeling scripts; load them from `outputs/feature_columns.json`.
