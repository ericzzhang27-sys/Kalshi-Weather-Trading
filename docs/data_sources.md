# Data Sources

## Label

The supervised label now uses official NOAA/NWS daily TMAX for Central Park:

```text
forecast_error = official_daily_high_f - forecast_high
```

`official_daily_high_f` is loaded from NOAA/NWS daily data when a CSV with `DATE`
and `TMAX` is found, preferring Central Park station `USW00094728`. Open-Meteo
historical daily max is not used as truth.

## Hourly Observations

Hourly observed features use IEM/NWS ASOS observations for station `NYC`/`KNYC`
when available. The raw `valid` timestamps in the current file are treated as
America/New_York local clock time, consistent with the METAR Z timestamps in the
file. Routine and special observations are retained.

Timestamp-safe observed features use only observations with:

```text
observation_timestamp <= prediction_time
```

This applies to current temperature, max/min temperature so far, temperature
changes, humidity, wind, precipitation, and sky-cover-derived variables.

## Forecasts

Forecast source priority is:

1. NDFD/NWS historical MaxT forecast archive at
   `data/processed/ndfd_knyc_daily_high_forecasts.csv`.
2. No training rebuild is written if NDFD coverage is incomplete.

When the NDFD archive exists, `scripts/run_day6_data_verification.py` expands
daily forecasts to hourly prediction rows and chooses the latest NDFD issue with:

```text
forecast_issue_time <= prediction_timestamp
```

Rows without an as-of-available NDFD forecast now fail the Day 6 rebuild instead
of falling back to Open-Meteo. This keeps `forecasts_clean.csv`,
`supervised_forecast_error_rows.csv`, and `modeling_rows_v1.csv` NWS-forecast
only for the training window. Open-Meteo hourly forecast files are ignored by the
feature builder unless they are replaced by an NWS-issued source. Because the
current NDFD archive is daily MaxT rather than hourly forecast temperature, the
restored 36-feature model contract reproduces hourly forecast-relative columns
from the timestamp-safe NDFD daily-high forecast.

## Caveats

ASOS hourly and special observations can miss brief highs between reports.
Official NOAA/NWS daily TMAX remains the settlement-quality label.

Final selected model settings and metrics are recorded in:

- `config/model_config.yaml`
- `outputs/ngboost_distribution_comparison.csv`
- `outputs/ngboost_hyperparameter_search.csv`
- `outputs/ngboost_evaluation_report.csv`
- `outputs/reports/source_usage_report.csv`
- `outputs/robust_laplace_baseline/comparison.md`

## Current Official-Migration Model

- Distribution: Normal NGBoost.
- Selected hyperparameters: `n_estimators=15`, `learning_rate=0.04`,
  `max_depth=3`, `min_samples_leaf=50`, `minibatch_frac=0.8`,
  `natural_gradient=true`, `random_state=42`.
- Feature set: `official_migration_current36`.
- Validation NLL: `1.755863`.
- Validation interval log loss: `1.089209`.
- Validation 80%/90% coverage: `0.8536` / `0.9248`.
- Test NLL: `1.687318`.
- Test interval log loss: `1.068653`.
- Test 80%/90% coverage: `0.8836` / `0.9415`.

The compact tuning grid was used because larger 300-700 estimator NGBoost runs
were impractical after the official-data migration in this environment.
