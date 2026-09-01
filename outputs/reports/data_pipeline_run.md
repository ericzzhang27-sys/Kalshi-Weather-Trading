# Data Pipeline Run Report

- Run finished (local): 2026-08-24 14:54:02
- Overall result: SUCCESS

| Stage | Script | Result | Elapsed (s) |
|---|---|---|---|
| day6_verification | `scripts/run_day6_data_verification.py` | PASS | 79.8 |
| day7_targets | `scripts/build_day7_supervised_table.py` | PASS | 3.2 |
| day8_features | `scripts/build_features.py` | PASS | 194.3 |
| feature_integrity | `scripts/verify_feature_integrity.py` | PASS | 80.9 |

## Output tail — day6_verification

```text
  parsed = pd.to_datetime(sample, errors="coerce")
C:\Weather Trading\Kalshi-Weather-Trading\src\data_audit.py:85: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  parsed = pd.to_datetime(sample, errors="coerce")
C:\Weather Trading\Kalshi-Weather-Trading\src\data_audit.py:85: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  parsed = pd.to_datetime(sample, errors="coerce")
C:\Weather Trading\Kalshi-Weather-Trading\src\data_audit.py:85: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  parsed = pd.to_datetime(sample, errors="coerce")
C:\Weather Trading\Kalshi-Weather-Trading\src\data_audit.py:85: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  parsed = pd.to_datetime(sample, errors="coerce")
C:\Weather Trading\Kalshi-Weather-Trading\src\data_audit.py:85: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  parsed = pd.to_datetime(sample, errors="coerce")
C:\Weather Trading\Kalshi-Weather-Trading\src\weather_data.py:206: DtypeWarning: Columns (0: ice_accretion_3hr) have mixed types. Specify dtype option on import or set low_memory=False.
  return pd.read_csv(path, na_values=ASOS_MISSING_VALUES, keep_default_na=True)
C:\Weather Trading\Kalshi-Weather-Trading\src\weather_data.py:206: DtypeWarning: Columns (0: ice_accretion_3hr) have mixed types. Specify dtype option on import or set low_memory=False.
  return pd.read_csv(path, na_values=ASOS_MISSING_VALUES, keep_default_na=True)
```

## Output tail — day7_targets

```text
    forecast_source: 0
  duplicate count by ['date', 'location']: 36823
  duplicate count by ['date', 'location', 'prediction_time']: 0
Day 7 supervised forecast-error table complete.
Daily target rows: 1,601
Supervised rows: 38,424
Date range: 2022-01-01 to 2026-05-20
Locations: NYC
Prediction times: 00:00, 01:00, 02:00, 03:00, 04:00, 05:00, 06:00, 07:00, 08:00, 09:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00, 17:00, 18:00, 19:00, 20:00, 21:00, 22:00, 23:00
Target forecast_error mean/std/min/max: 0.405 / 2.637 / -8.990 / 15.010
Daily targets: C:\Weather Trading\Kalshi-Weather-Trading\data\processed\daily_forecast_error_targets.csv
Supervised rows: C:\Weather Trading\Kalshi-Weather-Trading\data\processed\supervised_forecast_error_rows.csv
Target summary: C:\Weather Trading\Kalshi-Weather-Trading\outputs\day7_targets\target_summary.csv
Warnings:
- Prediction target join dropped dates because actual and forecast coverage does not fully overlap.
```

## Output tail — day8_features

```text
Random 20-row preview: C:\Weather Trading\Kalshi-Weather-Trading\outputs\day8_features\modeling_rows_v1_preview.csv
Skipped/limited feature notes:
- hourly_forecasts_clean.csv is Open-Meteo forecast data and was ignored; training features use NWS/NDFD forecast_high plus observed NWS/ASOS features.
- forecast_horizon_hours and minutes_until_typical_peak use 3 PM local time on target_date as the typical peak temperature time.
- temp_change_30m skipped; hourly_clean.csv cadence is hourly, so 30-minute temperature change would fake unavailable precision.
- Forecast-relative hourly feature columns were reproduced from the timestamp-safe NDFD daily-high forecast because no NWS hourly forecast temperature archive is available.
- mean_temp_error_so_far reproduced as mean observed temperature so far minus the timestamp-safe NDFD daily-high forecast because no NWS hourly forecast temperature archive is available.
- num_new_highs_last_3h counts strict new observed highs in the trailing (prediction_time - 3h, prediction_time] window.
- area_under_temp_curve_so_far is a cumulative hourly trapezoid integral of observed temperature from the start of target_date through prediction_time.
- near_boundary_duration_so_far follows the requested definition abs(temp - round(temp)) <= 0.5°F, so it counts non-missing hourly observations under normal numeric rounding.
- minutes_until_sunset skipped; no extra solar dependency was added for Day 8.
- Post-peak features (is_post_peak_hour, is_post_window_hour, temp_drop_since_max, hours_since_max, is_verified_peak) mirror settlement_state post-peak gating and give NGBoost heteroscedastic sigma a sharp after-18:00 signal.
- forecast_current_temp_gap_per_hour_to_peak is (forecast_high - current_temp) divided by hours until the 3 PM typical peak; zero-hour rows are left missing.
- needed_warming_rate_minus_recent_rate compares the forecast-required warming rate to the observed 3-hour warming rate temp_change_180m / 3.
- Dropped 1627 rows missing critical Day 8 fields.
```

## Output tail — feature_integrity

```text
Feature integrity verification complete: WARN
Rows verified: 36,797
Feature columns verified: 65
Report: C:\Weather Trading\Kalshi-Weather-Trading\outputs\day8_features\feature_integrity_report.md
Checks CSV: C:\Weather Trading\Kalshi-Weather-Trading\outputs\day8_features\feature_integrity_checks.csv
```
