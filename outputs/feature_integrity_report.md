# Feature Integrity Report

Overall status: **WARN**

## Summary

- Feature columns checked: 39
- PASS checks: 57
- WARN checks: 1
- FAIL checks: 0
- Detailed checks CSV: `outputs\feature_integrity_checks.csv`

## Important Limitations

- This verifies internal provenance and formulas against the local processed source tables.
- It does not prove Open-Meteo proxy forecasts equal official NWS forecasts or Kalshi-visible forecasts.
- Forecast issue/run timestamps are still absent, so true forecast as-of availability cannot be fully verified.
- `near_boundary_duration_so_far` follows the requested formula `abs(temp - round(temp)) <= 0.5`; this threshold counts every non-missing numeric hourly temperature.

## Warnings

- near_boundary_duration_so_far interpretation: 0 affected rows; The requested abs(temp - round(temp)) <= 0.5 formula counts every non-missing numeric hourly temperature.

## Feature Provenance

- `forecast_high`: forecasts_clean.forecast_high joined by date/location
- `day_of_year_sin`: sin(2*pi*prediction_time.dayofyear/366)
- `day_of_year_cos`: cos(2*pi*prediction_time.dayofyear/366)
- `hour_sin`: sin(2*pi*fractional_prediction_hour/24)
- `hour_cos`: cos(2*pi*fractional_prediction_hour/24)
- `month`: prediction_time.month
- `season`: encoded from prediction_time.month
- `forecast_horizon_hours`: hours from prediction_time to 15:00 on target_date
- `current_temp`: hourly_clean.temperature_2m at prediction_time
- `dew_point`: hourly_clean.dew_point_2m at prediction_time
- `cloud_cover_now`: hourly_clean.cloud_cover at prediction_time
- `wind_speed`: hourly_clean.wind_speed_10m at prediction_time
- `precipitation_now`: hourly_clean.precipitation at prediction_time
- `temp_minus_dew_point`: current_temp - dew_point
- `wind_dir_sin`: sin(hourly_clean.wind_direction_10m in radians)
- `wind_dir_cos`: cos(hourly_clean.wind_direction_10m in radians)
- `max_temp_so_far`: cumulative max of hourly_clean.temperature_2m within target_date
- `temp_change_60m`: current_temp - observed temp 60 minutes earlier
- `temp_change_120m`: current_temp - observed temp 120 minutes earlier
- `temp_change_180m`: current_temp - observed temp 180 minutes earlier
- `temp_change_240m`: current_temp - observed temp 240 minutes earlier
- `temp_change_300m`: current_temp - observed temp 300 minutes earlier
- `temp_acceleration_60m`: 2*temp_change_60m - temp_change_120m
- `temp_change_60m_minus_3h_avg_rate`: temp_change_60m - temp_change_180m/3
- `forecast_temp_current_hour`: hourly_forecasts_clean.temperature_2m at prediction_time
- `current_temp_minus_forecast_temp`: current_temp - forecast_temp_current_hour
- `forecast_max_so_far`: cumulative max of hourly_forecasts_clean.temperature_2m within target_date
- `max_so_far_minus_forecast_max_so_far`: max_temp_so_far - forecast_max_so_far
- `current_temp_minus_max_so_far`: current_temp - max_temp_so_far
- `minutes_since_max_temp_so_far`: prediction_time - latest timestamp of max_temp_so_far
- `hour_of_max_temp_so_far`: hour of latest timestamp of max_temp_so_far
- `max_so_far_minus_forecast_high`: max_temp_so_far - forecast_high
- `mean_temp_error_so_far`: cumulative mean(actual hourly temp - forecast hourly temp)
- `max_temp_error_so_far`: max_temp_so_far - forecast_max_so_far
- `num_new_highs_last_3h`: strict new observed highs in trailing (t - 3h, t] window
- `temp_range_so_far`: max_temp_so_far - min_temp_so_far within target_date
- `area_under_temp_curve_so_far`: cumulative hourly trapezoid integral of observed temp
- `near_boundary_duration_so_far`: count so far where abs(temp - round(temp)) <= 0.5
- `minutes_until_typical_peak`: minutes from prediction_time to 15:00 on target_date

## Check Results

| Check | Status | Affected rows | Columns | Details |
| --- | --- | ---: | --- | --- |
| Required schema: modeling_rows_v1 | PASS | 0 |  | All required source columns are present. |
| Required schema: hourly_clean | PASS | 0 |  | All required source columns are present. |
| Required schema: hourly_forecasts_clean | PASS | 0 |  | All required source columns are present. |
| Required schema: daily_clean | PASS | 0 |  | All required source columns are present. |
| Required schema: forecasts_clean | PASS | 0 |  | All required source columns are present. |
| Required schema: supervised_forecast_error_rows | PASS | 0 |  | All required source columns are present. |
| Feature spec columns exist in modeling_rows_v1 | PASS | 0 |  | feature_columns.json is aligned to modeling_rows_v1.csv. |
| Feature spec excludes target and final actuals | PASS | 0 |  | Target/audit columns are not model features. |
| Feature spec columns are numeric | PASS | 0 |  | All feature columns are numeric/model-safe. |
| Modeling row keys and hourly coverage | PASS | 0 | location, target_date, prediction_time | Rows are unique and every location/date has 24 hourly prediction rows. |
| Modeling rows match supervised rows | PASS | 0 | modeling_rows_v1, supervised_forecast_error_rows | modeling rows=38424, supervised rows=38424. |
| Target formula forecast_error = actual_high - forecast_high | PASS | 0 | forecast_error, actual_high, forecast_high | Target math matches stored columns. |
| daily_clean.actual_high matches max hourly_clean.temperature_2m | PASS | 0 | actual_high, temperature_2m | Tolerance: 0.11 F. |
| forecasts_clean.forecast_high matches max hourly_forecasts_clean.temperature_2m | PASS | 0 | forecast_high, temperature_2m | Tolerance: 0.11 F. |
| Feature formula: forecast_high | PASS | 0 | forecast_high | forecasts_clean.forecast_high joined by date/location |
| Feature formula: day_of_year_sin | PASS | 0 | day_of_year_sin | sin(2*pi*prediction_time.dayofyear/366) |
| Feature formula: day_of_year_cos | PASS | 0 | day_of_year_cos | cos(2*pi*prediction_time.dayofyear/366) |
| Feature formula: hour_sin | PASS | 0 | hour_sin | sin(2*pi*fractional_prediction_hour/24) |
| Feature formula: hour_cos | PASS | 0 | hour_cos | cos(2*pi*fractional_prediction_hour/24) |
| Feature formula: month | PASS | 0 | month | prediction_time.month |
| Feature formula: season | PASS | 0 | season | encoded from prediction_time.month |
| Feature formula: forecast_horizon_hours | PASS | 0 | forecast_horizon_hours | hours from prediction_time to 15:00 on target_date |
| Feature formula: current_temp | PASS | 0 | current_temp | hourly_clean.temperature_2m at prediction_time |
| Feature formula: dew_point | PASS | 0 | dew_point | hourly_clean.dew_point_2m at prediction_time |
| Feature formula: cloud_cover_now | PASS | 0 | cloud_cover_now | hourly_clean.cloud_cover at prediction_time |
| Feature formula: wind_speed | PASS | 0 | wind_speed | hourly_clean.wind_speed_10m at prediction_time |
| Feature formula: precipitation_now | PASS | 0 | precipitation_now | hourly_clean.precipitation at prediction_time |
| Feature formula: temp_minus_dew_point | PASS | 0 | temp_minus_dew_point | current_temp - dew_point |
| Feature formula: wind_dir_sin | PASS | 0 | wind_dir_sin | sin(hourly_clean.wind_direction_10m in radians) |
| Feature formula: wind_dir_cos | PASS | 0 | wind_dir_cos | cos(hourly_clean.wind_direction_10m in radians) |
| Feature formula: max_temp_so_far | PASS | 0 | max_temp_so_far | cumulative max of hourly_clean.temperature_2m within target_date |
| Feature formula: temp_change_60m | PASS | 0 | temp_change_60m | current_temp - observed temp 60 minutes earlier |
| Feature formula: temp_change_120m | PASS | 0 | temp_change_120m | current_temp - observed temp 120 minutes earlier |
| Feature formula: temp_change_180m | PASS | 0 | temp_change_180m | current_temp - observed temp 180 minutes earlier |
| Feature formula: temp_change_240m | PASS | 0 | temp_change_240m | current_temp - observed temp 240 minutes earlier |
| Feature formula: temp_change_300m | PASS | 0 | temp_change_300m | current_temp - observed temp 300 minutes earlier |
| Feature formula: temp_acceleration_60m | PASS | 0 | temp_acceleration_60m | 2*temp_change_60m - temp_change_120m |
| Feature formula: temp_change_60m_minus_3h_avg_rate | PASS | 0 | temp_change_60m_minus_3h_avg_rate | temp_change_60m - temp_change_180m/3 |
| Feature formula: forecast_temp_current_hour | PASS | 0 | forecast_temp_current_hour | hourly_forecasts_clean.temperature_2m at prediction_time |
| Feature formula: current_temp_minus_forecast_temp | PASS | 0 | current_temp_minus_forecast_temp | current_temp - forecast_temp_current_hour |
| Feature formula: forecast_max_so_far | PASS | 0 | forecast_max_so_far | cumulative max of hourly_forecasts_clean.temperature_2m within target_date |
| Feature formula: max_so_far_minus_forecast_max_so_far | PASS | 0 | max_so_far_minus_forecast_max_so_far | max_temp_so_far - forecast_max_so_far |
| Feature formula: current_temp_minus_max_so_far | PASS | 0 | current_temp_minus_max_so_far | current_temp - max_temp_so_far |
| Feature formula: minutes_since_max_temp_so_far | PASS | 0 | minutes_since_max_temp_so_far | prediction_time - latest timestamp of max_temp_so_far |
| Feature formula: hour_of_max_temp_so_far | PASS | 0 | hour_of_max_temp_so_far | hour of latest timestamp of max_temp_so_far |
| Feature formula: max_so_far_minus_forecast_high | PASS | 0 | max_so_far_minus_forecast_high | max_temp_so_far - forecast_high |
| Feature formula: mean_temp_error_so_far | PASS | 0 | mean_temp_error_so_far | cumulative mean(actual hourly temp - forecast hourly temp) |
| Feature formula: max_temp_error_so_far | PASS | 0 | max_temp_error_so_far | max_temp_so_far - forecast_max_so_far |
| Feature formula: num_new_highs_last_3h | PASS | 0 | num_new_highs_last_3h | strict new observed highs in trailing (t - 3h, t] window |
| Feature formula: temp_range_so_far | PASS | 0 | temp_range_so_far | max_temp_so_far - min_temp_so_far within target_date |
| Feature formula: area_under_temp_curve_so_far | PASS | 0 | area_under_temp_curve_so_far | cumulative hourly trapezoid integral of observed temp |
| Feature formula: near_boundary_duration_so_far | PASS | 0 | near_boundary_duration_so_far | count so far where abs(temp - round(temp)) <= 0.5 |
| Feature formula: minutes_until_typical_peak | PASS | 0 | minutes_until_typical_peak | minutes from prediction_time to 15:00 on target_date |
| Timestamp safety: current_temp_source_time <= prediction_time | PASS | 0 | current_temp_source_time, prediction_time | Source/valid timestamps are not after prediction_time. |
| Timestamp safety: max_temp_so_far_source_time <= prediction_time | PASS | 0 | max_temp_so_far_source_time, prediction_time | Source/valid timestamps are not after prediction_time. |
| Timestamp safety: forecast_temp_source_valid_time <= prediction_time | PASS | 0 | forecast_temp_source_valid_time, prediction_time | Source/valid timestamps are not after prediction_time. |
| Timestamp safety: forecast_max_so_far_source_valid_time <= prediction_time | PASS | 0 | forecast_max_so_far_source_valid_time, prediction_time | Source/valid timestamps are not after prediction_time. |
| near_boundary_duration_so_far interpretation | WARN | 0 | near_boundary_duration_so_far | The requested abs(temp - round(temp)) <= 0.5 formula counts every non-missing numeric hourly temperature. |