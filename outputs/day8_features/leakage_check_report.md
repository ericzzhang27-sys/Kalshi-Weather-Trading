# Day 8 Leakage Check Report

Overall status: **WARN**

## Checks

### Target leakage check: PASS

- Affected rows: 0
- Affected columns: none
- Explanation: No target or final actual high columns are included as features.

### Future timestamp check: WARN

- Affected rows: 0
- Affected columns: next_3h_forecast_source_issue_time
- Explanation: Available source timestamps are not after prediction_time, but forecast issue/run timestamp columns are missing, so forecast issuance safety is documented as a data limitation.

### Max-so-far sanity check: WARN

- Affected rows: 421
- Affected columns: max_temp_so_far, actual_high
- Explanation: max_temp_so_far exceeds official daily actual_high for some rows. Because observed_temperature_source is IEM/NWS ASOS, this is treated as a source-disagreement warning: hourly/special ASOS reports can differ from the official daily TMAX climate product. Future timestamp checks still guard leakage.

### Chronological validity check: PASS

- Affected rows: 0
- Affected columns: none
- Explanation: prediction_time and target_date parse cleanly, prediction_time is inside the target-date window, and rows can be sorted by ['location', 'target_date', 'prediction_time'].

### Feature reproducibility check: PASS

- Affected rows: 0
- Affected columns: none
- Explanation: Feature columns are present, numeric/model-safe, and do not include metadata columns.

## Feature Notes

- hourly_forecasts_clean.csv is Open-Meteo forecast data and was ignored; training features use NWS/NDFD forecast_high plus observed NWS/ASOS features.
- forecast_horizon_hours and minutes_until_typical_peak use 3 PM local time on target_date as the typical peak temperature time.
- temp_change_30m skipped; hourly_clean.csv cadence is hourly, so 30-minute temperature change would fake unavailable precision.
- Forecast-relative hourly feature columns were reproduced from the timestamp-safe NDFD daily-high forecast because no NWS hourly forecast temperature archive is available.
- mean_temp_error_so_far reproduced as mean observed temperature so far minus the timestamp-safe NDFD daily-high forecast because no NWS hourly forecast temperature archive is available.
- num_new_highs_last_3h counts strict new observed highs in the trailing (prediction_time - 3h, prediction_time] window.
- area_under_temp_curve_so_far is a cumulative hourly trapezoid integral of observed temperature from the start of target_date through prediction_time.
- near_boundary_duration_so_far follows the requested definition abs(temp - round(temp)) <= 0.5°F, so it counts non-missing hourly observations under normal numeric rounding.
- minutes_until_sunset skipped; no extra solar dependency was added for Day 8.
- forecast_current_temp_gap_per_hour_to_peak is (forecast_high - current_temp) divided by hours until the 3 PM typical peak; zero-hour rows are left missing.
- needed_warming_rate_minus_recent_rate compares the forecast-required warming rate to the observed 3-hour warming rate temp_change_180m / 3.
- Dropped 1627 rows missing critical Day 8 fields.
