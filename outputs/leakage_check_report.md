# Day 8 Leakage Check Report

Overall status: **WARN**

## Checks

### Target leakage check: PASS

- Affected rows: 0
- Affected columns: none
- Explanation: No target or final actual high columns are included as features.

### Future timestamp check: WARN

- Affected rows: 0
- Affected columns: forecast_source_issue_time, forecast_temp_source_issue_time, next_3h_forecast_source_issue_time
- Explanation: Available source timestamps are not after prediction_time, but forecast issue/run timestamp columns are missing, so forecast issuance safety is documented as a data limitation.

### Max-so-far sanity check: PASS

- Affected rows: 0
- Affected columns: none
- Explanation: max_temp_so_far never exceeds actual_high beyond the 0.5 degree tolerance.

### Chronological validity check: PASS

- Affected rows: 0
- Affected columns: none
- Explanation: prediction_time and target_date parse cleanly, prediction_time is inside the target-date window, and rows can be sorted by ['location', 'target_date', 'prediction_time'].

### Feature reproducibility check: PASS

- Affected rows: 0
- Affected columns: none
- Explanation: Feature columns are present, numeric/model-safe, and do not include metadata columns.

## Feature Notes

- hourly_forecasts_clean.csv has forecast valid timestamps but no issue/run/reference timestamp; forecast issuance safety cannot be directly verified.
- forecasts_clean.csv has daily forecast highs but no issue/run/reference timestamp; forecast_high is treated as the Day 7 baseline known at prediction time.
- forecast_horizon_hours and minutes_until_typical_peak use 3 PM local time on target_date as the typical peak temperature time.
- temp_change_30m skipped; hourly_clean.csv cadence is hourly, so 30-minute temperature change would fake unavailable precision.
- cloud_cover_next_3h and precip_probability_next_3h skipped because hourly forecast issue/run timestamps are unavailable; future valid times cannot be proven to come from a run issued at or before prediction_time.
- minutes_until_sunset skipped; no extra solar dependency was added for Day 8.
- recent_forecast_revision, forecast_spread, and model_disagreement skipped; forecasts_clean.csv has no repeated issue/run timestamp.
