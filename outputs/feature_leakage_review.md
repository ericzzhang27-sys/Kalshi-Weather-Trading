# Feature Leakage Review

Day 17 rule: a feature is allowed only if its value would genuinely be known at prediction timestamp `t`.

Forecast-based future-looking features are only valid if they come from a forecast issued at or before `t`. The current hourly forecast table does not expose an issue/run timestamp, so those features are treated conservatively.

## Suspicious Patterns

- Removing `time_season` improved validation NLL from 1.3224 to 1.2534 and improved bucket log loss from 0.9098 to 0.8984. This is not direct leakage, but it is a suspiciously large improvement from removing a timestamp-safe group and suggests noisy calendar/hour overfit or validation-period shift.
- Removing `forecast_relative` degraded validation NLL to 2.1472, confirming this group carries the strongest signal. The group mixes safe daily forecast-relative features with hourly forecast-derived features whose issue-time availability is uncertain, so the final model keeps only the safe daily/high-relative subset.
- The final safe reduced candidate has validation NLL 1.3324, about 0.010 worse than the full reference and 0.079 worse than the best required ablation, but it avoids unproven hourly forecast issue-time assumptions.

## Feature Table

| feature name | feature group | data source | construction method | available at t | leakage risk | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forecast_high | forecast_relative | supervised forecast row / daily forecast input | NWS forecast high carried into supervised row | yes | low | keep | Kept as the supervised NWS forecast high known by project construction at prediction time. |
| day_of_year_sin | time_season | prediction timestamp | sin(day of year) | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| day_of_year_cos | time_season | prediction timestamp | cos(day of year) | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| hour_sin | time_season | prediction timestamp | sin(prediction hour) | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| hour_cos | time_season | prediction timestamp | cos(prediction hour) | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| month | time_season | prediction timestamp | calendar month | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| season | time_season | prediction timestamp | calendar season code | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| forecast_horizon_hours | time_season | prediction timestamp and target date | hours from prediction time to typical 3 PM peak | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| current_temp | observed_temperature_path | hourly_clean observations | latest observed temperature at or before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| dew_point | humidity_dew_point | hourly_clean observations | latest observed dew point at or before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| cloud_cover_now | cloud_precipitation | hourly_clean observations | latest observed cloud cover at or before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| wind_speed | wind | hourly_clean observations | latest observed wind speed at or before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| precipitation_now | cloud_precipitation | hourly_clean observations | latest observed precipitation at or before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| temp_minus_dew_point | humidity_dew_point | derived current observation | current_temp - dew_point | yes | low | keep | Kept in final frozen feature set. |
| wind_dir_sin | wind | hourly_clean observations | sin(latest wind direction degrees) | yes | low | keep | Kept in final frozen feature set. |
| wind_dir_cos | wind | hourly_clean observations | cos(latest wind direction degrees) | yes | low | keep | Kept in final frozen feature set. |
| max_temp_so_far | observed_temperature_path | hourly_clean observations | cumulative same-day max through prediction_time | yes | low-medium | keep | Same-day aggregate is allowed because construction is cumulative only through prediction_time. |
| temp_change_60m | recent_temperature_changes | hourly_clean observations | current_temp minus latest temp about 60 minutes before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| temp_change_120m | recent_temperature_changes | hourly_clean observations | current_temp minus latest temp about 120 minutes before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| temp_change_180m | recent_temperature_changes | hourly_clean observations | current_temp minus latest temp about 180 minutes before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| temp_change_240m | recent_temperature_changes | hourly_clean observations | current_temp minus latest temp about 240 minutes before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| temp_change_300m | recent_temperature_changes | hourly_clean observations | current_temp minus latest temp about 300 minutes before prediction_time | yes | low | keep | Kept in final frozen feature set. |
| temp_acceleration_60m | recent_temperature_changes | derived observed temperature history | 2 * temp_change_60m - temp_change_120m | yes | low | keep | Kept in final frozen feature set. |
| temp_change_60m_minus_3h_avg_rate | recent_temperature_changes | derived observed temperature history | temp_change_60m - temp_change_180m / 3 | yes | low | keep | Kept in final frozen feature set. |
| forecast_temp_current_hour | forecast_relative | hourly_forecasts_clean | forecast temperature valid at or before prediction_time | uncertain | medium | remove | Hourly forecast table has valid timestamps but no issue/run timestamp, so issuance safety cannot be proven. |
| current_temp_minus_forecast_temp | forecast_relative | derived observed plus hourly forecast | current_temp - forecast_temp_current_hour | uncertain | medium | remove | Hourly forecast table has valid timestamps but no issue/run timestamp, so issuance safety cannot be proven. |
| forecast_max_so_far | forecast_relative | hourly_forecasts_clean | max forecast temperature valid through prediction_time | uncertain | medium | remove | Hourly forecast table has valid timestamps but no issue/run timestamp, so issuance safety cannot be proven. |
| max_so_far_minus_forecast_max_so_far | forecast_relative | derived observed plus hourly forecast | max_temp_so_far - forecast_max_so_far | uncertain | medium | remove | Hourly forecast table has valid timestamps but no issue/run timestamp, so issuance safety cannot be proven. |
| current_temp_minus_max_so_far | observed_temperature_path | derived observed temperature path | current_temp - max_temp_so_far | yes | low | keep | Kept in final frozen feature set. |
| minutes_since_max_temp_so_far | observed_temperature_path | hourly_clean observations | prediction_time - source time of max_temp_so_far | yes | low | keep | Kept in final frozen feature set. |
| hour_of_max_temp_so_far | observed_temperature_path | hourly_clean observations | hour of source time of max_temp_so_far | yes | low | keep | Kept in final frozen feature set. |
| max_so_far_minus_forecast_high | forecast_relative | derived observed plus daily forecast | max_temp_so_far - forecast_high | yes | low | keep | Kept in final frozen feature set. |
| mean_temp_error_so_far | forecast_relative | hourly_clean plus hourly_forecasts_clean | expanding mean of observed hourly temp - matching hourly forecast temp through prediction_time | uncertain | medium | remove | Hourly forecast table has valid timestamps but no issue/run timestamp, so issuance safety cannot be proven. |
| max_temp_error_so_far | forecast_relative | derived observed plus hourly forecast | max_temp_so_far - forecast_max_so_far | uncertain | medium | remove | Hourly forecast table has valid timestamps but no issue/run timestamp, so issuance safety cannot be proven. |
| num_new_highs_last_3h | observed_temperature_path | hourly_clean observations | count of strict new highs in trailing 3 hours through prediction_time | yes | low | keep | Kept in final frozen feature set. |
| temp_range_so_far | observed_temperature_path | hourly_clean observations | cumulative max minus min observed temp through prediction_time | yes | low-medium | keep | Same-day aggregate is allowed because construction is cumulative only through prediction_time. |
| area_under_temp_curve_so_far | observed_temperature_path | hourly_clean observations | cumulative trapezoid integral of observed temp through prediction_time | yes | low-medium | keep | Same-day aggregate is allowed because construction is cumulative only through prediction_time. |
| near_boundary_duration_so_far | observed_temperature_path | hourly_clean observations | count of hourly observations near integer-degree boundaries through prediction_time | yes | low-medium | keep | Same-day aggregate is allowed because construction is cumulative only through prediction_time. |
| minutes_until_typical_peak | time_season | prediction timestamp and target date | minutes from prediction time to typical 3 PM peak | yes | low | remove | Timestamp-safe, but removed because validation ablation improved without this group. |
| cloud_cover_next_3h | considered_not_in_day16_model | hourly_forecasts_clean | mean forecast cloud cover for next 3 forecast-valid hours | uncertain | high | remove | Future valid-time forecast window is only safe with issue-time provenance; current source lacks that proof. |
| precip_probability_next_3h | considered_not_in_day16_model | hourly_forecasts_clean | mean forecast precipitation probability for next 3 forecast-valid hours | uncertain | high | remove | Future valid-time forecast window is only safe with issue-time provenance; current source lacks that proof. |
| temp_change_30m | considered_not_in_day16_model | hourly_clean observations | 30-minute temp change, unavailable at hourly cadence | no | medium | remove | Hourly cadence cannot support a real 30-minute observed change without interpolation. |
| forecast_high_minus_current_temp | considered_not_in_day16_model | not present | candidate formula forecast_high - current_temp | no | low | remove | Not present in the modeling dataframe and not added without ablation. |
| forecast_high_minus_max_so_far | considered_not_in_day16_model | not present | candidate formula forecast_high - max_temp_so_far | no | low | remove | Not present in the modeling dataframe and not added without ablation. |
