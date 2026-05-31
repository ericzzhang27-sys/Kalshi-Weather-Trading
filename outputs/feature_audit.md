# Day 17 NGBoost Feature Audit

## Goal

Run a feature robustness and ablation study for the tuned Day 16 NGBoost model so the production/demo model uses a justified, timestamp-safe feature set rather than every available feature.

## Reference Model Setup

- Model: NGBoost Normal forecast-error distribution.
- Target: `forecast_error`.
- Day 16 selected candidate: `more_trees_lower_lr`.
- Hyperparameters: `n_estimators=500`, `learning_rate=0.02`, `max_depth=2`, `min_samples_leaf=20`, `minibatch_frac=1.0`, `natural_gradient=True`, `random_state=42`.
- Train split: 2022-01-01 to 2023-12-31.
- Validation split: 2024-01-01 to 2024-12-31.
- Test split reserved: 2025-01-01 to 2026-05-20.
- Test-set metrics were not used for feature selection.

## Ablation Comparison

| run | removed | features | NLL | bucket log loss | mean Brier | cov80 | cov90 | sigma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_feature_set |  | 39 | 1.3224 | 0.9098 | 0.0794 | 0.803 | 0.893 | 1.069 |
| remove_time_season | time_season | 31 | 1.2534 | 0.8984 | 0.0777 | 0.804 | 0.892 | 1.053 |
| remove_observed_temperature_path | observed_temperature_path | 30 | 1.3128 | 0.9089 | 0.0791 | 0.825 | 0.906 | 1.097 |
| remove_forecast_relative | forecast_relative | 31 | 2.1472 | 1.4300 | 0.1179 | 0.761 | 0.847 | 1.697 |
| remove_humidity_dew_point | humidity_dew_point | 37 | 1.3192 | 0.9131 | 0.0793 | 0.810 | 0.897 | 1.077 |
| remove_cloud_precipitation | cloud_precipitation | 37 | 1.3015 | 0.9014 | 0.0786 | 0.814 | 0.900 | 1.076 |
| remove_wind | wind | 36 | 1.3254 | 0.9106 | 0.0793 | 0.813 | 0.897 | 1.076 |
| remove_recent_temperature_changes | recent_temperature_changes | 32 | 1.3153 | 0.9081 | 0.0791 | 0.809 | 0.895 | 1.073 |
| minimal_feature_set | minimal_feature_set | 9 | 1.4870 | 0.9858 | 0.0858 | 0.814 | 0.897 | 1.185 |

## Strongest And Weakest Groups

| group | read | evidence |
| --- | --- | --- |
| forecast_relative | strongest but reduced | Removing the full group raised validation NLL by 0.8248, but hourly forecast-derived members were removed for timestamp safety. |
| time_season | weak/noisy | Removing the group improved validation NLL by 0.0690 and bucket log loss by 0.0114. |
| cloud_precipitation | weak/noisy current features | Removing current cloud/precip improved NLL by 0.0209, but final keeps them because they are timestamp-safe and may help robustness. |
| wind | near-neutral | Removing wind slightly worsened NLL by 0.0030; final keeps it because it is safe and low cost. |
| humidity_dew_point | near-neutral | Removing humidity/dew point barely changed NLL but worsened bucket log loss; final keeps it. |
| recent_temperature_changes | mildly noisy but plausible | Removing recent changes improved NLL by 0.0071; final keeps them as timestamp-safe dynamics with a caveat for future pruning. |

## Suspicious Features Or Groups

- Removing `time_season` improved validation NLL from 1.3224 to 1.2534 and improved bucket log loss from 0.9098 to 0.8984. This is not direct leakage, but it is a suspiciously large improvement from removing a timestamp-safe group and suggests noisy calendar/hour overfit or validation-period shift.
- Removing `forecast_relative` degraded validation NLL to 2.1472, confirming this group carries the strongest signal. The group mixes safe daily forecast-relative features with hourly forecast-derived features whose issue-time availability is uncertain, so the final model keeps only the safe daily/high-relative subset.
- The final safe reduced candidate has validation NLL 1.3324, about 0.010 worse than the full reference and 0.079 worse than the best required ablation, but it avoids unproven hourly forecast issue-time assumptions.

## Metrics Used For Decision-Making

Primary metric was chronological validation NLL/log score. Bucket interval log loss and mean Brier score were used as probability-quality checks. 50%, 80%, and 90% coverage plus average predicted sigma were used to avoid overconfident or overly wide distributions.

## Final Feature Set Decision

Final frozen feature count: 25.

| feature | group |
| --- | --- |
| forecast_high | forecast_relative |
| current_temp | observed_temperature_path |
| dew_point | humidity_dew_point |
| cloud_cover_now | cloud_precipitation |
| wind_speed | wind |
| precipitation_now | cloud_precipitation |
| temp_minus_dew_point | humidity_dew_point |
| wind_dir_sin | wind |
| wind_dir_cos | wind |
| max_temp_so_far | observed_temperature_path |
| temp_change_60m | recent_temperature_changes |
| temp_change_120m | recent_temperature_changes |
| temp_change_180m | recent_temperature_changes |
| temp_change_240m | recent_temperature_changes |
| temp_change_300m | recent_temperature_changes |
| temp_acceleration_60m | recent_temperature_changes |
| temp_change_60m_minus_3h_avg_rate | recent_temperature_changes |
| current_temp_minus_max_so_far | observed_temperature_path |
| minutes_since_max_temp_so_far | observed_temperature_path |
| hour_of_max_temp_so_far | observed_temperature_path |
| max_so_far_minus_forecast_high | forecast_relative |
| num_new_highs_last_3h | observed_temperature_path |
| temp_range_so_far | observed_temperature_path |
| area_under_temp_curve_so_far | observed_temperature_path |
| near_boundary_duration_so_far | observed_temperature_path |

The chosen set removes all `time_season` features and removes hourly forecast-derived features whose issue timestamp cannot be verified. It keeps observed-current, observed-cumulative, recent observed change, current humidity/cloud/wind/precipitation, `forecast_high`, and `max_so_far_minus_forecast_high` features.

## Final Validation Metrics

| candidate | features | NLL | bucket log loss | mean Brier | cov80 | cov90 | sigma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full reference | 39 | 1.3224 | 0.9098 | 0.0794 | 0.803 | 0.893 | 1.069 |
| best required ablation remove_time_season | 31 | 1.2534 | 0.8984 | 0.0777 | 0.804 | 0.892 | 1.053 |
| final safe reduced | 25 | 1.3324 | 0.9199 | 0.0809 | 0.830 | 0.908 | 1.156 |
| safe reduced with time check | 33 | 1.4177 | 0.9546 | 0.0838 | 0.807 | 0.895 | 1.160 |

## Timestamp-Safety Rationale

Every kept feature is either derived from the prediction timestamp itself, from observations at or before `prediction_time`, from cumulative same-day observations through `prediction_time`, or from the daily `forecast_high` already present in the supervised prediction row. No target, actual final high, settlement bucket, raw timestamp, future-valid observation, or unverified future forecast window appears in the frozen feature list.

## Limitations And Caveats

- The safest final set gives up some validation probability quality relative to the best required ablation.
- `forecast_high` is kept under the project contract that it is known for each prediction row; future data refreshes should preserve an explicit forecast issue timestamp if available.
- Recent temperature-change features are timestamp-safe but not clearly helpful in this one-year validation period; a future walk-forward study should prune them more granularly.
- The final test period remains untouched and should be used only once for final reporting.
