# Prediction Schema

- Generated at UTC: 2026-06-02T01:52:43.411495+00:00
- Default output path: `C:\Weather Trading\Kalshi-Weather-Trading\outputs\final_bucket_probability_predictions.csv`

## Inputs

- Prediction rows must contain every selected model feature in `outputs/final_feature_list.json`.
- Prediction rows must contain `forecast_high`; it is metadata for bucket conversion, not a model feature.
- Optional metadata columns are preserved when present: `row_id`, `date`, `prediction_time`, `location`, `actual_high`, `forecast_error`, `forecast_horizon_hours`, and split/station fields.
- Bucket definitions use lower-open, upper-closed final-temperature intervals: `lower_temp < actual_high <= upper_temp`.
- If no bucket schema is provided, Kalshi-style six-bucket schemas are built around each row's `forecast_high`.

## Output Columns

| Column | Meaning |
|---|---|
| `row_id` | Stable prediction-row identifier. Generated if missing. |
| `bucket_index`, `bucket_name` | Bucket position and display label. |
| `bucket_lower_temp`, `bucket_upper_temp` | Final-temperature interval bounds. Blank means open-ended. |
| `error_lower`, `error_upper` | Forecast-error interval after subtracting `forecast_high`. Blank means open-ended. |
| `probability` | `P(error_lower < forecast_error <= error_upper) = F(error_upper) - F(error_lower)`. |
| `mu` | Predicted forecast-error location. |
| `sigma` | Final calibrated forecast-error scale used for bucket pricing. |
| `raw_sigma` | Scale after model-level sigma scaling and before post-hoc calibration alpha. |
| `model_raw_sigma` | Raw scale emitted by the NGBoost artifact before engine adjustments. |
| `model_sigma_scale` | Model-level scale multiplier stored in the artifact. |
| `sigma_scaling_alpha`, `alpha` | Post-hoc calibration multiplier from `models/calibration_config.json`. |
| `distribution_type` | Distribution family used by the CDF, such as `laplace` or `normal`. |
| `model_name`, `calibration_method` | Artifact and calibration provenance. |
| `feature_*` diagnostics | Per-row missing/infinite feature values imputed or replaced before prediction. |

## Validation

- Required feature columns are checked before prediction and ordered to the model artifact's feature list.
- Feature values are coerced to numeric, infinities are treated as missing, and the saved train-only imputer fills missing values.
- Distribution parameters must be finite, and final `sigma` must be greater than zero.
- Bucket probabilities must be finite, nonnegative, no greater than one, and sum to one per `row_id`.

## Run Diagnostics

- `model_path`: C:\Weather Trading\Kalshi-Weather-Trading\models\ngboost_laplace_current36_default.pkl
- `model_name`: ngboost_laplace_current36_default
- `distribution_type`: laplace
- `feature_count`: 36
- `model_sigma_scale`: 1.3
- `calibration_alpha`: 0.7
- `calibration_method`: global_sigma_scaling
- `prediction_row_count`: 5
- `probability_row_count`: 30
- `bucket_count_per_prediction`: 6
- `max_abs_row_probability_sum_deviation`: 0.0
- `total_feature_values_imputed_or_replaced`: 20
