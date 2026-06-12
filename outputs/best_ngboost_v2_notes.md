# Best NGBoost v2 Selection Notes

## Summary

Selected `official_migration_depth3_subsample_15` as `ngboost_best_v2` using chronological validation probability metrics only.

## Chronological Split

Train dates: 2022-01-01 to 2023-12-31
Validation dates: 2024-01-01 to 2024-12-31
Test dates: 2025-01-01 to 2026-05-20
Test set used during tuning: No

## Search Space

- `official_migration_reference_10`: n_estimators=10, learning_rate=0.05, max_depth=2, min_samples_leaf=50, minibatch_frac=1.0
- `official_migration_leaf20_15`: n_estimators=15, learning_rate=0.04, max_depth=2, min_samples_leaf=20, minibatch_frac=1.0
- `official_migration_depth3_subsample_15`: n_estimators=15, learning_rate=0.04, max_depth=3, min_samples_leaf=50, minibatch_frac=0.8

## Metrics Used

Validation NLL/log score was the primary metric.
Coverage, bucket Brier score, interval log loss, and calibration error were used as safeguards.
MAE, RMSE, and bias were tracked as secondary diagnostics only.

## Best Candidate

- n_estimators: 15
- learning_rate: 0.04
- max_depth: 3
- min_samples_leaf: 50
- minibatch_frac: 0.8
- natural_gradient: True
- random_state: 42

## Validation Results

- validation NLL: 1.755863
- 50% coverage: 0.5680
- 80% coverage: 0.8536
- 90% coverage: 0.9248
- bucket Brier score: 0.605187
- interval log loss: 1.127420
- calibration error: 0.023351
- MAE: 1.105213
- RMSE: 1.445178
- bias: 0.024038

## Why This Candidate Was Selected

The best validation NLL was 1.755863. Candidates within 0.020 NLL were tie-broken by 80% and 90% coverage error, interval log loss, bucket Brier score, calibration error, and simpler tree settings.

## Comparison to v1

Existing v1 validation metrics were not available in a comparable summary file.

## Caveats

- This is one chronological validation period, so validation overfitting is still possible.
- Weather forecast errors can be seasonal, and a single validation year may not represent every regime.
- Heavy-tail behavior may remain imperfect under the configured NGBoost distribution.
- Calibration can still be imperfect even when NLL improves.
- The test set remains untouched and is reserved for final evaluation.

## Test Set Status

The test set remains untouched for final evaluation.

## Run Accounting

- Successful candidates: 3
- Failed candidates logged: 0
- Refit selected model on train + validation: Yes
