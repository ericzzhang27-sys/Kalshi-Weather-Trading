# Best NGBoost v2 Selection Notes

## Summary

Selected `more_trees_lower_lr` as `ngboost_best_v2` using chronological validation probability metrics only.

## Chronological Split

Train dates: 2022-01-01 to 2023-12-31
Validation dates: 2024-01-01 to 2024-12-31
Test dates: 2025-01-01 to 2026-05-20
Test set used during tuning: No

## Search Space

- `v1_reference_or_near_reference`: n_estimators=300, learning_rate=0.03, max_depth=2, min_samples_leaf=20, minibatch_frac=1.0
- `more_trees_lower_lr`: n_estimators=500, learning_rate=0.02, max_depth=2, min_samples_leaf=20, minibatch_frac=1.0
- `many_trees_low_lr`: n_estimators=700, learning_rate=0.01, max_depth=2, min_samples_leaf=20, minibatch_frac=1.0
- `slightly_deeper`: n_estimators=500, learning_rate=0.02, max_depth=3, min_samples_leaf=20, minibatch_frac=1.0
- `deeper_more_regularized`: n_estimators=700, learning_rate=0.01, max_depth=3, min_samples_leaf=30, minibatch_frac=1.0
- `shallow_large_leaf`: n_estimators=500, learning_rate=0.02, max_depth=2, min_samples_leaf=50, minibatch_frac=1.0
- `shallow_subsampled`: n_estimators=500, learning_rate=0.02, max_depth=2, min_samples_leaf=50, minibatch_frac=0.8

## Metrics Used

Validation NLL/log score was the primary metric.
Coverage, bucket Brier score, interval log loss, and calibration error were used as safeguards.
MAE, RMSE, and bias were tracked as secondary diagnostics only.

## Best Candidate

- n_estimators: 500
- learning_rate: 0.02
- max_depth: 2
- min_samples_leaf: 20
- minibatch_frac: 1.0
- natural_gradient: True
- random_state: 42

## Validation Results

- validation NLL: 1.322431
- 50% coverage: 0.5122
- 80% coverage: 0.8026
- 90% coverage: 0.8933
- bucket Brier score: 0.476141
- interval log loss: 0.909813
- calibration error: 0.013203
- MAE: 0.918882
- RMSE: 1.447715
- bias: 0.122936

## Why This Candidate Was Selected

The best validation NLL was 1.316139. Candidates within 0.020 NLL were tie-broken by 80% and 90% coverage error, interval log loss, bucket Brier score, calibration error, and simpler tree settings.

- validation NLL improvements are tiny between the top candidates
- best validation NLL candidate was not selected because coverage/tie-breakers favored more_trees_lower_lr

## Comparison to v1

Existing v1 metrics were loaded from `outputs/ngboost_distribution_comparison.csv`.

| metric | v1 validation | v2 validation |
|---|---:|---:|
| NLL | 1.324001 | 1.322431 |
| 80% coverage | 0.8300 | 0.8026 |
| 90% coverage | 0.9043 | 0.8933 |
| interval log loss | 0.907309 | 0.909813 |
| bucket Brier | 0.079078 prior mean-per-bucket | 0.476141 multiclass sum |
| calibration error | not available | 0.013203 |

## Caveats

- This is one chronological validation period, so validation overfitting is still possible.
- Weather forecast errors can be seasonal, and a single validation year may not represent every regime.
- Heavy-tail behavior may remain imperfect under the configured NGBoost distribution.
- Calibration can still be imperfect even when NLL improves.
- The test set remains untouched and is reserved for final evaluation.

## Test Set Status

The test set remains untouched for final evaluation.

## Run Accounting

- Successful candidates: 7
- Failed candidates logged: 0
- Refit selected model on train + validation: Yes
