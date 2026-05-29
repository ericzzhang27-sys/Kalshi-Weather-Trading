# NGBoost Improvement Plan

## Summary of Day 13 Findings

Day 13 evaluated `ngboost_normal_v0`, a Normal NGBoost/DGBM model trained on `forecast_error`.

- Validation NLL was 1.3240; test NLL worsened to 1.6855.
- Validation interval coverage was slightly high: 50% = 53.43%, 80% = 83.00%, 90% = 90.43%.
- Test interval coverage was low: 50% = 43.10%, 80% = 72.65%, 90% = 83.71%.
- Test interval log loss was 0.9550 for NGBoost versus 1.4889 for the empirical baseline.
- Test mean bucket Brier score was 0.1017 for NGBoost versus 0.1542 for the empirical baseline.
- Bucket probability conversion passed coherence checks: probabilities were nonnegative, row sums were 1, and no invalid rows were found.
- No dedicated feature-importance or feature-stability artifact was found, so feature robustness remains unproven.
- `outputs/best_model_notes.md` was missing before this Day 14 consolidation and has been created.

## Three Biggest Weaknesses

### 1. Test-period undercoverage and validation/test gap

The model is materially overconfident in the test period. `outputs/coverage_report.csv` shows test coverage of 72.65% for 80% intervals and 83.71% for 90% intervals. `outputs/coverage_by_group.csv` shows worse 80% test coverage by hour/horizon, including prediction hour 9 and horizon 6 at 63.37%, and season 0 at 64.04%. `outputs/ngboost_evaluation_report.csv` also shows NLL worsening from 1.3240 on validation to 1.6855 on test.

This matters because probability trading punishes overconfident distributions: underpriced tail or adjacent-bucket risk can create repeated false edges. Next fixes should test stronger validation discipline, scale calibration, and distribution choices that reduce out-of-sample undercoverage.

### 2. Tail and bucket calibration errors

The model assigns too little probability to some realized tail outcomes. `outputs/bucket_brier_scores.csv` shows test `market_bucket_0` predicted at 9.06% versus a 16.63% empirical frequency, and `market_bucket_5` predicted at 0.90% versus 1.98%. `outputs/calibration_tables.csv` also shows large tail-bin gaps, including test `(-inf, -3]` in the 0.4-0.5 bin with 43.94% mean prediction versus 70.68% realized frequency.

This matters because tail buckets can be where market prices look most attractive, and too-small tail probabilities can make the model miss real risk. Next fixes should test tail-aware calibration, heavier-tailed NGBoost distributions where available, and bucket-level reliability checks.

### 3. Predicted scale is too brittle

The Normal NGBoost scale does not absorb test-period error dispersion. `outputs/standardized_residual_summary.csv` shows test standardized residual std of 1.2978 and mean of -0.2566, compared with validation std of 1.0529 and mean of 0.0523. `outputs/ngboost_nll_v0.json` reports test sigma min of 0.2335 and median of 1.3314.

This matters because sigma controls the whole probability surface. If it is too small or unstable, interval probabilities and tail prices become overconfident even when the location estimate is reasonable. Next fixes should test sigma floors or shrinkage, scale calibration, robust distributions, and feature restrictions that reduce brittle same-day scale estimates.

## Frozen Model Hierarchy

- Primary model: NGBoost / DGBM on `forecast_error`.
- Required baseline: empirical historical forecast-error distribution.
- Optional future extensions only: CDF classifier, multiclass bucket model, quantile model, alternative DGBM implementations.
- Bucket probabilities must come from the model-implied CDF.
- Model selection should prioritize probability quality, calibration, interval coverage, and robustness over point accuracy.

## Robustness Roadmap

### Distribution Choice

- Compare Normal NGBoost against NGBoost-supported robust or heavier-tailed distributions, if available.
- Keep all candidates inside the same forecast-error distribution framework.
- Check PIT shape, standardized residuals, tail coverage, and NLL by split.

### Hyperparameters

- Tune tree depth, leaf size, number of estimators, learning rate, column sampling, and minibatch settings on validation only.
- Add explicit checks for over-narrow sigma and validation/test transfer.
- Prefer stable validation gains over small NLL improvements.

### Features

- Audit same-day temperature-derived features for brittleness by hour and horizon.
- Test feature subsets that reduce dependence on noisy late-day state variables.
- Add feature-stability diagnostics before trusting feature-driven sigma changes.

### Calibration

- Calibrate distribution scale and bucket probabilities on validation only.
- Track calibration curves by bucket, tail bucket, hour, season, and horizon.
- Avoid calibration layers that improve validation but degrade test-period coverage.

### Validation

- Keep chronological validation and reserve test for final reporting.
- Add walk-forward or rolling-window diagnostics before selecting a tuned model.
- Compare every tuned model against the empirical baseline on the fixed metric set.

### Interval Conversion

- Preserve probability coherence: nonnegative bucket probabilities, included tails, and row sums of 1.
- Validate endpoint conventions for Kalshi-style temperature buckets before interpreting probabilities.
- Fail fast if interval probabilities require numerical repair to become valid.
