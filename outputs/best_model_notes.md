# Best Model Notes

## Current Best NGBoost Configuration

- Model: `ngboost_normal_v0`
- Target: `forecast_error`
- Distribution: Normal
- Score: log score / negative log likelihood
- Base learner: `DecisionTreeRegressor(max_depth=2, min_samples_leaf=50, random_state=11)`
- NGBoost settings: `n_estimators=120`, `learning_rate=0.05`, `minibatch_frac=1.0`, `col_sample=1.0`, `random_state=11`
- Feature count: 39 leakage-screened numeric features
- Imputation: median fit on train only

## Current Metrics

From `outputs/ngboost_evaluation_report.csv` and `outputs/coverage_report.csv`:

- Validation NLL: 1.3240
- Test NLL: 1.6855
- Validation interval log loss: 0.8734
- Test interval log loss: 0.9550
- Empirical baseline test interval log loss: 1.4889
- Validation mean bucket Brier: 0.0938
- Test mean bucket Brier: 0.1017
- Empirical baseline test mean bucket Brier: 0.1542
- Validation coverage: 50% = 53.43%, 80% = 83.00%, 90% = 90.43%
- Test coverage: 50% = 43.10%, 80% = 72.65%, 90% = 83.71%

## Current Calibration Weaknesses

- Test-period undercoverage is material at 80% and 90%.
- Test NLL is much worse than validation NLL, so robustness has not transferred cleanly.
- Tail buckets receive too little probability in test, especially `market_bucket_0` and `market_bucket_5`.
- Test standardized residuals are shifted and too dispersed: mean = -0.2566, std = 1.2978.

## Day 15+ Improvement Priorities

1. Test distribution and sigma-calibration changes that reduce test-style undercoverage.
2. Improve tail and bucket calibration without breaking probability coherence.
3. Add walk-forward validation and feature-stability diagnostics before further tuning.

Do not tune on the test set. Use validation and walk-forward diagnostics for model selection; reserve test metrics for final reporting.
