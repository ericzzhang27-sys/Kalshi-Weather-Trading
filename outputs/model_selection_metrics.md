# Model Selection Metrics

These metrics are fixed before further tuning. Validation metrics guide model selection. Test metrics are reserved for final reporting and should not be used for iterative tuning.

## Validation NLL / Log Score

Measures the negative log probability density assigned to the realized `forecast_error`.

- Why it matters: rewards full-distribution quality, not just point accuracy.
- Better direction: lower is better.
- Model-selection use: primary validation metric for NGBoost distribution tuning.
- Caveats: continuous-density NLL is not directly comparable to discrete empirical bucket NLL; it can reward over-narrow distributions unless coverage and calibration are checked too.

## 50%, 80%, and 90% Interval Coverage

Measures how often realized forecast errors fall inside model prediction intervals at each nominal level.

- Why it matters: exposes overconfidence and underconfidence across the distribution.
- Better direction: closer to nominal coverage is better.
- Model-selection use: reject models with strong undercoverage, especially at 80% and 90%.
- Caveats: coverage alone does not reward sharper useful intervals; track average interval width alongside coverage.

## Bucket Brier Score

Measures squared error between predicted bucket probabilities and realized bucket outcomes.

- Why it matters: evaluates the market-facing probability vector.
- Better direction: lower is better.
- Model-selection use: compare NGBoost bucket probabilities against the empirical baseline on the same bucket schema.
- Caveats: common buckets can dominate the average; inspect tail buckets separately.

## Interval Log Loss

Measures the negative log probability assigned to the realized forecast-error interval or bucket.

- Why it matters: penalizes models that assign too little probability to the bucket that actually settles.
- Better direction: lower is better.
- Model-selection use: core discrete probability metric for comparing NGBoost against the empirical baseline.
- Caveats: very small probabilities can dominate the average; probability floors must be documented and not used to hide invalid probabilities.

## Calibration Error

Measures the gap between predicted probability and empirical frequency in probability bins.

- Why it matters: probability trading requires probabilities that mean what they say.
- Better direction: lower absolute calibration error is better.
- Model-selection use: inspect overall and bucket-specific calibration, especially tails, hour, season, and horizon groups.
- Caveats: small bins are noisy; use counts and grouped diagnostics before overreacting to isolated bins.

## Unacceptable Model Criteria

A model is not acceptable for trading research if any of these hold:

- Strong undercoverage on 80% or 90% intervals.
- Severe overconfidence in calibration curves.
- Predicted sigma is unstable or unrealistically small.
- Tail probabilities are obviously too small.
- Realized buckets often receive near-zero probability.
- Validation gains do not transfer to the test period.
- The model performs worse than the empirical baseline on core probability metrics.
- Any evidence of future leakage.
- Interval probabilities fail nonnegativity or sum-to-one checks before numerical repair.
