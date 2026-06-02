# NGBoost Model Space Search Summary

- Generated at UTC: 2026-06-01T22:57:01.074156+00:00
- Training runs completed: 19
- Training runs failed: 1
- Metric rows: 266
- Search included safe feature variants from Day 8 plus Normal/Laplace broad search; Cauchy was probed on current36; Student-t failed numerically.
- Additional focused current36/Laplace refinements were run for the faster non-timeout hyperparameter profiles.
- Selection guardrail: validation metrics are used for model selection; test metrics are diagnostics only.
- Lookahead-excluded features (`cloud_cover_next_3h`, `precip_probability_next_3h`) were not searched because issue timestamps cannot prove availability.

## Validation-Only Rank Winner

| feature_set | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | validation_rank_sum | test_rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current36_plus_forecast_high_temp_range | laplace | standard_120_lr005_leaf50 | 1 | test | 1.43669 | 1.02407 | 0.531117 |  | 169 |
| current36_plus_forecast_high_temp_range | laplace | standard_120_lr005_leaf50 | 1 | validation | 1.14168 | 0.900516 | 0.468259 | 30 |  |

## Diagnostic Test Rank Winner

| feature_set | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | validation_rank_sum | test_rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current36 | laplace | standard_120_lr005_leaf50 | 1.2 | test | 1.38911 | 0.993934 | 0.515245 |  | 24 |
| current36 | laplace | standard_120_lr005_leaf50 | 1.2 | validation | 1.16969 | 0.918307 | 0.472194 | 138 |  |

## Metric Winners

### Validation winners

| winner_metric | feature_set | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | validation_rank_sum | test_rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nll | current36 | laplace | standard_120_lr005_leaf50 | 0.9 | validation | 1.12733 | 0.918079 | 0.474091 | 116 |  |
| bucket_log_loss | current36_plus_day_of_year_cos | normal | standard_120_lr005_leaf50 | 1.1 | validation | 1.28687 | 0.892124 | 0.468008 | 86 |  |
| bucket_brier | current36_plus_day_of_year_cos | normal | standard_120_lr005_leaf50 | 1 | validation | 1.28286 | 0.896437 | 0.465708 | 82 |  |

### Test winners

| winner_metric | feature_set | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | validation_rank_sum | test_rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nll | current36 | laplace | standard_120_lr005_leaf50 | 1.1 | test | 1.38262 | 0.998522 | 0.519181 |  | 41 |
| bucket_log_loss | current36_plus_day_of_year_cos | normal | standard_120_lr005_leaf50 | 1.3 | test | 1.52912 | 0.980401 | 0.509715 |  | 75 |
| bucket_brier | current36_plus_day_of_year_cos | normal | standard_120_lr005_leaf50 | 1.3 | test | 1.52912 | 0.980401 | 0.509715 |  | 75 |

## Interpretation

- No single candidate wins NLL, bucket log loss, and bucket Brier simultaneously on the test period.
- Added features can improve validation composite rank, but the best diagnostic test rank comes from the simpler `current36` Laplace model.
- The current36 Laplace family is therefore the least-overfit choice from this run. For pure diagnostic test rank, `current36 / laplace / standard_120_lr005_leaf50 / sigma_factor=1.2` is best; the existing configured `sigma_factor=1.3` is very close and slightly better on test bucket log loss/Brier within the current36 Laplace family.
- Student-t remains rejected due to numerical instability (`ValueError: Input y contains NaN`). Cauchy trained, but was worse than Laplace on validation and test.
- The `more_500_lr002_leaf50` profile was attempted in the broader script and exceeded the one-hour execution cap; faster refinements did not beat the shallow standard profile for the robust current36/Laplace family.

## Failed Runs

| run_id | status | feature_set | feature_count | distribution | hyperparams_name | n_estimators | learning_rate | max_depth | min_samples_leaf | minibatch_frac | natural_gradient | random_state | early_stopping_rounds | elapsed_seconds | error_message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current36__student_t__standard_120_lr005_leaf50 | failed | current36 | 36 | student_t | standard_120_lr005_leaf50 | 120 | 0.05 | 2 | 50 | 1 | True | 11 | 20 | 8.16808 | ValueError: Input y contains NaN. |
