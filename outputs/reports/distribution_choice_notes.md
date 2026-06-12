# Distribution Choice Notes

The all-in-one `scripts/compare_ngboost_distributions.py` family comparison timed out during the official-data migration, including a 5-estimator smoke run. To avoid stale 54-feature output, this report compares actual current36 Normal and Laplace evaluations. Distribution selection used chronological validation metrics only; the test split stayed untouched until final evaluation.

## Current36 Results

### Validation
- `ngboost_normal_v0`: NLL=1.755863, interval_log_loss=1.089209, mean_bucket_brier=0.120448, cov80=0.8536, cov90=0.9248
- `ngboost_laplace_v0`: NLL=1.802805, interval_log_loss=1.134602, mean_bucket_brier=0.124198, cov80=0.8390, cov90=0.9444

### Test
- `ngboost_normal_v0`: NLL=1.687318, interval_log_loss=1.068653, mean_bucket_brier=0.117201, cov80=0.8836, cov90=0.9415
- `ngboost_laplace_v0`: NLL=1.688962, interval_log_loss=1.090709, mean_bucket_brier=0.118199, cov80=0.8730, cov90=0.9521


## Selection

Normal is the final selected distribution because it led the current36 validation comparison and the Normal hyperparameter grid selected `official_migration_depth3_subsample_15`. Test metrics were computed once after that selection.