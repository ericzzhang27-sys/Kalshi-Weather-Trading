# Robust Laplace Baseline Comparison

Single fixed robust model; no hyperparameter search, sigma scaling, or validation selection.

## Setup

- Feature list: `outputs\final_feature_list.json` (36 features)
- Dataset: `data\processed\modeling_rows_v1.csv`
- Fit split: train + validation through 2024-12-31
- Test split: 2025-01-01 to 2026-05-20
- Robust model params: `{"distribution": "laplace", "n_estimators": 120, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 50, "minibatch_frac": 1.0, "natural_gradient": true, "random_state": 11, "early_stopping_rounds": null}`

## Metrics

| model                            | split          |   n_rows |   continuous_nll |   interval_log_loss |   mean_bucket_brier |   top_interval_accuracy |   mean_probability_true_interval |   mean_sigma |    mean_mu |   mean_sample_size |
|:---------------------------------|:---------------|---------:|-----------------:|--------------------:|--------------------:|------------------------:|---------------------------------:|-------------:|-----------:|-------------------:|
| robust_laplace_ngboost_current36 | test_2025_plus |    11610 |         1.745881 |            1.069259 |            0.113795 |                0.550904 |                         0.426383 |     1.331258 |   0.359928 |         nan        |
| constant_normal_train_baseline   | test_2025_plus |    11610 |         2.367422 |            1.513647 |            0.152732 |                0.340396 |                         0.229165 |     2.650263 |   0.412479 |         nan        |
| empirical_baseline_day9          | test_2025_plus |    11610 |       nan        |            1.475525 |            0.150811 |                0.314470 |                         0.243686 |   nan        | nan        |         183.099053 |
| empirical_local_normal_baseline  | test_2025_plus |    11610 |       nan        |            1.469916 |            0.150234 |                0.317399 |                         0.242027 |     2.653500 |   0.543449 |         183.099053 |

## Notes

- `continuous_nll` is only directly meaningful for parametric density models.
- Interval metrics use fixed forecast-error buckets: `(-inf, -3]`, `(-3, -1]`, `(-1, 1]`, `(1, 3]`, `(3, inf)`.
- Empirical baselines are from `outputs/day9_empirical_baseline/empirical_baseline_predictions.csv` regenerated on the current feature table.
