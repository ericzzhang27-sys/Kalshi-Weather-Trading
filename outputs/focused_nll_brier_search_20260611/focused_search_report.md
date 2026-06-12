# Focused NGBoost NLL/Brier Search

- Generated UTC: 2026-06-11T22:08:58.559621+00:00
- Elapsed seconds: 2368.4
- Successful runs: 21
- Failed runs: 0
- Selection uses validation only; test rows are reported for the selected validation winners.

## Validation Winner: rank_sum

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | test | 1.82977 | 1.10182 | 0.116778 | 2.70072 | 2.61981 | 1 | 3 | 2 | 6 |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | validation | 1.88369 | 1.11955 | 0.121212 | 2.53678 | 2.56743 | 1 | 3 | 2 | 6 |

## Validation Winner: nll

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | test | 1.82977 | 1.10182 | 0.116778 | 2.70072 | 2.61981 | 1 | 3 | 2 | 6 |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | validation | 1.88369 | 1.11955 | 0.121212 | 2.53678 | 2.56743 | 1 | 3 | 2 | 6 |

## Validation Winner: brier

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.15 | test | 1.8414 | 1.10585 | 0.117047 | 2.82348 | 2.73889 | 4 | 1 | 1 | 6 |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.15 | validation | 1.88602 | 1.11951 | 0.121025 | 2.65209 | 2.68413 | 4 | 1 | 1 | 6 |

## Best Validation Rows

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.15 | validation | 1.88602 | 1.11951 | 0.121025 | 2.65209 | 2.68413 | 4 | 1 | 1 | 6 |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | validation | 1.88369 | 1.11955 | 0.121212 | 2.53678 | 2.56743 | 1 | 3 | 2 | 6 |
| plus_forecast_high_temp_range__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_temp_range | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.15 | validation | 1.88574 | 1.12132 | 0.121288 | 2.62733 | 2.65495 | 3 | 5 | 3 | 11 |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.2 | validation | 1.89223 | 1.12202 | 0.121068 | 2.7674 | 2.80083 | 10 | 2 | 4 | 16 |
| plus_forecast_high_temp_range__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_temp_range | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | validation | 1.88504 | 1.1225 | 0.121552 | 2.5131 | 2.53952 | 2 | 11 | 5 | 18 |
| plus_forecast_high_temp_range__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_temp_range | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.2 | validation | 1.89051 | 1.12281 | 0.121254 | 2.74156 | 2.77039 | 9 | 4 | 7 | 20 |
| current36__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | current36 | 36 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.15 | validation | 1.88853 | 1.12335 | 0.121526 | 2.62975 | 2.65607 | 7 | 10 | 8 | 25 |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.05 | validation | 1.88619 | 1.12272 | 0.121659 | 2.42148 | 2.45073 | 5 | 15 | 6 | 26 |
| current36__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | current36 | 36 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.2 | validation | 1.89295 | 1.12462 | 0.121466 | 2.74409 | 2.77156 | 11 | 9 | 9 | 29 |
| current36__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | current36 | 36 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | validation | 1.88822 | 1.12477 | 0.121817 | 2.51542 | 2.54059 | 6 | 18 | 10 | 34 |
| plus_forecast_high_day_cos__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_day_cos | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.15 | validation | 1.89347 | 1.125 | 0.121556 | 2.61658 | 2.63174 | 13 | 12 | 11 | 36 |
| plus_forecast_high_day_cos__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_day_cos | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.2 | validation | 1.897 | 1.1257 | 0.121461 | 2.73034 | 2.74616 | 17 | 8 | 12 | 37 |
| plus_forecast_high_temp_range__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_temp_range | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.25 | validation | 1.89855 | 1.1265 | 0.121425 | 2.85579 | 2.88582 | 19 | 7 | 13 | 39 |
| plus_forecast_high_temp_range__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_temp_range | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.05 | validation | 1.88941 | 1.12691 | 0.122075 | 2.39886 | 2.42409 | 8 | 24 | 16 | 48 |
| plus_forecast_high_day_cos__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_forecast_high_day_cos | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.1 | validation | 1.89418 | 1.12706 | 0.121887 | 2.50281 | 2.51731 | 14 | 20 | 17 | 51 |
| plus_ndfd_metadata__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.25 | validation | 1.90155 | 1.12665 | 0.121316 | 2.88271 | 2.91753 | 31 | 6 | 14 | 51 |
| current36__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | current36 | 36 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.25 | validation | 1.9007 | 1.12811 | 0.121611 | 2.85843 | 2.88704 | 27 | 14 | 21 | 62 |
| current36__skew_normal__mid_300_lr002_depth3_leaf50_sub08_direct | current36 | 36 | skew_normal | mid_300_lr002_depth3_leaf50_sub08_direct | 1.2 | validation | 1.89699 | 1.1272 | 0.122286 | 2.70172 | 2.65573 | 16 | 30 | 19 | 65 |
| curated_weather_plus__skew_normal__fast_120_lr005_depth2_leaf50_sub1_direct | curated_weather_plus | 46 | skew_normal | fast_120_lr005_depth2_leaf50_sub1_direct | 1.15 | validation | 1.90017 | 1.12875 | 0.121899 | 2.66945 | 2.71056 | 24 | 21 | 23 | 68 |
| plus_ndfd_metadata__skew_normal__mid_300_lr002_depth3_leaf50_sub08_direct | plus_ndfd_metadata | 38 | skew_normal | mid_300_lr002_depth3_leaf50_sub08_direct | 1.15 | validation | 1.89923 | 1.12712 | 0.122286 | 2.63522 | 2.57829 | 21 | 31 | 18 | 70 |