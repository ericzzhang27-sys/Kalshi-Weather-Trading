# Refined Skew-Normal NLL/Brier Search

- Generated UTC: 2026-06-11T22:52:03.393751+00:00
- Elapsed seconds: 2138.9
- Successful runs: 28
- Failed runs: 0
- Selection uses validation only; test rows are reported for the selected validation winners.

## Validation Winner: rank_sum

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.1 | test | 1.81812 | 1.09242 | 0.115849 | 2.69496 | 2.64219 | 4 | 5 | 2 | 11 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.1 | validation | 1.86868 | 1.11087 | 0.120375 | 2.50773 | 2.54306 | 4 | 5 | 2 | 11 |

## Validation Winner: nll

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.1 | test | 1.81299 | 1.09205 | 0.115951 | 2.65867 | 2.62223 | 1 | 25 | 6 | 32 |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.1 | validation | 1.86699 | 1.11151 | 0.120605 | 2.49289 | 2.51964 | 1 | 25 | 6 | 32 |

## Validation Winner: brier

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.15 | test | 1.83077 | 1.09721 | 0.11618 | 2.81746 | 2.76228 | 10 | 1 | 3 | 14 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.15 | validation | 1.87126 | 1.11096 | 0.120216 | 2.62171 | 2.65866 | 10 | 1 | 3 | 14 |

## Best Validation Rows

| run_id | feature_set | feature_count | distribution | hyperparams_name | sigma_factor | split | nll | bucket_log_loss | bucket_brier | mean_sigma | median_sigma | nll_rank | brier_rank | log_loss_rank | rank_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.12 | validation | 1.86919 | 1.11057 | 0.120283 | 2.55332 | 2.5893 | 6 | 4 | 1 | 11 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.1 | validation | 1.86868 | 1.11087 | 0.120375 | 2.50773 | 2.54306 | 4 | 5 | 2 | 11 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.15 | validation | 1.87126 | 1.11096 | 0.120216 | 2.62171 | 2.65866 | 10 | 1 | 3 | 14 |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.15 | validation | 1.86923 | 1.11145 | 0.12042 | 2.60621 | 2.63417 | 7 | 7 | 5 | 19 |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.12 | validation | 1.86737 | 1.11115 | 0.120502 | 2.53822 | 2.56546 | 2 | 14 | 4 | 20 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.08 | validation | 1.86892 | 1.11166 | 0.120508 | 2.46213 | 2.49683 | 5 | 16 | 7 | 28 |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.18 | validation | 1.87248 | 1.11266 | 0.12042 | 2.67419 | 2.70289 | 15 | 6 | 10 | 31 |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.1 | validation | 1.86699 | 1.11151 | 0.120605 | 2.49289 | 2.51964 | 1 | 25 | 6 | 32 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.18 | validation | 1.87471 | 1.11226 | 0.120231 | 2.69011 | 2.72801 | 24 | 2 | 8 | 34 |
| plus_day_cos_forecast_high__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_day_cos_forecast_high | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.18 | validation | 1.87462 | 1.11295 | 0.120434 | 2.62896 | 2.663 | 23 | 9 | 12 | 44 |
| plus_day_cos_forecast_high__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_day_cos_forecast_high | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.15 | validation | 1.87305 | 1.11282 | 0.12051 | 2.56212 | 2.59529 | 18 | 17 | 11 | 46 |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.08 | validation | 1.86739 | 1.11237 | 0.120748 | 2.44757 | 2.47383 | 3 | 40 | 9 | 52 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.2 | validation | 1.87768 | 1.11358 | 0.120282 | 2.7357 | 2.77425 | 35 | 3 | 16 | 54 |
| plus_ndfd_day_cos_forecast_high__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_day_cos_forecast_high | 40 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.15 | validation | 1.8754 | 1.1133 | 0.120507 | 2.5985 | 2.65151 | 28 | 15 | 13 | 56 |
| plus_ndfd_temp_range__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_temp_range | 39 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.2 | validation | 1.87533 | 1.11392 | 0.120462 | 2.71952 | 2.7487 | 27 | 10 | 20 | 57 |
| plus_day_cos_forecast_high__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_day_cos_forecast_high | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.2 | validation | 1.87642 | 1.11353 | 0.120426 | 2.67351 | 2.70813 | 34 | 8 | 15 | 57 |
| plus_ndfd_day_cos_forecast_high__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_day_cos_forecast_high | 40 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.12 | validation | 1.87418 | 1.11343 | 0.120608 | 2.53072 | 2.58234 | 21 | 26 | 14 | 61 |
| plus_day_cos_forecast_high__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_day_cos_forecast_high | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.12 | validation | 1.87301 | 1.11366 | 0.120669 | 2.49528 | 2.52759 | 17 | 29 | 18 | 64 |
| plus_ndfd_interactions__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_interactions | 40 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.15 | validation | 1.87289 | 1.11429 | 0.120598 | 2.59831 | 2.5891 | 16 | 24 | 25 | 65 |
| plus_ndfd_metadata__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.05 | validation | 1.87088 | 1.11386 | 0.120787 | 2.39374 | 2.42747 | 9 | 42 | 19 | 70 |
| plus_ndfd_interactions__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_interactions | 40 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.12 | validation | 1.87157 | 1.11437 | 0.120688 | 2.53053 | 2.52156 | 11 | 34 | 26 | 71 |
| plus_ndfd_day_cos_forecast_high__skew_normal__regularized_160_lr004_depth2_leaf75_sub1_direct | plus_ndfd_day_cos_forecast_high | 40 | skew_normal | regularized_160_lr004_depth2_leaf75_sub1_direct | 1.18 | validation | 1.87807 | 1.11411 | 0.120487 | 2.66629 | 2.72068 | 37 | 11 | 24 | 72 |
| plus_ndfd_metadata__skew_normal__quick_100_lr006_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | quick_100_lr006_depth2_leaf50_sub1_direct | 1.12 | validation | 1.87611 | 1.11361 | 0.120569 | 2.57115 | 2.64338 | 33 | 22 | 17 | 72 |
| plus_ndfd_metadata__skew_normal__quick_100_lr006_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | quick_100_lr006_depth2_leaf50_sub1_direct | 1.15 | validation | 1.87816 | 1.11395 | 0.120491 | 2.64002 | 2.71418 | 39 | 12 | 22 | 73 |
| plus_ndfd_metadata__skew_normal__quick_100_lr006_depth2_leaf50_sub1_direct | plus_ndfd_metadata | 38 | skew_normal | quick_100_lr006_depth2_leaf50_sub1_direct | 1.1 | validation | 1.8756 | 1.11394 | 0.120669 | 2.52524 | 2.59617 | 29 | 30 | 21 | 80 |