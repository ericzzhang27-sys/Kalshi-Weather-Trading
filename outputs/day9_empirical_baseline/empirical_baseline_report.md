# Empirical Historical Forecast-Error Baseline

## Goal
This baseline estimates forecast-error probabilities from similar historical forecast errors.

## Method
For each prediction row, the model gathers eligible historical forecast errors and forms an empirical CDF.

F(c) = count(error <= c) / n

The implementation uses the smoothed version:

F(c) = (count(error <= c) + alpha) / (n + 2 alpha)

Forecast-error interval probabilities are then computed as:

P(a < error <= b) = F(b) - F(a)

Open-ended intervals use F(b) for error <= b and 1 - F(a) for error > a. Interval probabilities are normalized and validated to sum to one.

## No-Future-Leakage Rule
Every prediction only used rows with date < prediction row date. Same-date and future rows are excluded.

## Fallback Hierarchy
1. same_station_doy_hour_horizon: same station, circular day-of-year window, same prediction hour, similar horizon.
2. same_season_hour_horizon: same season, same prediction hour, similar horizon.
3. same_station_month: same station and month.
4. all_past: all rows before the prediction row date.

## Validation Setup
- Train rows: 26,304
- Test rows: 12,120
- Train date range: 2022-01-01 to 2024-12-31
- Test date range: 2025-01-01 to 2026-05-20
- Intervals: (-inf, -3], (-3, -1], (-1, 1], (1, 3], (3, inf)
- min_samples: 30
- doy_window: 30
- horizon_window_hours: 6
- smoothing_alpha: 1

Forecast horizon note:
- Because the dataset uses all 24 prediction hours, forecast_horizon_hours is signed. Positive values indicate prediction times before the typical daily high period, zero indicates near-peak times, and negative values indicate post-peak prediction states. Negative values are preserved rather than clipped, because post-peak rows represent a different information regime.

## Forecast Horizon Diagnostics
| Split | Min | 25th pct | Median | 75th pct | Max | Negative | Zero | Positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | -8.00 | -2.25 | 3.50 | 9.25 | 15.00 | 8,768 | 1,096 | 16,440 |
| Test | -8.00 | -2.25 | 3.50 | 9.25 | 15.00 | 4,040 | 505 | 7,575 |

## Candidate Sample Size Diagnostics
| Min | 25th pct | Median | 75th pct | Max |
| ---: | ---: | ---: | ---: | ---: |
| 183 | 183 | 183 | 183 | 184 |

## Uniform Baseline Comparison
- Empirical mean NLL: 1.488905
- Normal mean NLL: 1.474166
- Uniform baseline NLL: 1.609438
- Empirical NLL improvement vs uniform: 0.120533
- Normal NLL improvement vs uniform: 0.135272

## Normal Distribution Baseline
This comparator uses the same leakage-safe historical candidate rows as the empirical baseline, then fits a normal distribution with the candidate mean and sample standard deviation.
- Mean NLL: 1.474166
- Median NLL: 1.236724
- Top-interval accuracy: 0.267327
- Average probability assigned to true interval: 0.255241
- Average fitted sigma: 2.122079
- Brier score over interval probabilities: 0.756382

## Results
- Number of test rows: 12,120
- Mean NLL: 1.488905
- Median NLL: 1.349155
- Top-interval accuracy: 0.243564
- Average probability assigned to true interval: 0.252455
- Average sample size: 183.18
- Brier score over interval probabilities: 0.771243

## Fallback Usage
| Fallback level | Count | Percent | Mean NLL | Average sample size |
| --- | ---: | ---: | ---: | ---: |
| same_station_doy_hour_horizon | 12,120 | 100.00% | 1.488905 | 183.18 |

## Limitations
- Sparse samples can make some fallback levels noisy.
- Estimates may be unstable for extreme weather.
- Weather and forecast systems can change over time, creating possible regime changes.
- Station-specific differences may remain even after fallback filtering.
- This is a benchmark, not the final model.

## Role in Project
NGBoost/DGBM should beat this baseline on chronological validation.
