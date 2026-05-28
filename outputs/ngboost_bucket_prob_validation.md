# NGBoost Bucket Probability Validation

- Generated at UTC: 2026-05-28T02:36:31.739259+00:00
- Source prediction file or model source: `C:\Weather Trading\Kalshi-Weather-Trading\outputs\ngboost_distribution_params_v0.csv`
- Prediction rows priced: 20904
- Probability rows generated: 125424
- Buckets per prediction row: 6
- Included splits: test, validation
- Distribution type used: normal
- Bucket mode: kalshi_around_forecast_rounding_nearest
- Min probability: 0
- Max probability: 0.999981606554
- Mean row probability sum: 1
- Max absolute deviation from row sum 1: 1.11022302463e-16
- Number of invalid rows found: 0
- Validation passed: True

## Manual CDF Examples

### Example 1

```text
forecast_high = 46
error | X_t ~ Normal(mu=-0.589201, sigma=1.4612)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.589201) / 1.4612) - NormalCDF((-1.5 - -0.589201) / 1.4612)
= 0.505454028782
```

### Example 2

```text
forecast_high = 46
error | X_t ~ Normal(mu=-0.998851, sigma=1.4612)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.998851) / 1.4612) - NormalCDF((-1.5 - -0.998851) / 1.4612)
= 0.481689646719
```

### Example 3

```text
forecast_high = 46
error | X_t ~ Normal(mu=-0.730148, sigma=1.4612)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.730148) / 1.4612) - NormalCDF((-1.5 - -0.730148) / 1.4612)
= 0.50092726338
```

### Example 4

```text
forecast_high = 46
error | X_t ~ Normal(mu=-0.606064, sigma=1.4612)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.606064) / 1.4612) - NormalCDF((-1.5 - -0.606064) / 1.4612)
= 0.505121496092
```

### Example 5

```text
forecast_high = 46
error | X_t ~ Normal(mu=-0.698531, sigma=1.4949)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.698531) / 1.4949) - NormalCDF((-1.5 - -0.698531) / 1.4949)
= 0.49271865217
```
