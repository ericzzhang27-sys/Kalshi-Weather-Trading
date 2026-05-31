# NGBoost Bucket Probability Validation

- Generated at UTC: 2026-05-31T22:42:46.661440+00:00
- Source prediction file or model source: `C:\Weather Trading\Kalshi-Weather-Trading\outputs\ngboost_distribution_params_v0.csv`
- Prediction rows priced: 20904
- Probability rows generated: 125424
- Buckets per prediction row: 6
- Included splits: test, validation
- Distribution type used: normal
- Bucket mode: kalshi_around_forecast_rounding_nearest
- Min probability: 0
- Max probability: 0.999991377408
- Mean row probability sum: 1
- Max absolute deviation from row sum 1: 1.11022302463e-16
- Number of invalid rows found: 0
- Validation passed: True

## Manual CDF Examples

### Example 1

```text
forecast_high = 46
error | X_t ~ normal(mu=-0.759827, scale=1.42195)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.759827) / 1.42195) - NormalCDF((-1.5 - -0.759827) / 1.42195)
= 0.510842740062
```

### Example 2

```text
forecast_high = 46
error | X_t ~ normal(mu=-1.10292, scale=1.45076)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -1.10292) / 1.45076) - NormalCDF((-1.5 - -1.10292) / 1.45076)
= 0.473238559971
```

### Example 3

```text
forecast_high = 46
error | X_t ~ normal(mu=-0.892646, scale=1.45076)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.892646) / 1.45076) - NormalCDF((-1.5 - -0.892646) / 1.45076)
= 0.493718995675
```

### Example 4

```text
forecast_high = 46
error | X_t ~ normal(mu=-0.710999, scale=1.46819)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.710999) / 1.46819) - NormalCDF((-1.5 - -0.710999) / 1.46819)
= 0.499766923142
```

### Example 5

```text
forecast_high = 46
error | X_t ~ normal(mu=-0.812041, scale=1.4684)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - -0.812041) / 1.4684) - NormalCDF((-1.5 - -0.812041) / 1.4684)
= 0.494499413389
```
