# NGBoost Bucket Probability Validation

- Generated at UTC: 2026-06-07T15:37:51.372390+00:00
- Source prediction file or model source: `C:\Weather Trading\Kalshi-Weather-Trading\outputs\ngboost_distribution_params_v0.csv`
- Prediction rows priced: 20006
- Probability rows generated: 120036
- Buckets per prediction row: 6
- Included splits: test, validation
- Distribution type used: normal
- Bucket mode: kalshi_around_forecast_rounding_nearest
- Min probability: 4.76142302132e-05
- Max probability: 0.539017639883
- Mean row probability sum: 1
- Max absolute deviation from row sum 1: 1.11022302463e-16
- Number of invalid rows found: 0
- Validation passed: True

## Manual CDF Examples

### Example 1

```text
forecast_high = 46
error | X_t ~ normal(mu=0.328647, scale=1.49297)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - 0.328647) / 1.49297) - NormalCDF((-1.5 - 0.328647) / 1.49297)
= 0.435368914377
```

### Example 2

```text
forecast_high = 46
error | X_t ~ normal(mu=0.328647, scale=1.49297)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - 0.328647) / 1.49297) - NormalCDF((-1.5 - 0.328647) / 1.49297)
= 0.435368914377
```

### Example 3

```text
forecast_high = 46
error | X_t ~ normal(mu=0.328647, scale=1.49297)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - 0.328647) / 1.49297) - NormalCDF((-1.5 - 0.328647) / 1.49297)
= 0.435368914377
```

### Example 4

```text
forecast_high = 46
error | X_t ~ normal(mu=0.328647, scale=1.49297)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - 0.328647) / 1.49297) - NormalCDF((-1.5 - 0.328647) / 1.49297)
= 0.435368914377
```

### Example 5

```text
forecast_high = 46
error | X_t ~ normal(mu=0.328647, scale=1.49297)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= NormalCDF((0.5 - 0.328647) / 1.49297) - NormalCDF((-1.5 - 0.328647) / 1.49297)
= 0.435368914377
```
