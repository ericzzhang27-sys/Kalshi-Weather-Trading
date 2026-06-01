# NGBoost Bucket Probability Validation

- Generated at UTC: 2026-06-01T01:26:09.972282+00:00
- Source prediction file or model source: `C:\Weather Trading\Kalshi-Weather-Trading\outputs\ngboost_distribution_params_v0.csv`
- Prediction rows priced: 20904
- Probability rows generated: 125424
- Buckets per prediction row: 6
- Included splits: test, validation
- Distribution type used: laplace
- Bucket mode: kalshi_around_forecast_rounding_nearest
- Min probability: 0
- Max probability: 0.998636445419
- Mean row probability sum: 1
- Max absolute deviation from row sum 1: 1.11022302463e-16
- Number of invalid rows found: 0
- Validation passed: True

## Manual CDF Examples

### Example 1

```text
forecast_high = 46
error | X_t ~ laplace(mu=-1.51333, scale=1.64224)

Final bucket:
42.5 < final_high <= 44.5

Convert to forecast-error interval:
-3.5 < error <= -1.5

Probability:
P(-3.5 < error <= -1.5)
= LaplaceCDF((-1.5 - -1.51333) / 1.64224) - LaplaceCDF((-3.5 - -1.51333) / 1.64224)
= 0.354902800698
```

### Example 2

```text
forecast_high = 46
error | X_t ~ laplace(mu=-1.67011, scale=1.64224)

Final bucket:
42.5 < final_high <= 44.5

Convert to forecast-error interval:
-3.5 < error <= -1.5

Probability:
P(-3.5 < error <= -1.5)
= LaplaceCDF((-1.5 - -1.67011) / 1.64224) - LaplaceCDF((-3.5 - -1.67011) / 1.64224)
= 0.385121060252
```

### Example 3

```text
forecast_high = 46
error | X_t ~ laplace(mu=-1.62525, scale=1.64517)

Final bucket:
42.5 < final_high <= 44.5

Convert to forecast-error interval:
-3.5 < error <= -1.5

Probability:
P(-3.5 < error <= -1.5)
= LaplaceCDF((-1.5 - -1.62525) / 1.64517) - LaplaceCDF((-3.5 - -1.62525) / 1.64517)
= 0.376670447475
```

### Example 4

```text
forecast_high = 46
error | X_t ~ laplace(mu=-1.49698, scale=1.73219)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= LaplaceCDF((0.5 - -1.49698) / 1.73219) - LaplaceCDF((-1.5 - -1.49698) / 1.73219)
= 0.343005727646
```

### Example 5

```text
forecast_high = 46
error | X_t ~ laplace(mu=-1.48995, scale=1.74925)

Final bucket:
44.5 < final_high <= 46.5

Convert to forecast-error interval:
-1.5 < error <= 0.5

Probability:
P(-1.5 < error <= 0.5)
= LaplaceCDF((0.5 - -1.48995) / 1.74925) - LaplaceCDF((-1.5 - -1.48995) / 1.74925)
= 0.342569765715
```
