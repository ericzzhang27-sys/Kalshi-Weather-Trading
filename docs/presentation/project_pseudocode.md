# Project Pseudocode

```text
load observation data
load forecast data
load target rows

validate required columns
parse timestamps
check row counts and date ranges
confirm feature source timestamps are at or before prediction_time
confirm target and final-high columns are excluded from model features

for each prediction timestamp:
    collect observations available up to prediction_time
    collect forecasts available up to prediction_time
    compute calendar and time-of-day features
    compute current weather features
    compute forecast-relative features
    compute path-dependent intraday features
    compute forecast_error = actual_high - forecast_high

split rows chronologically:
    train on historical dates
    validate on later dates
    test on the final held-out period

train NGBoost distribution model:
    input: timestamp-safe features
    target: forecast_error
    output: distribution parameters such as mu and sigma

for each prediction row:
    read forecast_high
    build Kalshi-style final-temperature buckets
    convert each final-temperature bucket to forecast-error bounds
    compute P(bucket) = F(error_upper) - F(error_lower)
    validate probabilities are finite, nonnegative, and sum to 1

evaluate predictions:
    compute negative log likelihood
    compute Brier score and interval log loss
    compute calibration and reliability tables
    compute prediction interval coverage
    plot PIT histogram and coverage diagnostics

if timestamp-correct market prices are available:
    for each bucket:
        edge = model_probability - market_probability
    filter edges by fees, spread, liquidity, slippage, and risk buffer

save outputs:
    distribution parameters
    bucket probabilities
    evaluation reports
    calibration reports
    figures
    presentation notebook
```

