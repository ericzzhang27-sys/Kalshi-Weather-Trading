# Weather Probability Modeling Project — Codex Build Spec

## 0. Project Purpose

Build a profitability-first, calculus-linked machine learning project for weather prediction markets, focused on Kalshi-style temperature bucket markets.

The system should estimate the **conditional distribution of NWS forecast error**, convert that distribution into probabilities for mutually exclusive final-temperature buckets, compare those probabilities to executable market prices, and decide whether a bucket or basket has positive expected value after costs, liquidity, and model uncertainty.

This is not just a school poster. The architecture should be realistic enough to support paper trading and later real trading research, while still producing clean explanations and visuals for a calculus class.

The project is **probability-estimation-first**, not trading-bot-first. Build the probability signal first. Then build paper trading and market logic around that signal.

---

## 1. Core Thesis

Kalshi temperature markets are often structured as mutually exclusive buckets, not just threshold contracts.

Example market structure:

```text
71°F or below
72°F
73°F
74°F
75°F
76°F or above
```

Exactly one bucket resolves YES. Therefore, the central problem is not simply:

```text
Will temperature be >= K?
```

The central problem is:

```text
What is the probability mass in each final-temperature bucket?
```

The updated modeling thesis is:

```text
Model NWS forecast error, not raw final temperature.
```

Define:

```text
error = actual_official_high - nws_forecast_high_available_at_time_t
```

This means the model is not trying to predict weather from scratch. It is learning the conditional distribution of how wrong the NWS forecast is likely to be, given information available at timestamp `t`.

The project should model:

```text
error | X_t
```

where:

- `error` = actual official high minus the NWS predicted high available at time `t`.
- `X_t` = all weather, forecast, and time-state information available at timestamp `t`.
- `actual_official_high` = settlement-source high temperature for the relevant station/day.
- `nws_forecast_high_available_at_time_t` = the high-temperature forecast known at timestamp `t`.

The primary model should output a full predictive distribution:

```text
error | X_t ~ Distribution(theta(X_t))
```

For example:

```text
error | X_t ~ Normal(mu(X_t), sigma(X_t))
```

or preferably, if practical:

```text
error | X_t ~ StudentT(df(X_t), loc(X_t), scale(X_t))
```

Then calculate bucket probabilities by converting final-temperature buckets into forecast-error intervals.

For final-temperature bucket `B_i = (L_i, U_i]`, convert to error bounds:

```text
error_lower_i = L_i - nws_forecast_high
error_upper_i = U_i - nws_forecast_high
```

Then compute:

```text
P(bucket_i) = P(error_lower_i < error <= error_upper_i)
            = F_error(error_upper_i | X_t) - F_error(error_lower_i | X_t)
```

The trading edge for bucket `i` is:

```text
edge_i = p_i - q_i - fees - slippage - model_margin
```

where:

- `p_i` = model probability of bucket `i`.
- `q_i` = executable Kalshi price for bucket `i`, not midpoint.
- `model_margin` = uncertainty haircut applied to avoid trading on fragile estimates.

---

## 2. Calculus Foundation

The project must explicitly demonstrate calculus knowledge.

### 2.1 Forecast-error distribution

The model estimates a probability distribution over forecast error:

```text
error = actual_official_high - nws_forecast_high
```

The PDF is:

```text
f(e | X_t)
```

The CDF is:

```text
F(c | X_t) = P(error <= c | X_t)
```

### 2.2 Bucket probability as a definite integral

For error bucket `E_i = (a_i, b_i]`:

```text
P(E_i) = integral from a_i to b_i of f(e | X_t) de
```

This means each market bucket corresponds to area under the forecast-error density curve.

### 2.3 CDF differences and the Fundamental Theorem of Calculus

Since the PDF is the derivative of the CDF:

```text
f(e | X_t) = d/de F(e | X_t)
```

The Fundamental Theorem of Calculus gives:

```text
P(a < error <= b) = F(b | X_t) - F(a | X_t)
```

This is the main formula used to price buckets.

### 2.4 Whole-degree reporting and half-degree boundaries

If official highs are reported as whole degrees, treat each integer as a half-degree interval.

If NWS forecast high is `73°F` and market buckets are:

```text
71 or lower, 72, 73, 74, 75, 76 or higher
```

then use half-degree final-temperature boundaries:

```text
71 or lower: T <= 71.5
72: 71.5 < T <= 72.5
73: 72.5 < T <= 73.5
74: 73.5 < T <= 74.5
75: 74.5 < T <= 75.5
76 or higher: T > 75.5
```

Convert to forecast-error boundaries by subtracting forecast high `73`:

```text
71 or lower: error <= -1.5
72: -1.5 < error <= -0.5
73: -0.5 < error <= +0.5
74: +0.5 < error <= +1.5
75: +1.5 < error <= +2.5
76 or higher: error > +2.5
```

Then:

```text
P(72) = F(-0.5) - F(-1.5)
P(73) = F(+0.5) - F(-0.5)
P(74) = F(+1.5) - F(+0.5)
```

For edge buckets:

```text
P(71 or lower) = F(-1.5)
P(76 or higher) = 1 - F(+2.5)
```

### 2.5 Expected value

For a $1 payout bucket:

```text
EV_i = p_i - q_i
```

after costs:

```text
EV_i = p_i - q_i - fees_i - slippage_i - model_margin_i
```

### 2.6 Optimization across buckets and baskets

Single bucket:

```text
choose i maximizing EV_i
```

Basket `S`:

```text
p_S = sum(p_i for i in S)
q_S = sum(q_i for i in S)
EV_S = p_S - q_S
```

### 2.7 Sensitivity to forecast-error center and uncertainty

If a simplified model assumes:

```text
error | X_t ~ Normal(mu, sigma)
```

then interval probabilities shift when `mu` and `sigma` change. This can be visualized by showing probability mass moving between adjacent error intervals.

For a CDF threshold:

```text
F(c) = P(error <= c)
```

For a Normal distribution:

```text
F(c) = Phi((c - mu) / sigma)
```

The PDF is the derivative of the CDF with respect to `c`:

```text
dF/dc = f(c)
```

### 2.8 Gradient boosting uses calculus

Gradient boosting minimizes a loss function through derivative-based updates.

For distributional GBM / NGBoost, the model learns parameters of a predictive distribution by optimizing a proper scoring rule such as negative log likelihood.

For a predicted density `p_theta(y | x)`:

```text
NLL = -sum(log p_theta(y_i | x_i))
```

The model improves distribution parameters using gradient-based updates.

For CDF-classification benchmarks, binary/logistic loss is:

```text
L = -sum[y_i log(p_i) + (1 - y_i) log(1 - p_i)]
dL/dF_i = p_i - y_i
```

---

## 3. Final Modeling Decision

Use a profitability-first, **forecast-error distribution-first** architecture.

### 3.1 Primary model: distributional GBM / NGBoost on forecast error

Primary model:

```text
Distributional gradient boosting / NGBoost on NWS forecast error
```

Target:

```text
y = error = actual_official_high - nws_forecast_high_available_at_time_t
```

Input:

```text
X_t = all timestamp-safe information available at time t
```

Output:

```text
predictive distribution of error | X_t
```

Example output:

```text
Normal(loc=mu(X_t), scale=sigma(X_t))
```

or, if supported and stable:

```text
StudentT(df, loc, scale)
```

Bucket probabilities:

```text
P(a < error <= b) = F_error(b | X_t) - F_error(a | X_t)
```

Recommended first implementation:

```text
NGBoost with Normal distribution, optimized by negative log likelihood.
```

Recommended stronger implementation:

```text
NGBoost with heavier-tailed distribution if practical, or a residual distribution adjustment if Normal tails are too thin.
```

Why this is better than quantile-first:

```text
The model directly outputs a full predictive distribution and CDF.
Bucket probabilities come directly from the model-implied CDF.
It avoids interpolating backward from a few quantiles.
It is more natural for probability pricing, interval probabilities, log score, and calibration.
```

### 3.2 Benchmark 1: CDF-based cumulative GBM classifiers

Train separate gradient-boosted binary classifiers for key error boundaries `c`.

For each boundary:

```text
target_c = 1 if error <= c else 0
```

Model output:

```text
F(c | X_t) = P(error <= c | X_t)
```

Then compute bucket probabilities:

```text
P(a < error <= b) = F(b | X_t) - F(a | X_t)
```

Important monotonicity constraint:

```text
F(c1) <= F(c2) for c1 < c2
```

If separate classifiers violate monotonicity, fix with:

```text
sorting cumulative probabilities
isotonic regression across boundary predictions
ordinal model structure if implemented later
```

This benchmark is valuable because it prices the exact bucket boundaries directly.

### 3.3 Benchmark 2: multiclass GBM on forecast-error buckets

Build a direct bucket classifier:

```text
X_t -> [P(B1), P(B2), P(B3), P(B4), P(B5), ...]
```

Target:

```text
y_bucket = bucket containing realized forecast error
```

This directly matches the market structure, but it may mishandle ordering and tails. It should be a benchmark or ensemble component, not the primary model.

### 3.4 Baseline: empirical historical forecast-error distribution

Model:

```text
error = actual_official_high - forecast_high_available_at_time_t
```

Use historical errors from similar contexts:

```text
station
season / day of year
time of day
forecast high level
current temp vs forecast path
max so far vs forecast path
cloud regime
wind regime
forecast horizon
```

Then compute bucket probabilities by checking how often similar errors fall inside each error interval.

This baseline is important because it is robust, interpretable, and less likely to overfit than a complex model.

### 3.5 Optional benchmark: quantile GBM on forecast error

Quantile GBM is now optional, not primary.

Target:

```text
y = error
```

Train:

```text
Q05, Q10, Q25, Q50, Q75, Q90, Q95
```

Then interpolate a CDF from quantiles and compute bucket probabilities.

Use this only as a secondary comparison because converting a few quantiles into bucket probabilities is weaker than directly modeling the predictive distribution or CDF.

### 3.6 Final ensemble

Blend probability estimates:

```text
p_i_final = w1 * p_i_dgbm + w2 * p_i_cdf_gbm + w3 * p_i_multiclass + w4 * p_i_empirical + w5 * p_i_quantile_optional
```

Weights should be selected based on out-of-sample:

```text
negative log likelihood / log score
bucket log loss
bucket Brier score
calibration
coverage
paper-trading EV by edge size
```

For the first build, use fixed weights or choose the best single model on chronological validation data. Later, optimize weights on validation data.

---

## 4. Sequential Nowcasting Framework

The system is not a one-time morning forecast. It continuously updates the forecast-error distribution throughout the day.

At each timestamp `t`:

```text
weather observations update
forecast path updates
max temp so far updates
forecast error distribution updates
final-temperature bucket probabilities update
market prices update
edge recalculates
```

The model is trained offline, but predictions update live.

Do not implement online retraining initially. Use:

```text
offline-trained model + live feature updates
```

Training data format:

```text
one station-day creates many timestamped rows
```

Example:

```text
2025-05-01 09:00 -> final forecast error
2025-05-01 10:00 -> final forecast error
2025-05-01 11:00 -> final forecast error
2025-05-01 12:00 -> final forecast error
...
```

The target final error is fixed for a day/forecast reference, but features change over time.

Critical rule:

```text
Every feature in X_{d,t} must be available at timestamp t.
```

---

## 5. Data Sources

### 5.1 Observation data: IEM ASOS/METAR

Use Iowa State IEM ASOS/METAR for historical and live station observations.

IEM provides automated airport weather observations and one-minute ASOS data for many U.S. ASOS sites. It is suitable for live/near-live station monitoring.

Important fields:

```text
station
valid
tmpf: air temperature in °F
dwpf: dew point in °F
relh: relative humidity
sknt: wind speed in knots
drct: wind direction degrees
p01i: 1-hour precipitation in inches
alti: altimeter pressure
mslp: sea-level pressure
vsby: visibility
gust: wind gust
skyc1-skyc4: sky coverage layers
skyl1-skyl4: cloud layer heights
wxcodes: present weather codes
feel: apparent temperature
metar: raw METAR string
```

Features from IEM:

```text
current_temp
max_temp_so_far
time_of_max_so_far
temp_change_15m
temp_change_30m
temp_change_60m
temp_change_120m
dew_point
temp_minus_dew_point
relative_humidity
wind_speed
wind_direction_sin
wind_direction_cos
wind_gust
pressure
pressure_change
precip_last_hour
visibility
observed_cloud_cover_proxy
present_weather_flags
actual_official_high or proxy final_daily_high
```

### 5.2 Forecast data: NWS forecast high preferred; Open-Meteo as proxy/feature source

The target is explicitly NWS forecast error:

```text
error = actual_official_high - NWS forecast_high_available_at_time_t
```

Preferred forecast source:

```text
NWS forecast high available at timestamp t
```

If archived NWS forecast data is difficult to obtain quickly, use Open-Meteo Historical Forecast API as a practical proxy for forecast-path features, while clearly documenting that it is not the true NWS forecast.

Use Open-Meteo Historical Forecast API for historical forecast features and Open-Meteo Forecast API for live forecast features.

Important forecast variables:

```text
temperature_2m
relative_humidity_2m
dew_point_2m
apparent_temperature
pressure_msl
surface_pressure
cloud_cover
cloud_cover_low
cloud_cover_mid
cloud_cover_high
wind_speed_10m
wind_direction_10m
wind_gusts_10m
shortwave_radiation
direct_radiation
diffuse_radiation
vapour_pressure_deficit
cape
precipitation
snowfall
precipitation_probability
rain
showers
weather_code
visibility
is_day
```

Forecast features:

```text
nws_forecast_high
forecast_temp_current_hour
forecast_high_remaining_day
forecast_high_full_day
forecast_max_so_far_by_this_time
current_temp_minus_forecast_temp
max_so_far_minus_forecast_max_so_far
forecast_revision_since_morning
forecast_revision_last_1h
cloud_cover_now_forecast
cloud_cover_next_1h
cloud_cover_next_3h
cloud_cover_next_6h
wind_speed_next_3h
wind_direction_next_3h
precip_probability_next_3h
shortwave_radiation_next_3h
is_day
model_disagreement if multiple forecast models are pulled
```

Do not use Open-Meteo Historical Weather / ERA5 as if it were a forecast. Reanalysis is useful for demos but can leak future information if treated as a live forecast.

### 5.3 Market data: Kalshi API

Use Kalshi API for market metadata and order books.

Kalshi order books return active bids for YES and NO sides only. Convert asks:

```text
YES ask = 100 - best NO bid
NO ask = 100 - best YES bid
```

Market/trading features:

```text
market_ticker
event_ticker
bucket_labels
bucket_boundaries
yes_bid
yes_ask
no_bid
no_ask
bid_ask_spread
orderbook_depth_by_price
average_executable_price
volume
volume_24h
open_interest
last_trade_time if available
price_change_last_10m
price_change_last_30m
price_change_last_60m
sum_of_bucket_asks
sum_of_bucket_bids
cross_bucket_mispricing
```

Market data should mostly be in the trading layer, not the pure weather probability model.

---

## 6. Data Source URLs / References

Use these links in docs/comments:

```text
IEM ASOS download:
https://mesonet.agron.iastate.edu/request/download.phtml

IEM ASOS script help:
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=

Open-Meteo Historical Forecast API:
https://open-meteo.com/en/docs/historical-forecast-api

Open-Meteo Forecast API:
https://open-meteo.com/en/docs

Kalshi orderbook docs:
https://docs.kalshi.com/api-reference/market/get-market-orderbook

Kalshi orderbook response guide:
https://docs.kalshi.com/getting_started/orderbook_responses

NGBoost project/docs:
https://stanfordmlgroup.github.io/projects/ngboost/

NGBoost paper:
https://proceedings.mlr.press/v119/duan20a.html
```

---

## 7. Feature Engineering Plan

### 7.1 Core observation features

From station observations up to timestamp `t`:

```text
current_temp
max_temp_so_far
time_of_max_so_far
minutes_since_time_of_max_so_far
temp_change_15m
temp_change_30m
temp_change_60m
temp_change_120m
dew_point
temp_minus_dew_point
relative_humidity
wind_speed
wind_direction_sin
wind_direction_cos
wind_gust
pressure
pressure_change_1h
precip_last_hour
visibility
observed_cloud_cover_proxy
present_weather_rain_flag
present_weather_fog_flag
```

### 7.2 Forecast-path features

From forecast path available at timestamp `t`:

```text
nws_forecast_high
forecast_temp_now
forecast_high_full_day
forecast_high_remaining_day
forecast_max_so_far_by_time_t
forecast_temp_next_1h
forecast_temp_next_3h_max
forecast_temp_next_6h_max
current_temp_minus_forecast_temp_now
max_so_far_minus_forecast_max_so_far
forecast_revision_since_morning
forecast_revision_last_1h
cloud_cover_now
cloud_cover_next_1h
cloud_cover_next_3h
cloud_cover_next_6h
wind_speed_next_3h
wind_direction_next_3h_sin
wind_direction_next_3h_cos
precip_probability_next_3h
shortwave_radiation_now
shortwave_radiation_next_3h_sum
```

### 7.3 Time features

```text
day_of_year
month
hour
minute
day_of_year_sin
day_of_year_cos
hour_sin
hour_cos
minutes_since_sunrise
minutes_until_sunset
minutes_until_typical_peak
fraction_of_heating_day_elapsed
is_day
```

### 7.4 Error-boundary and bucket features

For distributional GBM / NGBoost, bucket features are applied after prediction by converting temperature buckets into forecast-error intervals.

For CDF-classification benchmark rows, include boundary-specific features if training one stacked model:

```text
error_boundary_c
forecast_high_minus_final_temp_boundary
current_temp_minus_final_temp_boundary
max_so_far_minus_final_temp_boundary
```

But for the simplest CDF-classification benchmark, train separate models per boundary `c` using the same feature matrix.

### 7.5 Market/trading features

Used in trading layer:

```text
best_yes_bid
best_yes_ask
spread
volume
volume_24h
open_interest
orderbook_depth_within_edge
average_executable_price_by_size
sum_bucket_yes_asks
sum_bucket_yes_bids
market_probability_vector
market_price_change_10m
market_price_change_30m
```

---

## 8. Targets

### 8.1 Primary distributional target

For distributional GBM / NGBoost:

```text
y = error = actual_official_high - nws_forecast_high_available_at_time_t
```

This is the main supervised learning target.

### 8.2 CDF benchmark targets

For each important forecast-error boundary `c`:

```text
target_c = 1 if error <= c else 0
```

Examples:

```text
F(-1.5): target = 1 if error <= -1.5
F(-0.5): target = 1 if error <= -0.5
F(+0.5): target = 1 if error <= +0.5
F(+1.5): target = 1 if error <= +1.5
F(+2.5): target = 1 if error <= +2.5
```

### 8.3 Multiclass bucket target

For multiclass benchmark:

```text
y_error_bucket = bucket_containing_realized_forecast_error
```

Example with forecast high `73` and final-temperature buckets `<=71, 72, 73, 74, 75, >=76`:

```text
error <= -1.5 -> <=71
-1.5 < error <= -0.5 -> 72
-0.5 < error <= +0.5 -> 73
+0.5 < error <= +1.5 -> 74
+1.5 < error <= +2.5 -> 75
error > +2.5 -> >=76
```

### 8.4 Optional quantile target

For optional quantile GBM:

```text
y = error
```

Train conditional quantiles of forecast error.

---

## 9. Evaluation Metrics

### 9.1 Primary distributional model

Evaluate distribution quality, not just point prediction:

```text
negative log likelihood / log score
PIT histogram / probability integral transform diagnostics
coverage of prediction intervals, e.g. 50%, 80%, 90%
sharpness of intervals
CRPS if implemented
MAE of distribution mean or median only as secondary metric
RMSE of distribution mean only as secondary metric
bucket log loss after converting distribution to bucket probabilities
bucket Brier score after converting distribution to bucket probabilities
calibration by interval/bucket probability
```

### 9.2 CDF benchmark model

```text
binary log loss per boundary
Brier score per boundary
monotonicity violation rate
bucket log loss after differencing CDF values
bucket Brier score
calibration curves for F(c)
```

### 9.3 Multiclass bucket model

```text
multiclass log loss
multiclass Brier score
top-1 accuracy
mean absolute bucket error
confusion matrix
calibration curve by bucket
```

### 9.4 Trading evaluation

```text
edge by bucket
edge by time of day
paper PnL
PnL after fees
PnL after simulated slippage
hit rate by edge bucket
hit rate by entry time
return per unit risk
max drawdown
liquidity-adjusted edge
paper-trading EV by edge size
```

Accuracy is not the main metric. Probability quality and expected value after costs are the main metrics.

---

## 10. Trading Logic

### 10.1 Single bucket

For bucket `i`:

```text
edge_i = p_i - q_i - fees_i - slippage_i - model_margin_i
```

Trade only if:

```text
edge_i > min_edge_threshold
```

### 10.2 Basket trades

For basket `S`:

```text
p_S = sum(p_i for i in S)
q_S = sum(q_i for i in S)
edge_S = p_S - q_S - costs
```

Examples:

```text
upper buckets = {74, 75, >=76}
lower buckets = {<=71, 72}
middle buckets = {73, 74}
tail buckets = {<=71, >=76}
```

Early in the day, basket trades may be more robust than single-bucket trades.

### 10.3 Bucket-sum arbitrage

Because exactly one bucket resolves YES:

```text
sum_i true p_i = 1
```

If executable YES asks satisfy:

```text
sum_i ask_i + fees < 1
```

then buying all buckets is theoretical arbitrage if all legs fill.

If selling is possible and:

```text
sum_i bid_i - fees > 1
```

then selling all buckets is theoretical arbitrage, subject to margin and execution.

### 10.4 Liquidity-aware execution

For buying a bucket, compute average executable price for `n` contracts:

```text
q_bar(n) = weighted average ask price after walking the order book
```

Liquidity-adjusted edge:

```text
edge_liq(n) = p_adj - q_bar(n) - fees - impact
```

Only buy while marginal expected value remains positive:

```text
marginal_price < p_adj - fee_buffer - min_edge
```

### 10.5 Position sizing

Use conservative fractional Kelly.

For bucket/basket with probability `p`, cost `q`, and uncertainty haircut `m`:

```text
p_adj = p - m
f_full = (p_adj - q) / (1 - q)
f_actual = lambda * f_full
```

Use:

```text
lambda = 0.05 to 0.25
initial default lambda = 0.10
```

Apply hard caps:

```text
max risk per trade
max risk per station/day
max risk per region
max total open risk
liquidity cap
```

### 10.6 Early vs late capital allocation

Predictions usually improve as the day passes, but market prices also update. Use time/uncertainty-dependent hurdle rates.

Trade if:

```text
edge_t > fees + spread + slippage + uncertainty_margin_t
```

Early day:

```text
larger uncertainty margin
prefer broad basket trades
smaller size
```

Midday:

```text
best balance of information and non-obvious edge
specific bucket trades become more reasonable
```

Late day:

```text
model is most accurate, but edge may become latency-based
trade only non-obvious stale prices with strong settlement confidence
```

Confidence multiplier example:

```text
confidence_t = 1 - predicted_sigma_t / sigma_morning
size_t = fractional_kelly_size * confidence_t
```

---

## 11. Project Folder Structure

Build the repository like this:

```text
weather-bucket-model/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore

  config/
    stations.yaml
    markets_example.yaml
    model_config.yaml

  data/
    raw/
      observations/
      forecasts/
      kalshi/
    interim/
    processed/
    sample/

  notebooks/
    01_bucket_probability_demo.ipynb
    02_feature_engineering_check.ipynb
    03_model_diagnostics.ipynb
    04_distributional_gbm_diagnostics.ipynb
    05_visuals_for_paper.ipynb

  src/
    weather_bucket_model/
      __init__.py
      data/
        __init__.py
        iem_client.py
        open_meteo_client.py
        kalshi_client.py
        schemas.py
      features/
        __init__.py
        observation_features.py
        forecast_features.py
        time_features.py
        error_features.py
        build_dataset.py
      models/
        __init__.py
        empirical_error.py
        distributional_gbm.py
        cdf_classification_gbm.py
        multiclass_gbm.py
        quantile_gbm.py
        ensemble.py
        calibration.py
        distribution_utils.py
      trading/
        __init__.py
        bucket_pricing.py
        orderbook.py
        edge.py
        sizing.py
        policies.py
      evaluation/
        __init__.py
        metrics.py
        backtest.py
        calibration_plots.py
        coverage.py
      visualization/
        __init__.py
        density_plots.py
        probability_bars.py
        edge_plots.py
        intraday_plots.py
        distribution_plots.py
      utils/
        __init__.py
        dates.py
        validation.py
        math_utils.py

  scripts/
    fetch_iem_observations.py
    fetch_open_meteo_forecasts.py
    build_training_dataset.py
    train_empirical_error.py
    train_distributional_gbm.py
    train_cdf_classification_gbm.py
    train_multiclass_gbm.py
    train_quantile_gbm.py
    run_paper_scan.py
    make_project_figures.py

  tests/
    test_bucket_pricing.py
    test_orderbook_conversion.py
    test_no_future_leakage.py
    test_distribution_outputs.py
    test_cdf_monotonicity.py
    test_sizing.py

  reports/
    figures/
    paper_draft.md
    final_paper.md
```

---

## 12. Exact Module Responsibilities

### `data/iem_client.py`

Responsibilities:

```text
Download historical ASOS/METAR data from IEM.
Fetch recent/live station observations if possible.
Parse timestamps.
Clean missing values like 'M'.
Return standardized DataFrame.
```

Expected columns:

```text
station, valid, tmpf, dwpf, relh, drct, sknt, p01i, alti, mslp, vsby, gust, skyc1, skyc2, skyc3, skyc4, wxcodes, metar
```

### `data/open_meteo_client.py`

Responsibilities:

```text
Fetch historical forecast data for station coordinates and dates.
Fetch live forecast path.
Normalize variable names.
Return hourly forecast DataFrame.
```

Expected columns:

```text
time, temperature_2m, dew_point_2m, relative_humidity_2m, cloud_cover, cloud_cover_low, cloud_cover_mid, cloud_cover_high, wind_speed_10m, wind_direction_10m, wind_gusts_10m, pressure_msl, precipitation_probability, precipitation, rain, showers, shortwave_radiation, weather_code, visibility, is_day
```

### `data/kalshi_client.py`

Responsibilities:

```text
Fetch markets.
Fetch order books.
Convert YES/NO bids into executable asks.
Parse bucket labels if possible.
Return orderbook objects.
```

### `features/observation_features.py`

Responsibilities:

```text
Compute current temp.
Compute max temp so far.
Compute rolling temp changes.
Compute wind direction sin/cos.
Compute pressure changes.
Compute observed weather flags.
```

### `features/forecast_features.py`

Responsibilities:

```text
Align forecast path to prediction timestamp.
Compute NWS forecast high or forecast proxy high.
Compute forecast high full day.
Compute forecast high remaining day.
Compute forecast temp at current hour.
Compute current temp minus forecast temp.
Compute max-so-far minus forecast max-so-far.
Compute next 1h/3h/6h cloud/wind/precip features.
```

### `features/time_features.py`

Responsibilities:

```text
Create day/year cyclic features.
Create hour cyclic features.
Compute rough sunrise/sunset or use approximate daylight features.
Compute minutes until typical peak.
```

### `features/error_features.py`

Responsibilities:

```text
Compute forecast error target.
Convert final-temperature bucket boundaries into forecast-error boundaries.
Create CDF boundary targets for benchmark models.
Create multiclass error-bucket targets.
```

### `features/build_dataset.py`

Responsibilities:

```text
Create timestamped training rows.
For each station-day, create rows at selected times.
Ensure features only use data available up to timestamp.
Attach actual_official_high.
Attach nws_forecast_high_at_t.
Attach error target.
Attach error-boundary targets and bucket target for example market definitions.
```

Prediction timestamps for first version:

```text
09:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00 local time
```

### `models/empirical_error.py`

Responsibilities:

```text
Compute historical forecast errors.
Find similar historical rows by bins or nearest neighbors.
Return empirical distribution of forecast error.
Convert error distribution to bucket probabilities.
```

Simple similarity bins:

```text
station
month or season
hour bucket
forecast_high level
current_temp_minus_forecast_temp bin
max_so_far_minus_forecast_max_so_far bin
```

### `models/distributional_gbm.py`

Responsibilities:

```text
Train NGBoost or similar distributional GBM on forecast error.
Predict distribution parameters for new feature rows.
Return distribution object or CDF function.
Compute CDF values at error boundaries.
Compute interval probabilities F(b)-F(a).
Save/load trained model.
Evaluate NLL, coverage, PIT, and interval probabilities.
```

Suggested functions:

```python
def train_ngboost_error_model(X_train, y_train, config): ...
def predict_error_distribution(model, X): ...
def distribution_cdf(dist, values): ...
def error_interval_probabilities(dist, intervals): ...
def save_distributional_model(model, path): ...
def load_distributional_model(path): ...
```

### `models/cdf_classification_gbm.py`

Responsibilities:

```text
Train separate binary GBM classifiers for error CDF boundaries.
Predict F(c)=P(error <= c) for each boundary.
Enforce monotonicity across boundaries.
Convert CDF values to interval probabilities.
Evaluate boundary-level and bucket-level metrics.
```

Suggested functions:

```python
def make_cdf_targets(df, boundaries): ...
def train_cdf_classifiers(X, y_error, boundaries, config): ...
def predict_cdf_values(models, X, boundaries): ...
def enforce_cdf_monotonicity(cdf_values, method='isotonic_or_sort'): ...
def cdf_values_to_bucket_probs(cdf_values, boundaries): ...
```

### `models/multiclass_gbm.py`

Responsibilities:

```text
Train direct error-bucket classifier benchmark.
Predict bucket probability vector.
Evaluate multiclass log loss and Brier score.
```

### `models/quantile_gbm.py`

Responsibilities:

```text
Optional: train quantile GBM models for forecast error.
Predict quantiles for new feature rows.
Convert quantiles to an approximate CDF only for comparison.
```

### `models/distribution_utils.py`

Responsibilities:

```text
Normal/Student-t CDF wrappers.
PDF plotting helpers.
Interval probability calculation.
Probability normalization.
PIT calculation.
Prediction interval extraction.
```

### `models/ensemble.py`

Responsibilities:

```text
Blend probability vectors from distributional, CDF-classification, multiclass, empirical, and optional quantile models.
Normalize final probabilities.
Evaluate ensemble weights.
```

### `models/calibration.py`

Responsibilities:

```text
Implement calibration diagnostics.
Calibrate interval probabilities if needed.
Calibrate CDF classifier outputs if needed.
Produce reliability plots and bucket calibration tables.
```

### `trading/bucket_pricing.py`

Responsibilities:

```text
Define final-temperature bucket intervals.
Convert whole-degree buckets to half-degree bounds.
Convert final-temperature bounds to forecast-error bounds by subtracting forecast high.
Compute bucket probabilities from distribution CDF.
Validate bucket probabilities sum to 1.
```

### `trading/orderbook.py`

Responsibilities:

```text
Convert Kalshi yes/no bids to executable yes/no asks.
Compute best bid/ask.
Compute average executable price for size n.
Walk orderbook by price level.
```

### `trading/edge.py`

Responsibilities:

```text
Compute edge by bucket.
Compute basket edge.
Compute buy-all and sell-all arbitrage checks.
Include fees, spread, model margin, and slippage.
```

### `trading/sizing.py`

Responsibilities:

```text
Fractional Kelly sizing.
Model uncertainty haircut.
Liquidity cap.
Station/day/region exposure cap.
Basket sizing.
```

### `trading/policies.py`

Responsibilities:

```text
Dynamic hurdle by time of day.
Early/mid/late allocation policy.
Confirmation policy requiring edge persistence.
```

### `evaluation/backtest.py`

Responsibilities:

```text
Walk-forward evaluation by date.
No random train/test split for final result.
Simulate timestamped predictions.
Compute model metrics and trading metrics.
```

### `visualization/*`

Generate figures for paper:

```text
forecast-error density split into bucket intervals
CDF with error-boundary intervals
model probabilities vs market prices
edge by bucket
intraday temperature path vs forecast path
predicted distribution fan chart over time
calibration curve
PIT histogram
coverage plot
orderbook depth / liquidity-adjusted edge
```

---

## 13. Build Order for Codex

### Phase 1: Core math and bucket pricing

Build first:

```text
src/weather_bucket_model/trading/bucket_pricing.py
src/weather_bucket_model/utils/math_utils.py
src/weather_bucket_model/models/distribution_utils.py
tests/test_bucket_pricing.py
notebooks/01_bucket_probability_demo.ipynb
```

Requirements:

```text
Define final-temperature bucket intervals.
Convert integer buckets to half-degree continuous intervals.
Convert final-temperature bucket intervals to forecast-error intervals.
Compute Normal/Student-t CDF interval probabilities.
Support edge buckets.
Ensure probabilities sum to 1.
Generate forecast-error density plot split into buckets.
```

Acceptance checks:

```text
For forecast_high=73 and final buckets <=71, 72, 73, 74, 75, >=76, error intervals are correct.
For exhaustive error buckets, probabilities are nonnegative and sum to 1.
Moving error mu upward shifts mass to higher final-temperature buckets.
Increasing sigma increases tails.
```

### Phase 2: Data verification and dataset build

Build:

```text
src/weather_bucket_model/data/iem_client.py
src/weather_bucket_model/data/open_meteo_client.py
src/weather_bucket_model/data/schemas.py
src/weather_bucket_model/features/build_dataset.py
src/weather_bucket_model/features/error_features.py
tests/test_no_future_leakage.py
scripts/build_training_dataset.py
```

Requirements:

```text
Load/download observation data.
Load/download forecast data.
Verify timestamps and dates.
Create timestamped rows X_{d,t}.
Compute final official/proxy high.
Compute forecast_high_at_t.
Compute error target.
Reject rows where feature availability is unclear.
```

### Phase 3: Feature engineering

Build:

```text
features/observation_features.py
features/forecast_features.py
features/time_features.py
```

Requirements:

```text
Use only observations up to timestamp t.
Align forecasts by timestamp.
Compute core observation, forecast-path, and time features.
Add explicit anti-leakage checks.
```

### Phase 4: Empirical forecast-error baseline

Build:

```text
models/empirical_error.py
scripts/train_empirical_error.py
```

Requirements:

```text
Compute historical forecast errors.
Group or nearest-neighbor similar states.
Return empirical distribution of error.
Convert to bucket probabilities.
```

This is the first serious baseline because it is robust and explainable.

### Phase 5: Primary distributional GBM / NGBoost model

Build:

```text
models/distributional_gbm.py
scripts/train_distributional_gbm.py
tests/test_distribution_outputs.py
```

Requirements:

```text
Train NGBoost on forecast error.
Start with Normal distribution.
Output predictive distribution parameters.
Compute CDF at arbitrary error boundaries.
Compute interval probabilities.
Evaluate NLL/log score, coverage, PIT, and bucket Brier/log loss.
```

Acceptance checks:

```text
Predicted sigma/scale is positive.
CDF values are monotone in boundary c.
Interval probabilities are nonnegative and sum to 1.
NLL is finite on validation/test rows.
Prediction interval coverage can be computed.
```

### Phase 6: CDF-classification benchmark

Build:

```text
models/cdf_classification_gbm.py
scripts/train_cdf_classification_gbm.py
tests/test_cdf_monotonicity.py
```

Requirements:

```text
Train one binary GBM per important error boundary.
Predict F(c)=P(error <= c).
Enforce monotonicity across CDF boundaries.
Convert CDF values into bucket probabilities.
Compare to NGBoost.
```

### Phase 7: Multiclass and optional quantile benchmarks

Build:

```text
models/multiclass_gbm.py
models/quantile_gbm.py
scripts/train_multiclass_gbm.py
scripts/train_quantile_gbm.py
```

Requirements:

```text
Train direct bucket classifier benchmark.
Optionally train quantile models on forecast error.
Compare all models using chronological validation.
```

### Phase 8: Ensemble and paper-trading scanner

Build:

```text
models/ensemble.py
trading/orderbook.py
trading/edge.py
trading/sizing.py
scripts/run_paper_scan.py
evaluation/backtest.py
```

Requirements:

```text
Blend model probabilities.
Compare to sample or live Kalshi market prices.
Compute edge after costs.
Apply liquidity and sizing rules.
Output ranked opportunities.
```

### Phase 9: Visuals and report artifacts

Build:

```text
visualization/*.py
scripts/make_project_figures.py
reports/paper_draft.md
```

Requirements:

```text
Generate all project visuals.
Produce clear narrative connecting calculus, probabilistic ML, and trading.
```

---

## 14. Minimum Viable Product

If time/data is limited, build this:

```text
1. Forecast-error interval probability calculator using a parametric distribution.
2. Bucket probability visuals from forecast-error PDF/CDF.
3. Clean supervised table with X_{d,t} -> error.
4. Empirical forecast-error baseline.
5. NGBoost Normal model or fallback distributional model.
6. Conversion from predicted error distribution to final-temperature bucket probabilities.
7. Simulated market prices or manually entered Kalshi snapshot.
8. EV scanner by bucket.
9. Liquidity-aware orderbook example with fake orderbook.
10. Paper explaining full architecture and limitations.
```

MVP does not need a live trading bot.

---

## 15. Ambitious Version

If data and time permit:

```text
1. Pull real IEM observations for 5-10 U.S. stations.
2. Pull NWS forecast high history if accessible, or Open-Meteo historical forecast data as proxy.
3. Build timestamped training dataset over 2022-2026.
4. Train empirical error, NGBoost, CDF-classification, multiclass, and optional quantile models.
5. Add calibration diagnostics and ensemble.
6. Pull live IEM + forecast data.
7. Pull live Kalshi orderbooks.
8. Run paper scanner every 5-15 minutes.
9. Log model distributions, market prices, and hypothetical trades.
10. Evaluate paper PnL after settlement.
```

---

## 16. Four-Week Build Plan Status

### Completed foundation: Days 1-5

Do not redo these unless a small adjustment is necessary.

Completed / assumed completed:

```text
Project thesis and modeling direction established.
Forecast-error framing chosen.
Basic data collection started.
Raw and forecast datasets downloaded from Open-Meteo / related sources.
Initial variables and project structure discussed.
Architecture shifted from quantile GBM, to CDF-based classification, and now to distributional GBM / NGBoost as likely primary model.
```

### Day 6: Verify and clean downloaded datasets

Tasks:

```text
Inspect downloaded raw/weather forecast CSVs.
Standardize column names.
Parse timestamps and local time zones.
Validate date ranges.
Check missing values.
Check duplicate timestamps.
Check units: Fahrenheit vs Celsius, knots vs mph, etc.
Separate observation data from forecast data.
Create cleaning report.
```

Files:

```text
src/weather_bucket_model/data/schemas.py
src/weather_bucket_model/data/open_meteo_client.py
src/weather_bucket_model/utils/validation.py
scripts/verify_downloaded_data.py
```

Deliverables:

```text
data/interim/clean_observations_sample.csv
data/interim/clean_forecasts_sample.csv
reports/data_validation_report.md
```

### Day 7: Create supervised learning table skeleton

Tasks:

```text
Create one row per station-day-prediction_timestamp.
Attach forecast_high_available_at_t.
Attach actual/proxy final high.
Compute error target = actual_high - forecast_high_at_t.
Add row IDs and metadata.
Do not add future-leaking features.
```

Files:

```text
src/weather_bucket_model/features/build_dataset.py
src/weather_bucket_model/features/error_features.py
scripts/build_training_dataset.py
```

Deliverables:

```text
data/processed/training_rows_v0.csv
```

### Day 8: Build timestamp-safe observation features

Tasks:

```text
Compute current_temp.
Compute max_temp_so_far using observations <= t only.
Compute time_of_max_so_far.
Compute rolling temp changes.
Compute dew point spread.
Compute wind direction sin/cos.
Compute pressure/wind/cloud flags if available.
Add no-future-leakage tests.
```

Files:

```text
src/weather_bucket_model/features/observation_features.py
tests/test_no_future_leakage.py
```

Deliverables:

```text
data/processed/training_rows_obs_features.csv
```

### Day 9: Build timestamp-safe forecast features

Tasks:

```text
Align forecast path available at timestamp t.
Compute forecast_temp_now.
Compute forecast_high_full_day.
Compute forecast_high_remaining_day.
Compute forecast_max_so_far_by_time_t.
Compute current_temp_minus_forecast_temp_now.
Compute max_so_far_minus_forecast_max_so_far.
Compute cloud/wind/precip/shortwave next 1h/3h/6h features.
Compute forecast revision features if multiple forecast updates exist.
```

Files:

```text
src/weather_bucket_model/features/forecast_features.py
```

Deliverables:

```text
data/processed/training_rows_full_features_v1.csv
```

### Day 10: Build error-boundary and bucket conversion utilities

Tasks:

```text
Define final-temperature bucket schemas.
Convert integer buckets to half-degree final-temperature bounds.
Convert final-temperature bounds to forecast-error bounds by subtracting forecast_high_at_t.
Create example bucket labels and error intervals.
Validate edge buckets.
```

Files:

```text
src/weather_bucket_model/trading/bucket_pricing.py
src/weather_bucket_model/features/error_features.py
tests/test_bucket_pricing.py
```

Deliverables:

```text
outputs/example_error_intervals.csv
```

### Day 11: Build empirical forecast-error baseline

Tasks:

```text
Compute historical forecast-error distribution.
Group/similar-match by station, season, hour, forecast_high level, current-vs-forecast residual, max-so-far-vs-forecast residual.
Return empirical distribution for new row.
Compute interval probabilities by counting similar historical errors.
Add smoothing/fallback when sample size is small.
```

Files:

```text
src/weather_bucket_model/models/empirical_error.py
scripts/train_empirical_error.py
```

Deliverables:

```text
models/empirical_error.pkl
outputs/empirical_bucket_probs.csv
```

### Day 12: Implement primary NGBoost / distributional GBM model

Tasks:

```text
Install/verify NGBoost or prepare fallback.
Train distributional GBM on y=forecast_error.
Start with Normal distribution.
Save model.
Predict mu and sigma/scale for validation rows.
Compute negative log likelihood.
Compute prediction interval coverage.
```

Files:

```text
src/weather_bucket_model/models/distributional_gbm.py
scripts/train_distributional_gbm.py
tests/test_distribution_outputs.py
```

Deliverables:

```text
models/ngboost_error_normal.pkl
outputs/ngboost_distribution_predictions.csv
outputs/ngboost_validation_metrics.json
```

### Day 13: Convert NGBoost distributions to bucket probabilities

Tasks:

```text
For each validation row, convert market buckets into error intervals.
Use predicted distribution CDF to compute P(a < error <= b).
Validate probabilities are nonnegative and sum to 1.
Generate example probability table.
Compare bucket probabilities to empirical baseline.
```

Files:

```text
src/weather_bucket_model/models/distribution_utils.py
src/weather_bucket_model/trading/bucket_pricing.py
```

Deliverables:

```text
outputs/ngboost_bucket_probs.csv
outputs/ngboost_vs_empirical_comparison.csv
```

### Day 14: Implement distributional evaluation metrics

Tasks:

```text
Implement NLL/log score.
Implement PIT values and PIT histogram data.
Implement prediction interval coverage.
Implement bucket log loss.
Implement bucket Brier score.
Implement calibration tables for interval events.
Compare empirical baseline vs NGBoost.
```

Files:

```text
src/weather_bucket_model/evaluation/metrics.py
src/weather_bucket_model/evaluation/coverage.py
src/weather_bucket_model/evaluation/calibration_plots.py
```

Deliverables:

```text
outputs/model_scores_week2.csv
reports/figures/pit_histogram.png
reports/figures/coverage_plot.png
```

### Day 15: Build CDF-classification benchmark

Tasks:

```text
Choose relevant error boundaries.
Create target_c = 1 if error <= c.
Train one GBM classifier per boundary.
Predict F(c) for each validation row.
Enforce monotonicity across boundaries.
Convert CDF values to interval probabilities.
Evaluate vs NGBoost and empirical baseline.
```

Files:

```text
src/weather_bucket_model/models/cdf_classification_gbm.py
scripts/train_cdf_classification_gbm.py
tests/test_cdf_monotonicity.py
```

Deliverables:

```text
models/cdf_gbm_models.pkl
outputs/cdf_gbm_bucket_probs.csv
outputs/cdf_gbm_metrics.json
```

### Day 16: Build multiclass benchmark

Tasks:

```text
Create true error-bucket labels.
Train multiclass GBM benchmark.
Predict bucket probability vectors.
Evaluate multiclass log loss, Brier score, top-bucket accuracy, confusion matrix.
Compare to NGBoost and CDF benchmark.
```

Files:

```text
src/weather_bucket_model/models/multiclass_gbm.py
scripts/train_multiclass_gbm.py
```

Deliverables:

```text
models/multiclass_gbm.pkl
outputs/multiclass_bucket_probs.csv
outputs/multiclass_metrics.json
```

### Day 17: Optional quantile benchmark or fallback analysis

Tasks:

```text
If time permits, train quantile GBM on forecast error.
Otherwise, write a fallback note explaining why quantile GBM is optional.
Compare prediction intervals from quantile model vs NGBoost.
Use quantile model only as benchmark, not primary.
```

Files:

```text
src/weather_bucket_model/models/quantile_gbm.py
scripts/train_quantile_gbm.py
```

Deliverables:

```text
outputs/quantile_benchmark_results.csv
```

### Day 18: Model comparison and calibration review

Tasks:

```text
Compare empirical, NGBoost, CDF-classification, multiclass, and optional quantile models.
Rank by NLL/log score where applicable.
Rank by bucket log loss and Brier score.
Check calibration curves.
Check coverage.
Check model behavior by hour and edge size.
Choose primary probability source for trading demo.
```

Files:

```text
src/weather_bucket_model/models/ensemble.py
src/weather_bucket_model/evaluation/metrics.py
```

Deliverables:

```text
outputs/final_model_comparison.csv
reports/figures/model_comparison.png
reports/figures/calibration_curves.png
```

### Day 19: Market schema and orderbook executable pricing

Tasks:

```text
Define Kalshi bucket market schema.
Implement mock market snapshot loader.
Implement YES/NO bid to ask conversion.
Implement orderbook depth walking.
Compute average executable price by size.
```

Files:

```text
src/weather_bucket_model/data/kalshi_client.py
src/weather_bucket_model/trading/orderbook.py
tests/test_orderbook_conversion.py
```

Deliverables:

```text
data/sample/mock_kalshi_bucket_market.csv
outputs/orderbook_pricing_demo.csv
```

### Day 20: EV scanner and basket scanner

Tasks:

```text
Compute single-bucket edge.
Compute basket probabilities and basket prices.
Scan tail, middle, upper, lower, adjacent, and all-bucket baskets.
Include fees, slippage, and model margin.
Rank opportunities.
```

Files:

```text
src/weather_bucket_model/trading/edge.py
scripts/run_paper_scan.py
```

Deliverables:

```text
outputs/ev_scanner_results.csv
outputs/basket_scanner_results.csv
```

### Day 21: Sizing, policies, and risk controls

Tasks:

```text
Implement fractional Kelly with uncertainty haircut.
Implement liquidity caps.
Implement max exposure limits.
Implement early/mid/late-day hurdle rates.
Keep live trading disabled.
```

Files:

```text
src/weather_bucket_model/trading/sizing.py
src/weather_bucket_model/trading/policies.py
tests/test_sizing.py
```

Deliverables:

```text
outputs/sizing_demo.csv
```

### Day 22: Paper-trading scan end-to-end

Tasks:

```text
Run full pipeline from feature row to predicted distribution to bucket probabilities to EV scanner to paper decision.
Use mock or manually entered Kalshi market snapshot.
Log all decisions.
Log rejected trades and reasons.
```

Files:

```text
scripts/run_paper_scan.py
src/weather_bucket_model/evaluation/backtest.py
```

Deliverables:

```text
outputs/decisions.csv
outputs/paper_trades.csv
outputs/rejected_trades.csv
```

### Day 23: Final visualizations

Tasks:

```text
Create forecast-error density split into bucket intervals.
Create CDF plot with F(b)-F(a).
Create predicted distribution fan chart over time.
Create PIT histogram.
Create coverage chart.
Create model probabilities vs market prices.
Create edge-by-bucket and edge-by-basket charts.
Create intraday actual vs forecast path chart.
```

Files:

```text
src/weather_bucket_model/visualization/*.py
scripts/make_project_figures.py
```

Deliverables:

```text
reports/figures/*.png
```

### Day 24: Write calculus and model explanation

Tasks:

```text
Explain forecast-error distribution.
Explain CDF and PDF.
Explain interval probability as area under PDF.
Explain P(a < error <= b)=F(b)-F(a) by Fundamental Theorem of Calculus.
Explain gradient boosting / NGBoost as loss optimization using derivative-based updates.
Explain why probability quality matters more than point accuracy.
```

Files:

```text
reports/paper_draft.md
```

### Day 25: Write trading and limitations sections

Tasks:

```text
Explain executable prices.
Explain bucket and basket EV.
Explain liquidity-aware sizing.
Explain model uncertainty margin.
Explain no future leakage.
Explain settlement/source mismatch caveat.
Explain why paper trading is not proof of profitability.
```

Files:

```text
reports/paper_draft.md
```

### Day 26: Final demo notebook and README

Tasks:

```text
Create final notebook showing pipeline.
Document how to run scripts.
Document project structure.
Include expected outputs.
```

Files:

```text
notebooks/04_distributional_gbm_diagnostics.ipynb
README.md
```

### Day 27: Polish and validation

Tasks:

```text
Run tests.
Run scripts end-to-end.
Check paths.
Check all figures render.
Check model artifacts load.
Check paper references outputs correctly.
```

Deliverables:

```text
passing tests
clean outputs
final paper draft
```

### Day 28: Final submission package

Tasks:

```text
Export paper.
Clean repo.
Prepare final figures.
Prepare short presentation.
Prepare answers to skeptical questions.
```

Deliverables:

```text
reports/final_paper.md or pdf
README.md
figures
model comparison table
paper-trading demo output
```

---

## 17. Key Visualizations

Create these:

```text
1. Forecast-error density split into bucket intervals.
2. Error CDF curve with bucket probabilities shown as F(b)-F(a).
3. Example final-temperature buckets converted into forecast-error intervals.
4. NGBoost predicted mu/sigma or distribution parameters over time.
5. Prediction interval fan chart for forecast error / final high.
6. PIT histogram for distribution calibration.
7. Coverage plot for 50%, 80%, 90% intervals.
8. Model bucket probabilities vs market prices.
9. Edge-by-bucket chart.
10. Edge-by-basket chart.
11. Intraday actual temperature vs forecast path.
12. Orderbook depth / liquidity-adjusted edge chart.
13. Probability mass shift as the day progresses.
14. Tail vs middle probability mass chart.
```

---

## 18. Important Risks and Rules

### 18.1 Future leakage

Never use:

```text
final daily high as feature
future observations after timestamp t
future forecast updates not available at timestamp t
ERA5/reanalysis as if it were live forecast data
post-settlement information
```

### 18.2 Forecast-source mismatch

If using Open-Meteo forecast data as a proxy for NWS forecasts, do not claim it is exactly NWS forecast error.

State clearly:

```text
The ideal target is NWS forecast error. If archived NWS forecast data is unavailable, Open-Meteo historical forecast data is used as a proxy for the demonstration.
```

### 18.3 Market-data leakage

Do not put Kalshi price directly into the weather probability model initially. Use Kalshi data in the trading layer.

### 18.4 Calibration risk

A profitable model needs honest probabilities. Accuracy is not enough.

### 18.5 Thin liquidity

Never size off midpoint. Use executable orderbook prices.

### 18.6 Correlation

Buckets in the same market are mutually exclusive. Related city/date/weather markets are correlated. Do not apply independent Kelly sizing blindly.

### 18.7 Settlement matching

Kalshi settlement source/station must be matched carefully. If using a different station than settlement, results may be invalid.

### 18.8 Distribution misspecification

A Normal forecast-error model may underestimate tails. Check PIT, coverage, and tail bucket calibration. Consider Student-t or empirical residual adjustments if tails are too thin.

---

## 19. Codex Implementation Style

Prefer:

```text
small modules
pure functions where possible
pandas DataFrames with clear schemas
type hints
unit tests for math/market logic
no huge notebooks as main code
scripts for reproducible runs
chronological train/validation/test splits
explicit anti-leakage checks
```

Avoid:

```text
overengineering live trading first
assuming data exists without validation
random train/test split as final evaluation
using midpoint prices
using future observations
using reanalysis as forecast
claiming real profitability from tiny paper-trading samples
```

Use time-based splits:

```text
train: older dates
validation: later dates
test: newest dates
```

---

## 20. First Commands / First Files To Build From Current State

Because Days 1-5 are already completed, start at data verification and supervised table construction.

First commands:

```text
mkdir -p src/weather_bucket_model/{data,features,models,trading,evaluation,visualization,utils}
mkdir -p tests scripts notebooks reports/figures data/{raw,interim,processed,sample}
```

First files to prioritize now:

```text
src/weather_bucket_model/data/schemas.py
src/weather_bucket_model/utils/validation.py
src/weather_bucket_model/features/build_dataset.py
src/weather_bucket_model/features/error_features.py
src/weather_bucket_model/features/observation_features.py
src/weather_bucket_model/features/forecast_features.py
src/weather_bucket_model/models/empirical_error.py
src/weather_bucket_model/models/distributional_gbm.py
src/weather_bucket_model/models/distribution_utils.py
scripts/build_training_dataset.py
scripts/train_empirical_error.py
scripts/train_distributional_gbm.py
```

First functions:

```python
def compute_forecast_error(actual_high: float, forecast_high: float) -> float: ...

def final_temp_bucket_bounds(label: str) -> tuple[float | None, float | None]: ...

def temp_bounds_to_error_bounds(lower: float | None, upper: float | None, forecast_high: float) -> tuple[float | None, float | None]: ...

def interval_probability_from_cdf(cdf_fn, lower: float | None, upper: float | None) -> float: ...

def build_timestamped_rows(observations, forecasts, prediction_times) -> pd.DataFrame: ...

def validate_no_future_leakage(rows: pd.DataFrame) -> None: ...
```

First tests:

```text
forecast error is actual_high - forecast_high
integer bucket bounds use half-degree intervals
error bounds shift correctly when forecast_high changes
probabilities for exhaustive buckets sum to 1
features never use observations after timestamp t
NGBoost predicted scale/sigma is positive
CDF interval probabilities are nonnegative
```

---

## 21. Final Teacher-Facing Narrative

This project models weather prediction markets using calculus and machine learning. A temperature bucket market is divided into mutually exclusive intervals, and each bucket’s fair price is the probability that the official final high temperature lands inside that interval. Instead of trying to predict the weather from scratch, the model estimates the distribution of NWS forecast error: actual official high minus the NWS forecast high. This makes the project about learning when the forecast is likely to be too high or too low. The calculus foundation is that each bucket probability is a definite integral: the area under the forecast-error probability density curve between two error boundaries. Equivalently, each bucket probability is a difference in CDF values, `F(b)-F(a)`, which follows from the Fundamental Theorem of Calculus. The machine learning model estimates the forecast-error distribution throughout the day as new observations arrive. Then the system converts that distribution into final-temperature bucket probabilities, compares those probabilities to executable market prices, and identifies possible positive expected value trades after costs and liquidity. Calculus appears in the probability integrals, CDF/PDF relationship, expected value calculations, sensitivity analysis, optimization, and gradient-based model training.

---

## 22. Final Profitability-First Thesis

The profitable version is not a static classifier and not a simple temperature forecast. It is a sequential, forecast-error distribution modeling system:

```text
station observations + forecast path + time state
    -> distributional GBM / NGBoost model of forecast error
    -> predicted forecast-error distribution and CDF
    -> final-temperature bucket probabilities via F(b)-F(a)
    -> executable market edge after costs and liquidity
    -> conservative basket/position sizing
```

The central claim is:

```text
To trade bucket weather markets profitably, the model must estimate the full conditional distribution of forecast error, then convert that distribution into bucket probabilities using CDF differences. Forecast-error distribution modeling is more stable and more trading-relevant than predicting raw final temperature or only predicting a single threshold.
```
