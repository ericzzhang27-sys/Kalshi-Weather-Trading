# Weather Bucket-Market Forecasting Project — Codex Build Spec

## 0. Project Purpose

Build a profitability-first, calculus-linked machine learning project for weather prediction markets, focused on Kalshi-style temperature bucket markets.

The system should estimate the full probability distribution of the official final daily high temperature, convert that distribution into probabilities for mutually exclusive temperature buckets, compare those probabilities to executable market prices, and decide whether a trade has positive expected value after costs, liquidity, and uncertainty.

This is not just a school poster. The architecture should be realistic enough to support paper trading and later real trading research, while still producing clean explanations and visuals for a calculus class.

---

## 1. Core Thesis

Kalshi temperature markets are often structured as mutually exclusive buckets, not just threshold contracts.

Example market structure:

```text
60°F or below
61°F
62°F
63°F
64°F or above
```

Exactly one bucket resolves YES. Therefore, the central problem is not simply:

```text
Will temperature be >= K?
```

The central problem is:

```text
What is the probability mass in each final-temperature bucket?
```

The project should model:

```text
T_max | X_t
```

where:

- `T_max` = official final daily high temperature at the settlement station.
- `X_t` = all weather and forecast information available at time `t`.

Then compute:

```text
P(bucket_i) = F(bucket_upper_i | X_t) - F(bucket_lower_i | X_t)
```

where `F` is the predicted CDF of final high temperature.

The trading edge for bucket `i` is:

```text
edge_i = p_i - q_i - fees - slippage - model_margin
```

where:

- `p_i` = model probability of bucket `i`.
- `q_i` = executable Kalshi price for bucket `i`, not midpoint.

---

## 2. Calculus Foundation

The project must explicitly demonstrate calculus knowledge.

### 2.1 Bucket probability as a definite integral

For bucket `B_i = [a_i, b_i)`:

```text
P(B_i) = integral from a_i to b_i of f(t | X_t) dt
```

This means each bucket price corresponds to area under the probability density curve.

### 2.2 CDF differences

Equivalently:

```text
P(B_i) = F(b_i | X_t) - F(a_i | X_t)
```

### 2.3 Whole-degree reporting

If official daily highs are reported as whole degrees, treat a bucket like `62°F` as a rounded interval:

```text
P(reported high = 62) ≈ P(61.5 <= T_max < 62.5)
```

For a normal approximation:

```text
P(62) = Phi((62.5 - mu) / sigma) - Phi((61.5 - mu) / sigma)
```

Edge buckets:

```text
P(60 or below) = Phi((60.5 - mu) / sigma)
P(64 or above) = 1 - Phi((63.5 - mu) / sigma)
```

### 2.4 Expected value

For a $1 binary payout bucket:

```text
EV_i = p_i - q_i
```

after costs:

```text
EV_i = p_i - q_i - fees_i - slippage_i - model_margin_i
```

### 2.5 Optimization across buckets and baskets

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

### 2.6 Sensitivity to forecast center and uncertainty

If a simplified model assumes:

```text
T_max ~ Normal(mu, sigma)
```

then bucket probabilities shift when `mu` and `sigma` change. This can be visualized through probability mass moving between adjacent intervals.

For threshold probability:

```text
P(T >= K) = 1 - Phi((K - mu) / sigma)
dP/dmu = (1 / sigma) * phi((K - mu) / sigma)
```

For bucket probabilities, sensitivity can be shown by recalculating interval masses after changing `mu` or `sigma`.

### 2.7 Gradient boosting uses calculus

Gradient boosting minimizes a differentiable loss function.

For binary/logistic loss:

```text
L = -sum[y_i log(p_i) + (1 - y_i) log(1 - p_i)]
dL/dF_i = p_i - y_i
```

For regression/quantile loss, quantile models minimize pinball loss. The model learns conditional quantiles of final high temperature, which are then converted into an estimated CDF.

---

## 3. Final Modeling Decision

Use a profitability-first, distribution-first architecture.

### 3.1 Main model

Main model:

```text
Distributional / quantile gradient boosting nowcasting model
```

Train separate quantile models for final daily high temperature:

```text
Q10, Q25, Q50, Q75, Q90
```

At prediction time, the model outputs updated quantiles for `T_max | X_t`. Convert those quantiles into a CDF, then compute each bucket probability by CDF differences.

Possible libraries:

```text
LightGBM quantile regression
CatBoost quantile loss
sklearn GradientBoostingRegressor(loss='quantile')
XGBoost quantile objective if available
```

Recommended first implementation: `sklearn.ensemble.GradientBoostingRegressor` with quantile loss because it is easy and stable.

Recommended more serious implementation: LightGBM quantile regression.

### 3.2 Strong baseline

Build an empirical conditional forecast-error distribution model.

Model:

```text
T_max = forecast_high + error
```

Use historical errors:

```text
error = actual_final_high - forecast_high_available_at_time_t
```

Condition on approximate contexts:

```text
station
season / day of year
time of day
current temp vs forecast path
max so far vs forecast path
cloud regime
wind regime
forecast horizon
```

Then compute bucket probabilities by asking how often similar historical errors place `T_max` in each bucket.

This baseline is important because it is robust, interpretable, and less likely to overfit than a complex model.

### 3.3 Benchmark model

Build a multiclass calibrated gradient boosting classifier as a benchmark or ensemble component.

Model:

```text
X_t -> [P(B1), P(B2), P(B3), P(B4), P(B5)]
```

Possible implementations:

```text
CatBoostClassifier
LightGBM multiclass
XGBoost objective='multi:softprob'
sklearn HistGradientBoostingClassifier or GradientBoostingClassifier for prototype
```

This directly matches the market structure, but it may mishandle bucket ordering and tails. It should not be the only final model.

### 3.4 Final ensemble

Blend probability estimates:

```text
p_i_final = w1 * p_i_empirical + w2 * p_i_quantile + w3 * p_i_multiclass + w4 * p_i_market_prior
```

Weights should be selected based on out-of-sample:

```text
multiclass log loss
multiclass Brier score
calibration
paper-trading EV / PnL
```

For the first build, use fixed weights or equal weights. Later, optimize weights on validation data.

---

## 4. Sequential Nowcasting Framework

The system is not a one-time morning forecast. It continuously updates probabilities throughout the day.

At each timestamp `t`:

```text
weather observations update
forecast path updates
max temp so far updates
time remaining shrinks
model distribution updates
bucket probabilities update
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
2025-05-01 09:00 -> final bucket
2025-05-01 10:00 -> final bucket
2025-05-01 11:00 -> final bucket
2025-05-01 12:00 -> final bucket
...
```

The target final high is the same for all rows from a day, but features change over time.

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
final_daily_high
```

### 5.2 Forecast data: Open-Meteo Historical Forecast + Forecast API

Use Open-Meteo Historical Forecast API for historical forecast features and Open-Meteo Forecast API for live forecast features.

Open-Meteo Historical Forecast API archives forecast model data and is appropriate for ML training because it reflects model forecasts rather than reanalysis.

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
model_disagreement if multiple models are pulled
```

Do not use Open-Meteo Historical Weather / ERA5 as if it were a forecast. Reanalysis is useful for demos but can leak future information.

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

### 7.4 Bucket features

For distribution-first model, bucket features are used after prediction, not necessarily as model inputs.

For a general bucket-level binary model, include:

```text
bucket_lower
bucket_upper
bucket_center
bucket_width
forecast_high_minus_bucket_center
current_temp_minus_bucket_center
max_so_far_minus_bucket_center
```

For fixed 5-bucket markets, the distributional model predicts final high distribution, then bucket bounds are applied externally.

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

### 8.1 Regression target

For quantile gradient boosting:

```text
y = final_daily_high
```

### 8.2 Bucket target

For multiclass benchmark:

```text
y_bucket = bucket_containing_final_daily_high
```

Example:

```python
if final_high <= 60:
    y_bucket = "<=60"
elif final_high == 61:
    y_bucket = "61"
elif final_high == 62:
    y_bucket = "62"
elif final_high == 63:
    y_bucket = "63"
else:
    y_bucket = ">=64"
```

Exact bucket construction must use the actual Kalshi market bucket definitions.

---

## 9. Evaluation Metrics

### 9.1 Distribution / quantile model

```text
pinball loss by quantile
coverage of prediction intervals
MAE of median prediction
RMSE of median prediction
CRPS if implemented
bucket log loss after converting to bucket probabilities
bucket Brier score
calibration by bucket probability
```

### 9.2 Multiclass bucket model

```text
multiclass log loss
multiclass Brier score
top-1 accuracy
mean absolute bucket error
confusion matrix
calibration curve by bucket
```

### 9.3 Trading evaluation

```text
edge by bucket
edge by time of day
paper PnL
PnL after fees
PnL after simulated slippage
hit rate by entry time
return per unit risk
max drawdown
liquidity-adjusted edge
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
upper buckets = {63, >=64}
lower buckets = {<=60, 61}
middle buckets = {62, 63}
tail buckets = {<=60, >=64}
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
confidence_t = 1 - sigma_t / sigma_morning
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
    04_visuals_for_paper.ipynb

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
        bucket_features.py
        build_dataset.py
      models/
        __init__.py
        empirical_error.py
        quantile_gbm.py
        multiclass_gbm.py
        ensemble.py
        calibration.py
        cdf_from_quantiles.py
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
      visualization/
        __init__.py
        density_plots.py
        probability_bars.py
        edge_plots.py
        intraday_plots.py
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
    train_quantile_gbm.py
    train_multiclass_gbm.py
    run_paper_scan.py
    make_project_figures.py

  tests/
    test_bucket_pricing.py
    test_orderbook_conversion.py
    test_no_future_leakage.py
    test_cdf_from_quantiles.py
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

### `features/build_dataset.py`

Responsibilities:

```text
Create timestamped training rows.
For each station-day, create rows at selected times.
Ensure features only use data available up to timestamp.
Attach final_daily_high target.
Attach bucket target for example market definitions.
```

Prediction timestamps for first version:

```text
09:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00 local time
```

### `models/empirical_error.py`

Responsibilities:

```text
Compute forecast errors.
Find similar historical rows by bins or nearest neighbors.
Return empirical distribution of final high.
Convert to bucket probabilities.
```

Simple similarity bins:

```text
station
month or season
hour bucket
forecast_high_minus_current_bucket_center
current_temp_minus_forecast_temp bin
max_so_far_minus_forecast_max_so_far bin
```

### `models/quantile_gbm.py`

Responsibilities:

```text
Train quantile GBM models for multiple quantiles.
Predict quantiles for new feature rows.
Save/load trained models.
```

Quantiles:

```text
0.10, 0.25, 0.50, 0.75, 0.90
```

### `models/cdf_from_quantiles.py`

Responsibilities:

```text
Take predicted quantiles.
Construct monotonic CDF approximation.
Compute F(x) by interpolation.
Compute bucket probabilities F(b)-F(a).
Ensure probabilities are nonnegative and sum to 1 after normalization.
```

Simple initial method:

```text
Sort quantiles.
Use piecewise-linear interpolation between quantile points.
Add lower/upper tails using simple extrapolation.
Evaluate CDF at bucket bounds.
```

### `models/multiclass_gbm.py`

Responsibilities:

```text
Train direct bucket classifier benchmark.
Predict bucket probability vector.
Evaluate multiclass log loss and Brier score.
```

### `models/ensemble.py`

Responsibilities:

```text
Blend probability vectors from empirical, quantile, multiclass, and market-prior models.
Normalize final probabilities.
Evaluate ensemble weights.
```

### `models/calibration.py`

Responsibilities:

```text
Implement simple calibration diagnostics.
Optional: calibrate multiclass or one-vs-rest probabilities.
For first version, produce calibration plots rather than complex calibration.
```

### `trading/bucket_pricing.py`

Responsibilities:

```text
Define bucket intervals.
Convert whole-degree buckets to half-degree bounds.
Compute bucket probabilities from CDF.
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
normal density split into buckets
CDF with bucket intervals
model probabilities vs market prices
edge by bucket
intraday temperature path vs expected path
quantile fan chart over time
calibration curve
orderbook depth / liquidity-adjusted edge
```

---

## 13. Build Order for Codex

### Phase 1: Core math and bucket pricing

Build first:

```text
src/weather_bucket_model/trading/bucket_pricing.py
src/weather_bucket_model/utils/math_utils.py
tests/test_bucket_pricing.py
notebooks/01_bucket_probability_demo.ipynb
```

Requirements:

```text
Define bucket intervals.
Convert integer buckets to half-degree continuous intervals.
Compute normal CDF bucket probabilities.
Support edge buckets.
Ensure probabilities sum to 1.
Generate density plot split into buckets.
```

Acceptance checks:

```text
For buckets <=60, 61, 62, 63, >=64 and mu=62.4, sigma=1.3, probabilities are nonnegative and sum to 1.
Moving mu upward shifts mass to upper buckets.
Increasing sigma increases tails.
```

### Phase 2: Orderbook and EV logic

Build:

```text
src/weather_bucket_model/trading/orderbook.py
src/weather_bucket_model/trading/edge.py
src/weather_bucket_model/trading/sizing.py
tests/test_orderbook_conversion.py
tests/test_sizing.py
```

Requirements:

```text
Convert YES/NO bids to asks.
Compute executable average price by size.
Compute marginal EV by orderbook level.
Compute edge by bucket.
Compute basket edge.
Compute fractional Kelly size.
```

Acceptance checks:

```text
YES ask = 100 - best NO bid.
Average executable price rises as size walks the book.
Sizing returns zero if edge <= 0.
Sizing respects max cap.
```

### Phase 3: Data clients

Build:

```text
src/weather_bucket_model/data/iem_client.py
src/weather_bucket_model/data/open_meteo_client.py
src/weather_bucket_model/data/schemas.py
scripts/fetch_iem_observations.py
scripts/fetch_open_meteo_forecasts.py
```

Requirements:

```text
Fetch small sample of observations for one station.
Fetch small sample of forecast variables for one location/date.
Save raw CSV/parquet.
Standardize timestamps.
```

Keep robust error handling. Do not overbuild authentication unless needed.

### Phase 4: Feature engineering

Build:

```text
features/observation_features.py
features/forecast_features.py
features/time_features.py
features/build_dataset.py
scripts/build_training_dataset.py
tests/test_no_future_leakage.py
```

Requirements:

```text
Create timestamped rows.
Use only observations up to timestamp.
Align forecasts by timestamp.
Compute final daily high from observations after day complete.
Construct target.
```

Important: add explicit anti-leakage checks.

### Phase 5: Empirical forecast-error model

Build:

```text
models/empirical_error.py
scripts/train_empirical_error.py
```

Requirements:

```text
Compute historical forecast error.
Group or nearest-neighbor similar states.
Return empirical distribution.
Convert to bucket probabilities.
```

This should be the first serious model because it is robust and explainable.

### Phase 6: Quantile GBM model

Build:

```text
models/quantile_gbm.py
models/cdf_from_quantiles.py
scripts/train_quantile_gbm.py
tests/test_cdf_from_quantiles.py
```

Requirements:

```text
Train one model per quantile.
Predict quantiles.
Enforce monotonic quantiles if needed by sorting.
Convert quantiles to approximate CDF.
Compute bucket probabilities.
Evaluate bucket log loss.
```

### Phase 7: Multiclass benchmark

Build:

```text
models/multiclass_gbm.py
```

Requirements:

```text
Train bucket classifier.
Output probability vector.
Compare to quantile model.
```

### Phase 8: Ensemble and paper-trading scanner

Build:

```text
models/ensemble.py
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
Produce clear narrative connecting calculus, ML, and trading.
```

---

## 14. Minimum Viable Product

If time/data is limited, build this:

```text
1. Normal-distribution bucket probability calculator.
2. Bucket probability visuals.
3. Simulated market prices.
4. EV scanner by bucket.
5. Liquidity-aware orderbook example with fake orderbook.
6. Small synthetic or Open-Meteo/IEM sample dataset.
7. Simple quantile regression demo or empirical forecast-error demo.
8. Paper explaining full architecture and limitations.
```

MVP does not need a live trading bot.

---

## 15. Ambitious Version

If data and time permit:

```text
1. Pull real IEM observations for 5-10 U.S. stations.
2. Pull Open-Meteo historical forecast data for same stations.
3. Build timestamped training dataset over 2022-2025.
4. Train empirical error and quantile GBM models.
5. Add multiclass benchmark.
6. Pull live IEM + Open-Meteo data.
7. Pull live Kalshi orderbooks.
8. Run paper scanner every 5-15 minutes.
9. Log model probabilities, market prices, and hypothetical trades.
10. Evaluate paper PnL after settlement.
```

---

## 16. Three-Week Build Plan

### Week 1: Math engine, data schema, and bucket probability demo

#### Day 1

```text
Create repo structure.
Write README thesis.
Implement bucket interval definitions.
Implement normal CDF bucket probability calculator.
```

#### Day 2

```text
Create density plot split into buckets.
Create CDF difference plot.
Create probability-vs-bucket bar chart.
```

#### Day 3

```text
Implement fake market prices.
Compute edge by bucket.
Compute buy-all-buckets arbitrage check.
```

#### Day 4

```text
Implement orderbook conversion and average executable price.
Add tests for Kalshi YES/NO bid conversion.
```

#### Day 5

```text
Implement fractional Kelly sizing and liquidity caps.
Create first paper section explaining calculus.
```

#### Weekend checkpoint

Deliverables:

```text
Working bucket probability calculator.
Working EV scanner on simulated prices.
3-4 figures for paper.
Clear calculus explanation draft.
```

### Week 2: Data and feature engineering

#### Day 6

```text
Build IEM client.
Fetch sample station observations.
Clean tmpf/dwpf/wind fields.
```

#### Day 7

```text
Build Open-Meteo client.
Fetch sample historical forecast data.
Align forecast timestamps.
```

#### Day 8

```text
Build observation features.
Compute current temp, max so far, temp changes, dew point spread, wind sin/cos.
```

#### Day 9

```text
Build forecast features.
Compute forecast high, forecast temp now, current-vs-forecast residuals, cloud/wind/precip next 3h.
```

#### Day 10

```text
Build timestamped training dataset.
Add anti-leakage validation.
Construct final high targets and example buckets.
```

#### Weekend checkpoint

Deliverables:

```text
Small processed dataset.
Feature engineering code.
No-future-leakage test.
Initial intraday temperature path plot.
```

### Week 3: Models, evaluation, and final paper/demo

#### Day 11

```text
Implement empirical forecast-error model.
Generate empirical bucket probabilities.
Evaluate simple log loss / Brier score.
```

#### Day 12

```text
Implement quantile GBM models.
Train Q10/Q25/Q50/Q75/Q90.
Convert quantiles to bucket probabilities.
```

#### Day 13

```text
Implement multiclass benchmark if time permits.
Compare quantile vs multiclass vs empirical model.
```

#### Day 14

```text
Build model probability vs market price chart.
Build edge-by-bucket chart.
Build calibration plot if possible.
```

#### Day 15

```text
Write final report.
Explain calculus, ML, features, trading logic, limitations, and future work.
```

#### Final weekend

```text
Polish figures.
Run scripts end-to-end.
Make final demo notebook.
Prepare final paper/poster.
```

---

## 17. Key Visualizations

Create these:

```text
1. Normal density split into 5 shaded bucket intervals.
2. CDF curve with bucket probabilities shown as F(b)-F(a).
3. Model bucket probabilities vs market prices.
4. Edge-by-bucket chart.
5. Intraday actual temperature vs forecast path.
6. Quantile fan chart over time.
7. Calibration curve or reliability diagram.
8. Orderbook depth / liquidity-adjusted edge chart.
9. Probability mass shift as the day progresses.
10. Tail vs middle probability mass chart.
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
```

### 18.2 Market-data leakage

Do not put Kalshi price directly into the weather probability model initially. Use Kalshi data in the trading layer.

### 18.3 Calibration risk

A profitable model needs honest probabilities. Accuracy is not enough.

### 18.4 Thin liquidity

Never size off midpoint. Use executable orderbook prices.

### 18.5 Correlation

Buckets in same market are mutually exclusive. Related city/date/weather markets are correlated. Do not apply independent Kelly sizing blindly.

### 18.6 Settlement matching

Kalshi settlement source/station must be matched carefully. If using a different station than settlement, results may be invalid.

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
```

Avoid:

```text
overengineering live trading first
assuming data exists without validation
random train/test split as final evaluation
using midpoint prices
using future observations
```

Use time-based splits:

```text
train: older dates
validation: later dates
test: newest dates
```

---

## 20. First Commands / First Files to Build

Start with:

```text
mkdir -p src/weather_bucket_model/{trading,utils,visualization}
mkdir -p tests notebooks reports/figures
```

Create:

```text
src/weather_bucket_model/trading/bucket_pricing.py
src/weather_bucket_model/utils/math_utils.py
tests/test_bucket_pricing.py
scripts/make_project_figures.py
```

First functions:

```python
def normal_cdf(x: float, mu: float, sigma: float) -> float: ...

def bucket_probability_normal(lower: float | None, upper: float | None, mu: float, sigma: float) -> float: ...

def integer_bucket_bounds(label: str) -> tuple[float | None, float | None]: ...

def bucket_probabilities_normal(buckets, mu: float, sigma: float) -> dict[str, float]: ...
```

First test:

```text
probabilities for exhaustive buckets sum to 1
probabilities are nonnegative
upper bucket probability rises when mu rises
```

---

## 21. Final Teacher-Facing Narrative

This project models weather prediction markets using calculus and machine learning. A temperature market is divided into mutually exclusive buckets, and each bucket’s fair price is the probability that the official final high temperature lands inside that interval. The calculus foundation is that each bucket probability is a definite integral: the area under a probability density curve between two temperature boundaries. The machine learning model estimates the final-temperature distribution throughout the day as new observations arrive. Then the system converts that distribution into bucket probabilities using CDF differences, compares those probabilities to market prices, and identifies possible positive expected value trades after costs and liquidity. Calculus appears in the probability integrals, expected value calculations, sensitivity analysis, optimization, and gradient-based model training.

---

## 22. Final Profitability-First Thesis

The profitable version is not a static classifier and not a simple temperature forecast. It is a sequential, distribution-first nowcasting system:

```text
station observations + forecast path + time state
    -> distributional gradient boosting model
    -> predicted final-high CDF
    -> bucket probabilities via interval integrals
    -> executable market edge after costs and liquidity
    -> conservative basket/position sizing
```

The central claim is:

```text
To trade bucket weather markets profitably, the model must estimate the entire probability distribution of the final official high temperature, not just the most likely bucket or a single threshold probability.
```
