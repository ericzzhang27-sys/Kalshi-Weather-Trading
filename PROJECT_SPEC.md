# Weather Bucket-Market Trading Project Spec

## 1. Project Thesis

This project builds a machine-learning probability signal for weather bucket markets, especially temperature markets with mutually exclusive settlement buckets.

The core idea is not to predict the final temperature from scratch. Instead, the model estimates the distribution of **NWS forecast error** at a given prediction timestamp.

The main target is:

error = actual official high - NWS predicted high at timestamp t

The model learns when the NWS forecast is likely to be too high or too low, conditional on information available at timestamp t.

Because weather markets settle into mutually exclusive buckets, the model must output a full probability distribution across all possible buckets. The probabilities must be coherent, nonnegative, and sum to 1.

Current Day 14 direction: the final project is NGBoost/DGBM-focused. The primary model estimates a full distribution for forecast_error, and bucket probabilities come from the model-implied CDF:

F(c | X_t) = P(error <= c | X_t)

P(a < error <= b | X_t) = F(b | X_t) - F(a | X_t)

The required comparison remains the empirical historical forecast-error distribution. CDF classifiers, multiclass bucket models, quantile models, and alternative DGBM implementations are optional future extensions only. Probability quality, calibration, and interval coverage matter more than point accuracy.

The goal of the project is to build a calibrated probability signal first. Trading, paper-trading, position sizing, and execution logic come later.

Profitability is not assumed. A model is only potentially tradable if its calibrated probabilities differ from market prices by enough to overcome spreads, fees, liquidity constraints, model uncertainty, and adverse selection.

---

## 2. Forecast-Error Target

For each city, date, and prediction timestamp t, define:

error_t = actual official high for that date - NWS forecast high available at timestamp t

Where:

- actual official high is the final settlement temperature from the official weather station.
- NWS forecast high is the forecast that would have been available at timestamp t.
- timestamp t is the time when the model is making a prediction or hypothetical trading decision.

Interpretation:

- error_t > 0 means the actual high was warmer than the NWS forecast.
- error_t < 0 means the actual high was cooler than the NWS forecast.
- error_t = 0 means the NWS forecast exactly matched the official high.

This target is preferable to directly predicting the raw final temperature because the NWS forecast already contains a large amount of weather information. The model’s job is to learn systematic residual errors around that forecast.

---

## 3. Prediction Timestamp

All model predictions are made as of a timestamp t.

At timestamp t, the model may only use information that was published, observable, or computable at or before t.

Examples of possible prediction timestamps:

- 9:00 AM local time
- 12:00 PM local time
- 2:00 PM local time
- every 30 minutes during the trading day
- a fixed amount of time before market close

This project may eventually support multiple timestamps per day, but every row must clearly identify the timestamp t being used.

The most important rule:

No feature can use information from after timestamp t.

This rule applies to weather observations, forecast updates, market prices, realized temperature data, and any engineered feature derived from those sources.

---

## 4. NGBoost / DGBM Distribution Logic

The primary model estimates the conditional distribution of forecast error directly:

error_t | X_t ~ DGBM distribution

Current implementation:

- model family: NGBoost / DGBM
- target: forecast_error
- current distribution: Normal
- score: log score / negative log likelihood
- output parameters: distribution location and scale

For any forecast-error boundary c, the fitted distribution gives:

F(c | X_t) = P(error_t <= c | X_t)

Bucket probabilities are then computed by differencing CDF values:

P(a < error_t <= b | X_t) = F(b | X_t) - F(a | X_t)

The exact bucket boundaries should be chosen based on the market settlement rules and the empirical distribution of forecast errors.

Optional future extensions may include a CDF classifier, multiclass bucket model, quantile model, or alternative DGBM implementation, but they are not required benchmark branches for the current project.

---

## 5. Bucket Probability Conversion

Kalshi-style temperature markets settle into buckets of actual official high temperature.

The model operates in forecast-error space, so temperature bucket boundaries must be converted into error boundaries.

Given:

error = actual high - NWS forecast high at timestamp t

Then:

actual high = NWS forecast high at timestamp t + error

Suppose the market bucket is:

L < actual high <= U

And the NWS predicted high at timestamp t is:

nws_t

Then the bucket becomes:

L - nws_t < error <= U - nws_t

Therefore:

P(L < actual high <= U | X_t) = F(U - nws_t | X_t) - F(L - nws_t | X_t)

This is the central pricing equation of the project.

For lower-tail buckets:

P(actual high <= U | X_t) = F(U - nws_t | X_t)

For upper-tail buckets:

P(actual high > L | X_t) = 1 - F(L - nws_t | X_t)

All buckets must be priced together so that the final probabilities form a complete distribution.

---

## 6. Example

Suppose:

NWS forecast high at timestamp t = 73°F

Market buckets:

- <= 69
- 70-71
- 72-73
- 74-75
- >= 76

Convert each bucket into forecast-error space.

Approximate bucket conversion:

- <= 69 means error <= -4
- 70-71 means -3 <= error <= -2
- 72-73 means -1 <= error <= 0
- 74-75 means 1 <= error <= 2
- >= 76 means error >= 3

Then the model estimates CDF values such as:

- F(-4)
- F(-2)
- F(0)
- F(2)

Bucket probabilities are computed from CDF differences:

- P(actual <= 69) = F(-4)
- P(70-71) = F(-2) - F(-4)
- P(72-73) = F(0) - F(-2)
- P(74-75) = F(2) - F(0)
- P(actual >= 76) = 1 - F(2)

Important caveat:

Exact bucket conversion depends on the market’s settlement rules and endpoint conventions. Some markets may use inclusive integer buckets, while the model may treat errors as continuous intervals. Before coding, bucket definitions must be standardized carefully.

---

## 7. Required Model Output

For every market at timestamp t, the model should output a full probability table.

Example format:

| Bucket | Error Interval | Model Probability | Market Price | Edge |
|---|---:|---:|---:|---:|
| <=69 | error <= -4 | 0.12 | 0.10 | +0.02 |
| 70-71 | -3 to -2 | 0.20 | 0.24 | -0.04 |
| 72-73 | -1 to 0 | 0.31 | 0.29 | +0.02 |
| 74-75 | 1 to 2 | 0.25 | 0.27 | -0.02 |
| >=76 | error >= 3 | 0.12 | 0.10 | +0.02 |

The probabilities above are only examples.

The actual model output must satisfy:

- every bucket has a probability
- all probabilities are between 0 and 1
- probabilities are nonnegative
- probabilities sum to 1
- tail buckets are included
- probabilities are derived from one coherent CDF, not unrelated binary predictions

---

## 8. Feature Rules and No-Leakage Contract

Every feature must be available at timestamp t.

A feature is legal only if it could have been known at or before the prediction time.

Legal feature examples:

- NWS forecast high available at t
- forecast issue time
- time since latest forecast update
- current observed temperature at or before t
- hourly observations before t
- morning low so far, if computed only from observations before t
- wind, humidity, cloud cover, pressure, and precipitation observations before t
- forecast revisions up to t
- time of day
- day of year
- month
- city/station
- historical forecast error for prior settled days
- recent city-specific bias using only previous days
- model guidance available before t
- market prices available at or before t, if later used as features

Illegal or dangerous feature examples:

- actual official high for the same day
- any observation after timestamp t
- final daily maximum temperature if the day is not over
- later NWS forecast updates after t
- settlement bucket
- final market price after trading closes
- market result
- features computed from the full day’s weather when predicting earlier in the day
- rolling averages that accidentally include the current target day’s final result
- random train/test splits that mix future examples into training

Every feature should eventually have metadata:

- feature name
- source
- timestamp when published or observed
- timestamp when available to the model
- transformation logic
- whether it depends on same-day data
- whether it depends on prior-day-only data

---

## 9. Validation Requirements

The project must validate four separate things:

1. No leakage
2. Probability coherence
3. Calibration
4. Economic usefulness

### 9.1 Leakage Validation

Checks:

- Every row has a prediction timestamp t.
- Every feature has an availability timestamp.
- Every feature timestamp is less than or equal to t.
- Actual high is used only in the target, never in features.
- Same-day observations after t are excluded.
- Forecast updates after t are excluded.
- Market prices after t are excluded.
- Historical rolling features use only prior available data.
- Evaluation is chronological, not random.

The model should be assumed suspicious until these checks pass.

### 9.2 Probability Coherence Validation

For each prediction:

- CDF values must be nondecreasing across boundaries.
- CDF values must be between 0 and 1.
- Bucket probabilities must be nonnegative.
- Bucket probabilities must sum to 1.
- Tail probabilities must be included.
- No bucket should be priced independently from the others.

Required checks:

sum(bucket_probs) ≈ 1

min(bucket_probs) >= 0

max(bucket_probs) <= 1

F(c1) <= F(c2) for all c1 < c2

### 9.3 Calibration Validation

The model must be evaluated as a probability model, not just as a classifier.

Useful metrics:

- Brier score
- log loss
- reliability curves
- calibration by bucket
- calibration by city
- calibration by time of day
- calibration by forecast horizon
- expected calibration error

Important principle:

A model can have decent classification accuracy but still be useless for trading if its probabilities are miscalibrated.

For trading, calibrated probabilities matter more than simply predicting the most likely bucket.

### 9.4 Economic Validation

Later in the project, once market prices are included, evaluate:

edge_i = model_probability_i - market_price_i

A trade is only potentially attractive if:

edge_i > transaction costs + spread cost + liquidity risk + model uncertainty buffer

Profitability cannot be claimed from backtest returns alone unless the backtest includes:

- realistic historical market prices
- bid/ask spreads
- fees
- liquidity constraints
- fill assumptions
- timestamp-correct market data
- no use of future prices
- out-of-sample testing

Day 1 does not require building this trading system. Day 1 only defines the probability signal correctly.

---

## 10. Train/Test Splitting

The project should not use random train/test splits as the main evaluation method.

Weather forecasting and markets are time-dependent. Random splits can leak future regimes into the training set and produce overly optimistic results.

Preferred evaluation methods:

- chronological train/test split
- walk-forward validation
- expanding-window validation
- rolling-window validation

Example:

Train on 2021-2023  
Validate on 2024  
Test on 2025

Or:

Train up to date D  
Predict after date D  
Roll forward and repeat

The evaluation design must simulate the actual historical information flow.

---

## 11. Model Scope

The model hierarchy is frozen for the current project phase.

Primary model:

- NGBoost / DGBM on forecast_error

Required baseline:

- empirical historical forecast-error distribution

Prediction output:

- full probability distribution across market buckets, derived from the model-implied CDF

Optional future extensions only:

- CDF classifier
- multiclass bucket model
- quantile model
- alternative DGBM implementations

Not included in the current probability-model milestone:

- live trading
- automatic order execution
- Kelly sizing
- portfolio optimization
- market-making logic
- complex deep learning
- reinforcement learning
- direct profitability claims

Those should only be added after the NGBoost probability signal is proven coherent and reasonably calibrated against the empirical baseline.

---

## 12. Why Not Quantile Regression First?

Quantile regression predicts values of the target at specific probability levels.

For example:

- q10 = error value at the 10th percentile
- q50 = median error
- q90 = error value at the 90th percentile

That is useful, but it gives the inverse problem:

Given probability p, what is the error threshold?

Weather bucket markets need the opposite:

Given bucket boundary c, what is the probability error is below c?

That is a CDF question.

The market needs:

P(error <= c)

and

P(a < error <= b)

Therefore, the primary NGBoost model must expose a coherent CDF for bucket pricing.

Quantile regression may still be useful later as a secondary diagnostic, but it should not be the main architecture for bucket pricing.

---

## 13. Why Not Separate Bucket Classifiers?

A separate classifier for each bucket would estimate:

P(bucket_i wins)

But training each bucket independently can create incoherent probabilities.

Problems:

- probabilities may not sum to 1
- two adjacent buckets may both receive high probabilities
- all buckets may receive low probabilities
- tail risk may be mishandled
- the model does not naturally learn an ordered temperature distribution

The NGBoost distribution approach is better for the current project because temperature buckets are ordered and bucket probabilities can be derived from one coherent forecast-error CDF.

By modeling the forecast-error distribution and differencing CDF values, the model respects the ordered structure of the outcome.

---

## 14. Day 1 Success Criteria

Day 1 is complete when the project spec clearly answers these questions:

1. What is the model predicting?

The conditional distribution of NWS forecast error, exposed through a model-implied CDF.

2. What is the target?

error = actual official high - NWS predicted high at timestamp t

3. How are model probabilities created?

NGBoost/DGBM is trained on forecast_error with log score. The fitted distribution supplies CDF values at any boundary c.

4. How are bucket probabilities computed?

By converting temperature bucket boundaries into error boundaries and subtracting CDF values.

5. Why do probabilities sum to 1?

Because the full set of buckets partitions the forecast-error distribution, including lower and upper tails.

6. What prevents leakage?

Every feature must be available at or before timestamp t.

7. How will the model be evaluated?

Using chronological validation, probability coherence checks, calibration metrics, and later economic backtesting.

8. What does profitability require?

Calibrated model probabilities that differ from market prices by more than costs, spreads, liquidity risk, and model uncertainty.

---

## 15. Day 2 Handoff

Day 2 should focus on data schema and data sourcing, not model training yet.

Day 2 deliverables should include:

- raw data schema
- forecast table schema
- observation table schema
- actual official high table schema
- feature table schema
- target generation logic
- timestamp availability rules
- candidate data sources
- first no-leakage data pipeline design

The next step is to define exactly what data is needed and how each field will be timestamped.

No model should be trained until the data contract is clear enough to prevent future leakage.

---

## 16. Main Pitfalls

### Pitfall 1: Predicting raw temperature directly

The project is not mainly about predicting the final high from scratch. It is about modeling forecast error around the NWS forecast.

### Pitfall 2: Producing one probability at a time

The market contains mutually exclusive buckets. The model needs a full probability distribution, not isolated predictions.

### Pitfall 3: Ignoring tails

If the market has upper and lower tail buckets, they must be included. Otherwise probabilities will not sum to 1.

### Pitfall 4: Trusting model probabilities without calibration

Distributional model probabilities can be miscalibrated. Calibration must be tested directly.

### Pitfall 5: Using random splits

Random splits can make the model look better than it is. The project needs chronological or walk-forward validation.

### Pitfall 6: Accidentally leaking same-day weather data

Same-day observations are only legal if they occurred before timestamp t. Anything after t is future information.

### Pitfall 7: Claiming profitability too early

A good probability model is not automatically a profitable trading strategy. Profitability depends on market prices, costs, spreads, liquidity, and execution.

---

## 17. Final Project Definition

This project builds a timestamp-correct NGBoost/DGBM probability model for weather bucket markets.

The model estimates the conditional distribution of NWS forecast error using only information available at prediction time. It converts temperature market buckets into forecast-error intervals, computes coherent fair probabilities from the model-implied CDF, validates calibration and leakage, compares against the empirical baseline, and only later compares fair probabilities to market prices for possible trading edges.

The first milestone is not to make money.

The first milestone is to produce trustworthy bucket probabilities.
