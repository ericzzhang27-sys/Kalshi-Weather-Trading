# Internal Map — Kalshi Weather Pipeline (pre-implementation inspect)

## Weather Datasets Located
- **Processed hourly/daily obs**: `data/processed/hourly_clean.csv`, `daily_clean.csv` (official NOAA/NWS Central Park)
- **Supervised targets**: `data/processed/supervised_forecast_error_rows.csv` (prediction_time ≤ official_high, NDFD forecast_high)
- **Modeling rows**: `data/processed/modeling_rows_v1.csv` (36k rows, 2022-01-01 to 2026-05-20, 92 cols, timestamp-safe features, split chronological)
- **Raw NDFD archive**: `data/raw/ndfd/`, `ndfd_full_tmp/` (daily MaxT, issue-time filtered, 2021-2026)
- **Forecast clean**: `data/processed/forecasts_clean.csv` (27 cols, NWS/NDFD daily MaxT)

## Saved Model Predictions / Probabilities
- `outputs/final_bucket_probability_predictions.csv` — **only 1 day (2022-01-01, 30 rows)** — not usable for historical backtest (tiny sample, maybe leftover demo)
- `outputs/ngboost_bucket_probs_v0.csv` — **primary**: 120k rows, 6 buckets per prediction, Normal(0.328,1.49), forecast_high 42-54F, validation/test 2024-2026, schema: row_id,date,prediction_time,location,split,forecast_high,actual_high,forecast_error,bucket_index,bucket_name,bucket_lower_temp,bucket_upper_temp,error_lower,error_upper,mu,sigma,distribution_type,df,probability,timestamp,prediction_timestamp,forecast_horizon_hours,nll
- `outputs/ngboost_bucket_probabilities_calibrated.csv` — **authoritative for backtest**: 125k rows, laplace calibrated (sigma*0.7), same 2024-2026 window, more comprehensive — used as default prob source
- `outputs/ngboost_distribution_params_v0.csv` — raw mu/sigma per prediction (20k rows), source for bucket pricing
- `models/ngboost_*` — pkl artifacts (Normal/Laplace), feature list `outputs/final_feature_list.json` (42 features, current41_postpeak)

**Schema standardized** to: timestamp/prediction_time, target_date/date, city/location, target_date, bucket_lower/upper, bucket_label, probability/model_probability, model_name, actual_high, forecast_high

## Existing Kalshi API Utilities
- `src/trading/kalshi_client.py` — RSA-PSS signing, GET /markets with pagination, retries, REGISTERED TRANSIENT codes, but no historical endpoints
- `src/trading/market_discovery.py` — discovers live markets via location/weather terms, filtering, eligibility
- `src/trading/contract_mapping.py` — parses floor/cap strike → TemperatureBucket (lower-open upper-closed, half-degree), text fallback, validation
- `src/trading/orderbook.py` — YES/NO bid/ask normalization, spread, staleness, infer NO ask from YES bid
- `src/trading/edge.py` — Kalshi fee formula 0.07*p*(1-p) ceiling, edge = fair - ask - fee, liquidity/spread checks (already correct)
- `src/trading/config.py` — series_tickers per city (KXHIGHNY etc)
- `src/bucket_schema.py` — TemperatureBucket, Bucket, make_integer_temperature_buckets, validation

## Existing Backtesting/Trading Code
- **No historical backtester before** — only live loop `src/trading/live_loop.py` (shadow/paper), `paper_broker.py` (hypothetical fills), `portfolio.py`, `risk.py`
- `src/evaluation.py` — NLL, Brier, coverage for distributional model (offline, not trading)
- `src/calibration.py` — reliability tables

## Reuse Decisions
- Reuse `bucket_schema` for all bucket boundaries, never infer from strings if strike fields available
- Reuse `contract_mapping` logic inside `src/kalshi/normalize_markets.py` for bucket parsing
- Reuse `edge.estimate_kalshi_buy_fee` logic in `src/backtest/fees.py` (isolated, documented)
- Keep `KalshiHistoricalClient` as extension of `KalshiClient` for historical endpoints (/historical/markets, /historical/markets/{ticker}/candlesticks, /historical/trades, /historical/cutoff)
- Preserve frozen feature contract `outputs/final_feature_list.json` — no retraining

## Calendar Note
- Actual earliest usable 1-min candle with synthetic hourly sampling: 2024-01-01 00:00 UTC (real API would require credentials; earliest *actual* historical candle determined by API cutoff, reported in `data/kalshi/metadata/download_summary.json`)
