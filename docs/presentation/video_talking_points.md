# Video Talking Points

## 1. Purpose / Project Goal
- This project estimates probabilities for Kalshi-style New York City high-temperature bucket markets.
- The model predicts a full distribution for the NWS-reported final daily high temperature by modeling forecast error: `actual_high - forecast_high`.
- Market prices imply probabilities. The project compares model probabilities against those market-implied probabilities to identify possible edge.
- Calculus is central because each bucket probability is the area under a probability density curve between two temperature boundaries.
- The project demonstrates integrals, probability distributions, CDF differences, expected value, probability sensitivity, and gradient-based NGBoost training.
- This is a probability-signal research layer. It does not claim live trading profitability.

## 2. Main Code Walkthrough
- `src/weather_data.py` and `src/forecast_data.py` define cleaned weather and forecast inputs used by the pipeline.
- `src/target_builder.py` and `src/supervised_table.py` create supervised rows where each row has a prediction timestamp, forecast high, actual high, and forecast-error target.
- `src/features.py` builds timestamp-safe model features from observations, forecasts, calendar fields, and intraday temperature path context.
- `src/leakage_checks.py` validates that target columns and future timestamps are not included as model features.
- `src/splits.py` creates chronological train, validation, and test splits. The project avoids random splits because weather markets are time dependent.
- `src/train_ngboost.py` trains the NGBoost/DGBM-style distribution model and saves distribution parameters in `outputs/ngboost_distribution_params_v0.csv`.
- `src/distribution_pricing.py` converts final-temperature buckets into forecast-error intervals and prices each bucket with CDF differences.
- `src/evaluation.py` and `src/calibration.py` compute probability-quality diagnostics such as NLL, Brier score, interval coverage, PIT histograms, and calibration tables.
- `scripts/build_features.py`, `scripts/evaluate_ngboost.py`, and `scripts/calibrate_ngboost.py` are the main reproducible command-line workflow scripts.
- Data flow: raw weather and forecast data -> supervised target rows -> timestamp-safe features -> NGBoost distribution -> bucket probabilities -> evaluation and edge examples.

## 3. UML / Pseudocode
- The architecture diagram shows the full project path from raw data sources through the model, bucket converter, evaluator, edge scanner, and saved outputs.
- The main design idea is that bucket probabilities are not independent classifiers. They are all derived from one coherent forecast-error distribution.
- Pseudocode:
  1. Load data
  2. Validate data
  3. Build features
  4. Train model
  5. Convert distribution to bucket probabilities
  6. Evaluate predictions
  7. Compare to market prices

## 4. Execution Demo
- Install runtime dependencies:
  `pip install -r requirements.txt`
- Install notebook and test dependencies:
  `pip install -r requirements-dev.txt`
- Build timestamp-safe modeling rows:
  `python scripts/build_features.py`
- Train the configured NGBoost distribution model:
  `python -m src.train_ngboost`
- Convert predicted distributions into bucket probabilities:
  `python -m src.distribution_pricing`
- Evaluate probability quality:
  `python scripts/evaluate_ngboost.py`
- Run calibration diagnostics:
  `python scripts/calibrate_ngboost.py`
- Open the video walkthrough notebook:
  `jupyter notebook notebooks/project_walkthrough.ipynb`
- Viewers should see CSV outputs under `outputs/`, figures under `outputs/figures/`, and model artifacts under `models/`.

## 5. Output Discussion
- `outputs/ngboost_distribution_params_v0.csv` contains one distribution prediction per validation/test timestamp, including `mu`, `sigma`, and the realized `forecast_error`.
- `outputs/ngboost_bucket_probs_v0.csv` contains the bucket probabilities derived from the model CDF.
- `outputs/ngboost_evaluation_report.csv` summarizes NLL, interval log loss, Brier score, coverage error, and residual diagnostics.
- `outputs/ngboost_calibration_report.csv` compares raw and sigma-scaled probability calibration.
- Figures such as PIT histograms, coverage plots, and calibration curves show whether the model probabilities behave like calibrated probabilities.
- A positive edge example means `p_model - q_market` is positive, but real trading would also need executable prices, fees, spread, slippage, liquidity, and settlement-risk handling.
- Limitations: hourly data may miss true intrahour highs, NWS settlement rules matter, rounding and temperature conversion can change bucket mapping, and market slippage can erase apparent edge.
- Next steps are adding timestamp-correct market prices, order-book snapshots, paper-trading evaluation, and stronger walk-forward validation.
