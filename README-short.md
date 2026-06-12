# Quick Repo Overview

This is a short, simple readme for the Kalshi weather probability model.

## What it does
- Predicts the distribution of daily high temperature error, not the raw temperature.
- Uses Kalshi-style bucket markets as the reasoning: probabilities for temperature ranges matter.
- Trains a distributional model with NGBoost and turns the output into bucket probabilities.

## Current data normal
- Actual highs use official NOAA/NWS daily TMAX for Central Park.
- Intraday observed features use IEM/NWS ASOS station observations.
- Forecast highs use timestamp-safe NWS/NDFD historical MaxT forecasts.
- Open-Meteo forecast history is not the model's training forecast anchor.
- The restored 36-feature model contract reproduces hourly forecast-relative columns from the NDFD daily-high forecast because the current NWS archive is daily MaxT, not hourly forecast temperature.

## Important files
- `src/train_ngboost.py` - main training script and workflow.
- `config/model_config.yaml` - model settings like distribution, learning rate, tree depth, and sigma scale.
- `data/processed/modeling_rows_v1.csv` - main modeling table input.
- `outputs/final_feature_list.json` - feature list used by the model.
- `src/distributional_model.py` - distributional training helpers, target column definition, and NGBoost wrapper.
- `src/distribution_pricing.py` - converts model output into bucket probabilities.
- `src/evaluation.py` - evaluation metrics like NLL, Brier score, and coverage.
- `src/splits.py` - chronological train/validation/test split logic.

## Main variables and ideas
- `TARGET_COLUMN` - the model target, usually forecast error.
- `distribution` - distribution type used by NGBoost (Laplace or Normal style).
- `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `minibatch_frac` - tree and training settings.
- `sigma_scale` - rescales the predicted distribution spread after model output.
- `DEFAULT_CONFIG_PATH`, `DEFAULT_MODELING_TABLE_PATH`, `MODEL_OUTPUT_PATH` - default paths used by the training script.

## Key methods
- `run_standard_training()` - full train/eval workflow.
- `build_imputed_feature_frames()` - prepares feature matrices and fills missing values.
- `train_ngboost_distribution()` - fits the NGBoost distributional model.
- `predict_distribution_details()` - makes predictions for mu and sigma.
- `distribution_nll()` - computes negative log-likelihood for validation/testing.
- `price_buckets_for_dataframe()` - turns distribution output into bucket probabilities.

## Dependencies
Install the main dependencies with:

```bash
python -m pip install -r requirements.txt
```

Main packages used:
- `ngboost`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `PyYAML`
- `matplotlib`

## Run it
From the repo root:

```bash
python -m src.train_ngboost
```

Verify feature provenance and model-safe feature integrity:

```bash
python scripts/verify_feature_integrity.py
```

If you want dev/test packages too:

```bash
python -m pip install -r requirements-dev.txt
```

## Why this way
The code is built to estimate probability curves for forecast error, then compare those curves with weather bucket outcomes. That way the model is useful for markets where buckets settle on ranges instead of exact values.
