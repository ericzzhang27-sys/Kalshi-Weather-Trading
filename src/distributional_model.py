from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_COLUMNS_PATH = REPO_ROOT / "outputs" / "day8_features" / "feature_columns.json"
TARGET_COLUMN = "forecast_error"

LEAKAGE_EXACT_COLUMNS = {
    "forecast_error",
    "actual_high",
    "official_high",
    "final_high",
    "daily_high",
    "observed_high",
    "observed_daily_high",
    "settlement_temp",
    "max_temp_full_day",
    "target",
    "label",
    "date",
    "target_date",
    "prediction_time",
    "prediction_timestamp",
    "timestamp",
    "prediction_clock_time",
    "station",
    "station_id",
    "location",
    "forecast_source",
}

LEAKAGE_NAME_FRAGMENTS = (
    "forecast_error",
    "actual_high",
    "official_high",
    "final_high",
    "daily_high",
    "observed_high",
    "settlement",
    "max_temp_full_day",
    "full_day",
    "target_value",
    "bucket",
)

METADATA_OR_UNSAFE_FRAGMENTS = (
    "timestamp",
    "source_time",
    "issue_time",
    "valid_time",
    "created_at",
    "reference_time",
    "run_time",
    "as_of",
)

FUTURE_LOOKING_FRAGMENTS = (
    "future",
    "next_",
    "tomorrow",
    "post_settlement",
    "after_settlement",
)

MIN_SIGMA_FOR_NLL = 1e-6


def get_feature_columns(
    df: pd.DataFrame,
    feature_columns_path: str | Path = DEFAULT_FEATURE_COLUMNS_PATH,
) -> list[str]:
    """Return leakage-safe numeric feature columns for distributional modeling."""

    path = Path(feature_columns_path)
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
        raw_columns = list(spec.get("feature_columns", []))
        if not raw_columns:
            raise ValueError(f"Feature spec at {path} does not contain feature_columns")
    else:
        raw_columns = _infer_numeric_feature_columns(df)

    missing = [column for column in raw_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Feature columns are missing from dataframe: {missing}")

    unsafe = [column for column in raw_columns if is_unsafe_feature_name(column)]
    if unsafe:
        raise ValueError(f"Feature columns include leakage/metadata fields: {unsafe}")

    nonnumeric = [
        column
        for column in raw_columns
        if not (
            pd.api.types.is_numeric_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]
    if nonnumeric:
        raise ValueError(f"Feature columns must be numeric or boolean: {nonnumeric}")

    all_missing = [column for column in raw_columns if df[column].isna().all()]
    if all_missing:
        raise ValueError(f"Feature columns are entirely missing: {all_missing}")

    return raw_columns


def train_ngboost_normal(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | np.ndarray | None = None,
) -> Any:
    """Train an NGBoost Normal model for forecast-error density parameters."""

    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from ngboost.scores import LogScore
    except ImportError as exc:
        raise ImportError(
            "NGBoost is required for Day 11 distributional training. "
            "Install it with `python -m pip install ngboost`, then rerun "
            "`python -m src.train_ngboost`."
        ) from exc

    base_learner = DecisionTreeRegressor(
        max_depth=2,
        min_samples_leaf=50,
        random_state=11,
    )
    model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=base_learner,
        n_estimators=120,
        learning_rate=0.05,
        minibatch_frac=1.0,
        col_sample=1.0,
        random_state=11,
        verbose=False,
    )

    y_train_array = np.asarray(y_train, dtype=float)
    if X_val is not None and y_val is not None and len(X_val) > 0:
        model.fit(
            X_train,
            y_train_array,
            X_val=X_val,
            Y_val=np.asarray(y_val, dtype=float),
            early_stopping_rounds=20,
        )
    else:
        model.fit(X_train, y_train_array)
    return model


def predict_distribution_params(
    model: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_dist = model.pred_dist(X)
    mu = _extract_distribution_param(predicted_dist, ("loc", "mu", "mean"))
    sigma = _extract_distribution_param(predicted_dist, ("scale", "sigma", "std"))

    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    if mu_array.shape[0] != len(X) or sigma_array.shape[0] != len(X):
        raise ValueError(
            "Predicted distribution parameter lengths do not match input rows: "
            f"mu={mu_array.shape[0]}, sigma={sigma_array.shape[0]}, rows={len(X)}"
        )
    return mu_array, sigma_array


def normal_nll(
    y_true: pd.Series | np.ndarray,
    mu: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray,
    min_sigma: float = MIN_SIGMA_FOR_NLL,
) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    safe_sigma = np.clip(sigma_array, min_sigma, None)
    return 0.5 * np.log(2.0 * math.pi * safe_sigma**2) + ((y - mu_array) ** 2) / (
        2.0 * safe_sigma**2
    )


def is_unsafe_feature_name(column: str) -> bool:
    lower = column.lower()
    if lower in LEAKAGE_EXACT_COLUMNS:
        return True
    fragments = LEAKAGE_NAME_FRAGMENTS + METADATA_OR_UNSAFE_FRAGMENTS + FUTURE_LOOKING_FRAGMENTS
    return any(fragment in lower for fragment in fragments)


def validate_no_leakage_feature_columns(feature_columns: list[str]) -> None:
    unsafe = [column for column in feature_columns if is_unsafe_feature_name(column)]
    if unsafe:
        raise ValueError(f"Unsafe feature columns selected: {unsafe}")


def _infer_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if is_unsafe_feature_name(column):
            continue
        if df[column].isna().all():
            continue
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
            columns.append(column)
    if not columns:
        raise ValueError("No numeric leakage-safe feature columns were found")
    return columns


def _extract_distribution_param(predicted_dist: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(predicted_dist, name):
            return getattr(predicted_dist, name)

    params = getattr(predicted_dist, "params", None)
    if isinstance(params, dict):
        for name in names:
            if name in params:
                return params[name]

    raise ValueError(
        "Could not extract distribution parameter from NGBoost prediction. "
        f"Tried names: {', '.join(names)}"
    )
