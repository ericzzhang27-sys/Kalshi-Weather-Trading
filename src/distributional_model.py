from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import cauchy as scipy_cauchy
from scipy.stats import laplace as scipy_laplace
from scipy.stats import norm as scipy_norm
from scipy.stats import t as scipy_t
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

DISTRIBUTION_ALIASES = {
    "normal": "normal",
    "gaussian": "normal",
    "t": "student_t",
    "student_t": "student_t",
    "student-t": "student_t",
    "students_t": "student_t",
    "studentst": "student_t",
    "laplace": "laplace",
    "double_exponential": "laplace",
    "cauchy": "cauchy",
}

SIGNED_NGBOOST_DISTRIBUTIONS = {"normal", "student_t", "laplace", "cauchy"}
POSITIVE_ONLY_NGBOOST_DISTRIBUTIONS = {"gamma", "lognormal", "exponential", "weibull"}


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

    return train_ngboost_distribution(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        distribution="normal",
    )


def train_ngboost_distribution(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    distribution: str = "normal",
) -> Any:
    """Train an NGBoost signed forecast-error distribution."""

    try:
        from ngboost import NGBRegressor
        from ngboost.scores import LogScore
    except ImportError as exc:
        raise ImportError(
            "NGBoost is required for Day 11 distributional training. "
            "Install it with `python -m pip install ngboost`, then rerun "
            "`python -m src.train_ngboost`."
        ) from exc

    dist_name = normalize_distribution_name(distribution)
    if dist_name not in SIGNED_NGBOOST_DISTRIBUTIONS:
        raise ValueError(
            f"Distribution {distribution!r} is not supported for signed forecast_error. "
            f"Supported signed distributions: {sorted(SIGNED_NGBOOST_DISTRIBUTIONS)}"
        )
    dist_class = get_ngboost_distribution_class(dist_name)

    base_learner = DecisionTreeRegressor(
        max_depth=2,
        min_samples_leaf=50,
        random_state=11,
    )
    model = NGBRegressor(
        Dist=dist_class,
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


def normalize_distribution_name(distribution: str) -> str:
    normalized = str(distribution).strip().lower().replace(" ", "_")
    if normalized in DISTRIBUTION_ALIASES:
        return DISTRIBUTION_ALIASES[normalized]
    if normalized in POSITIVE_ONLY_NGBOOST_DISTRIBUTIONS:
        raise ValueError(
            f"{distribution!r} is positive-only and is not valid for raw signed forecast_error"
        )
    raise ValueError(
        f"Unsupported distribution {distribution!r}. "
        f"Supported signed distributions: {sorted(SIGNED_NGBOOST_DISTRIBUTIONS)}"
    )


def get_ngboost_distribution_class(distribution: str) -> Any:
    dist_name = normalize_distribution_name(distribution)
    try:
        from ngboost.distns import Cauchy, Laplace, Normal, T
    except ImportError as exc:
        raise ImportError("NGBoost is required for distributional training") from exc

    mapping = {
        "normal": Normal,
        "student_t": T,
        "laplace": Laplace,
        "cauchy": Cauchy,
    }
    return mapping[dist_name]


def infer_ngboost_distribution_name(model: Any) -> str:
    dist_class = getattr(model, "Dist", None)
    raw_name = getattr(dist_class, "__name__", None)
    if raw_name is None and dist_class is not None:
        raw_name = str(dist_class).split(".")[-1].strip("'>")
    if raw_name is None:
        return "normal"
    if raw_name == "T":
        return "student_t"
    return normalize_distribution_name(raw_name)


def predict_distribution_details(
    model: Any,
    X: pd.DataFrame,
    distribution: str | None = None,
) -> dict[str, np.ndarray | str | None]:
    predicted_dist = model.pred_dist(X)
    dist_name = (
        normalize_distribution_name(distribution)
        if distribution
        else infer_ngboost_distribution_name(model)
    )

    loc = _extract_distribution_param(predicted_dist, ("loc", "mu", "mean"))
    scale = _extract_distribution_param(predicted_dist, ("scale", "sigma", "std"))
    df = None
    if dist_name in {"student_t", "cauchy"}:
        df = _optional_distribution_param(predicted_dist, ("df", "nu"))

    loc_array = np.asarray(loc, dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    if loc_array.shape[0] != len(X) or scale_array.shape[0] != len(X):
        raise ValueError(
            "Predicted distribution parameter lengths do not match input rows: "
            f"loc={loc_array.shape[0]}, scale={scale_array.shape[0]}, rows={len(X)}"
        )
    if not np.isfinite(loc_array).all():
        raise ValueError("Predicted loc/mu contains non-finite values")
    if not np.isfinite(scale_array).all() or (scale_array <= 0.0).any():
        raise ValueError("Predicted scale/sigma must be finite and greater than 0")

    df_array: np.ndarray | None = None
    if df is not None:
        df_array = np.asarray(df, dtype=float)
        if df_array.shape[0] != len(X):
            raise ValueError(
                "Predicted distribution df length does not match input rows: "
                f"df={df_array.shape[0]}, rows={len(X)}"
            )
        if not np.isfinite(df_array).all() or (df_array <= 0.0).any():
            raise ValueError("Predicted df must be finite and greater than 0")

    return {
        "distribution_type": dist_name,
        "mu": loc_array,
        "sigma": scale_array,
        "scale": scale_array,
        "df": df_array,
    }


def predict_distribution_params(
    model: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    details = predict_distribution_details(model, X)
    return (
        np.asarray(details["mu"], dtype=float),
        np.asarray(details["sigma"], dtype=float),
    )


def normal_nll(
    y_true: pd.Series | np.ndarray,
    mu: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray,
    min_sigma: float = MIN_SIGMA_FOR_NLL,
) -> np.ndarray:
    return distribution_nll(
        y_true=y_true,
        mu=mu,
        sigma=sigma,
        distribution="normal",
        min_sigma=min_sigma,
    )


def distribution_nll(
    y_true: pd.Series | np.ndarray,
    mu: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray,
    distribution: str = "normal",
    df: pd.Series | np.ndarray | float | None = None,
    min_sigma: float = MIN_SIGMA_FOR_NLL,
) -> np.ndarray:
    logpdf = distribution_logpdf(
        y_true,
        mu=mu,
        sigma=sigma,
        distribution=distribution,
        df=df,
        min_sigma=min_sigma,
    )
    return -logpdf


def distribution_logpdf(
    x: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    distribution: str = "normal",
    df: pd.Series | np.ndarray | float | None = None,
    min_sigma: float = MIN_SIGMA_FOR_NLL,
) -> np.ndarray:
    dist_name = normalize_distribution_name(distribution)
    x_array, mu_array, sigma_array = _validate_distribution_arrays(
        x,
        mu,
        sigma,
        min_sigma=min_sigma,
    )
    df_array = _distribution_df_array(df, len(x_array), dist_name)

    if dist_name == "normal":
        values = scipy_norm.logpdf(x_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "student_t":
        values = scipy_t.logpdf(x_array, df=df_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "laplace":
        values = scipy_laplace.logpdf(x_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "cauchy":
        values = scipy_cauchy.logpdf(x_array, loc=mu_array, scale=sigma_array)
    else:
        raise AssertionError(f"Unhandled distribution after validation: {dist_name}")

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{dist_name} logpdf produced non-finite values")
    return values


def distribution_cdf(
    x: pd.Series | np.ndarray | list[float] | float,
    mu: pd.Series | np.ndarray | list[float] | float,
    sigma: pd.Series | np.ndarray | list[float] | float,
    distribution: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> np.ndarray:
    dist_name = normalize_distribution_name(distribution)
    x_array, mu_array, sigma_array = _broadcast_distribution_arrays(x, mu, sigma)
    df_array = _distribution_df_array(df, len(x_array), dist_name)

    if dist_name == "normal":
        values = scipy_norm.cdf(x_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "student_t":
        values = scipy_t.cdf(x_array, df=df_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "laplace":
        values = scipy_laplace.cdf(x_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "cauchy":
        values = scipy_cauchy.cdf(x_array, loc=mu_array, scale=sigma_array)
    else:
        raise AssertionError(f"Unhandled distribution after validation: {dist_name}")

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{dist_name} CDF produced non-finite values")
    if ((values < -1e-12) | (values > 1.0 + 1e-12)).any():
        raise ValueError(f"{dist_name} CDF produced values outside [0, 1]")
    return np.clip(values, 0.0, 1.0)


def distribution_ppf(
    q: pd.Series | np.ndarray | list[float] | float,
    mu: pd.Series | np.ndarray | list[float] | float,
    sigma: pd.Series | np.ndarray | list[float] | float,
    distribution: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> np.ndarray:
    dist_name = normalize_distribution_name(distribution)
    q_array, mu_array, sigma_array = _broadcast_distribution_arrays(q, mu, sigma)
    if ((q_array <= 0.0) | (q_array >= 1.0)).any():
        raise ValueError("Quantiles must be strictly between 0 and 1")
    df_array = _distribution_df_array(df, len(q_array), dist_name)

    if dist_name == "normal":
        values = scipy_norm.ppf(q_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "student_t":
        values = scipy_t.ppf(q_array, df=df_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "laplace":
        values = scipy_laplace.ppf(q_array, loc=mu_array, scale=sigma_array)
    elif dist_name == "cauchy":
        values = scipy_cauchy.ppf(q_array, loc=mu_array, scale=sigma_array)
    else:
        raise AssertionError(f"Unhandled distribution after validation: {dist_name}")

    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{dist_name} PPF produced non-finite interval bounds")
    return values


def distribution_std(
    sigma: pd.Series | np.ndarray | list[float],
    distribution: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> np.ndarray:
    dist_name = normalize_distribution_name(distribution)
    scale = _as_finite_1d_array(sigma, "sigma")
    if (scale <= 0.0).any():
        raise ValueError("sigma/scale must be greater than 0")
    df_array = _distribution_df_array(df, len(scale), dist_name)

    if dist_name == "normal":
        return scale
    if dist_name == "student_t":
        std = np.full_like(scale, np.nan, dtype=float)
        finite_variance = df_array > 2.0
        std[finite_variance] = scale[finite_variance] * np.sqrt(
            df_array[finite_variance] / (df_array[finite_variance] - 2.0)
        )
        return std
    if dist_name == "laplace":
        return scale * math.sqrt(2.0)
    if dist_name == "cauchy":
        return np.full_like(scale, np.nan, dtype=float)
    raise AssertionError(f"Unhandled distribution after validation: {dist_name}")


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


def _optional_distribution_param(predicted_dist: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if hasattr(predicted_dist, name):
            value = getattr(predicted_dist, name)
            if value is not None:
                return value

    params = getattr(predicted_dist, "params", None)
    if isinstance(params, dict):
        for name in names:
            if name in params and params[name] is not None:
                return params[name]
    return None


def _validate_distribution_arrays(
    x: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    min_sigma: float = MIN_SIGMA_FOR_NLL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = _as_finite_1d_array(x, "x")
    mu_array = _as_finite_1d_array(mu, "mu")
    sigma_array = _as_finite_1d_array(sigma, "sigma")
    lengths = {"x": len(x_array), "mu": len(mu_array), "sigma": len(sigma_array)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"x, mu, and sigma must have the same length: {lengths}")
    if len(x_array) == 0:
        raise ValueError("x, mu, and sigma cannot be empty")
    safe_sigma = np.clip(sigma_array, float(min_sigma), None)
    if not np.isfinite(safe_sigma).all() or (safe_sigma <= 0.0).any():
        raise ValueError("sigma must be finite and greater than 0")
    return x_array, mu_array, safe_sigma


def _broadcast_distribution_arrays(
    x: pd.Series | np.ndarray | list[float] | float,
    mu: pd.Series | np.ndarray | list[float] | float,
    sigma: pd.Series | np.ndarray | list[float] | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    try:
        x_b, mu_b, sigma_b = np.broadcast_arrays(x_array, mu_array, sigma_array)
    except ValueError as exc:
        raise ValueError("x, mu, and sigma could not be broadcast to a common shape") from exc

    flat_x = np.ravel(x_b).astype(float)
    flat_mu = np.ravel(mu_b).astype(float)
    flat_sigma = np.ravel(sigma_b).astype(float)
    if not np.isfinite(flat_x).all():
        raise ValueError("x contains non-finite values")
    if not np.isfinite(flat_mu).all():
        raise ValueError("mu contains non-finite values")
    if not np.isfinite(flat_sigma).all() or (flat_sigma <= 0.0).any():
        raise ValueError("sigma must be finite and greater than 0")
    return flat_x, flat_mu, flat_sigma


def _distribution_df_array(
    df: pd.Series | np.ndarray | list[float] | float | None,
    length: int,
    distribution: str,
) -> np.ndarray | None:
    dist_name = normalize_distribution_name(distribution)
    if dist_name == "student_t":
        if df is None:
            raise ValueError("Student-t distribution requires df/degrees-of-freedom values")
        df_array = np.asarray(df, dtype=float)
        if df_array.ndim == 0:
            df_array = np.full(length, float(df_array), dtype=float)
        else:
            df_array = np.ravel(df_array).astype(float)
        if len(df_array) != length:
            raise ValueError(f"df length must be {length}, got {len(df_array)}")
        if not np.isfinite(df_array).all() or (df_array <= 0.0).any():
            raise ValueError("df must be finite and greater than 0")
        return df_array
    if dist_name == "cauchy":
        return np.ones(length, dtype=float)
    return None


def _as_finite_1d_array(values: pd.Series | np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return array
