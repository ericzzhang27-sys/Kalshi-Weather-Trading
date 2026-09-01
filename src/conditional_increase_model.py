from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Exponential, HalfNormal, LogNormal, Poisson, Weibull
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

from src.negative_binomial import NegativeBinomial


TARGET_SHIFT = 1.0
CONTINUOUS_DISTRIBUTIONS = {
    "weibull": Weibull,
    "halfnormal": HalfNormal,
    "lognormal": LogNormal,
    "exponential": Exponential,
}


def positive_increase_rows(
    df: pd.DataFrame,
    target_col: str = "remaining_increase",
) -> pd.DataFrame:
    target = pd.to_numeric(df[target_col], errors="coerce")
    result = df.loc[target > 0].copy()
    if result.empty:
        raise ValueError("No positive remaining-increase rows are available")
    if not (pd.to_numeric(result[target_col], errors="raise") > 0).all():
        raise AssertionError("Conditional model received a non-positive target")
    return result


def conditional_dispersion(
    values: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    """Sample dispersion for Y=delta-1 on an already positive risk set."""
    y = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) == 0:
        return {"n": 0, "mean_y": float("nan"), "variance_y": float("nan"), "dispersion": float("nan")}
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ValueError("Dispersion target must contain non-negative integers")
    mean = float(y.mean())
    variance = float(y.var(ddof=1)) if len(y) > 1 else float("nan")
    return {
        "n": int(len(y)),
        "mean_y": mean,
        "variance_y": variance,
        "dispersion": float(variance / mean) if mean > 0 else float("nan"),
    }


def conditional_dispersion_by_group(
    frame: pd.DataFrame,
    group: pd.Series,
    *,
    target_col: str = "remaining_increase",
) -> pd.DataFrame:
    positive = positive_increase_rows(frame, target_col)
    aligned_group = pd.Series(group, index=frame.index).loc[positive.index]
    y = pd.to_numeric(positive[target_col], errors="raise") - TARGET_SHIFT
    rows = []
    for label, indexes in aligned_group.groupby(aligned_group, observed=True).groups.items():
        rows.append({"group": str(label), **conditional_dispersion(y.loc[indexes])})
    return pd.DataFrame(rows)


def _training_matrix(
    train_df: pd.DataFrame,
    features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, SimpleImputer]:
    positive = positive_increase_rows(train_df, target_col)
    missing = sorted(set(features).difference(positive.columns))
    if missing:
        raise ValueError(f"Missing conditional-increase features: {missing}")
    X = positive[features].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(positive[target_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Conditional-increase target must be finite and strictly positive")
    imputer = SimpleImputer(strategy="median")
    return X, target, imputer


def _ngboost_regressor(
    distribution: type,
    *,
    n_estimators: int,
    learning_rate: float,
    random_state: int,
    min_samples_leaf: int,
) -> NGBRegressor:
    base = DecisionTreeRegressor(
        criterion="friedman_mse",
        max_depth=3,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    return NGBRegressor(
        Dist=distribution,
        Base=base,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        minibatch_frac=0.8,
        natural_gradient=True,
        random_state=random_state,
        verbose=False,
    )


def train_positive_ngboost(
    train_df: pd.DataFrame,
    features: list[str],
    distribution: str,
    *,
    target_col: str = "remaining_increase",
    n_estimators: int = 200,
    learning_rate: float = 0.03,
    random_state: int = 42,
    min_samples_leaf: int = 50,
) -> dict[str, Any]:
    """Fit an NGBoost family on delta conditional on delta > 0.

    Continuous families are later discretized as P(k-1 < delta <= k). This
    keeps their probabilities aligned with the integer-Fahrenheit target.
    """
    key = distribution.lower().replace("_", "")
    if key not in CONTINUOUS_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported positive distribution {distribution!r}; "
            f"choose from {sorted(CONTINUOUS_DISTRIBUTIONS)}"
        )
    X, target, imputer = _training_matrix(train_df, features, target_col)
    transformed = imputer.fit_transform(X)
    model = _ngboost_regressor(
        CONTINUOUS_DISTRIBUTIONS[key],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(transformed, target)
    return {
        "type": "positive_continuous_ngboost",
        "distribution": key,
        "model": model,
        "imputer": imputer,
        "features": list(features),
        "target_col": target_col,
        "integer_settlement": True,
    }


def train_shifted_poisson_ngboost(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "remaining_increase",
    n_estimators: int = 200,
    learning_rate: float = 0.03,
    random_state: int = 42,
    min_samples_leaf: int = 50,
) -> dict[str, Any]:
    """Fit the discrete 1 + Poisson benchmark retained for diagnostics."""
    X, target, imputer = _training_matrix(train_df, features, target_col)
    shifted = target - TARGET_SHIFT
    if (shifted < 0).any() or not np.allclose(shifted, np.round(shifted)):
        raise ValueError("Shifted-Poisson target must be a non-negative integer")
    transformed = imputer.fit_transform(X)
    model = _ngboost_regressor(
        Poisson,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(transformed, shifted.astype(int))
    return {
        "type": "shifted_poisson_ngboost",
        "distribution": "shifted_poisson",
        "model": model,
        "imputer": imputer,
        "features": list(features),
        "target_shift": TARGET_SHIFT,
        "target_col": target_col,
        "integer_settlement": True,
    }


def train_shifted_negative_binomial_ngboost(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "remaining_increase",
    n_estimators: int = 200,
    learning_rate: float = 0.02,
    random_state: int = 42,
    min_samples_leaf: int = 50,
) -> dict[str, Any]:
    """Fit Y=delta-1 with NB2 variance mu + alpha*mu^2."""
    X, target, imputer = _training_matrix(train_df, features, target_col)
    shifted = target - TARGET_SHIFT
    if (shifted < 0).any() or not np.allclose(shifted, np.round(shifted)):
        raise ValueError("Shifted-Negative-Binomial target must be a non-negative integer")
    transformed = imputer.fit_transform(X)
    model = _ngboost_regressor(
        NegativeBinomial,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    # The custom NB2 score uses stable ordinary gradients; its two parameters
    # still vary with X through separate boosting trees.
    model.natural_gradient = False
    model.fit(transformed, shifted.astype(int))
    return {
        "type": "shifted_negative_binomial_ngboost",
        "distribution": "shifted_negative_binomial",
        "model": model,
        "imputer": imputer,
        "features": list(features),
        "target_shift": TARGET_SHIFT,
        "target_col": target_col,
        "integer_settlement": True,
        "variance": "mu + alpha * mu^2",
    }


def train_conditional_candidate(
    train_df: pd.DataFrame,
    features: list[str],
    distribution: str,
    **kwargs: Any,
) -> dict[str, Any]:
    if distribution.lower().replace("_", "") in {"poisson", "shiftedpoisson"}:
        return train_shifted_poisson_ngboost(train_df, features, **kwargs)
    if distribution.lower().replace("_", "") in {
        "negativebinomial",
        "shiftednegativebinomial",
        "nb",
        "nb2",
    }:
        return train_shifted_negative_binomial_ngboost(train_df, features, **kwargs)
    return train_positive_ngboost(train_df, features, distribution, **kwargs)


def predict_conditional_distribution(
    artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
) -> Any:
    if artifact.get("type") not in {
        "shifted_poisson_ngboost",
        "shifted_negative_binomial_ngboost",
        "positive_continuous_ngboost",
    }:
        raise ValueError("Unsupported conditional-increase artifact")
    frame = pd.DataFrame([features]) if isinstance(features, dict) else features
    missing = sorted(set(artifact["features"]).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing conditional-increase features: {missing}")
    X = frame[artifact["features"]].apply(pd.to_numeric, errors="coerce")
    transformed = artifact["imputer"].transform(X)
    return artifact["model"].pred_dist(transformed)


def predict_conditional_mean(
    artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
) -> np.ndarray:
    distribution = predict_conditional_distribution(artifact, features)
    if artifact["type"] in {"shifted_poisson_ngboost", "shifted_negative_binomial_ngboost"}:
        return np.asarray(distribution.params["mu"], dtype=float) + float(artifact["target_shift"])
    return np.asarray(distribution.dist.mean(), dtype=float)


def conditional_cdf(
    artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
    remaining_increase: float | np.ndarray,
) -> np.ndarray:
    """Evaluate P(delta <= x | delta > 0), on integer-degree support."""
    distribution = predict_conditional_distribution(artifact, features)
    threshold = np.floor(np.asarray(remaining_increase, dtype=float))
    if artifact["type"] in {"shifted_poisson_ngboost", "shifted_negative_binomial_ngboost"}:
        values = distribution.dist.cdf(threshold - artifact["target_shift"])
    else:
        values = distribution.dist.cdf(threshold)
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def conditional_interval_probability(
    artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
    lower: float | np.ndarray,
    upper: float | np.ndarray,
) -> np.ndarray:
    """Return P(lower < delta <= upper | delta > 0)."""
    return np.clip(
        conditional_cdf(artifact, features, upper)
        - conditional_cdf(artifact, features, lower),
        0.0,
        1.0,
    )


def _realized_interval_probabilities(
    artifact: dict[str, Any], distribution: Any, target: np.ndarray
) -> np.ndarray:
    if artifact["type"] in {"shifted_poisson_ngboost", "shifted_negative_binomial_ngboost"}:
        shifted = (target - artifact["target_shift"]).astype(int)
        return np.asarray(distribution.dist.pmf(shifted), dtype=float)
    upper = np.asarray(distribution.dist.cdf(target), dtype=float)
    lower = np.asarray(distribution.dist.cdf(target - 1.0), dtype=float)
    return np.clip(upper - lower, 0.0, 1.0)


def _central_integer_interval(distribution: Any, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(1.0, np.ceil(np.asarray(distribution.dist.ppf(alpha / 2.0), dtype=float)))
    upper = np.maximum(
        1.0, np.ceil(np.asarray(distribution.dist.ppf(1.0 - alpha / 2.0), dtype=float))
    )
    return lower, upper


def _quantile_crps(
    distribution: Any,
    target: np.ndarray,
    points: int = 99,
    quantile_shift: float = 0.0,
) -> float:
    """Numerical CRPS from the quantile-loss identity."""
    quantiles = np.linspace(0.01, 0.99, points)
    predicted = (
        np.asarray(distribution.dist.ppf(quantiles[:, None]), dtype=float) + quantile_shift
    )
    if predicted.shape != (points, len(target)):
        raise ValueError("Unexpected distribution quantile shape while computing CRPS")
    error = target[None, :] - predicted
    loss = np.where(
        error >= 0,
        quantiles[:, None] * error,
        (quantiles[:, None] - 1.0) * error,
    )
    return float(np.mean(2.0 * np.trapezoid(loss, quantiles, axis=0)))


def evaluate_conditional_distribution(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    target_col: str = "remaining_increase",
    *,
    compute_crps: bool = True,
) -> dict[str, float | int]:
    positive = positive_increase_rows(df, target_col)
    target = positive[target_col].to_numpy(dtype=float)
    distribution = predict_conditional_distribution(artifact, positive)
    if artifact["type"] in {"shifted_poisson_ngboost", "shifted_negative_binomial_ngboost"}:
        shifted = (target - artifact["target_shift"]).astype(int)
        density_nll = float(-np.mean(distribution.dist.logpmf(shifted)))
        mean = np.asarray(distribution.params["mu"], dtype=float) + artifact["target_shift"]
        lower_80 = distribution.dist.ppf(0.10) + artifact["target_shift"]
        upper_80 = distribution.dist.ppf(0.90) + artifact["target_shift"]
        lower_90 = distribution.dist.ppf(0.05) + artifact["target_shift"]
        upper_90 = distribution.dist.ppf(0.95) + artifact["target_shift"]
    else:
        density_nll = float(-np.mean(distribution.dist.logpdf(target)))
        mean = np.asarray(distribution.dist.mean(), dtype=float)
        lower_80, upper_80 = _central_integer_interval(distribution, 0.20)
        lower_90, upper_90 = _central_integer_interval(distribution, 0.10)
    realized_probability = np.clip(
        _realized_interval_probabilities(artifact, distribution, target), 1e-15, 1.0
    )
    result: dict[str, float | int] = {
        "n": int(len(target)),
        "mean_target": float(target.mean()),
        "mean_prediction": float(mean.mean()),
        "density_nll": density_nll,
        "interval_nll": float(-np.mean(np.log(realized_probability))),
        "mae": float(np.mean(np.abs(mean - target))),
        "rmse": float(np.sqrt(np.mean((mean - target) ** 2))),
        "coverage_80": float(np.mean((target >= lower_80) & (target <= upper_80))),
        "coverage_90": float(np.mean((target >= lower_90) & (target <= upper_90))),
    }
    quantile_shift = float(artifact.get("target_shift", 0.0))
    result["crps"] = (
        _quantile_crps(distribution, target, quantile_shift=quantile_shift)
        if compute_crps
        else float("nan")
    )
    result["nll"] = result["interval_nll"]
    return result


def evaluate_shifted_poisson(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    target_col: str = "remaining_increase",
) -> dict[str, float | int]:
    if artifact.get("type") != "shifted_poisson_ngboost":
        raise ValueError("evaluate_shifted_poisson requires a shifted-Poisson artifact")
    return evaluate_conditional_distribution(artifact, df, target_col)


def load_conditional_increase_model(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    if artifact.get("type") not in {
        "shifted_poisson_ngboost",
        "shifted_negative_binomial_ngboost",
        "positive_continuous_ngboost",
    }:
        raise ValueError(f"Invalid conditional-increase model: {path}")
    return artifact
