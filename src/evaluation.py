from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.distributional_model import (
    distribution_cdf,
    distribution_logpdf,
    distribution_ppf,
    distribution_std,
    normalize_distribution_name,
)


_DEFAULT_PROBABILITY_ATOL = 1e-6


def validate_distribution_params(
    df: pd.DataFrame,
    mu_col: str = "mu",
    sigma_col: str = "sigma",
    dist_type: str = "normal",
    df_col: str = "df",
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Distribution parameter DataFrame is empty")

    missing = [column for column in [mu_col, sigma_col] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing distribution parameter column(s): {missing}")

    mu = pd.to_numeric(df[mu_col], errors="coerce")
    sigma = pd.to_numeric(df[sigma_col], errors="coerce")
    if mu.isna().any():
        raise ValueError(f"{mu_col!r} contains missing or non-numeric values")
    if sigma.isna().any():
        raise ValueError(f"{sigma_col!r} contains missing or non-numeric values")
    if not np.isfinite(mu.to_numpy(dtype=float)).all():
        raise ValueError(f"{mu_col!r} contains non-finite values")
    sigma_values = sigma.to_numpy(dtype=float)
    if not np.isfinite(sigma_values).all():
        raise ValueError(f"{sigma_col!r} contains non-finite values")
    if (sigma_values <= 0.0).any():
        raise ValueError(f"{sigma_col!r} must be greater than 0 for every row")

    dist = normalize_distribution_name(dist_type)
    if dist == "student_t":
        if df_col not in df.columns:
            raise ValueError("Student-t distribution parameters require a df column")
        df_values = pd.to_numeric(df[df_col], errors="coerce")
        if df_values.isna().any():
            raise ValueError(f"{df_col!r} contains missing or non-numeric values")
        df_array = df_values.to_numpy(dtype=float)
        if not np.isfinite(df_array).all() or (df_array <= 0.0).any():
            raise ValueError(f"{df_col!r} must be finite and greater than 0")


def negative_log_likelihood(
    y_true: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    dist_type: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> float:
    y, mu_array, sigma_array = _validate_distribution_arrays(y_true, mu, sigma)

    logpdf = distribution_logpdf(
        y,
        mu=mu_array,
        sigma=sigma_array,
        distribution=dist_type,
        df=df,
    )
    if not np.isfinite(logpdf).all():
        raise ValueError("Distribution logpdf produced non-finite values")
    nll = float(-np.mean(logpdf))
    if not math.isfinite(nll):
        raise ValueError(f"NLL is not finite: {nll!r}")
    return nll


def prediction_interval_coverage(
    y_true: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    levels: Iterable[float] = (0.5, 0.8, 0.9),
    dist_type: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> pd.DataFrame:
    y, mu_array, sigma_array = _validate_distribution_arrays(y_true, mu, sigma)
    parsed_levels = _validate_interval_levels(levels)

    rows: list[dict[str, float | int]] = []
    for level in parsed_levels:
        lower_q = (1.0 - level) / 2.0
        upper_q = 1.0 - lower_q
        lower = distribution_ppf(
            np.full(len(y), lower_q, dtype=float),
            mu=mu_array,
            sigma=sigma_array,
            distribution=dist_type,
            df=df,
        )
        upper = distribution_ppf(
            np.full(len(y), upper_q, dtype=float),
            mu=mu_array,
            sigma=sigma_array,
            distribution=dist_type,
            df=df,
        )
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError(f"Prediction interval bounds are non-finite for level={level:g}")

        covered = (y >= lower) & (y <= upper)
        rows.append(
            {
                "level": float(level),
                "expected_coverage": float(level),
                "actual_coverage": float(np.mean(covered)),
                "coverage_error": float(np.mean(covered) - level),
                "avg_interval_width": float(np.mean(upper - lower)),
                "n": int(len(y)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "level",
            "expected_coverage",
            "actual_coverage",
            "coverage_error",
            "avg_interval_width",
            "n",
        ],
    )


def bucket_brier_scores(
    bucket_probs: pd.DataFrame,
    realized_bucket_labels: pd.Series,
) -> pd.DataFrame:
    probabilities = validate_bucket_probabilities(bucket_probs)
    labels = _validate_realized_bucket_labels(realized_bucket_labels, probabilities)

    rows: list[dict[str, float | int | str]] = []
    for bucket in probabilities.columns:
        predicted = probabilities[bucket].to_numpy(dtype=float)
        actual = (labels == bucket).astype(float).to_numpy(dtype=float)
        brier_score = float(np.mean((predicted - actual) ** 2))
        if not math.isfinite(brier_score) or brier_score < -1e-12 or brier_score > 1.0 + 1e-12:
            raise ValueError(
                f"Brier score for bucket {bucket!r} is outside [0, 1]: {brier_score!r}"
            )
        rows.append(
            {
                "bucket": str(bucket),
                "brier_score": brier_score,
                "mean_predicted_probability": float(np.mean(predicted)),
                "empirical_frequency": float(np.mean(actual)),
                "calibration_gap": float(np.mean(predicted) - np.mean(actual)),
                "count": int(len(probabilities)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "bucket",
            "brier_score",
            "mean_predicted_probability",
            "empirical_frequency",
            "calibration_gap",
            "count",
        ],
    )


def interval_log_loss(
    bucket_probs: pd.DataFrame,
    realized_bucket_labels: pd.Series,
    eps: float = 1e-12,
) -> float:
    if not math.isfinite(float(eps)) or not 0.0 < float(eps) < 1.0:
        raise ValueError(f"eps must be finite and between 0 and 1, got {eps!r}")

    probabilities = validate_bucket_probabilities(bucket_probs)
    labels = _validate_realized_bucket_labels(realized_bucket_labels, probabilities)
    column_index = pd.Index(probabilities.columns)
    positions = column_index.get_indexer(labels)
    if (positions < 0).any():
        missing = sorted({str(label) for label, position in zip(labels, positions) if position < 0})
        raise ValueError(f"Realized bucket labels not found in probability columns: {missing}")

    values = probabilities.to_numpy(dtype=float)
    realized_probabilities = values[np.arange(len(labels)), positions]
    loss = float(-np.mean(np.log(np.clip(realized_probabilities, float(eps), 1.0))))
    if not math.isfinite(loss):
        raise ValueError(f"Interval log loss is not finite: {loss!r}")
    return loss


def compute_pit_values(
    y_true: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    dist_type: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> pd.Series:
    y, mu_array, sigma_array = _validate_distribution_arrays(y_true, mu, sigma)
    pit = distribution_cdf(
        y,
        mu=mu_array,
        sigma=sigma_array,
        distribution=dist_type,
        df=df,
    )
    if not np.isfinite(pit).all():
        raise ValueError("PIT values contain non-finite values")
    if ((pit < 0.0) | (pit > 1.0)).any():
        raise ValueError("PIT values must be between 0 and 1")
    return pd.Series(pit, index=_maybe_index(y_true), name="pit")


def standardized_residuals(
    y_true: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    dist_type: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> pd.Series:
    y, mu_array, sigma_array = _validate_distribution_arrays(y_true, mu, sigma)
    denominator = distribution_std(sigma_array, distribution=dist_type, df=df)
    denominator = np.where(np.isfinite(denominator) & (denominator > 0.0), denominator, sigma_array)
    z = (y - mu_array) / denominator
    if not np.isfinite(z).all():
        raise ValueError("Standardized residuals contain non-finite values")
    return pd.Series(z, index=_maybe_index(y_true), name="standardized_residual")


def residual_summary(z: pd.Series) -> pd.DataFrame:
    if not isinstance(z, pd.Series):
        z = pd.Series(z)
    values = pd.to_numeric(z, errors="coerce")
    if values.empty:
        raise ValueError("Residual series is empty")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Residual series contains missing or non-finite values")

    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return pd.DataFrame(
        [
            {
                "mean": float(values.mean()),
                "std": std,
                "median": float(values.median()),
                "p05": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "n": int(len(values)),
            }
        ],
        columns=["mean", "std", "median", "p05", "p25", "p75", "p95", "n"],
    )


def coverage_by_group(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "forecast_error",
    mu_col: str = "mu",
    sigma_col: str = "sigma",
    dist_type: str = "normal",
    df_col: str = "df",
    level: float = 0.8,
    min_count: int = 30,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot compute grouped coverage on an empty DataFrame")
    required = [group_col, y_col, mu_col, sigma_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing grouped coverage column(s): {missing}")
    if int(min_count) < 1:
        raise ValueError("min_count must be at least 1")
    _validate_interval_levels([level])

    rows: list[dict[str, object]] = []
    for group_value, group_df in df.groupby(group_col, dropna=False, sort=True):
        coverage = prediction_interval_coverage(
            group_df[y_col],
            group_df[mu_col],
            group_df[sigma_col],
            levels=(level,),
            dist_type=dist_type,
            df=group_df[df_col] if normalize_distribution_name(dist_type) == "student_t" else None,
        ).iloc[0]
        rows.append(
            {
                group_col: group_value,
                "level": float(level),
                "count": int(len(group_df)),
                "actual_coverage": float(coverage["actual_coverage"]),
                "expected_coverage": float(coverage["expected_coverage"]),
                "coverage_error": float(coverage["coverage_error"]),
                "avg_interval_width": float(coverage["avg_interval_width"]),
                "enough_sample": bool(len(group_df) >= int(min_count)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            group_col,
            "level",
            "count",
            "actual_coverage",
            "expected_coverage",
            "coverage_error",
            "avg_interval_width",
            "enough_sample",
        ],
    )


def validate_bucket_probabilities(
    bucket_probs: pd.DataFrame,
    atol: float = _DEFAULT_PROBABILITY_ATOL,
    allow_renormalize: bool = False,
) -> pd.DataFrame:
    if not isinstance(bucket_probs, pd.DataFrame):
        raise TypeError("bucket_probs must be a pandas DataFrame")
    if bucket_probs.empty:
        raise ValueError("bucket_probs is empty")
    if not math.isfinite(float(atol)) or float(atol) < 0.0:
        raise ValueError(f"atol must be finite and nonnegative, got {atol!r}")
    if not bucket_probs.columns.is_unique:
        raise ValueError("bucket_probs columns must be unique bucket labels")

    probabilities = bucket_probs.apply(pd.to_numeric, errors="coerce")
    if probabilities.isna().any().any():
        missing_counts = probabilities.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"bucket_probs contains missing or non-numeric values: {missing_counts}")

    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("bucket_probs contains non-finite values")

    tolerance = float(atol)
    adjustments: list[str] = []
    if (values < -tolerance).any():
        raise ValueError(
            "bucket_probs contains probabilities negative beyond tolerance: "
            f"min={float(np.min(values)):.12g}, tolerance={tolerance:.12g}"
        )
    if (values > 1.0 + tolerance).any():
        raise ValueError(
            "bucket_probs contains probabilities above 1 beyond tolerance: "
            f"max={float(np.max(values)):.12g}, tolerance={tolerance:.12g}"
        )

    if ((values < 0.0) & (values >= -tolerance)).any():
        probabilities = probabilities.clip(lower=0.0)
        values = probabilities.to_numpy(dtype=float)
        adjustments.append("clipped tiny negative probabilities to zero")
    if ((values > 1.0) & (values <= 1.0 + tolerance)).any():
        probabilities = probabilities.clip(upper=1.0)
        values = probabilities.to_numpy(dtype=float)
        adjustments.append("clipped tiny probabilities above one to one")

    row_sums = probabilities.sum(axis=1)
    if not np.isfinite(row_sums.to_numpy(dtype=float)).all() or (row_sums <= 0.0).any():
        raise ValueError("bucket_probs row sums must be finite and positive")
    row_sum_deviation = (row_sums - 1.0).abs()
    max_deviation = float(row_sum_deviation.max())
    if max_deviation > tolerance:
        raise ValueError(
            "bucket_probs row sums must be close to 1. "
            f"Max absolute deviation={max_deviation:.12g}, tolerance={tolerance:.12g}"
        )

    if allow_renormalize and max_deviation > 0.0:
        probabilities = probabilities.div(row_sums, axis=0)
        adjustments.append("renormalized row sums to one")

    probabilities = probabilities.astype(float)
    probabilities.attrs["probability_validation_adjustments"] = adjustments
    probabilities.attrs["max_abs_row_probability_sum_deviation"] = max_deviation
    return probabilities


def _validate_dist_type(dist_type: str) -> str:
    return normalize_distribution_name(dist_type)


def _validate_distribution_arrays(
    y_true: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _as_finite_1d_array(y_true, "y_true")
    mu_array = _as_finite_1d_array(mu, "mu")
    sigma_array = _as_finite_1d_array(sigma, "sigma")
    lengths = {"y_true": len(y), "mu": len(mu_array), "sigma": len(sigma_array)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"y_true, mu, and sigma must have the same length: {lengths}")
    if len(y) == 0:
        raise ValueError("y_true, mu, and sigma cannot be empty")
    if (sigma_array <= 0.0).any():
        raise ValueError("sigma must be greater than 0 for every observation")
    return y, mu_array, sigma_array


def _as_finite_1d_array(values: pd.Series | np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return array


def _validate_interval_levels(levels: Iterable[float]) -> list[float]:
    parsed = [float(level) for level in levels]
    if not parsed:
        raise ValueError("At least one interval level is required")
    for level in parsed:
        if not math.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError(f"Coverage levels must be finite and between 0 and 1: {level!r}")
    return parsed


def _validate_realized_bucket_labels(
    realized_bucket_labels: pd.Series,
    bucket_probs: pd.DataFrame,
) -> pd.Series:
    labels = pd.Series(realized_bucket_labels).reset_index(drop=True)
    if len(labels) != len(bucket_probs):
        raise ValueError(
            "realized_bucket_labels length must match bucket_probs rows: "
            f"labels={len(labels)}, probabilities={len(bucket_probs)}"
        )
    if labels.isna().any():
        raise ValueError("realized_bucket_labels contains missing values")

    missing = sorted({str(label) for label in labels.unique() if label not in bucket_probs.columns})
    if missing:
        raise ValueError(f"Realized bucket labels not in probability columns: {missing}")
    return labels


def _maybe_index(values: object) -> pd.Index | None:
    return values.index if isinstance(values, pd.Series) else None
