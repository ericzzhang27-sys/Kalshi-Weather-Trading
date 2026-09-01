from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.conditional_increase_model import conditional_cdf


def hurdle_cdf(
    p_increase: np.ndarray | pd.Series | float,
    conditional_artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
    remaining_increase: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """CDF for the integer-support hurdle variable delta >= 0."""
    probability = np.asarray(p_increase, dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("p_increase must be finite and inside [0, 1]")
    threshold = np.asarray(remaining_increase, dtype=float)
    conditional = conditional_cdf(conditional_artifact, features, threshold)
    result = np.where(threshold < 0, 0.0, 1.0 - probability + probability * conditional)
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


def hurdle_interval_probability(
    p_increase: np.ndarray | pd.Series | float,
    conditional_artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
    lower: np.ndarray | pd.Series | float,
    upper: np.ndarray | pd.Series | float,
) -> np.ndarray:
    """Return P(lower < delta <= upper) for the combined hurdle model."""
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if np.any(lower_array >= upper_array):
        raise ValueError("Every hurdle interval must have lower < upper")
    return np.clip(
        hurdle_cdf(p_increase, conditional_artifact, features, upper_array)
        - hurdle_cdf(p_increase, conditional_artifact, features, lower_array),
        0.0,
        1.0,
    )


def final_temperature_cdf(
    final_temperature: np.ndarray | pd.Series | float,
    current_max: np.ndarray | pd.Series | float,
    p_increase: np.ndarray | pd.Series | float,
    conditional_artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
) -> np.ndarray:
    threshold = np.asarray(final_temperature, dtype=float) - np.asarray(current_max, dtype=float)
    return hurdle_cdf(p_increase, conditional_artifact, features, threshold)


def price_temperature_buckets(
    buckets: Iterable[Any],
    current_max: float,
    p_increase: float,
    conditional_artifact: dict[str, Any],
    features: pd.DataFrame | dict[str, Any],
) -> pd.DataFrame:
    """Price lower-open, upper-closed final-temperature buckets."""
    rows: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        lower = getattr(bucket, "lower_temp", None)
        upper = getattr(bucket, "upper_temp", None)
        label = getattr(bucket, "label", str(index))
        lower_cdf = 0.0 if lower is None else float(
            final_temperature_cdf(lower, current_max, p_increase, conditional_artifact, features)[0]
        )
        upper_cdf = 1.0 if upper is None else float(
            final_temperature_cdf(upper, current_max, p_increase, conditional_artifact, features)[0]
        )
        rows.append(
            {
                "bucket_index": index,
                "bucket_name": label,
                "bucket_lower_temp": lower,
                "bucket_upper_temp": upper,
                "probability": float(np.clip(upper_cdf - lower_cdf, 0.0, 1.0)),
            }
        )
    result = pd.DataFrame(rows)
    total = float(result["probability"].sum())
    if not np.isfinite(total) or abs(total - 1.0) > 1e-8:
        raise ValueError(f"Hurdle bucket probabilities sum to {total}, expected 1")
    return result


def integer_delta_probabilities(
    p_increase: np.ndarray | pd.Series,
    conditional_artifact: dict[str, Any],
    features: pd.DataFrame,
    max_delta: int = 10,
) -> np.ndarray:
    """Matrix for delta=0, 1, ..., max_delta, and a final >max_delta tail."""
    if max_delta < 1:
        raise ValueError("max_delta must be at least 1")
    probability = np.asarray(p_increase, dtype=float).reshape(-1)
    columns = [1.0 - probability]
    for delta in range(1, max_delta + 1):
        upper = conditional_cdf(conditional_artifact, features, float(delta))
        lower = conditional_cdf(conditional_artifact, features, float(delta - 1))
        columns.append(probability * np.clip(upper - lower, 0.0, 1.0))
    columns.append(probability * (1.0 - conditional_cdf(conditional_artifact, features, max_delta)))
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all() or (matrix < -1e-12).any():
        raise ValueError("Invalid full-hurdle probability matrix")
    totals = matrix.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-8):
        raise ValueError("Full-hurdle probability rows do not sum to one")
    return np.clip(matrix, 0.0, 1.0)


def categorical_scores(
    probability_matrix: np.ndarray,
    realized_delta: np.ndarray | pd.Series,
    max_delta: int = 10,
) -> dict[str, float]:
    target = np.asarray(realized_delta, dtype=int)
    category = np.where(target > max_delta, max_delta + 1, target)
    if (category < 0).any():
        raise ValueError("realized_delta cannot be negative")
    probability = np.asarray(probability_matrix, dtype=float)
    if probability.shape != (len(target), max_delta + 2):
        raise ValueError("Probability matrix has the wrong shape")
    realized = np.clip(probability[np.arange(len(target)), category], 1e-15, 1.0)
    one_hot = np.eye(max_delta + 2)[category]
    return {
        "multiclass_nll": float(-np.mean(np.log(realized))),
        "mean_bucket_brier": float(np.mean((probability - one_hot) ** 2)),
    }
