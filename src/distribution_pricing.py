from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .bucket_schema import (
        Bucket,
        TemperatureBucket,
        make_integer_temperature_buckets,
        validate_buckets,
        validate_temperature_buckets,
    )
    from .interval_probs import (
        final_bucket_to_error_bounds,
        interval_probability,
        normalize_probabilities,
    )
    from .error_boundaries import convert_nws_to_boundaries
    from .distributional_model import distribution_cdf, normalize_distribution_name
except ImportError:
    from bucket_schema import (
        Bucket,
        TemperatureBucket,
        make_integer_temperature_buckets,
        validate_buckets,
        validate_temperature_buckets,
    )
    from interval_probs import (
        final_bucket_to_error_bounds,
        interval_probability,
        normalize_probabilities,
    )
    from error_boundaries import convert_nws_to_boundaries
    from distributional_model import distribution_cdf, normalize_distribution_name


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_INPUT_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_params_v0.csv"
DEFAULT_BUCKET_PROBS_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_probs_v0.csv"
DEFAULT_VALIDATION_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_probs_validation_v0.csv"
DEFAULT_TEST_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_probs_test_v0.csv"
DEFAULT_VALIDATION_REPORT_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_prob_validation.md"

_PROBABILITY_NOISE_TOL = 1e-12

PREDICTION_METADATA_COLUMNS = [
    "row_id",
    "date",
    "timestamp",
    "prediction_time",
    "prediction_timestamp",
    "station",
    "station_id",
    "location",
    "split",
    "forecast_high",
    "actual_high",
    "official_high",
    "forecast_error",
    "model_version",
    "forecast_horizon_hours",
    "nll",
]

BUCKET_OUTPUT_COLUMNS = [
    "row_id",
    "date",
    "prediction_time",
    "station",
    "location",
    "split",
    "forecast_high",
    "actual_high",
    "forecast_error",
    "bucket_index",
    "bucket_name",
    "bucket_lower_temp",
    "bucket_upper_temp",
    "error_lower",
    "error_upper",
    "mu",
    "sigma",
    "distribution_type",
    "df",
    "probability",
]


@dataclass(frozen=True)
class ErrorBucket:
    label: str
    lower_error: float | None
    upper_error: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        _validate_optional_bound(self.lower_error, name="lower_error")
        _validate_optional_bound(self.upper_error, name="upper_error")
        if self.lower_error is None and self.upper_error is None:
            raise ValueError("lower_error and upper_error cannot both be None")
        if (
            self.lower_error is not None
            and self.upper_error is not None
            and self.upper_error <= self.lower_error
        ):
            raise ValueError("upper_error must be greater than lower_error")


def _validate_optional_bound(value: float | None, *, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _validate_forecast_high(forecast_high: float) -> float:
    value = float(forecast_high)
    if not math.isfinite(value):
        raise ValueError(f"forecast_high must be finite, got {forecast_high!r}")
    return value


def _validate_distribution_params(mu: float, sigma: float) -> tuple[float, float]:
    mu_value = float(mu)
    sigma_value = float(sigma)
    if not math.isfinite(mu_value):
        raise ValueError(f"mu must be finite, got {mu!r}")
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        raise ValueError(f"sigma must be finite and greater than 0, got {sigma!r}")
    return mu_value, sigma_value


def _validate_dist_type(dist_type: str) -> str:
    return normalize_distribution_name(dist_type)


def forecast_high_to_market_anchor(
    forecast_high: float,
    rounding: str = "nearest",
) -> int:
    """
    Convert a daily forecast high into the integer anchor used for market buckets.
    """
    forecast = _validate_forecast_high(forecast_high)
    method = str(rounding).strip().lower()
    if method in {"nearest", "round"}:
        return int(math.floor(forecast + 0.5))
    if method == "floor":
        return int(math.floor(forecast))
    if method == "ceil":
        return int(math.ceil(forecast))
    raise ValueError(f"Unsupported forecast rounding method: {rounding!r}")


def make_kalshi_buckets_around_forecast(
    forecast_high: float,
    location: str = "UNKNOWN",
    rounding: str = "nearest",
) -> list[Bucket]:
    """
    Build the existing Kalshi-style six-bucket market around one forecast high.

    Example for anchor 73:
    69 or lower, 70 to 71, 72 to 73, 74 to 75, 76 to 77, 78 or higher.
    """
    anchor = forecast_high_to_market_anchor(forecast_high, rounding=rounding)
    market_location = str(location).strip() or "UNKNOWN"
    return convert_nws_to_boundaries(anchor, market_location)


def _is_negative_infinity(value: float | None) -> bool:
    return value is not None and math.isinf(float(value)) and float(value) < 0.0


def _is_positive_infinity(value: float | None) -> bool:
    return value is not None and math.isinf(float(value)) and float(value) > 0.0


def _coerce_interval_boundary(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric):
        raise ValueError(f"{name} boundary cannot be NaN")
    return numeric


def _clean_probability(value: float, *, tolerance: float = _PROBABILITY_NOISE_TOL) -> float:
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"Probability is not finite: {value!r}")
    if probability < -tolerance:
        raise ValueError(f"Probability is negative beyond floating-point tolerance: {value!r}")
    if probability > 1.0 + tolerance:
        raise ValueError(f"Probability exceeds 1 beyond floating-point tolerance: {value!r}")
    if probability < 0.0:
        return 0.0
    if probability > 1.0:
        return 1.0
    return probability


def normal_cdf(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    mu_value, sigma_value = _validate_distribution_params(mu, sigma)

    scale = sigma_value * math.sqrt(2.0)

    if isinstance(x, np.ndarray):
        z = (x.astype(float) - mu_value) / scale
        erf_values = np.vectorize(math.erf, otypes=[float])(z)
        return 0.5 * (1.0 + erf_values)

    z = (float(x) - mu_value) / scale
    return float(0.5 * (1.0 + math.erf(z)))


def cdf_from_params(
    x: float | None,
    mu: float,
    sigma: float,
    dist_type: str = "normal",
    df: float | None = None,
) -> float:
    """
    Evaluate a forecast-error CDF from distribution parameters.

    Open interval endpoints represented by None are handled by
    interval_probability_from_cdf, because the CDF value depends on whether the
    missing endpoint is the lower or upper tail.
    """
    dist = _validate_dist_type(dist_type)
    mu_value, sigma_value = _validate_distribution_params(mu, sigma)
    if x is None:
        raise ValueError("x=None is only valid as an open interval endpoint")

    x_value = float(x)
    if math.isnan(x_value):
        raise ValueError("x cannot be NaN")
    if x_value == -math.inf:
        return 0.0
    if x_value == math.inf:
        return 1.0

    value = float(
        distribution_cdf(
            x_value,
            mu=mu_value,
            sigma=sigma_value,
            distribution=dist,
            df=df,
        )[0]
    )

    if not math.isfinite(value):
        raise ValueError(f"CDF returned non-finite value at {x_value!r}: {value!r}")
    if value < -_PROBABILITY_NOISE_TOL or value > 1.0 + _PROBABILITY_NOISE_TOL:
        raise ValueError(f"CDF value must be in [0, 1], got {value!r}")
    return min(1.0, max(0.0, value))


def interval_probability_from_cdf(
    lower: float | None,
    upper: float | None,
    mu: float,
    sigma: float,
    dist_type: str = "normal",
    df: float | None = None,
) -> float:
    """
    Compute P(lower < error <= upper) as F(upper) - F(lower).
    """
    _validate_dist_type(dist_type)
    _validate_distribution_params(mu, sigma)
    lower_value = _coerce_interval_boundary(lower, name="lower")
    upper_value = _coerce_interval_boundary(upper, name="upper")

    if lower_value is not None and upper_value is not None and upper_value <= lower_value:
        raise ValueError(
            f"Interval upper boundary must exceed lower boundary: {(lower, upper)!r}"
        )
    if _is_positive_infinity(lower_value):
        raise ValueError("lower boundary cannot be +inf")
    if _is_negative_infinity(upper_value):
        raise ValueError("upper boundary cannot be -inf")

    lower_cdf = (
        0.0
        if lower_value is None or _is_negative_infinity(lower_value)
        else cdf_from_params(lower_value, mu=mu, sigma=sigma, dist_type=dist_type, df=df)
    )
    upper_cdf = (
        1.0
        if upper_value is None or _is_positive_infinity(upper_value)
        else cdf_from_params(upper_value, mu=mu, sigma=sigma, dist_type=dist_type, df=df)
    )
    return _clean_probability(upper_cdf - lower_cdf)


def plot_and_save_cdf(mu: float = 1, sigma: float = 1, filename: str = "normal_cdf.png") -> None:
    """Plot normal CDF and save to figures folder."""
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    plt.figure(figsize=(10, 6))
    plt.plot(x, normal_cdf(x, mu, sigma))
    plt.title(f"Normal CDF (mu={mu}, sigma={sigma})")
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)

    output_dir = Path(__file__).parent.parent / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def normal_bucket_prob(
    lower_bound: float | None,
    upper_bound: float | None,
    mu: float,
    sigma: float,
) -> float:
    if lower_bound is None and upper_bound is None:
        raise ValueError("Both lower and upper bounds cannot be None")

    return interval_probability(
        lambda boundary: normal_cdf(boundary, mu=mu, sigma=sigma),
        lower_bound,
        upper_bound,
    )


def normal_bucket_probs(buckets: list[Bucket], mu: float, sigma: float) -> dict[str, float]:
    probability_list: dict[str, float] = {}
    for bucket in buckets:
        probability_list[bucket.name] = normal_bucket_prob(
            bucket.lower_bound,
            bucket.upper_bound,
            mu,
            sigma,
        )
    return probability_list


def temperature_bucket_to_error_bucket(
    bucket: TemperatureBucket,
    forecast_high: float,
) -> ErrorBucket:
    """
    Convert a final-temperature bucket into a forecast-error bucket.

    forecast_error = actual official high - NWS forecast high.
    """
    forecast = _validate_forecast_high(forecast_high)
    lower_error, upper_error = final_bucket_to_error_bounds(bucket, forecast)

    return ErrorBucket(
        label=bucket.label,
        lower_error=lower_error,
        upper_error=upper_error,
    )


def convert_temperature_buckets_to_error_buckets(
    buckets: list[TemperatureBucket],
    forecast_high: float,
) -> list[ErrorBucket]:
    validate_temperature_buckets(buckets)
    return [
        temperature_bucket_to_error_bucket(bucket, forecast_high)
        for bucket in buckets
    ]


def price_temperature_buckets_from_cdf(
    buckets: list[TemperatureBucket],
    forecast_high: float,
    cdf: Callable[[float], float],
    normalize: bool = True,
) -> list[dict[str, float | str | None]]:
    """
    Price final-temperature buckets from any forecast-error CDF callable.
    """
    error_buckets = convert_temperature_buckets_to_error_buckets(
        buckets=buckets,
        forecast_high=forecast_high,
    )

    probabilities = [
        interval_probability(cdf, bucket.lower_error, bucket.upper_error)
        for bucket in error_buckets
    ]
    if normalize:
        probabilities = normalize_probabilities(probabilities)

    forecast = _validate_forecast_high(forecast_high)
    rows: list[dict[str, float | str | None]] = []
    for temperature_bucket, error_bucket, probability in zip(
        buckets,
        error_buckets,
        probabilities,
    ):
        rows.append(
            {
                "bucket": temperature_bucket.label,
                "lower_temp": temperature_bucket.lower_temp,
                "upper_temp": temperature_bucket.upper_temp,
                "forecast_high": forecast,
                "lower_error": error_bucket.lower_error,
                "upper_error": error_bucket.upper_error,
                "probability": float(probability),
            }
        )

    return rows


def _bucket_label(bucket: Bucket | TemperatureBucket | dict[str, Any], index: int) -> str:
    if isinstance(bucket, dict):
        for key in ("bucket_name", "bucket", "label", "name"):
            if key in bucket and str(bucket[key]).strip():
                return str(bucket[key])
    else:
        for attr in ("label", "name"):
            if hasattr(bucket, attr):
                value = getattr(bucket, attr)
                if str(value).strip():
                    return str(value)
    return f"bucket_{index}"


def _bucket_temperature_bounds(
    bucket: Bucket | TemperatureBucket | dict[str, Any],
) -> tuple[float | None, float | None]:
    lower, upper = final_bucket_to_error_bounds(bucket, forecast_high=0.0)
    return lower, upper


def _validate_bucket_schema(
    buckets: list[Bucket | TemperatureBucket | dict[str, Any]],
) -> None:
    if len(buckets) == 0:
        raise ValueError("Bucket schema cannot be empty")
    if all(isinstance(bucket, TemperatureBucket) for bucket in buckets):
        validate_temperature_buckets(buckets)  # type: ignore[arg-type]
        return
    if all(isinstance(bucket, Bucket) for bucket in buckets):
        validate_buckets(buckets)  # type: ignore[arg-type]
        return

    previous_upper: float | None = None
    for index, bucket in enumerate(buckets):
        lower, upper = _bucket_temperature_bounds(bucket)
        if lower is None and upper is None:
            raise ValueError(f"Bucket {index} has both bounds open")
        if lower is not None and upper is not None and upper <= lower:
            raise ValueError(f"Bucket {index} upper_temp must exceed lower_temp")
        if index == 0:
            if lower is not None:
                raise ValueError("First bucket should be open-ended below")
        else:
            if lower is None:
                raise ValueError("Only the first bucket can be open-ended below")
            if previous_upper is None:
                raise ValueError("Previous bucket was open-ended above before final bucket")
            if float(lower) != float(previous_upper):
                raise ValueError(
                    "Bucket schema has a gap or overlap: "
                    f"previous upper={previous_upper}, current lower={lower}"
                )
        if index == len(buckets) - 1:
            if upper is not None:
                raise ValueError("Last bucket should be open-ended above")
        else:
            if upper is None:
                raise ValueError("Only the last bucket can be open-ended above")
        previous_upper = upper


def _row_to_mapping(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)


def _optional_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _prediction_metadata(row_values: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for column in PREDICTION_METADATA_COLUMNS:
        if column in row_values:
            metadata[column] = _optional_scalar(row_values[column])
    return metadata


def price_buckets_for_row(
    row: pd.Series | dict[str, Any],
    buckets: list[Bucket | TemperatureBucket | dict[str, Any]] | None = None,
    dist_type: str = "normal",
    forecast_rounding: str = "nearest",
) -> list[dict[str, Any]]:
    """
    Price every final-temperature bucket for one forecast-error distribution row.
    """
    row_values = _row_to_mapping(row)
    if "row_id" not in row_values and isinstance(row, pd.Series) and row.name is not None:
        row_values["row_id"] = row.name

    missing = [column for column in ("forecast_high", "mu", "sigma") if column not in row_values]
    if missing:
        raise ValueError(f"Prediction row is missing required columns: {missing}")

    forecast_high = _validate_forecast_high(row_values["forecast_high"])
    mu, sigma = _validate_distribution_params(row_values["mu"], row_values["sigma"])
    dist = _validate_dist_type(dist_type)
    df_value = _row_degrees_of_freedom(row_values, dist)
    if buckets is None:
        buckets = make_kalshi_buckets_around_forecast(
            forecast_high=forecast_high,
            location=str(row_values.get("location", "UNKNOWN")),
            rounding=forecast_rounding,
        )
    _validate_bucket_schema(buckets)
    metadata = _prediction_metadata(row_values)
    metadata["forecast_high"] = forecast_high

    records: list[dict[str, Any]] = []
    for bucket_index, bucket in enumerate(buckets):
        lower_temp, upper_temp = _bucket_temperature_bounds(bucket)
        error_lower, error_upper = final_bucket_to_error_bounds(bucket, forecast_high)
        probability = interval_probability_from_cdf(
            error_lower,
            error_upper,
            mu=mu,
            sigma=sigma,
            dist_type=dist,
            df=df_value,
        )
        record = {
            **metadata,
            "bucket_index": bucket_index,
            "bucket_name": _bucket_label(bucket, bucket_index),
            "bucket_lower_temp": lower_temp,
            "bucket_upper_temp": upper_temp,
            "error_lower": error_lower,
            "error_upper": error_upper,
            "mu": mu,
            "sigma": sigma,
            "distribution_type": dist,
            "df": df_value,
            "probability": probability,
        }
        records.append(record)

    return records


def _validate_prediction_params_frame(
    pred_df: pd.DataFrame,
    dist_type: str = "normal",
) -> pd.DataFrame:
    required_columns = ["forecast_high", "mu", "sigma"]
    missing = [column for column in required_columns if column not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction parameter DataFrame is missing required columns: {missing}")

    working = pred_df.copy()
    if "row_id" not in working.columns:
        working.insert(0, "row_id", np.arange(len(working), dtype=int))

    for column in required_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    if not np.isfinite(working["forecast_high"].to_numpy(dtype=float)).all():
        raise ValueError("forecast_high must be finite for every prediction row")
    if not np.isfinite(working["mu"].to_numpy(dtype=float)).all():
        raise ValueError("mu must be finite for every prediction row")
    sigma = working["sigma"].to_numpy(dtype=float)
    if not np.isfinite(sigma).all() or (sigma <= 0.0).any():
        raise ValueError("sigma must be finite and greater than 0 for every prediction row")
    if normalize_distribution_name(dist_type) == "student_t":
        if "df" not in working.columns:
            raise ValueError("Student-t bucket pricing requires a df column")
        working["df"] = pd.to_numeric(working["df"], errors="coerce")
        df_values = working["df"].to_numpy(dtype=float)
        if not np.isfinite(df_values).all() or (df_values <= 0.0).any():
            raise ValueError("df must be finite and greater than 0 for Student-t pricing")

    for column in ("actual_high", "official_high", "forecast_error", "forecast_horizon_hours", "nll"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    return working


def _row_degrees_of_freedom(row_values: dict[str, Any], dist_type: str) -> float | None:
    if dist_type != "student_t":
        return None
    if "df" not in row_values:
        raise ValueError("Student-t row pricing requires df")
    value = float(row_values["df"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"df must be finite and greater than 0, got {row_values['df']!r}")
    return value


def _probabilities_for_bucket_arrays(
    lower_error: np.ndarray | None,
    upper_error: np.ndarray | None,
    mu: np.ndarray,
    sigma: np.ndarray,
    dist_type: str,
    df: np.ndarray | None = None,
) -> np.ndarray:
    dist = _validate_dist_type(dist_type)
    lower_cdf = (
        np.zeros_like(mu, dtype=float)
        if lower_error is None
        else distribution_cdf(lower_error, mu=mu, sigma=sigma, distribution=dist, df=df)
    )
    upper_cdf = (
        np.ones_like(mu, dtype=float)
        if upper_error is None
        else distribution_cdf(upper_error, mu=mu, sigma=sigma, distribution=dist, df=df)
    )
    probabilities = np.asarray(upper_cdf - lower_cdf, dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Computed bucket probabilities contain non-finite values")
    serious_negative = probabilities < -_PROBABILITY_NOISE_TOL
    serious_above_one = probabilities > 1.0 + _PROBABILITY_NOISE_TOL
    if serious_negative.any() or serious_above_one.any():
        min_probability = float(np.min(probabilities))
        max_probability = float(np.max(probabilities))
        raise ValueError(
            "Computed bucket probabilities fall outside [0, 1]: "
            f"min={min_probability:.12g}, max={max_probability:.12g}"
        )
    return np.clip(probabilities, 0.0, 1.0)


def price_buckets_for_dataframe(
    pred_df: pd.DataFrame,
    buckets: list[Bucket | TemperatureBucket | dict[str, Any]] | None = None,
    dist_type: str = "normal",
    forecast_rounding: str = "nearest",
) -> pd.DataFrame:
    """
    Price every bucket for every NGBoost distribution-parameter row.

    Returns a long-format DataFrame with one row per prediction row per bucket.
    """
    dist = _validate_dist_type(dist_type)
    working = _validate_prediction_params_frame(pred_df, dist_type=dist)
    if buckets is None:
        records: list[dict[str, Any]] = []
        for _, row in working.iterrows():
            records.extend(
                price_buckets_for_row(
                    row,
                    buckets=None,
                    dist_type=dist,
                    forecast_rounding=forecast_rounding,
                )
            )
        result = pd.DataFrame.from_records(records)
        ordered_columns = [column for column in BUCKET_OUTPUT_COLUMNS if column in result.columns]
        remaining_columns = [
            column for column in result.columns if column not in ordered_columns
        ]
        return result[ordered_columns + remaining_columns]

    _validate_bucket_schema(buckets)
    working["_source_row_order"] = np.arange(len(working), dtype=int)

    forecast_high = working["forecast_high"].to_numpy(dtype=float)
    mu = working["mu"].to_numpy(dtype=float)
    sigma = working["sigma"].to_numpy(dtype=float)
    df_values = working["df"].to_numpy(dtype=float) if dist == "student_t" else None
    metadata_columns = [
        column for column in PREDICTION_METADATA_COLUMNS if column in working.columns
    ]
    base_columns = ["_source_row_order", *metadata_columns]

    frames: list[pd.DataFrame] = []
    for bucket_index, bucket in enumerate(buckets):
        lower_temp, upper_temp = _bucket_temperature_bounds(bucket)
        lower_error = None if lower_temp is None else lower_temp - forecast_high
        upper_error = None if upper_temp is None else upper_temp - forecast_high
        probabilities = _probabilities_for_bucket_arrays(
            lower_error=lower_error,
            upper_error=upper_error,
            mu=mu,
            sigma=sigma,
            dist_type=dist,
            df=df_values,
        )

        frame = working[base_columns].copy()
        frame["bucket_index"] = bucket_index
        frame["bucket_name"] = _bucket_label(bucket, bucket_index)
        frame["bucket_lower_temp"] = np.nan if lower_temp is None else float(lower_temp)
        frame["bucket_upper_temp"] = np.nan if upper_temp is None else float(upper_temp)
        frame["error_lower"] = np.nan if lower_error is None else lower_error
        frame["error_upper"] = np.nan if upper_error is None else upper_error
        frame["mu"] = mu
        frame["sigma"] = sigma
        frame["distribution_type"] = dist
        frame["df"] = df_values if df_values is not None else np.nan
        frame["probability"] = probabilities
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["_source_row_order", "bucket_index"], kind="stable", inplace=True)
    result.drop(columns=["_source_row_order"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    ordered_columns = [column for column in BUCKET_OUTPUT_COLUMNS if column in result.columns]
    remaining_columns = [
        column for column in result.columns if column not in ordered_columns
    ]
    return result[ordered_columns + remaining_columns]


def validate_bucket_probabilities(
    bucket_probs_df: pd.DataFrame,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """
    Validate long-format bucket probabilities and return a compact summary.
    """
    if len(bucket_probs_df) == 0:
        raise ValueError("Bucket probability DataFrame is empty")
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError(f"tolerance must be finite and nonnegative, got {tolerance!r}")

    required_columns = ["row_id", "bucket_name", "probability", "mu", "sigma"]
    missing = [column for column in required_columns if column not in bucket_probs_df.columns]
    if missing:
        raise ValueError(f"Bucket probability DataFrame is missing columns: {missing}")

    probabilities = pd.to_numeric(bucket_probs_df["probability"], errors="coerce")
    mu = pd.to_numeric(bucket_probs_df["mu"], errors="coerce")
    sigma = pd.to_numeric(bucket_probs_df["sigma"], errors="coerce")
    if not np.isfinite(probabilities.to_numpy(dtype=float)).all():
        raise ValueError("Every probability must be finite")
    if not np.isfinite(mu.to_numpy(dtype=float)).all():
        raise ValueError("Every mu must be finite")
    sigma_values = sigma.to_numpy(dtype=float)
    if not np.isfinite(sigma_values).all() or (sigma_values <= 0.0).any():
        raise ValueError("Every sigma must be finite and greater than 0")
    if (probabilities < -tolerance).any():
        min_probability = float(probabilities.min())
        raise ValueError(f"Probability below -tolerance: min={min_probability:.12g}")
    if (probabilities > 1.0 + tolerance).any():
        max_probability = float(probabilities.max())
        raise ValueError(f"Probability above 1+tolerance: max={max_probability:.12g}")

    clipped_probabilities = probabilities.clip(lower=0.0, upper=1.0)
    if ((clipped_probabilities < 0.0) | (clipped_probabilities > 1.0)).any():
        raise ValueError("Probabilities remain outside [0, 1] after tiny clipping")

    if {"error_lower", "error_upper"}.issubset(bucket_probs_df.columns):
        lower = pd.to_numeric(bucket_probs_df["error_lower"], errors="coerce")
        upper = pd.to_numeric(bucket_probs_df["error_upper"], errors="coerce")
        finite_lower = lower.notna() & np.isfinite(lower.to_numpy(dtype=float))
        finite_upper = upper.notna() & np.isfinite(upper.to_numpy(dtype=float))
        bad_bounds = finite_lower & finite_upper & (upper <= lower)
        if bad_bounds.any():
            bad_count = int(bad_bounds.sum())
            raise ValueError(f"Found {bad_count} rows where error_lower >= error_upper")

    if "bucket_index" in bucket_probs_df.columns:
        expected_bucket_count = int(bucket_probs_df["bucket_index"].nunique(dropna=False))
    else:
        expected_bucket_count = int(bucket_probs_df["bucket_name"].nunique(dropna=False))
    counts_by_row = bucket_probs_df.groupby("row_id", dropna=False).size()
    incomplete_rows = counts_by_row[counts_by_row != expected_bucket_count]
    if len(incomplete_rows) > 0:
        raise ValueError(
            "Every row_id must have the full bucket set. "
            f"Expected {expected_bucket_count}, found mismatches for {len(incomplete_rows)} rows."
        )

    row_sums = clipped_probabilities.groupby(bucket_probs_df["row_id"], dropna=False).sum()
    row_sum_deviation = (row_sums - 1.0).abs()
    max_abs_deviation = float(row_sum_deviation.max())
    if max_abs_deviation > tolerance:
        raise ValueError(
            "Bucket probabilities must sum to 1 by row_id. "
            f"Max absolute deviation={max_abs_deviation:.12g}, tolerance={tolerance:.12g}"
        )

    return {
        "validation_passed": True,
        "prediction_row_count": int(row_sums.shape[0]),
        "probability_row_count": int(len(bucket_probs_df)),
        "bucket_count_per_prediction": expected_bucket_count,
        "min_probability": float(clipped_probabilities.min()),
        "max_probability": float(clipped_probabilities.max()),
        "mean_row_probability_sum": float(row_sums.mean()),
        "max_abs_row_probability_sum_deviation": max_abs_deviation,
        "invalid_rows_found": 0,
    }


def load_prediction_params(
    path: str | Path,
    splits: list[str] | None = None,
    dist_type: str = "auto",
) -> pd.DataFrame:
    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(f"NGBoost distribution parameter file not found: {params_path}")
    df = pd.read_csv(params_path)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    if splits is not None and len(splits) > 0:
        if "split" not in df.columns:
            raise ValueError("Cannot filter by split because the prediction file has no split column")
        df = df[df["split"].isin(splits)].copy()
    inferred_dist = infer_prediction_distribution_type(df, dist_type)
    return _validate_prediction_params_frame(df, dist_type=inferred_dist)


def infer_prediction_distribution_type(
    pred_df: pd.DataFrame,
    requested_dist_type: str = "auto",
) -> str:
    requested = str(requested_dist_type).strip().lower()
    if requested not in {"", "auto", "infer"}:
        return normalize_distribution_name(str(requested_dist_type))
    if "distribution_type" not in pred_df.columns:
        return "normal"
    values = pred_df["distribution_type"].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "normal"
    normalized = {normalize_distribution_name(value) for value in values.unique()}
    if len(normalized) != 1:
        raise ValueError(f"Prediction file contains multiple distribution types: {sorted(normalized)}")
    return next(iter(normalized))


def _csv_bound_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def load_bucket_schema(path: str | Path) -> list[TemperatureBucket]:
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Bucket schema file not found: {schema_path}")
    df = pd.read_csv(schema_path)

    label_column = next(
        (
            column
            for column in ("bucket_name", "bucket", "label", "name")
            if column in df.columns
        ),
        None,
    )
    lower_column = next(
        (
            column
            for column in ("bucket_lower_temp", "lower_temp", "lower_bound", "lower")
            if column in df.columns
        ),
        None,
    )
    upper_column = next(
        (
            column
            for column in ("bucket_upper_temp", "upper_temp", "upper_bound", "upper")
            if column in df.columns
        ),
        None,
    )
    if label_column is None or lower_column is None or upper_column is None:
        raise ValueError(
            "Bucket schema CSV must include label and bound columns. "
            "Accepted labels: bucket_name, bucket, label, name; "
            "accepted bounds: lower_temp/lower_bound and upper_temp/upper_bound."
        )

    buckets = [
        TemperatureBucket(
            label=str(row[label_column]),
            lower_temp=_csv_bound_value(row[lower_column]),
            upper_temp=_csv_bound_value(row[upper_column]),
        )
        for _, row in df.iterrows()
    ]
    validate_temperature_buckets(buckets)
    return buckets


def build_default_bucket_schema(
    pred_df: pd.DataFrame,
    min_temp: int | None = None,
    max_temp: int | None = None,
) -> list[TemperatureBucket]:
    """
    Build a broad one-degree final-temperature schema for the prediction file.
    """
    if min_temp is None or max_temp is None:
        candidates: list[pd.Series] = []
        if "actual_high" in pred_df.columns:
            candidates.append(pd.to_numeric(pred_df["actual_high"], errors="coerce"))
        candidates.append(
            pd.to_numeric(pred_df["forecast_high"], errors="coerce")
            + pd.to_numeric(pred_df["mu"], errors="coerce")
        )
        values = pd.concat(candidates, ignore_index=True).dropna()
        if len(values) == 0:
            raise ValueError("Cannot infer default bucket range from prediction file")
        inferred_min = int(math.floor(float(values.min()))) - 2
        inferred_max = int(math.ceil(float(values.max()))) + 2
        min_temp = inferred_min if min_temp is None else min_temp
        max_temp = inferred_max if max_temp is None else max_temp

    return make_integer_temperature_buckets(int(min_temp), int(max_temp))


def write_bucket_probability_outputs(
    bucket_probs_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_BUCKET_PROBS_OUTPUT_PATH,
    validation_output_path: str | Path = DEFAULT_VALIDATION_OUTPUT_PATH,
    test_output_path: str | Path = DEFAULT_TEST_OUTPUT_PATH,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bucket_probs_df.to_csv(output, index=False)
    if "split" in bucket_probs_df.columns:
        for split_name, split_path in [
            ("validation", Path(validation_output_path)),
            ("test", Path(test_output_path)),
        ]:
            split_df = bucket_probs_df[bucket_probs_df["split"] == split_name]
            if len(split_df) > 0:
                split_path.parent.mkdir(parents=True, exist_ok=True)
                split_df.to_csv(split_path, index=False)


def _format_bound(value: Any) -> str:
    if value is None or pd.isna(value):
        return "open"
    return f"{float(value):.6g}"


def _format_bucket_interval(lower: Any, upper: Any) -> str:
    if lower is None or pd.isna(lower):
        return f"final_high <= {_format_bound(upper)}"
    if upper is None or pd.isna(upper):
        return f"{_format_bound(lower)} < final_high"
    return f"{_format_bound(lower)} < final_high <= {_format_bound(upper)}"


def _format_error_interval(lower: Any, upper: Any) -> str:
    if lower is None or pd.isna(lower):
        return f"error <= {_format_bound(upper)}"
    if upper is None or pd.isna(upper):
        return f"{_format_bound(lower)} < error"
    return f"{_format_bound(lower)} < error <= {_format_bound(upper)}"


def _manual_cdf_expression(row: pd.Series) -> str:
    mu = float(row["mu"])
    sigma = float(row["sigma"])
    dist = str(row.get("distribution_type", "normal"))
    dist_label = normalize_distribution_name(dist)
    cdf_label = "StudentTCDF" if dist_label == "student_t" else f"{dist_label.title()}CDF"
    lower = row.get("error_lower")
    upper = row.get("error_upper")
    upper_text = "1" if pd.isna(upper) else f"{cdf_label}(({float(upper):.6g} - {mu:.6g}) / {sigma:.6g})"
    lower_text = "0" if pd.isna(lower) else f"{cdf_label}(({float(lower):.6g} - {mu:.6g}) / {sigma:.6g})"
    return f"{upper_text} - {lower_text}"


def build_manual_cdf_examples(
    bucket_probs_df: pd.DataFrame,
    count: int = 5,
) -> list[str]:
    if len(bucket_probs_df) == 0:
        return []
    finite_rows = bucket_probs_df[
        bucket_probs_df["bucket_lower_temp"].notna()
        & bucket_probs_df["bucket_upper_temp"].notna()
    ]
    meaningful_rows = finite_rows[finite_rows["probability"] > 1e-4]
    if len(meaningful_rows) > 0:
        sample = (
            meaningful_rows.sort_values(
                ["row_id", "probability"],
                ascending=[True, False],
                kind="stable",
            )
            .groupby("row_id", sort=False)
            .head(1)
            .head(count)
        )
    else:
        sample = finite_rows.head(count)
    if len(sample) < count:
        sample = bucket_probs_df.head(count)

    examples: list[str] = []
    for _, row in sample.iterrows():
        forecast_high = float(row["forecast_high"])
        mu = float(row["mu"])
        sigma = float(row["sigma"])
        dist = normalize_distribution_name(str(row.get("distribution_type", "normal")))
        df_value = row.get("df")
        df_text = ""
        if dist == "student_t" and df_value is not None and not pd.isna(df_value):
            df_text = f", df={float(df_value):.6g}"
        probability = float(row["probability"])
        lower_temp = row.get("bucket_lower_temp")
        upper_temp = row.get("bucket_upper_temp")
        error_lower = row.get("error_lower")
        error_upper = row.get("error_upper")
        examples.append(
            "\n".join(
                [
                    f"forecast_high = {forecast_high:.6g}",
                    f"error | X_t ~ {dist}(mu={mu:.6g}, scale={sigma:.6g}{df_text})",
                    "",
                    "Final bucket:",
                    _format_bucket_interval(lower_temp, upper_temp),
                    "",
                    "Convert to forecast-error interval:",
                    _format_error_interval(error_lower, error_upper),
                    "",
                    "Probability:",
                    f"P({_format_error_interval(error_lower, error_upper)})",
                    f"= {_manual_cdf_expression(row)}",
                    f"= {probability:.12g}",
                ]
            )
        )
    return examples


def write_validation_report(
    bucket_probs_df: pd.DataFrame,
    validation_summary: dict[str, Any],
    source_prediction_file: str | Path,
    report_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
    dist_type: str = "normal",
    bucket_mode: str = "kalshi_around_forecast",
) -> None:
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    splits = (
        sorted(str(value) for value in bucket_probs_df["split"].dropna().unique())
        if "split" in bucket_probs_df.columns
        else []
    )
    examples = build_manual_cdf_examples(bucket_probs_df, count=5)
    lines = [
        "# NGBoost Bucket Probability Validation",
        "",
        f"- Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Source prediction file or model source: `{source_prediction_file}`",
        f"- Prediction rows priced: {validation_summary['prediction_row_count']}",
        f"- Probability rows generated: {validation_summary['probability_row_count']}",
        f"- Buckets per prediction row: {validation_summary['bucket_count_per_prediction']}",
        f"- Included splits: {', '.join(splits) if splits else 'not available'}",
        f"- Distribution type used: {dist_type}",
        f"- Bucket mode: {bucket_mode}",
        f"- Min probability: {validation_summary['min_probability']:.12g}",
        f"- Max probability: {validation_summary['max_probability']:.12g}",
        f"- Mean row probability sum: {validation_summary['mean_row_probability_sum']:.12g}",
        (
            "- Max absolute deviation from row sum 1: "
            f"{validation_summary['max_abs_row_probability_sum_deviation']:.12g}"
        ),
        f"- Number of invalid rows found: {validation_summary['invalid_rows_found']}",
        f"- Validation passed: {validation_summary['validation_passed']}",
        "",
        "## Manual CDF Examples",
        "",
    ]
    for index, example in enumerate(examples, start=1):
        lines.extend([f"### Example {index}", "", "```text", example, "```", ""])
    report.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NGBoost Normal forecast-error distributions into temperature bucket probabilities."
    )
    parser.add_argument(
        "--params-path",
        default=str(DEFAULT_PARAMS_INPUT_PATH),
        help="CSV of NGBoost distribution parameters.",
    )
    parser.add_argument(
        "--bucket-schema-path",
        default=None,
        help="Optional fixed CSV bucket schema. If omitted, Kalshi-style buckets are built around each forecast_high.",
    )
    parser.add_argument(
        "--bucket-mode",
        choices=["kalshi", "exhaustive"],
        default="kalshi",
        help="Bucket construction when --bucket-schema-path is omitted.",
    )
    parser.add_argument("--bucket-min-temp", type=int, default=None)
    parser.add_argument("--bucket-max-temp", type=int, default=None)
    parser.add_argument(
        "--forecast-rounding",
        choices=["nearest", "floor", "ceil"],
        default="nearest",
        help="How decimal forecast_high values are converted to integer market anchors.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["validation", "test"],
        help="Splits to include from the parameter CSV.",
    )
    parser.add_argument(
        "--dist-type",
        default="auto",
        help="Distribution to use for bucket pricing. Defaults to auto-infer from params.",
    )
    parser.add_argument("--output-path", default=str(DEFAULT_BUCKET_PROBS_OUTPUT_PATH))
    parser.add_argument(
        "--validation-output-path",
        default=str(DEFAULT_VALIDATION_OUTPUT_PATH),
    )
    parser.add_argument("--test-output-path", default=str(DEFAULT_TEST_OUTPUT_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_VALIDATION_REPORT_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pred_df = load_prediction_params(args.params_path, splits=args.splits, dist_type=args.dist_type)
    dist_type = infer_prediction_distribution_type(pred_df, args.dist_type)
    if args.bucket_schema_path:
        buckets = load_bucket_schema(args.bucket_schema_path)
        bucket_mode = f"fixed_schema:{args.bucket_schema_path}"
    elif args.bucket_mode == "exhaustive":
        buckets = build_default_bucket_schema(
            pred_df,
            min_temp=args.bucket_min_temp,
            max_temp=args.bucket_max_temp,
        )
        bucket_mode = "exhaustive_one_degree"
    else:
        buckets = None
        bucket_mode = f"kalshi_around_forecast_rounding_{args.forecast_rounding}"

    bucket_probs = price_buckets_for_dataframe(
        pred_df,
        buckets,
        dist_type=dist_type,
        forecast_rounding=args.forecast_rounding,
    )
    validation_summary = validate_bucket_probabilities(bucket_probs)
    write_bucket_probability_outputs(
        bucket_probs,
        output_path=args.output_path,
        validation_output_path=args.validation_output_path,
        test_output_path=args.test_output_path,
    )
    write_validation_report(
        bucket_probs,
        validation_summary,
        source_prediction_file=args.params_path,
        report_path=args.report_path,
        dist_type=dist_type,
        bucket_mode=bucket_mode,
    )
    print(
        "Saved NGBoost bucket probabilities: "
        f"{len(bucket_probs):,} rows across "
        f"{validation_summary['prediction_row_count']:,} prediction rows and "
        f"{validation_summary['bucket_count_per_prediction']:,} buckets."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
