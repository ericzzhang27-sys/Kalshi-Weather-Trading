from __future__ import annotations

import math
from collections.abc import Callable
import warnings
from typing import Any


Interval = tuple[float | None, float | None]

_TINY_NEGATIVE_TOL = 1e-12
_DEFAULT_SUM_TOL = 1e-8


def _finite_boundary(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} boundary is required for this interval")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} boundary must be finite, got {value!r}")
    return numeric


def _cdf_endpoint_boundary(
    value: float | None,
    *,
    name: str,
    lower_endpoint: bool,
) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric):
        raise ValueError(f"{name} boundary cannot be NaN")
    if math.isinf(numeric):
        if lower_endpoint and numeric < 0.0:
            return None
        if not lower_endpoint and numeric > 0.0:
            return None
        raise ValueError(f"{name} boundary has the wrong infinite direction: {value!r}")
    return numeric


def _clip_tiny_negative(value: float) -> float:
    if value < 0.0 and value >= -_TINY_NEGATIVE_TOL:
        return 0.0
    return value


def _clean_float_artifact(value: float) -> float:
    rounded = round(value, 15)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-15):
        return float(rounded)
    return value


def _extract_bucket_bound(bucket: Any, candidates: tuple[str, ...]) -> float | None:
    if isinstance(bucket, dict):
        for name in candidates:
            if name in bucket:
                value = bucket[name]
                break
        else:
            raise ValueError(f"Bucket is missing one of these bound fields: {candidates}")
    else:
        for name in candidates:
            if hasattr(bucket, name):
                value = getattr(bucket, name)
                break
        else:
            raise ValueError(f"Bucket is missing one of these bound attributes: {candidates}")

    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Bucket boundary must be finite or None, got {value!r}")
    return numeric


def _extract_bucket_label(bucket: Any, index: int) -> str:
    if isinstance(bucket, dict):
        for name in ("bucket", "bucket_name", "label", "name"):
            if name in bucket and str(bucket[name]).strip():
                return str(bucket[name])
    else:
        for name in ("label", "name"):
            if hasattr(bucket, name):
                value = getattr(bucket, name)
                if str(value).strip():
                    return str(value)
    return f"bucket_{index}"


def _bucket_temperature_bounds(bucket: Any) -> Interval:
    lower_temp = _extract_bucket_bound(
        bucket,
        ("lower_temp", "lower_bound", "bucket_lower_temp", "lower"),
    )
    upper_temp = _extract_bucket_bound(
        bucket,
        ("upper_temp", "upper_bound", "bucket_upper_temp", "upper"),
    )
    return lower_temp, upper_temp


def final_bucket_to_error_bounds(
    bucket: Any,
    forecast_high: float,
) -> Interval:
    """
    Convert final-temperature bucket bounds into forecast-error bounds.

    Buckets follow the project convention lower < final_high <= upper, so the
    converted interval is lower - forecast_high < error <= upper - forecast_high.
    """
    forecast = _finite_boundary(forecast_high, name="forecast_high")
    lower_temp, upper_temp = _bucket_temperature_bounds(bucket)

    if lower_temp is None and upper_temp is None:
        raise ValueError("Bucket lower and upper bounds cannot both be open")
    if lower_temp is not None and upper_temp is not None and upper_temp <= lower_temp:
        raise ValueError(
            f"Bucket upper boundary must exceed lower boundary: {(lower_temp, upper_temp)!r}"
        )

    lower_error = None if lower_temp is None else lower_temp - forecast
    upper_error = None if upper_temp is None else upper_temp - forecast

    return lower_error, upper_error


def _cdf_probability_value(cdf: Callable[[float], float], boundary: float) -> float:
    value = float(cdf(boundary))
    if not math.isfinite(value):
        raise ValueError(f"CDF returned non-finite value at {boundary:g}: {value!r}")
    if value < -_TINY_NEGATIVE_TOL or value > 1.0 + _TINY_NEGATIVE_TOL:
        raise ValueError(
            f"CDF value at {boundary:g} must be between 0 and 1, got {value!r}"
        )
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def interval_probability(
    cdf: Callable[[float], float],
    lower: float | None,
    upper: float | None,
) -> float:
    """
    Compute P(lower < error <= upper) from any callable CDF.

    None lower means F(-inf)=0, None upper means F(+inf)=1.
    Explicit -inf lower and +inf upper are accepted as equivalent open tails.
    """
    lower_float = _cdf_endpoint_boundary(lower, name="Lower", lower_endpoint=True)
    upper_float = _cdf_endpoint_boundary(upper, name="Upper", lower_endpoint=False)

    if lower_float is None and upper_float is None:
        return 1.0

    if lower_float is not None and upper_float is not None and upper_float <= lower_float:
        raise ValueError(
            f"Interval upper boundary must exceed lower boundary: {(lower, upper)!r}"
        )

    lower_cdf = 0.0 if lower_float is None else _cdf_probability_value(cdf, lower_float)
    upper_cdf = 1.0 if upper_float is None else _cdf_probability_value(cdf, upper_float)
    probability = float(upper_cdf - lower_cdf)

    if not math.isfinite(probability):
        raise ValueError(
            f"Interval probability is not finite for {(lower, upper)!r}: {probability!r}"
        )

    if probability < -_TINY_NEGATIVE_TOL:
        raise ValueError(
            "CDF produced a negative interval probability. "
            f"Interval={(lower, upper)!r}, F(lower)={lower_cdf:.12g}, "
            f"F(upper)={upper_cdf:.12g}"
        )

    return _clean_float_artifact(_clip_tiny_negative(probability))


def cdf_to_interval_probs(
    cdf_values: dict[float, float],
    intervals: list[Interval],
) -> dict[Interval, float]:
    """
    Convert finite-boundary CDF values into interval probabilities.

    Intervals follow the (a, b] convention. Open-ended intervals are
    represented with None.
    """
    cdf_lookup: dict[float, float] = {}
    for boundary, value in cdf_values.items():
        boundary_float = float(boundary)
        value_float = float(value)
        if not math.isfinite(boundary_float):
            raise ValueError(f"CDF boundary must be finite, got {boundary!r}")
        if not math.isfinite(value_float):
            raise ValueError(f"CDF value must be finite for boundary {boundary!r}")
        cdf_lookup[boundary_float] = value_float

    def lookup(boundary: float | None) -> float:
        boundary_float = _finite_boundary(boundary, name="CDF")
        if boundary_float not in cdf_lookup:
            raise ValueError(f"Missing CDF value for boundary {boundary_float:g}")
        return cdf_lookup[boundary_float]

    probs: dict[Interval, float] = {}
    for interval in intervals:
        lower, upper = interval
        if lower is None and upper is None:
            raise ValueError("Interval cannot have both boundaries open")

        probs[interval] = interval_probability(lookup, lower, upper)

    return probs


def validate_probability_vector(
    probs: list[float],
    tolerance: float = _DEFAULT_SUM_TOL,
    *,
    require_sum: bool = True,
) -> None:
    """
    Validate finite probabilities, [0, 1] bounds, and optional sum-to-one mass.
    """
    if len(probs) == 0:
        raise ValueError("Probability list cannot be empty")
    tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f"tolerance must be nonnegative and finite, got {tolerance!r}")

    total = 0.0
    for probability in probs:
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(f"Probability is not finite: {probability!r}")
        if value < -tol:
            raise ValueError(f"Probability is negative beyond tolerance: {probability!r}")
        if value > 1.0 + tol:
            raise ValueError(f"Probability exceeds 1 beyond tolerance: {probability!r}")
        total += min(1.0, max(0.0, value))

    if require_sum and abs(total - 1.0) > tol:
        raise ValueError(f"Probabilities must sum to 1; got {total:.12g}")


def normalize_probabilities(
    probs: list[float],
    tolerance: float = _DEFAULT_SUM_TOL,
) -> list[float]:
    """
    Clip tiny negative probabilities and renormalize to sum to one.
    """
    if len(probs) == 0:
        raise ValueError("Probability list cannot be empty")
    if tolerance < 0.0 or not math.isfinite(float(tolerance)):
        raise ValueError(f"tolerance must be nonnegative and finite, got {tolerance!r}")

    cleaned: list[float] = []
    clipped_mass = 0.0
    for probability in probs:
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(f"Probability is not finite: {probability!r}")
        if value < 0.0:
            if value < -tolerance:
                raise ValueError(f"Probability is negative beyond tolerance: {probability!r}")
            clipped_mass += abs(value)
            value = 0.0
        if value > 1.0:
            if value > 1.0 + tolerance:
                raise ValueError(f"Probability exceeds 1 beyond tolerance: {probability!r}")
            value = 1.0
        cleaned.append(value)

    if clipped_mass > tolerance:
        warnings.warn(
            f"Clipped {clipped_mass:.12g} negative probability mass before normalization",
            RuntimeWarning,
            stacklevel=2,
        )

    total = float(sum(cleaned))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"Cannot normalize probabilities with invalid total mass: {total!r}")

    if abs(total - 1.0) > tolerance:
        warnings.warn(
            f"Normalizing probabilities with total mass {total:.12g}",
            RuntimeWarning,
            stacklevel=2,
        )

    normalized = [probability / total for probability in cleaned]
    validate_probability_vector(normalized, tolerance=tolerance, require_sum=True)
    return normalized


def final_buckets_to_error_intervals(
    buckets: list[Any],
    forecast_high: float,
) -> list[Interval]:
    """
    Convert final-temperature bucket schemas into forecast-error intervals.
    """
    if len(buckets) == 0:
        raise ValueError("Bucket list cannot be empty")
    return [
        final_bucket_to_error_bounds(bucket, forecast_high)
        for bucket in buckets
    ]


def bucket_probabilities_from_cdf(
    buckets: list[Any],
    forecast_high: float,
    cdf: Callable[[float], float],
    *,
    normalize: bool = True,
    tolerance: float = _DEFAULT_SUM_TOL,
) -> list[dict[str, float | str | None]]:
    """
    Price final-temperature buckets from a forecast-error CDF.

    Each row uses the lower-open, upper-closed convention:
    P(bucket) = P(lower_error < error <= upper_error) = F(upper_error) - F(lower_error).
    """
    if len(buckets) == 0:
        raise ValueError("Bucket list cannot be empty")

    forecast = _finite_boundary(forecast_high, name="forecast_high")
    rows: list[dict[str, float | str | None]] = []
    probabilities: list[float] = []

    for index, bucket in enumerate(buckets):
        lower_temp, upper_temp = _bucket_temperature_bounds(bucket)
        lower_error, upper_error = final_bucket_to_error_bounds(bucket, forecast)
        probability = interval_probability(cdf, lower_error, upper_error)
        rows.append(
            {
                "bucket": _extract_bucket_label(bucket, index),
                "lower_temp": lower_temp,
                "upper_temp": upper_temp,
                "forecast_high": forecast,
                "lower_error": lower_error,
                "upper_error": upper_error,
                "probability": probability,
            }
        )
        probabilities.append(probability)

    if normalize:
        validate_probability_vector(probabilities, tolerance=tolerance, require_sum=True)
        probabilities = normalize_probabilities(probabilities, tolerance=tolerance)
        validate_probability_vector(probabilities, tolerance=tolerance, require_sum=True)
    else:
        validate_probability_vector(probabilities, tolerance=tolerance, require_sum=False)

    for row, probability in zip(rows, probabilities):
        row["probability"] = float(probability)

    return rows


def normalize_probs(
    probs: dict[Interval, float],
) -> dict[Interval, float]:
    """
    Normalize interval probabilities to sum to one.
    """
    intervals = list(probs)
    probabilities = normalize_probabilities(
        [float(probs[interval]) for interval in intervals],
        tolerance=_TINY_NEGATIVE_TOL,
    )
    return {
        interval: probability
        for interval, probability in zip(intervals, probabilities)
    }


def validate_interval_probs(
    probs: dict[Interval, float],
    tol: float = 1e-6,
) -> None:
    """
    Validate that interval probabilities are finite, bounded, and sum to one.
    """
    if not probs:
        raise ValueError("Interval probability dictionary is empty")

    for interval, probability in probs.items():
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(f"Probability for interval {interval!r} is not finite: {probability!r}")
        if value < -tol:
            raise ValueError(f"Probability for interval {interval!r} is negative: {probability!r}")
        if value > 1.0 + tol:
            raise ValueError(f"Probability for interval {interval!r} exceeds 1: {probability!r}")

    validate_probability_vector(list(probs.values()), tolerance=tol, require_sum=True)
