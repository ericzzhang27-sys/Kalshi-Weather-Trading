from __future__ import annotations

import math


Interval = tuple[float | None, float | None]

_TINY_NEGATIVE_TOL = 1e-12


def _finite_boundary(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} boundary is required for this interval")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} boundary must be finite, got {value!r}")
    return numeric


def _clip_tiny_negative(value: float) -> float:
    if value < 0.0 and value >= -_TINY_NEGATIVE_TOL:
        return 0.0
    return value


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

        if lower is None:
            probability = lookup(upper)
        elif upper is None:
            probability = 1.0 - lookup(lower)
        else:
            lower_float = _finite_boundary(lower, name="Lower")
            upper_float = _finite_boundary(upper, name="Upper")
            if upper_float <= lower_float:
                raise ValueError(f"Interval upper boundary must exceed lower boundary: {interval!r}")
            probability = lookup(upper_float) - lookup(lower_float)

        probs[interval] = _clip_tiny_negative(float(probability))

    return probs


def normalize_probs(
    probs: dict[Interval, float],
) -> dict[Interval, float]:
    """
    Normalize interval probabilities to sum to one.
    """
    cleaned: dict[Interval, float] = {}
    for interval, probability in probs.items():
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(f"Probability for interval {interval!r} is not finite: {probability!r}")
        value = _clip_tiny_negative(value)
        if value < 0.0:
            raise ValueError(f"Probability for interval {interval!r} is negative: {probability!r}")
        cleaned[interval] = value

    total = float(sum(cleaned.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"Cannot normalize probabilities with invalid total mass: {total!r}")

    return {interval: probability / total for interval, probability in cleaned.items()}


def validate_interval_probs(
    probs: dict[Interval, float],
    tol: float = 1e-6,
) -> None:
    """
    Validate that interval probabilities are finite, bounded, and sum to one.
    """
    if not probs:
        raise ValueError("Interval probability dictionary is empty")

    total = 0.0
    for interval, probability in probs.items():
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError(f"Probability for interval {interval!r} is not finite: {probability!r}")
        if value < -tol:
            raise ValueError(f"Probability for interval {interval!r} is negative: {probability!r}")
        if value > 1.0 + tol:
            raise ValueError(f"Probability for interval {interval!r} exceeds 1: {probability!r}")
        total += value

    if abs(total - 1.0) > tol:
        raise ValueError(f"Interval probabilities must sum to 1; got {total:.12g}")
