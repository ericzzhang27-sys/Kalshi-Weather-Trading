from __future__ import annotations

import math
from math import erf, isclose, sqrt

import pytest

from src.bucket_schema import (
    TemperatureBucket,
    integer_temperature_bucket_bounds,
    make_integer_temperature_buckets,
)
from src.interval_probs import (
    bucket_probabilities_from_cdf,
    cdf_to_interval_probs,
    final_bucket_to_error_bounds,
    final_buckets_to_error_intervals,
    interval_probability,
    normalize_probabilities,
    validate_interval_probs,
    validate_probability_vector,
)


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    z = (float(x) - mu) / (sigma * sqrt(2.0))
    return float(0.5 * (1.0 + erf(z)))


def test_integer_settlement_bucket_bounds_are_half_degree_and_lower_open() -> None:
    lower, upper = integer_temperature_bucket_bounds(73)
    bucket = TemperatureBucket("73", lower_temp=lower, upper_temp=upper)

    assert (lower, upper) == (72.5, 73.5)
    assert bucket.contains_actual_high(72.5001)
    assert bucket.contains_actual_high(73.5)
    assert not bucket.contains_actual_high(72.5)
    assert not bucket.contains_actual_high(73.5001)


def test_final_temperature_buckets_convert_to_forecast_error_intervals() -> None:
    buckets = make_integer_temperature_buckets(72, 74)

    intervals = final_buckets_to_error_intervals(buckets, forecast_high=72.5)

    assert intervals == [(None, 0.0), (0.0, 1.0), (1.0, None)]
    assert final_bucket_to_error_bounds(buckets[1], forecast_high=72.5) == (0.0, 1.0)


def test_interval_probability_accepts_none_and_explicit_infinite_tails() -> None:
    def finite_only_cdf(x: float) -> float:
        if not math.isfinite(float(x)):
            raise AssertionError("interval_probability should not call cdf at infinity")
        return _normal_cdf(x)

    assert interval_probability(finite_only_cdf, None, 0.0) == 0.5
    assert interval_probability(finite_only_cdf, -math.inf, 0.0) == 0.5
    assert interval_probability(finite_only_cdf, 0.0, None) == 0.5
    assert interval_probability(finite_only_cdf, 0.0, math.inf) == 0.5
    assert interval_probability(finite_only_cdf, -math.inf, math.inf) == 1.0


def test_bucket_probabilities_from_cdf_matches_hand_checkable_standard_normal() -> None:
    buckets = make_integer_temperature_buckets(72, 74)
    rows = bucket_probabilities_from_cdf(
        buckets,
        forecast_high=72.5,
        cdf=_normal_cdf,
        normalize=True,
        tolerance=1e-12,
    )

    phi_1 = _normal_cdf(1.0)
    expected = [0.5, phi_1 - 0.5, 1.0 - phi_1]
    probabilities = [float(row["probability"]) for row in rows]

    assert [row["bucket"] for row in rows] == ["72 or lower", "73", "74 or higher"]
    assert [(row["lower_error"], row["upper_error"]) for row in rows] == [
        (None, 0.0),
        (0.0, 1.0),
        (1.0, None),
    ]
    for probability, expected_probability in zip(probabilities, expected):
        assert isclose(probability, expected_probability, rel_tol=0.0, abs_tol=1e-12)
    assert isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_cdf_to_interval_probs_supports_open_ended_intervals() -> None:
    phi_1 = _normal_cdf(1.0)

    probs = cdf_to_interval_probs(
        cdf_values={0.0: 0.5, 1.0: phi_1},
        intervals=[(None, 0.0), (0.0, 1.0), (1.0, None)],
    )

    assert probs[(None, 0.0)] == 0.5
    assert isclose(probs[(0.0, 1.0)], phi_1 - 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert isclose(probs[(1.0, None)], 1.0 - phi_1, rel_tol=0.0, abs_tol=1e-12)
    validate_interval_probs(probs, tol=1e-12)


def test_interval_probability_rejects_non_monotone_cdf() -> None:
    def bad_cdf(x: float) -> float:
        return 0.8 if x <= 0.0 else 0.7

    with pytest.raises(ValueError, match="negative interval probability"):
        interval_probability(bad_cdf, 0.0, 1.0)


def test_bucket_probabilities_do_not_normalize_away_non_exhaustive_schema() -> None:
    buckets = [TemperatureBucket("72 or lower", lower_temp=None, upper_temp=72.5)]

    with pytest.raises(ValueError, match="sum to 1"):
        bucket_probabilities_from_cdf(
            buckets,
            forecast_high=72.5,
            cdf=_normal_cdf,
            normalize=True,
            tolerance=1e-12,
        )


def test_probability_validation_catches_bad_values_and_sums() -> None:
    with pytest.raises(ValueError, match="sum"):
        validate_probability_vector([0.25, 0.25], tolerance=1e-12)

    with pytest.raises(ValueError, match="exceeds 1"):
        normalize_probabilities([1.01, 0.0], tolerance=1e-6)
