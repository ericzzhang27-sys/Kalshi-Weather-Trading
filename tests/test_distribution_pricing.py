from __future__ import annotations

from math import isclose

import pandas as pd
import pytest
from scipy.stats import norm

from src.bucket_schema import TemperatureBucket, make_integer_temperature_buckets
from src.distribution_pricing import (
    forecast_high_to_market_anchor,
    make_kalshi_buckets_around_forecast,
    interval_probability_from_cdf,
    price_buckets_for_dataframe,
    validate_bucket_probabilities,
)
from src.interval_probs import final_bucket_to_error_bounds


def test_normal_cdf_interval_probability_matches_scipy() -> None:
    lower = -1.25
    upper = 2.5
    mu = 0.4
    sigma = 1.7

    probability = interval_probability_from_cdf(lower, upper, mu=mu, sigma=sigma)
    expected = norm.cdf(upper, loc=mu, scale=sigma) - norm.cdf(lower, loc=mu, scale=sigma)

    assert isclose(probability, expected, rel_tol=0.0, abs_tol=1e-12)


def test_lower_open_ended_bucket_uses_upper_cdf() -> None:
    upper = -0.5
    probability = interval_probability_from_cdf(None, upper, mu=0.0, sigma=2.0)

    assert isclose(probability, norm.cdf(upper, loc=0.0, scale=2.0), rel_tol=0.0, abs_tol=1e-12)


def test_upper_open_ended_bucket_uses_one_minus_lower_cdf() -> None:
    lower = 1.5
    probability = interval_probability_from_cdf(lower, None, mu=0.0, sigma=2.0)

    assert isclose(
        probability,
        1.0 - norm.cdf(lower, loc=0.0, scale=2.0),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_full_bucket_set_sums_to_one() -> None:
    buckets = make_integer_temperature_buckets(80, 83)
    pred_df = pd.DataFrame(
        {
            "row_id": [10],
            "date": ["2025-07-01"],
            "split": ["validation"],
            "forecast_high": [82.0],
            "actual_high": [83.0],
            "forecast_error": [1.0],
            "mu": [0.25],
            "sigma": [1.3],
        }
    )

    priced = price_buckets_for_dataframe(pred_df, buckets)
    summary = validate_bucket_probabilities(priced, tolerance=1e-12)

    assert len(priced) == len(buckets)
    assert summary["validation_passed"] is True
    assert isclose(float(priced["probability"].sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_final_bucket_to_error_bounds_subtracts_forecast_high() -> None:
    bucket = TemperatureBucket("80 to 85", lower_temp=80.0, upper_temp=85.0)

    assert final_bucket_to_error_bounds(bucket, forecast_high=82.0) == (-2.0, 3.0)


def test_invalid_sigma_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="sigma"):
        interval_probability_from_cdf(0.0, 1.0, mu=0.0, sigma=0.0)


def test_bad_bucket_boundaries_raise_clear_error() -> None:
    pred_df = pd.DataFrame(
        {
            "row_id": [1],
            "forecast_high": [82.0],
            "mu": [0.0],
            "sigma": [1.0],
        }
    )
    bad_buckets = [{"label": "bad", "lower_temp": 85.0, "upper_temp": 80.0}]

    with pytest.raises(ValueError, match="upper"):
        price_buckets_for_dataframe(pred_df, bad_buckets)


def test_kalshi_buckets_are_created_around_daily_forecast() -> None:
    buckets = make_kalshi_buckets_around_forecast(73.4, location="NYC")

    assert [bucket.name for bucket in buckets] == [
        "69 or lower",
        "70 to 71",
        "72 to 73",
        "74 to 75",
        "76 to 77",
        "78 or higher",
    ]
    assert buckets[0].lower_bound is None
    assert buckets[0].upper_bound == 69.5
    assert buckets[2].lower_bound == 71.5
    assert buckets[2].upper_bound == 73.5
    assert buckets[-1].lower_bound == 77.5
    assert buckets[-1].upper_bound is None


def test_decimal_forecast_anchor_uses_half_up_nearest_integer() -> None:
    assert forecast_high_to_market_anchor(73.4) == 73
    assert forecast_high_to_market_anchor(73.5) == 74


def test_dataframe_default_prices_six_kalshi_buckets_per_row() -> None:
    pred_df = pd.DataFrame(
        {
            "row_id": [1],
            "location": ["NYC"],
            "forecast_high": [73.0],
            "actual_high": [72.0],
            "mu": [0.0],
            "sigma": [1.0],
        }
    )

    priced = price_buckets_for_dataframe(pred_df)

    assert len(priced) == 6
    assert priced["bucket_name"].tolist() == [
        "69 or lower",
        "70 to 71",
        "72 to 73",
        "74 to 75",
        "76 to 77",
        "78 or higher",
    ]
    assert priced.loc[2, "error_lower"] == -1.5
    assert priced.loc[2, "error_upper"] == 0.5
    assert validate_bucket_probabilities(priced)["validation_passed"] is True
