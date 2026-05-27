from __future__ import annotations

from math import erf, isclose, sqrt
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_day10_distribution_pricing_outputs import (  # noqa: E402
    generate_forecast_error_bucket_conversion_examples,
    generate_interval_probability_demo,
)
from src.bucket_schema import (  # noqa: E402
    TemperatureBucket,
    make_integer_temperature_buckets,
    validate_temperature_buckets,
)
from src.distribution_pricing import (  # noqa: E402
    convert_temperature_buckets_to_error_buckets,
    price_temperature_buckets_from_cdf,
)
from src.interval_probs import interval_probability, normalize_probabilities  # noqa: E402


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.5) -> float:
    z = (float(x) - mu) / (sigma * sqrt(2.0))
    return float(0.5 * (1.0 + erf(z)))


def test_integer_temperature_buckets_use_half_degree_boundaries() -> None:
    buckets = make_integer_temperature_buckets(71, 76)

    assert len(buckets) == 6
    assert [bucket.label for bucket in buckets] == [
        "71 or lower",
        "72",
        "73",
        "74",
        "75",
        "76 or higher",
    ]
    assert buckets[0].lower_temp is None
    assert buckets[0].upper_temp == 71.5
    assert buckets[-1].lower_temp == 75.5
    assert buckets[-1].upper_temp is None


def test_temperature_bucket_validation_rejects_gaps() -> None:
    buckets = [
        TemperatureBucket("71 or lower", None, 71.5),
        TemperatureBucket("72", 72.0, 72.5),
        TemperatureBucket("73 or higher", 72.5, None),
    ]

    with pytest.raises(ValueError, match="Gap or overlap"):
        validate_temperature_buckets(buckets)


def test_forecast_high_73_conversion_to_error_buckets() -> None:
    buckets = make_integer_temperature_buckets(71, 76)
    error_buckets = convert_temperature_buckets_to_error_buckets(
        buckets=buckets,
        forecast_high=73,
    )
    by_label = {bucket.label: bucket for bucket in error_buckets}

    assert by_label["71 or lower"].lower_error is None
    assert by_label["71 or lower"].upper_error == -1.5
    assert by_label["72"].lower_error == -1.5
    assert by_label["72"].upper_error == -0.5
    assert by_label["73"].lower_error == -0.5
    assert by_label["73"].upper_error == 0.5
    assert by_label["74"].lower_error == 0.5
    assert by_label["74"].upper_error == 1.5
    assert by_label["75"].lower_error == 1.5
    assert by_label["75"].upper_error == 2.5
    assert by_label["76 or higher"].lower_error == 2.5
    assert by_label["76 or higher"].upper_error is None


def test_interval_probability_from_fake_cdf() -> None:
    def fake_cdf(x: float) -> float:
        return float(x)

    assert interval_probability(fake_cdf, 0.2, 0.7) == 0.5
    assert interval_probability(fake_cdf, None, 0.7) == 0.7
    assert interval_probability(fake_cdf, 0.2, None) == 0.8
    assert interval_probability(fake_cdf, None, None) == 1.0


def test_normalize_probabilities_clips_tiny_negatives() -> None:
    probabilities = normalize_probabilities([-1e-12, 0.5, 0.5])

    assert probabilities[0] == 0.0
    assert all(probability >= 0.0 for probability in probabilities)
    assert isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_normalize_probabilities_rejects_large_negative_values() -> None:
    with pytest.raises(ValueError, match="negative beyond tolerance"):
        normalize_probabilities([-0.01, 0.5, 0.5])


def test_price_temperature_buckets_from_normal_cdf_sums_to_one() -> None:
    buckets = make_integer_temperature_buckets(71, 76)

    rows = price_temperature_buckets_from_cdf(
        buckets=buckets,
        forecast_high=73,
        cdf=_normal_cdf,
        normalize=True,
    )

    probabilities = [float(row["probability"]) for row in rows]
    by_bucket = {str(row["bucket"]): float(row["probability"]) for row in rows}

    assert isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert all(probability >= 0.0 for probability in probabilities)
    assert max(rows, key=lambda row: float(row["probability"]))["bucket"] == "73"
    assert isclose(by_bucket["72"], by_bucket["74"], rel_tol=0.0, abs_tol=1e-12)


def test_day10_conversion_output_generator_uses_forecast_high_73(tmp_path: Path) -> None:
    output_file = tmp_path / "forecast_error_bucket_conversion_examples.csv"
    rows = generate_forecast_error_bucket_conversion_examples(output_file)

    assert output_file.exists()
    assert rows[0]["bucket"] == "71 or lower"
    assert rows[0]["upper_error"] == -1.5
    assert rows[1]["bucket"] == "72"
    assert rows[1]["lower_error"] == -1.5
    assert rows[1]["upper_error"] == -0.5
    assert rows[2]["bucket"] == "73"
    assert rows[2]["lower_error"] == -0.5
    assert rows[2]["upper_error"] == 0.5
    assert rows[-1]["bucket"] == "76 or higher"
    assert rows[-1]["lower_error"] == 2.5
    assert rows[-1]["upper_error"] is None


def test_interval_probability_demo_output_sums_to_one(tmp_path: Path) -> None:
    output_file = tmp_path / "interval_probability_demo.csv"
    rows = generate_interval_probability_demo(output_file)

    probabilities = [float(row["probability"]) for row in rows]

    assert output_file.exists()
    assert isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert all(probability >= 0.0 for probability in probabilities)
    assert max(rows, key=lambda row: float(row["probability"]))["bucket"] == "73"
