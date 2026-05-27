from __future__ import annotations

from csv import DictWriter
from math import erf, sqrt
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bucket_schema import make_integer_temperature_buckets  # noqa: E402
from src.distribution_pricing import (  # noqa: E402
    convert_temperature_buckets_to_error_buckets,
    price_temperature_buckets_from_cdf,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
FORECAST_HIGH = 73.0


def normal_error_cdf(x: float, mu: float = 0.0, sigma: float = 1.5) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be greater than 0")
    z = (float(x) - mu) / (sigma * sqrt(2.0))
    return float(0.5 * (1.0 + erf(z)))


def _write_rows(
    output_file: Path,
    rows: list[dict[str, float | str | None]],
    fieldnames: list[str],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_forecast_error_bucket_conversion_examples(
    output_file: Path = OUTPUT_DIR / "forecast_error_bucket_conversion_examples.csv",
) -> list[dict[str, float | str | None]]:
    buckets = make_integer_temperature_buckets(71, 76)
    error_buckets = convert_temperature_buckets_to_error_buckets(buckets, FORECAST_HIGH)

    rows = [
        {
            "bucket": bucket.label,
            "lower_temp": bucket.lower_temp,
            "upper_temp": bucket.upper_temp,
            "forecast_high": FORECAST_HIGH,
            "lower_error": error_bucket.lower_error,
            "upper_error": error_bucket.upper_error,
        }
        for bucket, error_bucket in zip(buckets, error_buckets)
    ]
    _write_rows(
        output_file,
        rows,
        [
            "bucket",
            "lower_temp",
            "upper_temp",
            "forecast_high",
            "lower_error",
            "upper_error",
        ],
    )
    return rows


def generate_interval_probability_demo(
    output_file: Path = OUTPUT_DIR / "interval_probability_demo.csv",
) -> list[dict[str, float | str | None]]:
    buckets = make_integer_temperature_buckets(71, 76)
    rows = price_temperature_buckets_from_cdf(
        buckets=buckets,
        forecast_high=FORECAST_HIGH,
        cdf=lambda x: normal_error_cdf(x, mu=0.0, sigma=1.5),
        normalize=True,
    )
    _write_rows(
        output_file,
        rows,
        [
            "bucket",
            "lower_temp",
            "upper_temp",
            "forecast_high",
            "lower_error",
            "upper_error",
            "probability",
        ],
    )
    return rows


def main() -> None:
    conversion_rows = generate_forecast_error_bucket_conversion_examples()
    probability_rows = generate_interval_probability_demo()
    print(
        "Saved Day 10 outputs: "
        f"{len(conversion_rows)} conversion rows and "
        f"{len(probability_rows)} probability rows"
    )


if __name__ == "__main__":
    main()
