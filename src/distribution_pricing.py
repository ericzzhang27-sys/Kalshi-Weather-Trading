from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .bucket_schema import Bucket, TemperatureBucket, validate_temperature_buckets
    from .interval_probs import interval_probability, normalize_probabilities
except ImportError:
    from bucket_schema import Bucket, TemperatureBucket, validate_temperature_buckets
    from interval_probs import interval_probability, normalize_probabilities


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


def normal_cdf(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0")

    scale = float(sigma) * math.sqrt(2.0)

    if isinstance(x, np.ndarray):
        z = (x.astype(float) - float(mu)) / scale
        erf_values = np.vectorize(math.erf, otypes=[float])(z)
        return 0.5 * (1.0 + erf_values)

    z = (float(x) - float(mu)) / scale
    return float(0.5 * (1.0 + math.erf(z)))


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
    lower_error = (
        None
        if bucket.lower_temp is None
        else float(bucket.lower_temp) - forecast
    )
    upper_error = (
        None
        if bucket.upper_temp is None
        else float(bucket.upper_temp) - forecast
    )

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
