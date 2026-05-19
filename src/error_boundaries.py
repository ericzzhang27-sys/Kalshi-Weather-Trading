from dataclasses import dataclass

from bucket_schema import Bucket


@dataclass(frozen=True)
class ErrorInterval:
    location: str
    bucket_name: str
    lower_error: float | None
    upper_error: float | None


def bucket_to_error_interval(bucket: Bucket, forecast_high: float) -> ErrorInterval:
    """
    Converts one actual-temperature bucket into a forecast-error interval.

    error = actual official high - NWS forecast high
    """
    lower_error = (
        None
        if bucket.lower_bound is None
        else bucket.lower_bound - forecast_high
    )

    upper_error = (
        None
        if bucket.upper_bound is None
        else bucket.upper_bound - forecast_high
    )

    return ErrorInterval(
        location=bucket.location,
        bucket_name=bucket.name,
        lower_error=lower_error,
        upper_error=upper_error,
    )


def buckets_to_error_intervals(
    buckets: list[Bucket],
    forecast_high: float,
) -> list[ErrorInterval]:
    """
    Converts all market buckets into forecast-error intervals.
    """
    if len(buckets) == 0:
        raise ValueError("buckets cannot be empty")

    return [
        bucket_to_error_interval(bucket, forecast_high)
        for bucket in buckets
    ]


def extract_cdf_boundaries(error_intervals: list[ErrorInterval]) -> list[float]:
    """
    Extracts finite CDF cutpoints from ordered forecast-error intervals.

    For buckets:
    error <= -1.5
    -1.5 < error <= -0.5
    -0.5 < error <= +0.5
    ...

    The needed CDF boundaries are:
    [-1.5, -0.5, +0.5, ...]
    """
    if len(error_intervals) == 0:
        raise ValueError("error_intervals cannot be empty")

    boundaries: list[float] = []

    for interval in error_intervals:
        if interval.upper_error is not None:
            boundaries.append(interval.upper_error)

    if boundaries != sorted(boundaries):
        raise ValueError(f"CDF boundaries must be ordered. Got: {boundaries}")

    if len(set(boundaries)) != len(boundaries):
        raise ValueError(f"CDF boundaries cannot contain duplicates. Got: {boundaries}")

    return boundaries


def format_error_interval(interval: ErrorInterval) -> str:
    """
    Human-readable representation of the forecast-error interval.
    """
    lower = interval.lower_error
    upper = interval.upper_error

    if lower is None:
        return f"error <= {upper:+.1f}"

    if upper is None:
        return f"error > {lower:+.1f}"

    return f"{lower:+.1f} < error <= {upper:+.1f}"