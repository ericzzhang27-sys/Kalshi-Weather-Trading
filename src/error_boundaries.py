from dataclasses import dataclass

try:
    from .bucket_schema import Bucket, validate_buckets
except ImportError:
    from bucket_schema import Bucket, validate_buckets


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


def convert_market_to_boundaries(temperatures: list[int], location: str) -> list[Bucket]:
    """
    Builds Kalshi-style two-degree interior buckets from sorted temperatures.

    For [69, 70, 71, 72, 73, 74, 75, 76, 77, 78], this returns:
    69 or lower, 70 to 71, 72 to 73, 74 to 75, 76 to 77, 78 or higher.
    """
    if len(temperatures) < 4:
        raise ValueError("temperatures must contain at least four bucket labels")

    if temperatures != sorted(temperatures):
        raise ValueError(f"temperatures must be sorted ascending. Got: {temperatures}")

    if len(set(temperatures)) != len(temperatures):
        raise ValueError(f"temperatures cannot contain duplicates. Got: {temperatures}")

    if len(temperatures) % 2 != 0:
        raise ValueError(
            "temperatures must contain an even count so interior labels form pairs"
        )

    buckets = [
        Bucket(
            location=location,
            name=f"{temperatures[0]} or lower",
            lower_bound=None,
            upper_bound=temperatures[0] + 0.5,
        )
    ]

    for i in range(1, len(temperatures) - 1, 2):
        lower_temperature = temperatures[i]
        upper_temperature = temperatures[i + 1]
        buckets.append(
            Bucket(
                location=location,
                name=f"{lower_temperature} to {upper_temperature}",
                lower_bound=lower_temperature - 0.5,
                upper_bound=upper_temperature + 0.5,
            )
        )

    buckets.append(
        Bucket(
            location=location,
            name=f"{temperatures[-1]} or higher",
            lower_bound=temperatures[-1] - 0.5,
            upper_bound=None,
        )
    )

    validate_buckets(buckets)
    return buckets


def convert_nws_to_boundaries(temperature: int, location: str) -> list[Bucket]:
    """
    Builds a six-bucket market centered near an NWS forecast high.
    """
    return convert_market_to_boundaries(
        list(range(temperature - 4, temperature + 6)),
        location,
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
