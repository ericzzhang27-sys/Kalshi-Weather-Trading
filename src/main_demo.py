from bucket_schema import (
    Bucket,
    validate_buckets,
    bucket_for_actual_temp,
    format_bucket_interval,
)
from error_boundaries import (
    buckets_to_error_intervals,
    extract_cdf_boundaries,
)


def format_bound(value: float | None) -> str:
    if value is None:
        return "None"
    if value > 0:
        return f"+{value:.1f}"
    return f"{value:.1f}"


def format_error_interval(interval) -> str:
    lower = interval.lower_error
    upper = interval.upper_error

    if lower is None:
        return f"error <= {format_bound(upper)}"

    if upper is None:
        return f"error > {format_bound(lower)}"

    return f"{format_bound(lower)} < error <= {format_bound(upper)}"


def build_demo_buckets(location: str) -> list[Bucket]:
    return [
        Bucket(
            location=location,
            name="71 or lower",
            reported_lower=None,
            reported_upper=71,
        ),
        Bucket(
            location=location,
            name="72",
            reported_lower=72,
            reported_upper=72,
        ),
        Bucket(
            location=location,
            name="73",
            reported_lower=73,
            reported_upper=73,
        ),
        Bucket(
            location=location,
            name="74",
            reported_lower=74,
            reported_upper=74,
        ),
        Bucket(
            location=location,
            name="75",
            reported_lower=75,
            reported_upper=75,
        ),
        Bucket(
            location=location,
            name="76 or higher",
            reported_lower=76,
            reported_upper=None,
        ),
    ]


def run_manual_resolution_checks(buckets: list[Bucket]) -> None:
    test_actual_highs = [60, 71.5, 72, 72.5, 72.5001, 73, 73.5, 75.5, 80]

    print("\nManual bucket-resolution checks:")

    for actual_high in test_actual_highs:
        bucket = bucket_for_actual_temp(actual_high, buckets)
        print(
            f"actual_high={actual_high}: "
            f"resolves YES for '{bucket.name}'"
        )


def main() -> None:
    location = "NYC"
    forecast_high = 73.0

    buckets = build_demo_buckets(location)

    validate_buckets(buckets)

    error_intervals = buckets_to_error_intervals(
        buckets=buckets,
        forecast_high=forecast_high,
    )

    cdf_boundaries = extract_cdf_boundaries(error_intervals)

    print(f"Location: {location}")
    print(f"NWS forecast high: {forecast_high:.1f}°F")

    print("\nMarket buckets:")
    for bucket in buckets:
        print(f"{bucket.name}: {format_bucket_interval(bucket)}")

    print("\nForecast-error intervals:")
    for interval in error_intervals:
        print(f"{interval.bucket_name}: {format_error_interval(interval)}")

    print("\nCDF boundaries needed for pricing:")
    print(cdf_boundaries)

    run_manual_resolution_checks(buckets)


if __name__ == "__main__":
    main()