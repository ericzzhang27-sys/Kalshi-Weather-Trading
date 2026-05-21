try:
    from .bucket_schema import (
        Bucket,
        bucket_for_actual_temp,
        format_bucket_interval,
        validate_buckets,
    )
    from .error_boundaries import (
        buckets_to_error_intervals,
        convert_market_to_boundaries,
        extract_cdf_boundaries,
    )
except ImportError:
    from bucket_schema import (
        Bucket,
        bucket_for_actual_temp,
        format_bucket_interval,
        validate_buckets,
    )
    from error_boundaries import (
        buckets_to_error_intervals,
        convert_market_to_boundaries,
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
    return convert_market_to_boundaries(
        [69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
        location,
    )


def run_manual_resolution_checks(buckets: list[Bucket]) -> None:
    test_actual_highs = [60, 69.5, 70, 71.5, 71.5001, 72, 73.5, 77.5, 80]

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
