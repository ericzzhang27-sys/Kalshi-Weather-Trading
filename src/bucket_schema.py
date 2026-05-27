from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_label(label: str, *, field_name: str) -> None:
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_bound(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number or None")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite, got {value!r}")


def _validate_interval_bounds(
    lower: float | None,
    upper: float | None,
    *,
    lower_name: str,
    upper_name: str,
) -> None:
    _validate_bound(lower, field_name=lower_name)
    _validate_bound(upper, field_name=upper_name)

    if lower is None and upper is None:
        raise ValueError(f"{lower_name} and {upper_name} cannot both be None")

    if lower is not None and upper is not None and float(upper) <= float(lower):
        raise ValueError(f"{upper_name} must be greater than {lower_name}")


@dataclass(frozen=True)
class TemperatureBucket:
    """
    Final-temperature market bucket using the lower-open, upper-closed convention.

    A bucket represents: lower_temp < actual_high <= upper_temp.
    None means the interval is open-ended on that side.
    """

    label: str
    lower_temp: float | None = None
    upper_temp: float | None = None

    def __post_init__(self) -> None:
        _validate_label(self.label, field_name="label")
        _validate_interval_bounds(
            self.lower_temp,
            self.upper_temp,
            lower_name="lower_temp",
            upper_name="upper_temp",
        )

    @property
    def name(self) -> str:
        return self.label

    @property
    def lower_bound(self) -> float | None:
        return self.lower_temp

    @property
    def upper_bound(self) -> float | None:
        return self.upper_temp

    def contains_actual_high(self, actual_high: float) -> bool:
        numeric_value = float(actual_high)
        if not math.isfinite(numeric_value):
            raise ValueError(f"actual_high must be finite, got {actual_high!r}")

        if self.lower_temp is not None and numeric_value <= self.lower_temp:
            return False

        if self.upper_temp is not None and numeric_value > self.upper_temp:
            return False

        return True


@dataclass
class Bucket:
    location: str
    name: str
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        _validate_label(self.location, field_name="location")
        _validate_label(self.name, field_name="name")
        _validate_interval_bounds(
            self.lower_bound,
            self.upper_bound,
            lower_name="lower_bound",
            upper_name="upper_bound",
        )

    @property
    def label(self) -> str:
        return self.name

    @property
    def lower_temp(self) -> float | None:
        return self.lower_bound

    @property
    def upper_temp(self) -> float | None:
        return self.upper_bound

    def contains_actual_high(self, actual_high: float) -> bool:
        numeric_value = float(actual_high)
        if not math.isfinite(numeric_value):
            raise ValueError(f"actual_high must be finite, got {actual_high!r}")

        if self.lower_bound is not None and numeric_value <= self.lower_bound:
            return False

        if self.upper_bound is not None and numeric_value > self.upper_bound:
            return False

        return True


def _validate_contiguous_temperature_intervals(
    intervals: list[tuple[str, float | None, float | None]],
) -> None:
    if len(intervals) == 0:
        raise ValueError("Bucket list cannot be empty")

    previous_upper: float | None = None

    for i, (label, lower, upper) in enumerate(intervals):
        _validate_interval_bounds(
            lower,
            upper,
            lower_name="lower boundary",
            upper_name="upper boundary",
        )

        if i == 0:
            if lower is not None:
                raise ValueError("First bucket should be open-ended below")
        else:
            if lower is None:
                raise ValueError(f"Only the first bucket can be open-ended below: {label}")

            if previous_upper is None:
                raise ValueError("Previous bucket was open-ended above before the final bucket")

            if float(lower) != float(previous_upper):
                raise ValueError(
                    f"Gap or overlap near {label}: "
                    f"previous upper={previous_upper}, current lower={lower}"
                )

        if i == len(intervals) - 1:
            if upper is not None:
                raise ValueError("Last bucket should be open-ended above")
        else:
            if upper is None:
                raise ValueError(f"Only the last bucket can be open-ended above: {label}")

        previous_upper = upper


def validate_temperature_buckets(buckets: list[TemperatureBucket]) -> None:
    intervals = [
        (bucket.label, bucket.lower_temp, bucket.upper_temp)
        for bucket in buckets
    ]
    _validate_contiguous_temperature_intervals(intervals)


def make_integer_temperature_buckets(min_temp: int, max_temp: int) -> list[TemperatureBucket]:
    """
    Build one-degree settlement buckets with half-degree boundaries.

    For min_temp=71 and max_temp=76, returns:
    71 or lower, 72, 73, 74, 75, 76 or higher.
    """
    if not isinstance(min_temp, int) or not isinstance(max_temp, int):
        raise TypeError("min_temp and max_temp must be integers")

    if min_temp >= max_temp:
        raise ValueError("min_temp must be less than max_temp")

    buckets = [
        TemperatureBucket(
            label=f"{min_temp} or lower",
            lower_temp=None,
            upper_temp=min_temp + 0.5,
        )
    ]

    for temperature in range(min_temp + 1, max_temp):
        buckets.append(
            TemperatureBucket(
                label=str(temperature),
                lower_temp=temperature - 0.5,
                upper_temp=temperature + 0.5,
            )
        )

    buckets.append(
        TemperatureBucket(
            label=f"{max_temp} or higher",
            lower_temp=max_temp - 0.5,
            upper_temp=None,
        )
    )

    validate_temperature_buckets(buckets)
    return buckets


def validate_buckets(buckets: list[Bucket]) -> None:
    if len(buckets) == 0:
        raise ValueError("Bucket list cannot be empty")

    locations = {bucket.location for bucket in buckets}
    if len(locations) != 1:
        raise ValueError(f"All buckets should have the same location. Found: {locations}")

    intervals = [
        (bucket.name, bucket.lower_bound, bucket.upper_bound)
        for bucket in buckets
    ]
    _validate_contiguous_temperature_intervals(intervals)


def bucket_for_actual_temp(actual_high: float, buckets: list[Bucket]) -> Bucket:
    matching_buckets = [
        bucket for bucket in buckets
        if bucket.contains_actual_high(actual_high)
    ]

    if len(matching_buckets) == 0:
        raise ValueError(f"No bucket matched actual_high={actual_high}")

    if len(matching_buckets) > 1:
        names = [bucket.name for bucket in matching_buckets]
        raise ValueError(f"Multiple buckets matched actual_high={actual_high}: {names}")

    return matching_buckets[0]


def format_bucket_interval(bucket: Bucket | TemperatureBucket) -> str:
    lower = bucket.lower_bound
    upper = bucket.upper_bound

    if lower is None:
        return f"actual <= {upper:.1f}"

    if upper is None:
        return f"actual > {lower:.1f}"

    return f"{lower:.1f} < actual <= {upper:.1f}"
