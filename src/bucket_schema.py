from dataclasses import dataclass, field


@dataclass
class Bucket:
    location: str
    name: str
    reported_lower: int | None = None
    reported_upper: int | None = None
    lower_bound: float | None = field(init=False)
    upper_bound: float | None = field(init=False)

    def __post_init__(self) -> None:
        if self.reported_lower is None and self.reported_upper is None:
            raise ValueError("reported_lower and reported_upper cannot both be None")

        if not isinstance(self.reported_lower, int) and self.reported_lower is not None:
            raise TypeError("reported_lower must be an int or None")

        if not isinstance(self.reported_upper, int) and self.reported_upper is not None:
            raise TypeError("reported_upper must be an int or None")

        if (
            self.reported_lower is not None
            and self.reported_upper is not None
            and self.reported_upper < self.reported_lower
        ):
            raise ValueError("reported_upper must be >= reported_lower")

        self.lower_bound, self.upper_bound = self._calculate_continuous_bounds()

    def _calculate_continuous_bounds(self) -> tuple[float | None, float | None]:
        if self.reported_lower is None:
            return None, self.reported_upper + 0.5

        if self.reported_upper is None:
            return self.reported_lower - 0.5, None

        return self.reported_lower - 0.5, self.reported_upper + 0.5

    def contains_actual_high(self, actual_high: float) -> bool:
        if self.lower_bound is not None and actual_high <= self.lower_bound:
            return False

        if self.upper_bound is not None and actual_high > self.upper_bound:
            return False

        return True


def validate_buckets(buckets: list[Bucket]) -> None:
    if len(buckets) == 0:
        raise ValueError("Bucket list cannot be empty")

    locations = {bucket.location for bucket in buckets}
    if len(locations) != 1:
        raise ValueError(f"All buckets should have the same location. Found: {locations}")

    previous_upper: float | None = None

    for i, bucket in enumerate(buckets):
        lower = bucket.lower_bound
        upper = bucket.upper_bound

        if lower is not None and upper is not None and upper <= lower:
            raise ValueError(f"Invalid bucket interval for {bucket.name}: upper must be > lower")

        if i == 0:
            if lower is not None:
                raise ValueError("First bucket should be open-ended below")
        else:
            if lower is None:
                raise ValueError(f"Only the first bucket can be open-ended below: {bucket.name}")

            if previous_upper is None:
                raise ValueError("Previous bucket was open-ended above before the final bucket")

            if lower != previous_upper:
                raise ValueError(
                    f"Gap or overlap near {bucket.name}: "
                    f"previous upper={previous_upper}, current lower={lower}"
                )

        if i == len(buckets) - 1:
            if upper is not None:
                raise ValueError("Last bucket should be open-ended above")
        else:
            if upper is None:
                raise ValueError(f"Only the last bucket can be open-ended above: {bucket.name}")

        previous_upper = upper


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


def format_bucket_interval(bucket: Bucket) -> str:
    lower = bucket.lower_bound
    upper = bucket.upper_bound

    if lower is None:
        return f"actual <= {upper:.1f}"

    if upper is None:
        return f"actual > {lower:.1f}"

    return f"{lower:.1f} < actual <= {upper:.1f}"