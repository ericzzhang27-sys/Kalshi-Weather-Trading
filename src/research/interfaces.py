from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_hash(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


@dataclass(frozen=True)
class TemperatureBucket:
    """One exhaustive Kalshi outcome using lower-open, upper-closed bounds."""

    label: str
    lower_f: float | None = None
    upper_f: float | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("bucket label cannot be empty")
        if self.lower_f is None and self.upper_f is None:
            raise ValueError("a temperature bucket needs at least one finite bound")
        for name, value in (("lower_f", self.lower_f), ("upper_f", self.upper_f)):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when supplied")
        if self.lower_f is not None and self.upper_f is not None:
            if float(self.lower_f) >= float(self.upper_f):
                raise ValueError("bucket lower bound must be below its upper bound")

    def contains(self, temperature_f: float) -> bool:
        value = float(temperature_f)
        return (self.lower_f is None or value > self.lower_f) and (
            self.upper_f is None or value <= self.upper_f
        )


def validate_bucket_schema(buckets: Sequence[TemperatureBucket]) -> tuple[TemperatureBucket, ...]:
    result = tuple(buckets)
    if len(result) < 2:
        raise ValueError("an event bucket schema must contain at least two outcomes")
    if len({item.label for item in result}) != len(result):
        raise ValueError("bucket labels must be unique")
    ordered = sorted(result, key=lambda item: float("-inf") if item.lower_f is None else item.lower_f)
    if ordered[0].lower_f is not None or ordered[-1].upper_f is not None:
        raise ValueError("bucket schema must include open lower and upper tails")
    for left, right in zip(ordered, ordered[1:]):
        if left.upper_f != right.lower_f:
            raise ValueError("bucket schema must be ordered, contiguous, and exhaustive")
    return tuple(ordered)


@dataclass(frozen=True)
class ForecastRequest:
    target_date: date
    as_of_utc: datetime
    location: str
    buckets: tuple[TemperatureBucket, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of_utc", _utc(self.as_of_utc))
        object.__setattr__(self, "buckets", validate_bucket_schema(self.buckets))
        if self.location.upper() != "NYC":
            raise ValueError("the current research scope is NYC only")
        if self.as_of_utc.date() > self.target_date:
            raise ValueError("forecast request cannot be created after the target date")


def _probability_vector(name: str, values: Sequence[float], size: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} has {len(result)} values but the schema has {size} buckets")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in result):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    if not math.isclose(sum(result), 1.0, abs_tol=1e-8):
        raise ValueError(f"{name} must sum to one, got {sum(result):.12g}")
    return result


@dataclass(frozen=True)
class ForecastDistribution:
    request: ForecastRequest
    integer_support_f: tuple[int, ...]
    p_weather: tuple[float, ...]
    p_market: tuple[float, ...]
    p_fair: tuple[float, ...]
    p_lower: tuple[float, ...]
    p_upper: tuple[float, ...]
    weather_model_version: str
    calibration_version: str
    market_pool_version: str
    feature_snapshot_hash: str

    def __post_init__(self) -> None:
        size = len(self.request.buckets)
        for name in ("p_weather", "p_market", "p_fair"):
            object.__setattr__(self, name, _probability_vector(name, getattr(self, name), size))
        lower = tuple(float(value) for value in self.p_lower)
        upper = tuple(float(value) for value in self.p_upper)
        if len(lower) != size or len(upper) != size:
            raise ValueError("probability uncertainty vectors must match the bucket schema")
        if any(not 0.0 <= lo <= hi <= 1.0 for lo, hi in zip(lower, upper)):
            raise ValueError("each uncertainty interval must satisfy 0 <= lower <= upper <= 1")
        if any(not lo - 1e-12 <= p <= hi + 1e-12 for lo, p, hi in zip(lower, self.p_fair, upper)):
            raise ValueError("fair probabilities must lie within their uncertainty intervals")
        object.__setattr__(self, "p_lower", lower)
        object.__setattr__(self, "p_upper", upper)
        support = tuple(int(value) for value in self.integer_support_f)
        if not support or support != tuple(sorted(set(support))):
            raise ValueError("integer temperature support must be nonempty, sorted, and unique")
        object.__setattr__(self, "integer_support_f", support)
        if len(self.feature_snapshot_hash) != 64:
            raise ValueError("feature_snapshot_hash must be a SHA-256 hex digest")


@dataclass(frozen=True)
class FixedPointLevel:
    """Exact fixed-point price/quantity pair from the 2026 Kalshi API."""

    price_dollars: Decimal
    quantity: Decimal

    @classmethod
    def parse(cls, price_dollars: str | int | float | Decimal, quantity: str | int | float | Decimal) -> "FixedPointLevel":
        try:
            price = Decimal(str(price_dollars))
            count = Decimal(str(quantity))
        except InvalidOperation as exc:
            raise ValueError("invalid fixed-point order-book value") from exc
        if not price.is_finite() or price < 0 or price > 1:
            raise ValueError("fixed-point price must be in [0, 1]")
        if not count.is_finite() or count < 0:
            raise ValueError("fixed-point quantity must be nonnegative")
        return cls(price_dollars=price, quantity=count)


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    event_ticker: str
    timestamp_ms: int
    sequence_number: int
    yes_bids: tuple[FixedPointLevel, ...]
    no_bids: tuple[FixedPointLevel, ...]
    lifecycle_state: str
    raw_payload_hash: str

    def __post_init__(self) -> None:
        if self.timestamp_ms <= 0 or self.sequence_number < 0:
            raise ValueError("snapshot timestamp and sequence number must be nonnegative")
        if len(self.raw_payload_hash) != 64:
            raise ValueError("raw_payload_hash must be a SHA-256 hex digest")
        if not self.ticker or not self.event_ticker:
            raise ValueError("snapshot tickers cannot be empty")


@dataclass(frozen=True)
class TradeDecision:
    action: Literal["TRADE", "NO_TRADE"]
    side: Literal["YES", "NO"] | None
    limit_price: Decimal | None
    contracts: int
    point_edge: float
    lower_confidence_edge: float
    estimated_costs: float
    reason_codes: tuple[str, ...]
    model_version: str
    risk_state: str
    mode: Literal["shadow"] = "shadow"

    def __post_init__(self) -> None:
        if self.mode != "shadow":
            raise ValueError("research decisions are shadow-only")
        if self.action == "TRADE" and (self.side is None or self.limit_price is None or self.contracts < 1):
            raise ValueError("trade decisions require a side, limit price, and whole contract")
        if self.action == "NO_TRADE" and self.contracts != 0:
            raise ValueError("no-trade decisions must have zero contracts")


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    created_at_utc: datetime
    hypothesis: str
    family: str
    seed: int
    data_hash: str
    model_hash: str
    config_hash: str
    package_hash: str
    folds: tuple[Mapping[str, Any], ...]
    trial_count: int
    elapsed_seconds: float
    probability_metrics: Mapping[str, Any]
    trading_metrics: Mapping[str, Any]
    robustness_tests: Mapping[str, Any]
    promotion_decision: str
    evidence_label: str
    status: Literal["succeeded", "failed"] = "succeeded"
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at_utc", _utc(self.created_at_utc))
        for name in ("data_hash", "model_hash", "config_hash", "package_hash"):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if self.trial_count < 0 or self.elapsed_seconds < 0:
            raise ValueError("trial_count and elapsed_seconds must be nonnegative")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed experiments require a failure reason")

    def as_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at_utc"] = self.created_at_utc.isoformat()
        return payload

    @property
    def record_hash(self) -> str:
        return canonical_json_hash(self.as_json_dict())
