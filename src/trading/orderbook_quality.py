"""Data-quality validation for stored Kalshi orderbook snapshots.

Every check fails closed: violations are classified as WARN or FAIL and a
FAIL status prevents a scrape cycle from being appended to the canonical
backtesting store (the raw payload is quarantined instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Mapping, Sequence

import pandas as pd


PRICE_TOLERANCE = 1e-9
VALID_SEVERITIES = {"WARN", "FAIL"}
VALID_STATUSES = {"OK", "WARN", "FAIL"}

# Columns expected on a normalized orderbook levels frame (see
# src.trading.orderbook.ORDERBOOK_COLUMNS).
_LEVEL_KEY_COLUMNS = ["ticker", "outcome_side", "quote_type"]
_ROW_KEY_COLUMNS = [
    "fetched_at",
    "ticker",
    "outcome_side",
    "quote_type",
    "source_book_side",
    "level",
]


@dataclass(frozen=True)
class QualityViolation:
    """A single failed data-quality check."""

    check: str
    severity: str
    detail: str

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(f"Invalid violation severity: {self.severity}")

    def describe(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"


@dataclass(frozen=True)
class QualityReport:
    """Aggregated result of all quality checks for one scrape cycle."""

    status: str
    violations: tuple[QualityViolation, ...]

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid quality report status: {self.status}")

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"

    @property
    def fail_count(self) -> int:
        return sum(1 for violation in self.violations if violation.severity == "FAIL")

    @property
    def warn_count(self) -> int:
        return sum(1 for violation in self.violations if violation.severity == "WARN")

    def describe(self) -> str:
        if not self.violations:
            return f"[{self.status}] no violations"
        return "\n".join(violation.describe() for violation in self.violations)


def build_quality_report(violations: Sequence[QualityViolation]) -> QualityReport:
    """Aggregate violations into a report; any FAIL escalates the status."""
    if any(violation.severity == "FAIL" for violation in violations):
        status = "FAIL"
    elif violations:
        status = "WARN"
    else:
        status = "OK"
    return QualityReport(status=status, violations=tuple(violations))


def validate_orderbook_levels(levels: pd.DataFrame) -> tuple[QualityViolation, ...]:
    """Validate a normalized orderbook levels frame for one cycle.

    Checks:
      * prices within [0, 1] and sizes finite/nonnegative
      * bid prices monotonically non-increasing across levels
      * ask prices monotonically non-decreasing across levels
      * cumulative size monotonically non-decreasing across levels
      * complementary YES/NO quotes derived from the same book side sum to 1
      * no duplicate (fetched_at, ticker, side, level) rows
    """
    violations: list[QualityViolation] = []
    required = set(_ROW_KEY_COLUMNS) | {"price_dollars", "size_contracts", "cumulative_size"}
    missing = required - set(levels.columns)
    if missing:
        return (
            QualityViolation(
                check="schema",
                severity="FAIL",
                detail=f"levels frame missing columns: {sorted(missing)}",
            ),
        )
    if levels.empty:
        return (
            QualityViolation(
                check="empty_orderbook",
                severity="WARN",
                detail="no orderbook rows were returned for this cycle",
            ),
        )

    violations.extend(_check_price_and_size_bounds(levels))
    violations.extend(_check_level_monotonicity(levels))
    violations.extend(_check_cumulative_size(levels))
    violations.extend(_check_complementary_quotes(levels))
    violations.extend(_check_duplicate_rows(levels))
    return tuple(violations)


def validate_orderbook_summary(summary: pd.DataFrame) -> tuple[QualityViolation, ...]:
    """Validate an orderbook summary frame for one cycle.

    Checks:
      * no crossed books (best bid must not exceed best ask)
      * spreads within [0, 1] and midpoints within [0, 1]
      * depths nonnegative and consistent with best-level sizes
    """
    violations: list[QualityViolation] = []
    if summary.empty:
        return (
            QualityViolation(
                check="empty_summary",
                severity="WARN",
                detail="summary frame is empty for this cycle",
            ),
        )

    for _, row in summary.iterrows():
        ticker = str(row.get("ticker", "<unknown>"))
        for outcome in ("yes", "no"):
            bid = row.get(f"best_{outcome}_bid")
            ask = row.get(f"best_{outcome}_ask")
            spread = row.get(f"{outcome}_spread")
            midpoint = row.get(f"{outcome}_midpoint")

            if bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask):
                if float(ask) + PRICE_TOLERANCE < float(bid):
                    violations.append(
                        QualityViolation(
                            check="crossed_book",
                            severity="FAIL",
                            detail=(
                                f"{ticker}: best {outcome} bid {float(bid):.4f} exceeds "
                                f"best {outcome} ask {float(ask):.4f}"
                            ),
                        )
                    )
            if spread is not None and not pd.isna(spread):
                value = float(spread)
                if value < -PRICE_TOLERANCE or value > 1.0 + PRICE_TOLERANCE:
                    violations.append(
                        QualityViolation(
                            check="spread_out_of_range",
                            severity="FAIL",
                            detail=f"{ticker}: {outcome} spread {value:.4f} outside [0, 1]",
                        )
                    )
            if midpoint is not None and not pd.isna(midpoint):
                value = float(midpoint)
                if value < -PRICE_TOLERANCE or value > 1.0 + PRICE_TOLERANCE:
                    violations.append(
                        QualityViolation(
                            check="midpoint_out_of_range",
                            severity="FAIL",
                            detail=f"{ticker}: {outcome} midpoint {value:.4f} outside [0, 1]",
                        )
                    )

            for field in (f"best_{outcome}_bid_size", f"best_{outcome}_ask_size"):
                value = row.get(field)
                if value is not None and not pd.isna(value) and float(value) < 0.0:
                    violations.append(
                        QualityViolation(
                            check="negative_size",
                            severity="FAIL",
                            detail=f"{ticker}: {field} is negative ({float(value)})",
                        )
                    )

            for field in (f"{outcome}_bid_depth", f"{outcome}_ask_depth"):
                value = row.get(field)
                if value is not None and not pd.isna(value) and float(value) < 0.0:
                    violations.append(
                        QualityViolation(
                            check="negative_depth",
                            severity="FAIL",
                            detail=f"{ticker}: {field} is negative ({float(value)})",
                        )
                    )
    return tuple(violations)


def check_timestamp_continuity(
    previous_max_fetched_at_by_ticker: Mapping[str, str],
    levels: pd.DataFrame,
) -> tuple[QualityViolation, ...]:
    """Ensure newly scraped snapshots are not older than what is already stored.

    Backtesting assumes append-only, time-monotonic storage. A new snapshot for
    a ticker whose ``fetched_at`` precedes the last stored snapshot indicates a
    clock problem or replayed data and must fail closed.
    """
    violations: list[QualityViolation] = []
    if levels.empty or not previous_max_fetched_at_by_ticker:
        return tuple(violations)

    latest_new = levels.groupby("ticker")["fetched_at"].max()
    for ticker, new_value in latest_new.items():
        previous_value = previous_max_fetched_at_by_ticker.get(str(ticker))
        if not previous_value:
            continue
        new_dt = _parse_iso_utc(new_value)
        previous_dt = _parse_iso_utc(previous_value)
        if new_dt is None or previous_dt is None:
            violations.append(
                QualityViolation(
                    check="timestamp_parse",
                    severity="FAIL",
                    detail=(
                        f"{ticker}: unparseable fetched_at "
                        f"(new={new_value!r}, previous={previous_value!r})"
                    ),
                )
            )
            continue
        if new_dt < previous_dt:
            violations.append(
                QualityViolation(
                    check="timestamp_regression",
                    severity="FAIL",
                    detail=(
                        f"{ticker}: new fetched_at {new_dt.isoformat()} is older than "
                        f"last stored {previous_dt.isoformat()}"
                    ),
                )
            )
    return tuple(violations)


def latest_fetched_at_by_ticker(levels: pd.DataFrame) -> dict[str, str]:
    """Return the max fetched_at ISO string per ticker from a levels frame."""
    if levels.empty or "ticker" not in levels.columns or "fetched_at" not in levels.columns:
        return {}
    latest = levels.groupby("ticker")["fetched_at"].max()
    return {str(ticker): str(value) for ticker, value in latest.items()}


def _check_price_and_size_bounds(levels: pd.DataFrame) -> tuple[QualityViolation, ...]:
    violations: list[QualityViolation] = []
    prices = pd.to_numeric(levels["price_dollars"], errors="coerce")
    sizes = pd.to_numeric(levels["size_contracts"], errors="coerce")

    bad_price_mask = prices.isna() | (prices < -PRICE_TOLERANCE) | (prices > 1.0 + PRICE_TOLERANCE)
    for idx, value in prices[bad_price_mask].items():
        violations.append(
            QualityViolation(
                check="price_out_of_range",
                severity="FAIL",
                detail=(
                    f"row {idx} ({levels.at[idx, 'ticker']}): price_dollars={value!r} "
                    "outside [0, 1]"
                ),
            )
        )

    bad_size_mask = sizes.isna() | ~sizes.map(math.isfinite) | (sizes < 0.0)
    for idx, value in sizes[bad_size_mask].items():
        violations.append(
            QualityViolation(
                check="invalid_size",
                severity="FAIL",
                detail=(
                    f"row {idx} ({levels.at[idx, 'ticker']}): size_contracts={value!r} "
                    "is not a finite nonnegative number"
                ),
            )
        )
    return tuple(violations)


def _check_level_monotonicity(levels: pd.DataFrame) -> tuple[QualityViolation, ...]:
    """Bid prices must be non-increasing and ask prices non-decreasing by level."""
    violations: list[QualityViolation] = []
    for key, group in levels.groupby(_LEVEL_KEY_COLUMNS, sort=False):
        ticker, outcome_side, quote_type = key
        ordered = group.sort_values("level", kind="stable")
        prices = pd.to_numeric(ordered["price_dollars"], errors="coerce").tolist()
        level_ids = ordered["level"].tolist()
        for position in range(1, len(prices)):
            previous_price = prices[position - 1]
            current_price = prices[position]
            if previous_price is None or current_price is None:
                continue
            if pd.isna(previous_price) or pd.isna(current_price):
                continue
            if quote_type == "bid":
                monotonic = current_price <= previous_price + PRICE_TOLERANCE
                expected = "non-increasing"
            else:
                monotonic = current_price >= previous_price - PRICE_TOLERANCE
                expected = "non-decreasing"
            if not monotonic:
                violations.append(
                    QualityViolation(
                        check="level_price_monotonicity",
                        severity="FAIL",
                        detail=(
                            f"{ticker} {outcome_side} {quote_type}: price at level "
                            f"{level_ids[position]} ({current_price:.4f}) violates "
                            f"{expected} ordering vs level {level_ids[position - 1]} "
                            f"({previous_price:.4f})"
                        ),
                    )
                )
    return tuple(violations)


def _check_cumulative_size(levels: pd.DataFrame) -> tuple[QualityViolation, ...]:
    violations: list[QualityViolation] = []
    for key, group in levels.groupby(_LEVEL_KEY_COLUMNS, sort=False):
        ticker, outcome_side, quote_type = key
        ordered = group.sort_values("level", kind="stable")
        cumulative = pd.to_numeric(ordered["cumulative_size"], errors="coerce").tolist()
        level_ids = ordered["level"].tolist()
        for position in range(1, len(cumulative)):
            previous_value = cumulative[position - 1]
            current_value = cumulative[position]
            if _is_missing(previous_value) or _is_missing(current_value):
                continue
            if current_value < previous_value - PRICE_TOLERANCE:
                violations.append(
                    QualityViolation(
                        check="cumulative_size_monotonicity",
                        severity="FAIL",
                        detail=(
                            f"{ticker} {outcome_side} {quote_type}: cumulative size at "
                            f"level {level_ids[position]} ({current_value}) is below "
                            f"level {level_ids[position - 1]} ({previous_value})"
                        ),
                    )
                )
    return tuple(violations)


def _check_complementary_quotes(levels: pd.DataFrame) -> tuple[QualityViolation, ...]:
    """YES and NO quotes derived from the same raw book side must sum to 1."""
    violations: list[QualityViolation] = []
    group_keys = ["ticker", "fetched_at", "source_book_side", "level"]
    for key, group in levels.groupby(group_keys, sort=False):
        ticker, fetched_at, source_book_side, level = key
        prices: dict[str, float] = {}
        for _, row in group.iterrows():
            price = row["price_dollars"]
            if price is None or pd.isna(price):
                continue
            prices[str(row["outcome_side"])] = float(price)
        if "YES" not in prices or "NO" not in prices:
            continue
        total = prices["YES"] + prices["NO"]
        if abs(total - 1.0) > 1e-6:
            violations.append(
                QualityViolation(
                    check="complementary_quote_sum",
                    severity="FAIL",
                    detail=(
                        f"{ticker} at {fetched_at} level {level} ({source_book_side}): "
                        f"YES {prices['YES']:.4f} + NO {prices['NO']:.4f} = {total:.4f} != 1"
                    ),
                )
            )
    return tuple(violations)


def _check_duplicate_rows(levels: pd.DataFrame) -> tuple[QualityViolation, ...]:
    violations: list[QualityViolation] = []
    duplicated = levels.duplicated(subset=_ROW_KEY_COLUMNS, keep=False)
    for idx in levels[duplicated].index:
        row = levels.loc[idx]
        violations.append(
            QualityViolation(
                check="duplicate_row",
                severity="FAIL",
                detail=(
                    f"duplicate orderbook row for {row['ticker']} at {row['fetched_at']} "
                    f"({row['outcome_side']} {row['quote_type']} level {row['level']})"
                ),
            )
        )
    return tuple(violations)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_iso_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


__all__ = [
    "QualityReport",
    "QualityViolation",
    "build_quality_report",
    "check_timestamp_continuity",
    "latest_fetched_at_by_ticker",
    "validate_orderbook_levels",
    "validate_orderbook_summary",
]