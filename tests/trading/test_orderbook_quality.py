from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.trading.orderbook import normalize_orderbook
from src.trading.orderbook_quality import (
    build_quality_report,
    check_timestamp_continuity,
    latest_fetched_at_by_ticker,
    validate_orderbook_levels,
    validate_orderbook_summary,
)


FETCHED_AT = datetime(2026, 6, 3, 15, 0, tzinfo=timezone.utc)
TICKER = "KXHIGHNY-26JUN03-T72"


def _clean_levels() -> pd.DataFrame:
    return normalize_orderbook(
        TICKER,
        {
            "orderbook_fp": {
                "yes_dollars": [[0.40, 10], [0.35, 5]],
                "no_dollars": [[0.55, 7], [0.60, 2]],
            }
        },
        FETCHED_AT,
    )


def test_validate_levels_passes_clean_book() -> None:
    violations = validate_orderbook_levels(_clean_levels())
    assert violations == ()


def test_validate_levels_detects_non_monotonic_bid_prices() -> None:
    levels = _clean_levels()
    yes_bids = levels[(levels["outcome_side"] == "YES") & (levels["quote_type"] == "bid")]
    # Swap prices between level 1 and 2 so bids increase with level.
    worse = yes_bids[yes_bids["level"] == 2].index[0]
    better = yes_bids[yes_bids["level"] == 1].index[0]
    levels.loc[worse, "price_dollars"], levels.loc[better, "price_dollars"] = (
        levels.loc[better, "price_dollars"],
        levels.loc[worse, "price_dollars"],
    )

    violations = validate_orderbook_levels(levels)

    checks = {violation.check for violation in violations}
    assert "level_price_monotonicity" in checks
    assert all(violation.severity == "FAIL" for violation in violations)


def test_validate_levels_detects_price_out_of_bounds() -> None:
    levels = _clean_levels()
    levels.loc[levels.index[0], "price_dollars"] = 1.5

    violations = validate_orderbook_levels(levels)

    checks = {violation.check for violation in violations}
    assert "price_out_of_range" in checks


def test_validate_levels_detects_negative_size() -> None:
    levels = _clean_levels()
    levels.loc[levels.index[0], "size_contracts"] = -3.0

    violations = validate_orderbook_levels(levels)

    checks = {violation.check for violation in violations}
    assert "invalid_size" in checks


def test_validate_levels_detects_duplicate_rows() -> None:
    levels = pd.concat([_clean_levels(), _clean_levels()], ignore_index=True)

    violations = validate_orderbook_levels(levels)

    checks = {violation.check for violation in violations}
    assert "duplicate_row" in checks


def test_validate_levels_detects_complementary_sum_violation() -> None:
    levels = _clean_levels()
    # Break the YES/NO complement relationship for one row.
    mask = (levels["outcome_side"] == "NO") & (levels["quote_type"] == "ask")
    idx = levels[mask].index[0]
    levels.loc[idx, "price_dollars"] = round(float(levels.loc[idx, "price_dollars"]) + 0.05, 6)

    violations = validate_orderbook_levels(levels)

    checks = {violation.check for violation in violations}
    assert "complementary_quote_sum" in checks


def test_validate_summary_detects_crossed_book() -> None:
    summary = pd.DataFrame(
        [
            {
                "fetched_at": FETCHED_AT.isoformat(),
                "ticker": TICKER,
                "best_yes_bid": 0.70,
                "best_yes_ask": 0.40,
                "yes_spread": -0.30,
                "yes_midpoint": 0.55,
                "yes_bid_depth": 5.0,
                "yes_ask_depth": 5.0,
            }
        ]
    )

    violations = validate_orderbook_summary(summary)

    checks = {violation.check for violation in violations}
    assert "crossed_book" in checks
    assert "spread_out_of_range" in checks


def test_validate_summary_passes_clean_row() -> None:
    summary = pd.DataFrame(
        [
            {
                "fetched_at": FETCHED_AT.isoformat(),
                "ticker": TICKER,
                "best_yes_bid": 0.40,
                "best_yes_ask": 0.45,
                "yes_spread": 0.05,
                "yes_midpoint": 0.425,
                "yes_bid_depth": 10.0,
                "yes_ask_depth": 7.0,
            }
        ]
    )

    assert validate_orderbook_summary(summary) == ()


def test_check_timestamp_continuity_detects_regression() -> None:
    previous = {TICKER: datetime(2026, 6, 3, 16, 0, tzinfo=timezone.utc).isoformat()}
    levels = _clean_levels()  # fetched at 15:00, older than stored 16:00

    violations = check_timestamp_continuity(previous, levels)

    assert len(violations) == 1
    assert violations[0].check == "timestamp_regression"
    assert violations[0].severity == "FAIL"


def test_check_timestamp_continuity_allows_newer_snapshots() -> None:
    previous = {TICKER: datetime(2026, 6, 3, 14, 0, tzinfo=timezone.utc).isoformat()}
    levels = _clean_levels()

    assert check_timestamp_continuity(previous, levels) == ()


def test_latest_fetched_at_by_ticker_returns_max_per_ticker() -> None:
    levels = _clean_levels()
    latest = latest_fetched_at_by_ticker(levels)
    assert latest == {TICKER: FETCHED_AT.isoformat()}


def test_build_quality_report_escalates_to_fail() -> None:
    from src.trading.orderbook_quality import QualityViolation

    warn_only = build_quality_report(
        [QualityViolation(check="empty", severity="WARN", detail="x")]
    )
    assert warn_only.status == "WARN"
    assert warn_only.ok

    with_fail = build_quality_report(
        [
            QualityViolation(check="empty", severity="WARN", detail="x"),
            QualityViolation(check="bad", severity="FAIL", detail="y"),
        ]
    )
    assert with_fail.status == "FAIL"
    assert not with_fail.ok
    assert with_fail.fail_count == 1