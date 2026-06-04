from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.trading.portfolio import (
    normalize_balance_payload,
    normalize_orders_payload,
    normalize_positions_payload,
)


FETCHED_AT = datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc)


def test_normalize_balance_payload_converts_kalshi_cents() -> None:
    balance = normalize_balance_payload(
        {"balance": 12345, "portfolio_value": 15000, "updated_ts": 123},
        FETCHED_AT,
        source="kalshi",
    )

    assert balance.iloc[0]["cash_balance_dollars"] == pytest.approx(123.45)
    assert balance.iloc[0]["portfolio_value_dollars"] == pytest.approx(150.0)


def test_normalize_positions_payload_keeps_exposure_context() -> None:
    positions = normalize_positions_payload(
        {
            "market_positions": [
                {
                    "ticker": "KXHIGHNY-26JUN04-B82.5",
                    "position_fp": "2.00",
                    "market_exposure_dollars": "1.20",
                    "realized_pnl_dollars": "-0.10",
                    "fees_paid_dollars": "0.02",
                    "resting_orders_count": 1,
                }
            ]
        },
        FETCHED_AT,
        source="kalshi",
    )

    row = positions.iloc[0]
    assert row["event_ticker"] == "KXHIGHNY-26JUN04"
    assert row["position_contracts"] == pytest.approx(2.0)
    assert row["market_exposure_dollars"] == pytest.approx(1.20)


def test_normalize_orders_payload_filters_closed_orders() -> None:
    orders = normalize_orders_payload(
        {
            "orders": [
                {
                    "order_id": "open-1",
                    "client_order_id": "client-1",
                    "ticker": "KXHIGHNY-26JUN04-B82.5",
                    "outcome_side": "yes",
                    "action": "buy",
                    "status": "resting",
                    "yes_price_dollars": "0.56",
                    "remaining_count_fp": "3.00",
                    "initial_count_fp": "3.00",
                },
                {
                    "order_id": "filled-1",
                    "ticker": "KXHIGHNY-26JUN04-B84.5",
                    "status": "filled",
                    "remaining_count_fp": "0.00",
                },
            ]
        },
        FETCHED_AT,
        source="kalshi",
    )

    assert len(orders) == 1
    assert orders.iloc[0]["order_exposure_dollars"] == pytest.approx(1.68)
