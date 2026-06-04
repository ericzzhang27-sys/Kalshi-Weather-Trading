from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.trading.order_intents import build_order_intents, ready_order_intents


GENERATED_AT = datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc)


def test_build_order_intents_marks_ready_and_blocked_rows() -> None:
    decisions = pd.DataFrame(
        [
            {
                "evaluated_at": GENERATED_AT.isoformat(),
                "row_id": "row-1",
                "event_ticker": "KXHIGHNY-26JUN04",
                "ticker": "KXHIGHNY-26JUN04-B82.5",
                "outcome_side": "YES",
                "action": "BUY_YES",
                "risk_status": "APPROVED",
                "risk_reason": "",
                "proposed_contracts": 1,
                "executable_price": 0.60,
                "estimated_fee_dollars": 0.02,
                "estimated_cost_dollars": 0.62,
                "net_edge": 0.04,
            },
            {
                "evaluated_at": GENERATED_AT.isoformat(),
                "row_id": "row-2",
                "event_ticker": "KXHIGHNY-26JUN04",
                "ticker": "KXHIGHNY-26JUN04-B84.5",
                "outcome_side": "NO",
                "action": "BUY_NO",
                "risk_status": "NO_TRADE",
                "risk_reason": "insufficient_cash_reserve",
            },
        ]
    )

    intents = build_order_intents(decisions, generated_at=GENERATED_AT)
    second = build_order_intents(decisions, generated_at=GENERATED_AT)

    assert list(intents["intent_status"]) == ["READY", "BLOCKED"]
    assert intents.iloc[0]["client_order_id"].startswith("kwt-")
    assert intents.iloc[0]["client_order_id"] == second.iloc[0]["client_order_id"]
    assert intents.iloc[0]["v2_side"] == "bid"
    assert intents.iloc[0]["v2_price_dollars"] == 0.60
    assert ready_order_intents(intents).iloc[0]["contracts"] == 1


def test_build_order_intents_converts_buy_no_to_v2_yes_book_ask() -> None:
    decisions = pd.DataFrame(
        [
            {
                "evaluated_at": GENERATED_AT.isoformat(),
                "row_id": "row-1",
                "event_ticker": "KXHIGHNY-26JUN04",
                "ticker": "KXHIGHNY-26JUN04-B82.5",
                "outcome_side": "NO",
                "action": "BUY_NO",
                "risk_status": "APPROVED",
                "risk_reason": "",
                "proposed_contracts": 1,
                "executable_price": 0.35,
                "estimated_fee_dollars": 0.02,
                "estimated_cost_dollars": 0.37,
                "net_edge": 0.04,
            }
        ]
    )

    intent = build_order_intents(decisions, generated_at=GENERATED_AT).iloc[0]

    assert intent["legacy_side"] == "no"
    assert intent["legacy_no_price_dollars"] == 0.35
    assert intent["v2_side"] == "ask"
    assert intent["v2_price_dollars"] == 0.65
