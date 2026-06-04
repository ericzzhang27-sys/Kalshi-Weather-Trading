from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.trading.config import PaperSettings
from src.trading.paper_broker import execute_paper_orders


EXECUTED_AT = datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc)


def test_execute_paper_orders_fills_ready_intent_and_updates_state(tmp_path) -> None:
    intents = _intents()
    orders_path = tmp_path / "paper_orders.csv"
    positions_path = tmp_path / "paper_positions.csv"
    pnl_path = tmp_path / "paper_pnl.csv"

    result = execute_paper_orders(
        intents,
        settings=PaperSettings(starting_cash_dollars=10.0),
        orders_path=orders_path,
        positions_path=positions_path,
        pnl_path=pnl_path,
        executed_at=EXECUTED_AT,
    )

    assert result.filled_count == 1
    assert result.orders.iloc[0]["paper_status"] == "FILLED"
    assert result.positions.iloc[0]["position_contracts"] == 1
    assert result.pnl.iloc[-1]["cash_balance_dollars"] == pytest.approx(9.38)
    assert orders_path.exists()
    assert positions_path.exists()
    assert pnl_path.exists()


def test_execute_paper_orders_is_idempotent_by_client_order_id(tmp_path) -> None:
    intents = _intents()
    kwargs = {
        "settings": PaperSettings(starting_cash_dollars=10.0),
        "orders_path": tmp_path / "paper_orders.csv",
        "positions_path": tmp_path / "paper_positions.csv",
        "pnl_path": tmp_path / "paper_pnl.csv",
        "executed_at": EXECUTED_AT,
    }

    first = execute_paper_orders(intents, **kwargs)
    second = execute_paper_orders(intents, **kwargs)

    assert first.filled_count == 1
    assert second.filled_count == 0
    assert len(second.orders) == 1


def _intents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "generated_at": EXECUTED_AT.isoformat(),
                "client_order_id": "kwt-demo",
                "row_id": "row-1",
                "event_ticker": "KXHIGHNY-26JUN04",
                "ticker": "KXHIGHNY-26JUN04-B82.5",
                "outcome_side": "YES",
                "action": "BUY_YES",
                "intent_status": "READY",
                "contracts": 1,
                "limit_price_dollars": 0.60,
                "estimated_fee_dollars": 0.02,
                "estimated_cost_dollars": 0.62,
            }
        ]
    )
