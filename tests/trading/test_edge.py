from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.trading.edge import (
    EdgeSettings,
    compute_edge_table,
    estimate_kalshi_buy_fee,
)


EVALUATED_AT = datetime(2026, 6, 3, 16, 0, tzinfo=timezone.utc)


def test_estimate_kalshi_buy_fee_includes_fee_and_balance_rounding() -> None:
    assert estimate_kalshi_buy_fee(0.50, contracts=1) == pytest.approx(0.02)
    assert estimate_kalshi_buy_fee(0.50, contracts=100) == pytest.approx(1.75)


def test_compute_edge_table_marks_cost_adjusted_candidate() -> None:
    probabilities = _probabilities(probability=0.70)
    orderbooks = _orderbook_summary(best_yes_ask=0.60, best_no_ask=0.45)
    settings = EdgeSettings(min_edge_dollars=0.01, slippage_buffer_dollars=0.0)

    edge = compute_edge_table(probabilities, orderbooks, settings=settings, evaluated_at=EVALUATED_AT)

    yes = edge[edge["action"] == "BUY_YES"].iloc[0]
    no = edge[edge["action"] == "BUY_NO"].iloc[0]
    assert yes["edge_status"] == "CANDIDATE"
    assert yes["net_edge"] > settings.min_edge_dollars
    assert yes["no_trade_reason"] == ""
    assert no["edge_status"] == "NO_TRADE"
    assert "gross_edge_nonpositive" in no["no_trade_reason"]


def test_compute_edge_table_rejects_edge_that_does_not_survive_costs() -> None:
    probabilities = _probabilities(probability=0.70)
    orderbooks = _orderbook_summary(best_yes_ask=0.69, best_no_ask=0.45)
    settings = EdgeSettings(min_edge_dollars=0.01, slippage_buffer_dollars=0.005)

    edge = compute_edge_table(probabilities, orderbooks, settings=settings, evaluated_at=EVALUATED_AT)

    yes = edge[edge["action"] == "BUY_YES"].iloc[0]
    assert yes["edge_status"] == "NO_TRADE"
    assert "edge_below_minimum" in yes["no_trade_reason"]


def test_compute_edge_table_rejects_stale_or_missing_orderbooks() -> None:
    probabilities = _probabilities(probability=0.70)
    orderbooks = _orderbook_summary(
        best_yes_ask=0.60,
        best_no_ask=0.45,
        staleness_seconds=600,
        orderbook_status="NO_TRADE",
        orderbook_reason="stale_orderbook",
    )

    edge = compute_edge_table(probabilities, orderbooks, evaluated_at=EVALUATED_AT)

    assert set(edge["edge_status"]) == {"NO_TRADE"}
    assert edge["no_trade_reason"].str.contains("stale_orderbook").all()


def test_compute_edge_table_rejects_settlement_no_trade_state() -> None:
    probabilities = _probabilities(probability=0.70)
    probabilities["settlement_status"] = "POST_PEAK_NO_TRADE"
    probabilities["settlement_reason"] = "post_peak_temperature_path_no_verified_settlement"
    probabilities["settlement_trading_allowed"] = False
    probabilities["probability_mode"] = "diagnostic_no_trade"
    orderbooks = _orderbook_summary(best_yes_ask=0.60, best_no_ask=0.45)

    edge = compute_edge_table(probabilities, orderbooks, evaluated_at=EVALUATED_AT)

    assert set(edge["edge_status"]) == {"NO_TRADE"}
    assert edge["no_trade_reason"].str.contains("settlement_state:").all()
    assert set(edge["settlement_status"]) == {"POST_PEAK_NO_TRADE"}


def _probabilities(probability: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "row-1",
                "event_ticker": "KXHIGHNY-26JUN03",
                "ticker": "KXHIGHNY-26JUN03-B80.5",
                "bucket_index": 3,
                "bucket_name": "80 to 81",
                "bucket_lower_temp": 79.5,
                "bucket_upper_temp": 81.5,
                "probability": probability,
                "probability_signal_status": "OK",
                "probability_signal_reason": "",
            }
        ]
    )


def _orderbook_summary(
    *,
    best_yes_ask: float,
    best_no_ask: float,
    staleness_seconds: float = 0.0,
    orderbook_status: str = "OK",
    orderbook_reason: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fetched_at": EVALUATED_AT.isoformat(),
                "evaluated_at": EVALUATED_AT.isoformat(),
                "ticker": "KXHIGHNY-26JUN03-B80.5",
                "staleness_seconds": staleness_seconds,
                "best_yes_bid": 0.55,
                "best_yes_bid_size": 5,
                "best_yes_ask": best_yes_ask,
                "best_yes_ask_size": 10,
                "yes_spread": best_yes_ask - 0.55,
                "best_no_bid": 1.0 - best_yes_ask,
                "best_no_bid_size": 10,
                "best_no_ask": best_no_ask,
                "best_no_ask_size": 8,
                "no_spread": best_no_ask - (1.0 - best_yes_ask),
                "orderbook_status": orderbook_status,
                "orderbook_reason": orderbook_reason,
            }
        ]
    )
