from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.trading.config import RiskSettings
from src.trading.portfolio import empty_portfolio_snapshot
from src.trading.risk import evaluate_risk


EVALUATED_AT = datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc)


def test_evaluate_risk_approves_candidate_with_cash_and_limits(tmp_path) -> None:
    settings = RiskSettings(kill_switch_path=tmp_path / "KILL")
    portfolio = empty_portfolio_snapshot(
        fetched_at=EVALUATED_AT,
        source="paper",
        cash_balance_dollars=100.0,
    )

    decisions = evaluate_risk(_edge_table(), portfolio, settings, evaluated_at=EVALUATED_AT)

    approved = decisions[decisions["risk_status"] == "APPROVED"]
    blocked = decisions[decisions["risk_status"] == "NO_TRADE"]
    assert len(approved) == 1
    assert approved.iloc[0]["proposed_contracts"] == 1
    assert blocked.iloc[0]["risk_reason"].startswith("edge:")


def test_evaluate_risk_blocks_when_kill_switch_exists(tmp_path) -> None:
    kill = tmp_path / "KILL"
    kill.write_text("stop", encoding="utf-8")
    settings = RiskSettings(kill_switch_path=kill)
    portfolio = empty_portfolio_snapshot(
        fetched_at=EVALUATED_AT,
        source="paper",
        cash_balance_dollars=100.0,
    )

    decisions = evaluate_risk(_edge_table().head(1), portfolio, settings, evaluated_at=EVALUATED_AT)

    assert decisions.iloc[0]["risk_status"] == "NO_TRADE"
    assert "kill_switch_active" in decisions.iloc[0]["risk_reason"]


def test_evaluate_risk_reserves_cash_across_approved_rows(tmp_path) -> None:
    settings = RiskSettings(
        kill_switch_path=tmp_path / "KILL",
        max_dollars_per_order=2.0,
        min_cash_reserve_dollars=0.0,
    )
    portfolio = empty_portfolio_snapshot(
        fetched_at=EVALUATED_AT,
        source="paper",
        cash_balance_dollars=1.0,
    )
    edge = pd.concat([_edge_table().head(1), _edge_table().head(1)], ignore_index=True)
    edge.loc[1, "ticker"] = "KXHIGHNY-26JUN04-B84.5"
    edge.loc[1, "row_id"] = "row-2"

    decisions = evaluate_risk(edge, portfolio, settings, evaluated_at=EVALUATED_AT)

    assert list(decisions["risk_status"]) == ["APPROVED", "NO_TRADE"]
    assert "insufficient_cash_reserve" in decisions.iloc[1]["risk_reason"]


def _edge_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evaluated_at": EVALUATED_AT.isoformat(),
                "row_id": "row-1",
                "event_ticker": "KXHIGHNY-26JUN04",
                "ticker": "KXHIGHNY-26JUN04-B82.5",
                "outcome_side": "YES",
                "action": "BUY_YES",
                "edge_status": "CANDIDATE",
                "no_trade_reason": "",
                "net_edge": 0.05,
                "executable_price": 0.60,
                "executable_size": 5,
                "fee_per_contract": 0.02,
            },
            {
                "evaluated_at": EVALUATED_AT.isoformat(),
                "row_id": "row-1",
                "event_ticker": "KXHIGHNY-26JUN04",
                "ticker": "KXHIGHNY-26JUN04-B82.5",
                "outcome_side": "NO",
                "action": "BUY_NO",
                "edge_status": "NO_TRADE",
                "no_trade_reason": "gross_edge_nonpositive",
                "net_edge": -0.10,
                "executable_price": 0.50,
                "executable_size": 5,
                "fee_per_contract": 0.02,
            },
        ]
    )
