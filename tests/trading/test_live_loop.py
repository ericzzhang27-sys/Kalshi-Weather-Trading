from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.trading.config import parse_trading_config
from src.trading.dashboard_data import DashboardState
from src.trading.live_loop import run_trading_cycle


EVALUATED_AT = datetime(2026, 6, 4, 16, 0, tzinfo=timezone.utc)


def test_run_trading_cycle_writes_day4_artifacts_with_paper_broker(tmp_path, monkeypatch) -> None:
    config = parse_trading_config(
        {
            "paper": {"starting_cash_dollars": 10.0},
            "risk": {"kill_switch_path": str(tmp_path / "KILL")},
            "outputs": {
                "portfolio_snapshot_path": str(tmp_path / "portfolio_snapshot.csv"),
                "risk_decisions_path": str(tmp_path / "risk_decisions.csv"),
                "order_intents_path": str(tmp_path / "order_intents.csv"),
                "paper_orders_path": str(tmp_path / "paper_orders.csv"),
                "paper_positions_path": str(tmp_path / "paper_positions.csv"),
                "paper_pnl_path": str(tmp_path / "paper_pnl.csv"),
                "trading_cycle_log_path": str(tmp_path / "trading_cycle_log.csv"),
            },
        }
    )

    monkeypatch.setattr(
        "src.trading.live_loop.load_dashboard_state",
        lambda *args, **kwargs: _dashboard_state(),
    )

    result = run_trading_cycle(config, paper_enabled=True)

    assert len(result.risk_decisions) == 1
    assert result.risk_decisions.iloc[0]["risk_status"] == "APPROVED"
    assert result.order_intents.iloc[0]["intent_status"] == "READY"
    assert result.paper_result is not None
    assert result.paper_result.filled_count == 1
    assert config.outputs.risk_decisions_path.exists()
    assert config.outputs.order_intents_path.exists()
    assert config.outputs.paper_orders_path.exists()
    assert config.outputs.trading_cycle_log_path.exists()


def _dashboard_state() -> DashboardState:
    return DashboardState(
        status={
            "event_ticker": "KXHIGHNY-26JUN04",
            "target_date": "2026-06-04",
            "dashboard_status": "OK",
            "settlement_status": "PRE_PEAK_FORECAST",
            "settlement_trading_allowed": True,
            "probability_mode": "ngboost_forecast",
            "probability_rows": 1,
            "warnings": [],
        },
        market_discovery=pd.DataFrame(),
        mapping=pd.DataFrame(),
        live_weather=pd.DataFrame(),
        live_feature_rows=pd.DataFrame(),
        feature_freshness=pd.DataFrame(),
        bucket_probabilities=pd.DataFrame(),
        distribution_params=pd.DataFrame(),
        settlement_state=pd.DataFrame(),
        orderbook=pd.DataFrame(),
        orderbook_summary=pd.DataFrame(),
        edge_table=pd.DataFrame(
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
                }
            ]
        ),
        bucket_board=pd.DataFrame(),
    )
