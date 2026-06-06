from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.trading.config import TradingConfig
from src.trading.dashboard_data import DashboardState, load_dashboard_state
from src.trading.kalshi_client import KalshiClient
from src.trading.order_intents import build_order_intents, save_order_intents
from src.trading.paper_broker import PaperExecutionResult, execute_paper_orders
from src.trading.portfolio import (
    PortfolioSnapshot,
    fetch_portfolio_snapshot,
    portfolio_from_paper_state,
    save_portfolio_snapshot,
)
from src.trading.risk import evaluate_risk, save_risk_decisions


TRADING_CYCLE_LOG_COLUMNS = [
    "cycle_started_at",
    "cycle_completed_at",
    "mode",
    "paper_enabled",
    "event_ticker",
    "target_date",
    "dashboard_status",
    "settlement_status",
    "settlement_trading_allowed",
    "probability_mode",
    "probability_rows",
    "edge_rows",
    "candidate_edges",
    "risk_rows",
    "risk_approved",
    "ready_intents",
    "paper_new_orders",
    "paper_filled",
    "paper_rejected",
    "status",
    "warnings",
    "portfolio_snapshot_path",
    "risk_decisions_path",
    "order_intents_path",
    "paper_orders_path",
    "paper_positions_path",
    "paper_pnl_path",
]


@dataclass(frozen=True)
class TradingCycleResult:
    dashboard_state: DashboardState
    portfolio: PortfolioSnapshot
    risk_decisions: pd.DataFrame
    order_intents: pd.DataFrame
    paper_result: PaperExecutionResult | None
    cycle_log: pd.DataFrame


def run_trading_cycle(
    config: TradingConfig,
    *,
    event_ticker: str | None = None,
    target_date: date | str | None = None,
    depth: int = 20,
    kalshi_client: KalshiClient | None = None,
    weather_client: Any | None = None,
    prediction_time: datetime | None = None,
    auth_market_data: bool = False,
    auth_orderbooks: bool = False,
    auth_portfolio: bool = False,
    paper_enabled: bool | None = None,
    write_outputs: bool = True,
) -> TradingCycleResult:
    """
    Run one fail-closed live-data cycle through risk, intents, and paper broker.
    """
    started_at = datetime.now(timezone.utc)
    state = load_dashboard_state(
        config,
        event_ticker=event_ticker,
        target_date=target_date,
        depth=depth,
        kalshi_client=kalshi_client,
        weather_client=weather_client,
        prediction_time=prediction_time,
        auth_market_data=auth_market_data,
        auth_orderbooks=auth_orderbooks,
        write_outputs=write_outputs,
    )
    client = kalshi_client or KalshiClient(
        base_url=config.kalshi.base_url,
        timeout_seconds=config.kalshi.request_timeout_seconds,
        max_retries=config.kalshi.max_retries,
        retry_backoff_seconds=config.kalshi.retry_backoff_seconds,
    )
    portfolio = (
        fetch_portfolio_snapshot(client, auth=True, fetched_at=started_at)
        if auth_portfolio
        else portfolio_from_paper_state(
            positions_path=config.outputs.paper_positions_path,
            orders_path=config.outputs.paper_orders_path,
            pnl_path=config.outputs.paper_pnl_path,
            starting_cash_dollars=config.paper.starting_cash_dollars,
            fetched_at=started_at,
        )
    )
    risk_decisions = evaluate_risk(
        state.edge_table,
        portfolio,
        config.risk,
        evaluated_at=started_at,
    )
    order_intents = build_order_intents(risk_decisions, generated_at=started_at)

    run_paper = config.mode == "paper" if paper_enabled is None else bool(paper_enabled)
    paper_result = None
    if run_paper:
        paper_result = execute_paper_orders(
            order_intents,
            settings=config.paper,
            orders_path=config.outputs.paper_orders_path,
            positions_path=config.outputs.paper_positions_path,
            pnl_path=config.outputs.paper_pnl_path,
            executed_at=started_at,
        )

    completed_at = datetime.now(timezone.utc)
    if write_outputs:
        save_portfolio_snapshot(portfolio, config.outputs.portfolio_snapshot_path)
        save_risk_decisions(risk_decisions, config.outputs.risk_decisions_path)
        save_order_intents(order_intents, config.outputs.order_intents_path)
    cycle_log = append_trading_cycle_log(
        config=config,
        state=state,
        risk_decisions=risk_decisions,
        order_intents=order_intents,
        paper_result=paper_result,
        paper_enabled=run_paper,
        started_at=started_at,
        completed_at=completed_at,
        output_path=config.outputs.trading_cycle_log_path if write_outputs else None,
    )
    return TradingCycleResult(
        dashboard_state=state,
        portfolio=portfolio,
        risk_decisions=risk_decisions,
        order_intents=order_intents,
        paper_result=paper_result,
        cycle_log=cycle_log,
    )


def append_trading_cycle_log(
    *,
    config: TradingConfig,
    state: DashboardState,
    risk_decisions: pd.DataFrame,
    order_intents: pd.DataFrame,
    paper_result: PaperExecutionResult | None,
    paper_enabled: bool,
    started_at: datetime,
    completed_at: datetime,
    output_path: str | Path | None,
) -> pd.DataFrame:
    warnings = state.status.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    row = {
        "cycle_started_at": started_at.isoformat(),
        "cycle_completed_at": completed_at.isoformat(),
        "mode": config.mode,
        "paper_enabled": bool(paper_enabled),
        "event_ticker": state.status.get("event_ticker", ""),
        "target_date": state.status.get("target_date", ""),
        "dashboard_status": state.status.get("dashboard_status", ""),
        "settlement_status": state.status.get("settlement_status", ""),
        "settlement_trading_allowed": state.status.get("settlement_trading_allowed", ""),
        "probability_mode": state.status.get("probability_mode", ""),
        "probability_rows": state.status.get("probability_rows", 0),
        "edge_rows": len(state.edge_table),
        "candidate_edges": _count_equals(state.edge_table, "edge_status", "CANDIDATE"),
        "risk_rows": len(risk_decisions),
        "risk_approved": _count_equals(risk_decisions, "risk_status", "APPROVED"),
        "ready_intents": _count_equals(order_intents, "intent_status", "READY"),
        "paper_new_orders": 0 if paper_result is None else paper_result.new_order_count,
        "paper_filled": 0 if paper_result is None else paper_result.filled_count,
        "paper_rejected": 0 if paper_result is None else paper_result.rejected_count,
        "status": "OK",
        "warnings": ";".join(str(item) for item in warnings if str(item)),
        "portfolio_snapshot_path": str(config.outputs.portfolio_snapshot_path),
        "risk_decisions_path": str(config.outputs.risk_decisions_path),
        "order_intents_path": str(config.outputs.order_intents_path),
        "paper_orders_path": str(config.outputs.paper_orders_path),
        "paper_positions_path": str(config.outputs.paper_positions_path),
        "paper_pnl_path": str(config.outputs.paper_pnl_path),
    }
    new_log = pd.DataFrame([row]).reindex(columns=TRADING_CYCLE_LOG_COLUMNS)
    if output_path is None:
        return new_log
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=TRADING_CYCLE_LOG_COLUMNS)
    combined = pd.concat([existing, new_log], ignore_index=True, sort=False).reindex(
        columns=TRADING_CYCLE_LOG_COLUMNS
    )
    combined.to_csv(path, index=False)
    return combined


def _count_equals(frame: pd.DataFrame, column: str, value: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int((frame[column].astype(str) == value).sum())
