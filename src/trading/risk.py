from __future__ import annotations

from datetime import date, datetime, timezone
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.trading.config import RiskSettings
from src.trading.portfolio import (
    PortfolioSnapshot,
    contracts_for_ticker,
    exposure_for_event,
    exposure_for_ticker,
    realized_pnl,
)


RISK_DECISION_COLUMNS = [
    "evaluated_at",
    "row_id",
    "event_ticker",
    "ticker",
    "outcome_side",
    "action",
    "edge_status",
    "edge_reason",
    "settlement_status",
    "settlement_reason",
    "settlement_trading_allowed",
    "probability_mode",
    "net_edge",
    "executable_price",
    "executable_size",
    "fee_per_contract",
    "proposed_contracts",
    "estimated_fee_dollars",
    "estimated_cost_dollars",
    "cash_balance_dollars",
    "reserved_cash_dollars",
    "current_contracts_for_ticker",
    "current_market_exposure_dollars",
    "current_event_exposure_dollars",
    "current_total_exposure_dollars",
    "current_open_orders",
    "kill_switch_active",
    "max_contracts_per_order",
    "max_contracts_per_market",
    "max_dollars_per_order",
    "max_dollars_per_market",
    "max_dollars_per_event",
    "max_correlated_event_exposure_dollars",
    "max_total_exposure",
    "max_open_orders",
    "max_daily_loss_dollars",
    "min_cash_reserve_dollars",
    "risk_status",
    "risk_reason",
]


def evaluate_risk(
    edge_table: pd.DataFrame,
    portfolio: PortfolioSnapshot,
    settings: RiskSettings,
    *,
    evaluated_at: datetime | None = None,
) -> pd.DataFrame:
    """
    Apply portfolio/risk constraints to edge candidates before order intents.
    """
    evaluated_at_dt = evaluated_at or datetime.now(timezone.utc)
    if edge_table.empty:
        return pd.DataFrame(columns=RISK_DECISION_COLUMNS)

    working = edge_table.copy()
    working["_candidate_rank"] = working.get("edge_status", "").map(
        lambda value: 0 if str(value) == "CANDIDATE" else 1
    )
    working["_net_edge_rank"] = pd.to_numeric(working.get("net_edge", 0.0), errors="coerce").fillna(
        float("-inf")
    )
    working = working.sort_values(
        ["_candidate_rank", "_net_edge_rank"],
        ascending=[True, False],
        kind="stable",
    )

    reserved_cash = 0.0
    reserved_market_exposure: dict[str, float] = {}
    reserved_event_exposure: dict[str, float] = {}
    reserved_contracts: dict[str, float] = {}
    reserved_open_orders = 0
    records: list[dict[str, Any]] = []

    for _, row in working.iterrows():
        ticker = str(row.get("ticker", "") or "")
        event_ticker = str(row.get("event_ticker", "") or _event_ticker_from_market(ticker))
        price = _optional_float(row.get("executable_price"))
        size = _optional_float(row.get("executable_size"))
        fee_per_contract = _optional_float(row.get("fee_per_contract")) or 0.0
        proposed_contracts = _proposed_contracts(
            price=price,
            size=size,
            fee_per_contract=fee_per_contract,
            settings=settings,
        )
        estimated_fee = fee_per_contract * proposed_contracts
        estimated_cost = (0.0 if price is None else price * proposed_contracts) + estimated_fee

        current_contracts = contracts_for_ticker(portfolio, ticker) + reserved_contracts.get(ticker, 0.0)
        current_market_exposure = (
            exposure_for_ticker(portfolio, ticker) + reserved_market_exposure.get(ticker, 0.0)
        )
        current_event_exposure = (
            exposure_for_event(portfolio, event_ticker)
            + reserved_event_exposure.get(event_ticker, 0.0)
        )
        current_total_exposure = (
            portfolio.total_market_exposure_dollars
            + sum(reserved_market_exposure.values())
        )
        current_open_orders = portfolio.open_order_count + reserved_open_orders

        reasons = _risk_reasons(
            row=row,
            ticker=ticker,
            event_ticker=event_ticker,
            price=price,
            size=size,
            proposed_contracts=proposed_contracts,
            estimated_cost=estimated_cost,
            portfolio=portfolio,
            settings=settings,
            reserved_cash=reserved_cash,
            current_contracts=current_contracts,
            current_market_exposure=current_market_exposure,
            current_event_exposure=current_event_exposure,
            current_total_exposure=current_total_exposure,
            current_open_orders=current_open_orders,
        )
        risk_status = "NO_TRADE" if reasons else "APPROVED"
        if risk_status == "APPROVED":
            reserved_cash += estimated_cost
            reserved_market_exposure[ticker] = reserved_market_exposure.get(ticker, 0.0) + estimated_cost
            reserved_event_exposure[event_ticker] = (
                reserved_event_exposure.get(event_ticker, 0.0) + estimated_cost
            )
            reserved_contracts[ticker] = reserved_contracts.get(ticker, 0.0) + proposed_contracts
            reserved_open_orders += 1

        records.append(
            {
                "evaluated_at": evaluated_at_dt.isoformat(),
                "row_id": row.get("row_id", ""),
                "event_ticker": event_ticker,
                "ticker": ticker,
                "outcome_side": row.get("outcome_side", ""),
                "action": row.get("action", ""),
                "edge_status": row.get("edge_status", ""),
                "edge_reason": row.get("no_trade_reason", ""),
                "settlement_status": row.get("settlement_status", ""),
                "settlement_reason": row.get("settlement_reason", ""),
                "settlement_trading_allowed": row.get("settlement_trading_allowed", ""),
                "probability_mode": row.get("probability_mode", ""),
                "net_edge": row.get("net_edge", ""),
                "executable_price": price,
                "executable_size": size,
                "fee_per_contract": fee_per_contract,
                "proposed_contracts": proposed_contracts,
                "estimated_fee_dollars": estimated_fee,
                "estimated_cost_dollars": estimated_cost,
                "cash_balance_dollars": portfolio.cash_balance_dollars,
                "reserved_cash_dollars": reserved_cash,
                "current_contracts_for_ticker": current_contracts,
                "current_market_exposure_dollars": current_market_exposure,
                "current_event_exposure_dollars": current_event_exposure,
                "current_total_exposure_dollars": current_total_exposure,
                "current_open_orders": current_open_orders,
                "kill_switch_active": settings.kill_switch_path.exists(),
                "max_contracts_per_order": settings.max_contracts_per_order,
                "max_contracts_per_market": settings.max_contracts_per_market,
                "max_dollars_per_order": settings.max_dollars_per_order,
                "max_dollars_per_market": settings.max_dollars_per_market,
                "max_dollars_per_event": settings.max_dollars_per_event,
                "max_correlated_event_exposure_dollars": (
                    settings.max_correlated_event_exposure_dollars
                ),
                "max_total_exposure": settings.max_total_exposure,
                "max_open_orders": settings.max_open_orders,
                "max_daily_loss_dollars": settings.max_daily_loss_dollars,
                "min_cash_reserve_dollars": settings.min_cash_reserve_dollars,
                "risk_status": risk_status,
                "risk_reason": ";".join(_dedupe(reasons)),
            }
        )

    return pd.DataFrame.from_records(records).reindex(columns=RISK_DECISION_COLUMNS)


def save_risk_decisions(decisions: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions.reindex(columns=RISK_DECISION_COLUMNS).to_csv(path, index=False)


def _risk_reasons(
    *,
    row: pd.Series,
    ticker: str,
    event_ticker: str,
    price: float | None,
    size: float | None,
    proposed_contracts: int,
    estimated_cost: float,
    portfolio: PortfolioSnapshot,
    settings: RiskSettings,
    reserved_cash: float,
    current_contracts: float,
    current_market_exposure: float,
    current_event_exposure: float,
    current_total_exposure: float,
    current_open_orders: int,
) -> list[str]:
    reasons: list[str] = []
    if settings.kill_switch_path.exists():
        reasons.append("kill_switch_active")
    if ticker in settings.denylist_tickers:
        reasons.append("ticker_denylisted")
    if event_ticker in settings.denylist_event_tickers:
        reasons.append("event_denylisted")
    event_date = _event_date_from_ticker(event_ticker)
    if event_date is not None and event_date.isoformat() in settings.denylist_target_dates:
        reasons.append("target_date_denylisted")

    edge_status = str(row.get("edge_status", "") or "")
    if edge_status != "CANDIDATE":
        reason = str(row.get("no_trade_reason", "") or edge_status or "edge_not_candidate")
        reasons.append(f"edge:{reason}")

    if price is None or not math.isfinite(price) or price <= 0.0 or price >= 1.0:
        reasons.append("invalid_executable_price")
    if size is None or not math.isfinite(size) or size <= 0.0:
        reasons.append("missing_executable_size")
    if proposed_contracts < 1:
        reasons.append("proposed_size_zero")

    if current_contracts + proposed_contracts > settings.max_contracts_per_market:
        reasons.append("max_contracts_per_market")
    if estimated_cost > settings.max_dollars_per_order + 1e-9:
        reasons.append("max_dollars_per_order")
    if current_market_exposure + estimated_cost > settings.max_dollars_per_market + 1e-9:
        reasons.append("max_dollars_per_market")
    if current_event_exposure + estimated_cost > settings.max_dollars_per_event + 1e-9:
        reasons.append("max_dollars_per_event")
    if (
        current_event_exposure + estimated_cost
        > settings.max_correlated_event_exposure_dollars + 1e-9
    ):
        reasons.append("correlated_event_exposure_limit")
    if current_total_exposure + estimated_cost > settings.max_total_exposure + 1e-9:
        reasons.append("max_total_exposure")
    if current_open_orders + 1 > settings.max_open_orders:
        reasons.append("max_open_orders")
    if portfolio.cash_balance_dollars - reserved_cash - estimated_cost < settings.min_cash_reserve_dollars - 1e-9:
        reasons.append("insufficient_cash_reserve")
    if realized_pnl(portfolio) <= -settings.max_daily_loss_dollars:
        reasons.append("max_daily_loss")
    return reasons


def _proposed_contracts(
    *,
    price: float | None,
    size: float | None,
    fee_per_contract: float,
    settings: RiskSettings,
) -> int:
    if price is None or size is None or price <= 0.0:
        return 0
    per_contract_cost = price + max(0.0, fee_per_contract)
    if per_contract_cost <= 0.0:
        return 0
    by_order_cost = math.floor(settings.max_dollars_per_order / per_contract_cost)
    return max(
        0,
        min(
            int(settings.max_contracts_per_order),
            int(math.floor(size)),
            int(by_order_cost),
        ),
    )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_ticker_from_market(ticker: str) -> str:
    parts = str(ticker).rsplit("-", 1)
    return parts[0] if len(parts) == 2 else str(ticker)


def _event_date_from_ticker(event_ticker: str) -> date | None:
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", str(event_ticker))
    if not match:
        return None
    month_lookup = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    month = month_lookup.get(match.group(2))
    if month is None:
        return None
    return date(2000 + int(match.group(1)), month, int(match.group(3)))


def _dedupe(reasons: list[str]) -> list[str]:
    return list(dict.fromkeys(reason for reason in reasons if reason))
