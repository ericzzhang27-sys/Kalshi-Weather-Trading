from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.trading.config import PaperSettings
from src.trading.order_intents import ready_order_intents


PAPER_ORDER_COLUMNS = [
    "submitted_at",
    "paper_order_id",
    "client_order_id",
    "row_id",
    "event_ticker",
    "ticker",
    "outcome_side",
    "action",
    "contracts",
    "limit_price_dollars",
    "fill_price_dollars",
    "fee_dollars",
    "gross_cost_dollars",
    "total_cost_dollars",
    "cash_before_dollars",
    "cash_after_dollars",
    "paper_status",
    "paper_reason",
]

PAPER_POSITION_COLUMNS = [
    "updated_at",
    "source",
    "ticker",
    "event_ticker",
    "outcome_side",
    "position_contracts",
    "avg_price_dollars",
    "cost_basis_dollars",
    "market_exposure_dollars",
    "realized_pnl_dollars",
    "fees_paid_dollars",
    "resting_orders_count",
]

PAPER_PNL_COLUMNS = [
    "updated_at",
    "starting_cash_dollars",
    "cash_balance_dollars",
    "open_position_cost_dollars",
    "open_exposure_dollars",
    "realized_pnl_dollars",
    "fees_paid_dollars",
    "portfolio_value_proxy_dollars",
    "filled_order_count",
    "open_order_count",
    "rejected_order_count",
]


@dataclass(frozen=True)
class PaperExecutionResult:
    orders: pd.DataFrame
    positions: pd.DataFrame
    pnl: pd.DataFrame
    new_order_count: int
    filled_count: int
    rejected_count: int


def execute_paper_orders(
    intents: pd.DataFrame,
    *,
    settings: PaperSettings,
    orders_path: str | Path,
    positions_path: str | Path,
    pnl_path: str | Path,
    executed_at: datetime | None = None,
) -> PaperExecutionResult:
    """
    Apply READY order intents to local paper cash/positions.
    """
    executed_at_dt = executed_at or datetime.now(timezone.utc)
    existing_orders = _read_csv(orders_path).reindex(columns=PAPER_ORDER_COLUMNS)
    positions = _read_csv(positions_path).reindex(columns=PAPER_POSITION_COLUMNS)
    pnl = _read_csv(pnl_path).reindex(columns=PAPER_PNL_COLUMNS)

    cash = _latest_cash(pnl, settings.starting_cash_dollars)
    seen_client_ids = set(existing_orders["client_order_id"].dropna().astype(str)) if not existing_orders.empty else set()
    new_records: list[dict[str, Any]] = []
    filled_count = 0
    rejected_count = 0

    for _, intent in ready_order_intents(intents).iterrows():
        client_order_id = str(intent.get("client_order_id", "") or "")
        if not client_order_id or client_order_id in seen_client_ids:
            continue
        seen_client_ids.add(client_order_id)
        contracts = int(float(intent.get("contracts", 0) or 0))
        price = _optional_float(intent.get("limit_price_dollars")) or 0.0
        fee = _optional_float(intent.get("estimated_fee_dollars")) or 0.0
        gross_cost = contracts * price
        total_cost = gross_cost + fee
        cash_before = cash
        status = "OPEN" if settings.fill_mode == "none" else "FILLED"
        reason = ""
        fill_price = "" if status == "OPEN" else price

        if contracts < 1:
            status = "REJECTED"
            reason = "invalid_contract_count"
        elif price <= 0.0 or price >= 1.0:
            status = "REJECTED"
            reason = "invalid_limit_price"
        elif status == "FILLED" and total_cost > cash + 1e-9:
            status = "REJECTED"
            reason = "paper_insufficient_cash"

        if status == "FILLED":
            cash -= total_cost
            positions = _apply_fill_to_positions(
                positions,
                intent=intent,
                contracts=contracts,
                fill_price=price,
                fee=fee,
                total_cost=total_cost,
                updated_at=executed_at_dt,
            )
            filled_count += 1
        elif status == "REJECTED":
            rejected_count += 1

        new_records.append(
            {
                "submitted_at": executed_at_dt.isoformat(),
                "paper_order_id": f"paper-{client_order_id}",
                "client_order_id": client_order_id,
                "row_id": intent.get("row_id", ""),
                "event_ticker": intent.get("event_ticker", ""),
                "ticker": intent.get("ticker", ""),
                "outcome_side": intent.get("outcome_side", ""),
                "action": intent.get("action", ""),
                "contracts": contracts,
                "limit_price_dollars": price,
                "fill_price_dollars": fill_price,
                "fee_dollars": fee if status == "FILLED" else 0.0,
                "gross_cost_dollars": gross_cost if status == "FILLED" else 0.0,
                "total_cost_dollars": total_cost if status == "FILLED" else 0.0,
                "cash_before_dollars": cash_before,
                "cash_after_dollars": cash,
                "paper_status": status,
                "paper_reason": reason,
            }
        )

    new_orders = pd.DataFrame.from_records(new_records).reindex(columns=PAPER_ORDER_COLUMNS)
    orders = (
        pd.concat([existing_orders, new_orders], ignore_index=True, sort=False)
        if not new_orders.empty
        else existing_orders
    ).reindex(columns=PAPER_ORDER_COLUMNS)
    pnl = _append_pnl_row(
        pnl,
        positions=positions,
        orders=orders,
        starting_cash=settings.starting_cash_dollars,
        cash=cash,
        updated_at=executed_at_dt,
    )

    _write_csv(orders, orders_path, PAPER_ORDER_COLUMNS)
    _write_csv(positions, positions_path, PAPER_POSITION_COLUMNS)
    _write_csv(pnl, pnl_path, PAPER_PNL_COLUMNS)
    return PaperExecutionResult(
        orders=orders,
        positions=positions,
        pnl=pnl,
        new_order_count=len(new_orders),
        filled_count=filled_count,
        rejected_count=rejected_count,
    )


def _apply_fill_to_positions(
    positions: pd.DataFrame,
    *,
    intent: pd.Series,
    contracts: int,
    fill_price: float,
    fee: float,
    total_cost: float,
    updated_at: datetime,
) -> pd.DataFrame:
    key = (
        str(intent.get("ticker", "") or ""),
        str(intent.get("outcome_side", "") or "").upper(),
    )
    if positions.empty:
        positions = pd.DataFrame(columns=PAPER_POSITION_COLUMNS)
    mask = (
        (positions["ticker"].astype(str) == key[0])
        & (positions["outcome_side"].astype(str).str.upper() == key[1])
    )
    if mask.any():
        idx = positions[mask].index[0]
        old_contracts = _optional_float(positions.at[idx, "position_contracts"]) or 0.0
        old_cost = _optional_float(positions.at[idx, "cost_basis_dollars"]) or 0.0
        old_fees = _optional_float(positions.at[idx, "fees_paid_dollars"]) or 0.0
        new_contracts = old_contracts + contracts
        new_cost = old_cost + total_cost
        positions.at[idx, "updated_at"] = updated_at.isoformat()
        positions.at[idx, "position_contracts"] = new_contracts
        positions.at[idx, "cost_basis_dollars"] = new_cost
        positions.at[idx, "market_exposure_dollars"] = new_cost
        positions.at[idx, "avg_price_dollars"] = new_cost / new_contracts if new_contracts else 0.0
        positions.at[idx, "fees_paid_dollars"] = old_fees + fee
        return positions.reindex(columns=PAPER_POSITION_COLUMNS)

    row = {
        "updated_at": updated_at.isoformat(),
        "source": "paper",
        "ticker": key[0],
        "event_ticker": intent.get("event_ticker", ""),
        "outcome_side": key[1],
        "position_contracts": contracts,
        "avg_price_dollars": total_cost / contracts,
        "cost_basis_dollars": total_cost,
        "market_exposure_dollars": total_cost,
        "realized_pnl_dollars": 0.0,
        "fees_paid_dollars": fee,
        "resting_orders_count": 0,
    }
    if positions.empty:
        return pd.DataFrame([row]).reindex(columns=PAPER_POSITION_COLUMNS)
    return pd.concat(
        [positions, pd.DataFrame([row])],
        ignore_index=True,
        sort=False,
    ).reindex(columns=PAPER_POSITION_COLUMNS)


def _append_pnl_row(
    pnl: pd.DataFrame,
    *,
    positions: pd.DataFrame,
    orders: pd.DataFrame,
    starting_cash: float,
    cash: float,
    updated_at: datetime,
) -> pd.DataFrame:
    open_position_cost = _sum_numeric(positions, "cost_basis_dollars")
    fees = _sum_numeric(positions, "fees_paid_dollars")
    realized = _sum_numeric(positions, "realized_pnl_dollars")
    open_order_count = 0
    rejected_count = 0
    filled_count = 0
    if not orders.empty and "paper_status" in orders.columns:
        statuses = orders["paper_status"].astype(str).str.upper()
        open_order_count = int(statuses.isin({"OPEN", "PENDING", "RESTING"}).sum())
        rejected_count = int((statuses == "REJECTED").sum())
        filled_count = int((statuses == "FILLED").sum())
    row = {
        "updated_at": updated_at.isoformat(),
        "starting_cash_dollars": starting_cash,
        "cash_balance_dollars": cash,
        "open_position_cost_dollars": open_position_cost,
        "open_exposure_dollars": open_position_cost,
        "realized_pnl_dollars": realized,
        "fees_paid_dollars": fees,
        "portfolio_value_proxy_dollars": cash + open_position_cost + realized,
        "filled_order_count": filled_count,
        "open_order_count": open_order_count,
        "rejected_order_count": rejected_count,
    }
    return pd.concat([pnl, pd.DataFrame([row])], ignore_index=True, sort=False).reindex(
        columns=PAPER_PNL_COLUMNS
    )


def _latest_cash(pnl: pd.DataFrame, starting_cash: float) -> float:
    if pnl.empty or "cash_balance_dollars" not in pnl.columns:
        return float(starting_cash)
    values = pd.to_numeric(pnl["cash_balance_dollars"], errors="coerce").dropna()
    return float(starting_cash) if values.empty else float(values.iloc[-1])


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


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


def _read_csv(path: str | Path) -> pd.DataFrame:
    candidate = Path(path)
    if not candidate.exists():
        return pd.DataFrame()
    return pd.read_csv(candidate)


def _write_csv(frame: pd.DataFrame, path: str | Path, columns: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=columns).to_csv(output, index=False)
