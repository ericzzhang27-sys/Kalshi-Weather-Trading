from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.trading.kalshi_client import KalshiClient


POSITION_COLUMNS = [
    "fetched_at",
    "source",
    "ticker",
    "event_ticker",
    "outcome_side",
    "position_contracts",
    "market_exposure_dollars",
    "realized_pnl_dollars",
    "fees_paid_dollars",
    "resting_orders_count",
]

OPEN_ORDER_COLUMNS = [
    "fetched_at",
    "source",
    "order_id",
    "client_order_id",
    "ticker",
    "event_ticker",
    "outcome_side",
    "action",
    "order_status",
    "price_dollars",
    "remaining_contracts",
    "initial_contracts",
    "order_exposure_dollars",
]

BALANCE_COLUMNS = [
    "fetched_at",
    "source",
    "cash_balance_dollars",
    "portfolio_value_dollars",
    "updated_ts",
]

PORTFOLIO_SNAPSHOT_COLUMNS = [
    "row_type",
    "fetched_at",
    "source",
    "cash_balance_dollars",
    "portfolio_value_dollars",
    "updated_ts",
    "ticker",
    "event_ticker",
    "outcome_side",
    "position_contracts",
    "market_exposure_dollars",
    "realized_pnl_dollars",
    "fees_paid_dollars",
    "resting_orders_count",
    "order_id",
    "client_order_id",
    "action",
    "order_status",
    "price_dollars",
    "remaining_contracts",
    "initial_contracts",
    "order_exposure_dollars",
]


@dataclass(frozen=True)
class PortfolioSnapshot:
    fetched_at: datetime
    source: str
    balance: pd.DataFrame
    positions: pd.DataFrame
    open_orders: pd.DataFrame

    @property
    def cash_balance_dollars(self) -> float:
        if self.balance.empty or "cash_balance_dollars" not in self.balance.columns:
            return 0.0
        value = pd.to_numeric(self.balance["cash_balance_dollars"], errors="coerce").dropna()
        return 0.0 if value.empty else float(value.iloc[-1])

    @property
    def total_market_exposure_dollars(self) -> float:
        if self.positions.empty or "market_exposure_dollars" not in self.positions.columns:
            return 0.0
        values = pd.to_numeric(self.positions["market_exposure_dollars"], errors="coerce")
        return float(values.fillna(0.0).sum())

    @property
    def open_order_count(self) -> int:
        if self.open_orders.empty:
            return 0
        if "remaining_contracts" not in self.open_orders.columns:
            return len(self.open_orders)
        remaining = pd.to_numeric(self.open_orders["remaining_contracts"], errors="coerce")
        return int((remaining.fillna(0.0) > 0.0).sum())


def fetch_portfolio_snapshot(
    client: KalshiClient,
    *,
    auth: bool,
    fetched_at: datetime | None = None,
) -> PortfolioSnapshot:
    """
    Fetch authenticated Kalshi portfolio state, or return an empty snapshot.
    """
    fetched_at_dt = fetched_at or datetime.now(timezone.utc)
    if not auth:
        return empty_portfolio_snapshot(fetched_at=fetched_at_dt, source="no_auth")

    balance_payload = client.get("/portfolio/balance", auth=True)
    positions_payload = client.get("/portfolio/positions", auth=True)
    orders_payload = client.get("/portfolio/orders", auth=True)
    return PortfolioSnapshot(
        fetched_at=fetched_at_dt,
        source="kalshi",
        balance=normalize_balance_payload(balance_payload, fetched_at_dt, source="kalshi"),
        positions=normalize_positions_payload(positions_payload, fetched_at_dt, source="kalshi"),
        open_orders=normalize_orders_payload(orders_payload, fetched_at_dt, source="kalshi"),
    )


def empty_portfolio_snapshot(
    *,
    fetched_at: datetime | None = None,
    source: str = "empty",
    cash_balance_dollars: float = 0.0,
) -> PortfolioSnapshot:
    fetched_at_dt = fetched_at or datetime.now(timezone.utc)
    balance = pd.DataFrame(
        [
            {
                "fetched_at": fetched_at_dt.isoformat(),
                "source": source,
                "cash_balance_dollars": float(cash_balance_dollars),
                "portfolio_value_dollars": float(cash_balance_dollars),
                "updated_ts": "",
            }
        ],
        columns=BALANCE_COLUMNS,
    )
    return PortfolioSnapshot(
        fetched_at=fetched_at_dt,
        source=source,
        balance=balance,
        positions=pd.DataFrame(columns=POSITION_COLUMNS),
        open_orders=pd.DataFrame(columns=OPEN_ORDER_COLUMNS),
    )


def portfolio_from_paper_state(
    *,
    positions_path: str | Path,
    orders_path: str | Path,
    pnl_path: str | Path,
    starting_cash_dollars: float,
    fetched_at: datetime | None = None,
) -> PortfolioSnapshot:
    fetched_at_dt = fetched_at or datetime.now(timezone.utc)
    pnl = _read_csv(pnl_path)
    cash = float(starting_cash_dollars)
    if not pnl.empty and "cash_balance_dollars" in pnl.columns:
        values = pd.to_numeric(pnl["cash_balance_dollars"], errors="coerce").dropna()
        if not values.empty:
            cash = float(values.iloc[-1])

    positions = _paper_positions_frame(_read_csv(positions_path), fetched_at_dt)
    orders = _paper_open_orders_frame(_read_csv(orders_path), fetched_at_dt)
    balance = pd.DataFrame(
        [
            {
                "fetched_at": fetched_at_dt.isoformat(),
                "source": "paper",
                "cash_balance_dollars": cash,
                "portfolio_value_dollars": cash + _exposure_sum(positions),
                "updated_ts": fetched_at_dt.isoformat(),
            }
        ],
        columns=BALANCE_COLUMNS,
    )
    return PortfolioSnapshot(
        fetched_at=fetched_at_dt,
        source="paper",
        balance=balance,
        positions=positions,
        open_orders=orders,
    )


def normalize_balance_payload(
    payload: Mapping[str, Any],
    fetched_at: datetime,
    *,
    source: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fetched_at": fetched_at.isoformat(),
                "source": source,
                "cash_balance_dollars": _dollars_from_cents_or_dollars(
                    payload.get("balance"),
                    payload.get("balance_dollars"),
                ),
                "portfolio_value_dollars": _dollars_from_cents_or_dollars(
                    payload.get("portfolio_value"),
                    payload.get("portfolio_value_dollars"),
                ),
                "updated_ts": payload.get("updated_ts", ""),
            }
        ],
        columns=BALANCE_COLUMNS,
    )


def normalize_positions_payload(
    payload: Mapping[str, Any],
    fetched_at: datetime,
    *,
    source: str,
) -> pd.DataFrame:
    raw_positions = payload.get("market_positions", [])
    if not isinstance(raw_positions, list):
        raw_positions = []
    records: list[dict[str, Any]] = []
    for raw in raw_positions:
        if not isinstance(raw, Mapping):
            continue
        ticker = str(raw.get("ticker", "") or "")
        records.append(
            {
                "fetched_at": fetched_at.isoformat(),
                "source": source,
                "ticker": ticker,
                "event_ticker": str(raw.get("event_ticker", "") or _event_ticker_from_market(ticker)),
                "outcome_side": str(raw.get("outcome_side", "") or "YES").upper(),
                "position_contracts": _optional_float(raw.get("position_fp"), raw.get("position")),
                "market_exposure_dollars": _optional_float(raw.get("market_exposure_dollars")),
                "realized_pnl_dollars": _optional_float(raw.get("realized_pnl_dollars")),
                "fees_paid_dollars": _optional_float(raw.get("fees_paid_dollars")),
                "resting_orders_count": _optional_float(raw.get("resting_orders_count")),
            }
        )
    return pd.DataFrame.from_records(records).reindex(columns=POSITION_COLUMNS)


def normalize_orders_payload(
    payload: Mapping[str, Any],
    fetched_at: datetime,
    *,
    source: str,
) -> pd.DataFrame:
    raw_orders = payload.get("orders", [])
    if not isinstance(raw_orders, list):
        raw_orders = []
    records: list[dict[str, Any]] = []
    for raw in raw_orders:
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status", "") or raw.get("order_status", "") or "").lower()
        remaining = _optional_float(raw.get("remaining_count_fp"), raw.get("remaining_count"))
        if status in {"filled", "canceled", "cancelled", "rejected"} or (remaining is not None and remaining <= 0.0):
            continue
        ticker = str(raw.get("ticker", "") or raw.get("market_ticker", "") or "")
        outcome_side = str(raw.get("outcome_side", "") or raw.get("side", "") or "").upper()
        price = _order_price(raw, outcome_side)
        remaining_contracts = 0.0 if remaining is None else remaining
        records.append(
            {
                "fetched_at": fetched_at.isoformat(),
                "source": source,
                "order_id": raw.get("order_id", ""),
                "client_order_id": raw.get("client_order_id", ""),
                "ticker": ticker,
                "event_ticker": str(raw.get("event_ticker", "") or _event_ticker_from_market(ticker)),
                "outcome_side": outcome_side or "YES",
                "action": str(raw.get("action", "") or "buy").upper(),
                "order_status": status or "open",
                "price_dollars": price,
                "remaining_contracts": remaining_contracts,
                "initial_contracts": _optional_float(raw.get("initial_count_fp"), raw.get("initial_count")),
                "order_exposure_dollars": None if price is None else price * remaining_contracts,
            }
        )
    return pd.DataFrame.from_records(records).reindex(columns=OPEN_ORDER_COLUMNS)


def save_portfolio_snapshot(snapshot: PortfolioSnapshot, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_snapshot_frame(snapshot).to_csv(path, index=False)


def portfolio_snapshot_frame(snapshot: PortfolioSnapshot) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not snapshot.balance.empty:
        balance = snapshot.balance.copy()
        balance["row_type"] = "balance"
        frames.append(balance)
    if not snapshot.positions.empty:
        positions = snapshot.positions.copy()
        positions["row_type"] = "position"
        frames.append(positions)
    if not snapshot.open_orders.empty:
        orders = snapshot.open_orders.copy()
        orders["row_type"] = "open_order"
        frames.append(orders)
    if not frames:
        return pd.DataFrame(columns=PORTFOLIO_SNAPSHOT_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False).reindex(columns=PORTFOLIO_SNAPSHOT_COLUMNS)


def exposure_for_ticker(snapshot: PortfolioSnapshot, ticker: str) -> float:
    if snapshot.positions.empty or "ticker" not in snapshot.positions.columns:
        return 0.0
    rows = snapshot.positions[snapshot.positions["ticker"].astype(str) == str(ticker)]
    return _exposure_sum(rows)


def exposure_for_event(snapshot: PortfolioSnapshot, event_ticker: str) -> float:
    if snapshot.positions.empty or "event_ticker" not in snapshot.positions.columns:
        return 0.0
    rows = snapshot.positions[snapshot.positions["event_ticker"].astype(str) == str(event_ticker)]
    return _exposure_sum(rows)


def contracts_for_ticker(snapshot: PortfolioSnapshot, ticker: str) -> float:
    if snapshot.positions.empty or "ticker" not in snapshot.positions.columns:
        return 0.0
    rows = snapshot.positions[snapshot.positions["ticker"].astype(str) == str(ticker)]
    if "position_contracts" not in rows.columns:
        return 0.0
    values = pd.to_numeric(rows["position_contracts"], errors="coerce")
    return float(values.fillna(0.0).abs().sum())


def realized_pnl(snapshot: PortfolioSnapshot) -> float:
    if snapshot.positions.empty or "realized_pnl_dollars" not in snapshot.positions.columns:
        return 0.0
    values = pd.to_numeric(snapshot.positions["realized_pnl_dollars"], errors="coerce")
    return float(values.fillna(0.0).sum())


def _paper_positions_frame(frame: pd.DataFrame, fetched_at: datetime) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    result = frame.copy()
    result["fetched_at"] = fetched_at.isoformat()
    result["source"] = "paper"
    if "market_exposure_dollars" not in result.columns and "cost_basis_dollars" in result.columns:
        result["market_exposure_dollars"] = result["cost_basis_dollars"]
    if "resting_orders_count" not in result.columns:
        result["resting_orders_count"] = 0
    for column in POSITION_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    return result.reindex(columns=POSITION_COLUMNS)


def _paper_open_orders_frame(frame: pd.DataFrame, fetched_at: datetime) -> pd.DataFrame:
    if frame.empty or "paper_status" not in frame.columns:
        return pd.DataFrame(columns=OPEN_ORDER_COLUMNS)
    open_statuses = {"OPEN", "PENDING", "RESTING"}
    working = frame[frame["paper_status"].astype(str).str.upper().isin(open_statuses)].copy()
    if working.empty:
        return pd.DataFrame(columns=OPEN_ORDER_COLUMNS)
    working["fetched_at"] = fetched_at.isoformat()
    working["source"] = "paper"
    working["order_id"] = working.get("paper_order_id", "")
    working["order_status"] = working["paper_status"]
    working["price_dollars"] = working.get("limit_price_dollars", working.get("fill_price_dollars", ""))
    working["remaining_contracts"] = working.get("contracts", 0)
    working["initial_contracts"] = working.get("contracts", 0)
    working["order_exposure_dollars"] = (
        pd.to_numeric(working["price_dollars"], errors="coerce")
        * pd.to_numeric(working["remaining_contracts"], errors="coerce")
    )
    for column in OPEN_ORDER_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    return working.reindex(columns=OPEN_ORDER_COLUMNS)


def _dollars_from_cents_or_dollars(cents_value: Any, dollars_value: Any = None) -> float:
    if dollars_value is not None:
        parsed = _optional_float(dollars_value)
        if parsed is not None:
            return parsed
    parsed = _optional_float(cents_value)
    if parsed is None:
        return 0.0
    return parsed / 100.0


def _order_price(raw: Mapping[str, Any], outcome_side: str) -> float | None:
    outcome = outcome_side.lower()
    if outcome == "no":
        return _optional_float(raw.get("no_price_dollars"), raw.get("price"))
    return _optional_float(raw.get("yes_price_dollars"), raw.get("price"))


def _optional_float(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _exposure_sum(frame: pd.DataFrame) -> float:
    if frame.empty or "market_exposure_dollars" not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame["market_exposure_dollars"], errors="coerce")
    return float(values.fillna(0.0).sum())


def _event_ticker_from_market(ticker: str) -> str:
    parts = str(ticker).rsplit("-", 1)
    return parts[0] if len(parts) == 2 else str(ticker)


def _read_csv(path: str | Path) -> pd.DataFrame:
    candidate = Path(path)
    if not candidate.exists():
        return pd.DataFrame()
    return pd.read_csv(candidate)
