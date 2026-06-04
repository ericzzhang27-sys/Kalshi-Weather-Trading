from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


ORDER_INTENT_COLUMNS = [
    "generated_at",
    "client_order_id",
    "row_id",
    "event_ticker",
    "ticker",
    "outcome_side",
    "action",
    "intent_status",
    "intent_reason",
    "contracts",
    "limit_price_dollars",
    "v2_price_dollars",
    "legacy_yes_price_dollars",
    "legacy_no_price_dollars",
    "estimated_fee_dollars",
    "estimated_cost_dollars",
    "net_edge",
    "time_in_force",
    "post_only",
    "reduce_only",
    "cancel_order_on_pause",
    "legacy_side",
    "legacy_action",
    "v2_side",
    "source_evaluated_at",
]


def build_order_intents(
    risk_decisions: pd.DataFrame,
    *,
    generated_at: datetime | None = None,
) -> pd.DataFrame:
    """
    Convert risk decisions into deterministic limit-order intent records.
    """
    generated_at_dt = generated_at or datetime.now(timezone.utc)
    if risk_decisions.empty:
        return pd.DataFrame(columns=ORDER_INTENT_COLUMNS)

    records: list[dict[str, Any]] = []
    for _, row in risk_decisions.iterrows():
        risk_status = str(row.get("risk_status", "") or "")
        ready = risk_status == "APPROVED"
        outcome_side = str(row.get("outcome_side", "") or "").upper()
        contracts = int(float(row.get("proposed_contracts", 0) or 0)) if ready else 0
        price = _optional_float(row.get("executable_price")) if ready else None
        v2_price = _v2_yes_book_price(outcome_side, price) if ready else None
        record = {
            "generated_at": generated_at_dt.isoformat(),
            "client_order_id": _client_order_id(row) if ready else "",
            "row_id": row.get("row_id", ""),
            "event_ticker": row.get("event_ticker", ""),
            "ticker": row.get("ticker", ""),
            "outcome_side": outcome_side,
            "action": row.get("action", ""),
            "intent_status": "READY" if ready else "BLOCKED",
            "intent_reason": "" if ready else str(row.get("risk_reason", "") or risk_status),
            "contracts": contracts,
            "limit_price_dollars": price,
            "v2_price_dollars": v2_price,
            "legacy_yes_price_dollars": price if outcome_side == "YES" and ready else "",
            "legacy_no_price_dollars": price if outcome_side == "NO" and ready else "",
            "estimated_fee_dollars": row.get("estimated_fee_dollars", 0.0) if ready else 0.0,
            "estimated_cost_dollars": row.get("estimated_cost_dollars", 0.0) if ready else 0.0,
            "net_edge": row.get("net_edge", ""),
            "time_in_force": "fill_or_kill",
            "post_only": False,
            "reduce_only": False,
            "cancel_order_on_pause": True,
            "legacy_side": outcome_side.lower(),
            "legacy_action": "buy",
            "v2_side": "bid" if outcome_side == "YES" else "ask",
            "source_evaluated_at": row.get("evaluated_at", ""),
        }
        records.append(record)
    return pd.DataFrame.from_records(records).reindex(columns=ORDER_INTENT_COLUMNS)


def save_order_intents(intents: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intents.reindex(columns=ORDER_INTENT_COLUMNS).to_csv(path, index=False)


def ready_order_intents(intents: pd.DataFrame) -> pd.DataFrame:
    if intents.empty or "intent_status" not in intents.columns:
        return pd.DataFrame(columns=ORDER_INTENT_COLUMNS)
    return intents[intents["intent_status"].astype(str) == "READY"].copy()


def _client_order_id(row: pd.Series) -> str:
    payload = "|".join(
        str(row.get(column, "") or "")
        for column in [
            "evaluated_at",
            "row_id",
            "event_ticker",
            "ticker",
            "outcome_side",
            "action",
            "executable_price",
            "proposed_contracts",
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]
    return f"kwt-{digest}"


def _v2_yes_book_price(outcome_side: str, outcome_price: float | None) -> float | None:
    if outcome_price is None:
        return None
    if outcome_side == "NO":
        return round(1.0 - outcome_price, 4)
    return round(outcome_price, 4)


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
