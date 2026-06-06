from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import math
from pathlib import Path
from typing import Any

import pandas as pd


EDGE_COLUMNS = [
    "evaluated_at",
    "row_id",
    "event_ticker",
    "ticker",
    "bucket_index",
    "bucket_name",
    "bucket_lower_temp",
    "bucket_upper_temp",
    "probability",
    "outcome_side",
    "action",
    "fair_yes",
    "fair_no",
    "fair_value",
    "executable_price",
    "executable_size",
    "best_bid",
    "best_ask",
    "spread",
    "gross_edge",
    "fee_estimate",
    "fee_per_contract",
    "slippage_buffer",
    "net_edge",
    "net_edge_percent",
    "min_edge_dollars",
    "min_edge_percent",
    "min_liquidity_contracts",
    "max_spread_dollars",
    "staleness_seconds",
    "max_staleness_seconds",
    "orderbook_status",
    "orderbook_reason",
    "settlement_status",
    "settlement_reason",
    "settlement_trading_allowed",
    "probability_mode",
    "probability_signal_status",
    "probability_signal_reason",
    "fee_rate",
    "evaluation_contracts",
    "trade_fee_rounding_increment",
    "balance_rounding_increment",
    "edge_status",
    "no_trade_reason",
]


@dataclass(frozen=True)
class EdgeSettings:
    min_edge_dollars: float = 0.02
    min_edge_percent: float = 0.0
    slippage_buffer_dollars: float = 0.005
    min_liquidity_contracts: float = 1.0
    max_spread_dollars: float = 0.25
    max_staleness_seconds: float = 300.0
    fee_rate: float = 0.07
    evaluation_contracts: float = 1.0
    trade_fee_rounding_increment: float = 0.0001
    balance_rounding_increment: float = 0.01

    def __post_init__(self) -> None:
        _require_nonnegative(self.min_edge_dollars, "min_edge_dollars")
        _require_nonnegative(self.min_edge_percent, "min_edge_percent")
        _require_nonnegative(self.slippage_buffer_dollars, "slippage_buffer_dollars")
        _require_positive(self.min_liquidity_contracts, "min_liquidity_contracts")
        _require_positive(self.max_spread_dollars, "max_spread_dollars")
        _require_positive(self.max_staleness_seconds, "max_staleness_seconds")
        _require_nonnegative(self.fee_rate, "fee_rate")
        _require_positive(self.evaluation_contracts, "evaluation_contracts")
        _require_positive(self.trade_fee_rounding_increment, "trade_fee_rounding_increment")
        _require_positive(self.balance_rounding_increment, "balance_rounding_increment")


def estimate_kalshi_buy_fee(
    price_dollars: float,
    contracts: float = 1.0,
    *,
    fee_rate: float = 0.07,
    trade_fee_rounding_increment: float = 0.0001,
    balance_rounding_increment: float = 0.01,
) -> float:
    """
    Estimate the single-fill taker fee for buying contracts at an executable ask.
    """
    _validate_price(price_dollars)
    _require_positive(contracts, "contracts")
    _require_nonnegative(fee_rate, "fee_rate")
    _require_positive(trade_fee_rounding_increment, "trade_fee_rounding_increment")
    _require_positive(balance_rounding_increment, "balance_rounding_increment")

    price = _decimal(price_dollars)
    quantity = _decimal(contracts)
    rate = _decimal(fee_rate)
    gross_cost = price * quantity
    raw_trade_fee = rate * quantity * price * (Decimal("1") - price)
    trade_fee = _ceil_to_increment(
        raw_trade_fee,
        _decimal(trade_fee_rounding_increment),
    )
    rounded_total_cost = _ceil_to_increment(
        gross_cost + trade_fee,
        _decimal(balance_rounding_increment),
    )
    return float(rounded_total_cost - gross_cost)


def compute_edge_table(
    bucket_probabilities: pd.DataFrame,
    orderbook_summary: pd.DataFrame,
    settings: EdgeSettings | None = None,
    evaluated_at: datetime | None = None,
) -> pd.DataFrame:
    """
    Compare model fair values with executable YES/NO asks after costs.
    """
    if not isinstance(bucket_probabilities, pd.DataFrame):
        raise TypeError("bucket_probabilities must be a pandas DataFrame")
    if not isinstance(orderbook_summary, pd.DataFrame):
        raise TypeError("orderbook_summary must be a pandas DataFrame")

    settings = settings or EdgeSettings()
    evaluated_at_dt = evaluated_at or datetime.now(timezone.utc)
    if bucket_probabilities.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    _validate_probability_inputs(bucket_probabilities)

    merged = _merge_probability_and_books(bucket_probabilities, orderbook_summary)
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        probability = _optional_float(row.get("probability"))
        fair_yes = probability
        fair_no = None if probability is None else 1.0 - probability
        records.append(
            _edge_record(
                row,
                outcome_side="YES",
                action="BUY_YES",
                fair_yes=fair_yes,
                fair_no=fair_no,
                evaluated_at=evaluated_at_dt,
                settings=settings,
            )
        )
        records.append(
            _edge_record(
                row,
                outcome_side="NO",
                action="BUY_NO",
                fair_yes=fair_yes,
                fair_no=fair_no,
                evaluated_at=evaluated_at_dt,
                settings=settings,
            )
        )

    return pd.DataFrame.from_records(records).reindex(columns=EDGE_COLUMNS)


def save_edge_table(edge_table: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edge_table.reindex(columns=EDGE_COLUMNS).to_csv(path, index=False)


def _edge_record(
    row: pd.Series,
    *,
    outcome_side: str,
    action: str,
    fair_yes: float | None,
    fair_no: float | None,
    evaluated_at: datetime,
    settings: EdgeSettings,
) -> dict[str, Any]:
    prefix = outcome_side.lower()
    fair_value = fair_yes if outcome_side == "YES" else fair_no
    executable_price = _optional_float(row.get(f"best_{prefix}_ask"))
    executable_size = _optional_float(row.get(f"best_{prefix}_ask_size"))
    best_bid = _optional_float(row.get(f"best_{prefix}_bid"))
    best_ask = _optional_float(row.get(f"best_{prefix}_ask"))
    spread = _optional_float(row.get(f"{prefix}_spread"))
    staleness = _staleness_seconds(row, evaluated_at)

    fee_estimate: float | None = None
    fee_per_contract: float | None = None
    gross_edge: float | None = None
    net_edge: float | None = None
    net_edge_percent: float | None = None
    if _finite(fair_value) and _finite(executable_price):
        fee_estimate = estimate_kalshi_buy_fee(
            executable_price,
            settings.evaluation_contracts,
            fee_rate=settings.fee_rate,
            trade_fee_rounding_increment=settings.trade_fee_rounding_increment,
            balance_rounding_increment=settings.balance_rounding_increment,
        )
        fee_per_contract = fee_estimate / settings.evaluation_contracts
        gross_edge = float(fair_value) - float(executable_price)
        net_edge = gross_edge - fee_per_contract - settings.slippage_buffer_dollars
        if float(executable_price) > 0.0:
            net_edge_percent = net_edge / float(executable_price)

    reasons = _no_trade_reasons(
        row=row,
        fair_value=fair_value,
        executable_price=executable_price,
        executable_size=executable_size,
        spread=spread,
        gross_edge=gross_edge,
        net_edge=net_edge,
        net_edge_percent=net_edge_percent,
        staleness_seconds=staleness,
        settings=settings,
    )
    edge_status = "NO_TRADE" if reasons else "CANDIDATE"

    return {
        "evaluated_at": evaluated_at.isoformat(),
        "row_id": row.get("row_id"),
        "event_ticker": row.get("event_ticker"),
        "ticker": row.get("ticker"),
        "bucket_index": row.get("bucket_index"),
        "bucket_name": row.get("bucket_name"),
        "bucket_lower_temp": row.get("bucket_lower_temp"),
        "bucket_upper_temp": row.get("bucket_upper_temp"),
        "probability": row.get("probability"),
        "outcome_side": outcome_side,
        "action": action,
        "fair_yes": fair_yes,
        "fair_no": fair_no,
        "fair_value": fair_value,
        "executable_price": executable_price,
        "executable_size": executable_size,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "gross_edge": gross_edge,
        "fee_estimate": fee_estimate,
        "fee_per_contract": fee_per_contract,
        "slippage_buffer": settings.slippage_buffer_dollars,
        "net_edge": net_edge,
        "net_edge_percent": net_edge_percent,
        "min_edge_dollars": settings.min_edge_dollars,
        "min_edge_percent": settings.min_edge_percent,
        "min_liquidity_contracts": settings.min_liquidity_contracts,
        "max_spread_dollars": settings.max_spread_dollars,
        "staleness_seconds": staleness,
        "max_staleness_seconds": settings.max_staleness_seconds,
        "orderbook_status": row.get("orderbook_status"),
        "orderbook_reason": row.get("orderbook_reason"),
        "settlement_status": row.get("settlement_status"),
        "settlement_reason": row.get("settlement_reason"),
        "settlement_trading_allowed": row.get("settlement_trading_allowed"),
        "probability_mode": row.get("probability_mode"),
        "probability_signal_status": row.get("probability_signal_status"),
        "probability_signal_reason": row.get("probability_signal_reason"),
        "fee_rate": settings.fee_rate,
        "evaluation_contracts": settings.evaluation_contracts,
        "trade_fee_rounding_increment": settings.trade_fee_rounding_increment,
        "balance_rounding_increment": settings.balance_rounding_increment,
        "edge_status": edge_status,
        "no_trade_reason": ";".join(reasons),
    }


def _no_trade_reasons(
    *,
    row: pd.Series,
    fair_value: float | None,
    executable_price: float | None,
    executable_size: float | None,
    spread: float | None,
    gross_edge: float | None,
    net_edge: float | None,
    net_edge_percent: float | None,
    staleness_seconds: float | None,
    settings: EdgeSettings,
) -> list[str]:
    reasons: list[str] = []
    probability_status = str(row.get("probability_signal_status", "OK") or "OK")
    if probability_status != "OK":
        reason = str(row.get("probability_signal_reason", "") or probability_status).strip()
        reasons.append(f"probability_signal:{reason}")

    if _explicit_false(row.get("settlement_trading_allowed")):
        status = str(row.get("settlement_status", "") or "settlement_state").strip()
        reason = str(row.get("settlement_reason", "") or status).strip()
        reasons.append(f"settlement_state:{reason}")

    orderbook_status = str(row.get("orderbook_status", "") or "").strip()
    orderbook_reason = str(row.get("orderbook_reason", "") or "").strip()
    if not orderbook_status:
        reasons.append("missing_orderbook_context")
    elif orderbook_status != "OK":
        reasons.append(orderbook_reason or f"orderbook_status:{orderbook_status}")

    if not _finite(fair_value):
        reasons.append("missing_fair_value")
    if not _finite(executable_price):
        reasons.append("missing_executable_price")
    if not _finite(executable_size):
        reasons.append("missing_executable_size")
    elif float(executable_size) < settings.min_liquidity_contracts:
        reasons.append("insufficient_liquidity")
    if not _finite(spread):
        reasons.append("missing_spread")
    elif float(spread) < -1e-9:
        reasons.append("crossed_orderbook")
    elif float(spread) > settings.max_spread_dollars + 1e-9:
        reasons.append("unusually_wide_orderbook")
    if _finite(staleness_seconds) and float(staleness_seconds) > settings.max_staleness_seconds:
        reasons.append("stale_orderbook")

    if not _finite(gross_edge):
        reasons.append("missing_gross_edge")
    elif float(gross_edge) <= 0.0:
        reasons.append("gross_edge_nonpositive")
    if not _finite(net_edge):
        reasons.append("missing_net_edge")
    elif float(net_edge) < settings.min_edge_dollars:
        reasons.append("edge_below_minimum")
    if (
        settings.min_edge_percent > 0.0
        and _finite(net_edge_percent)
        and float(net_edge_percent) < settings.min_edge_percent
    ):
        reasons.append("edge_percent_below_minimum")

    return _dedupe_reasons(reasons)


def _merge_probability_and_books(
    bucket_probabilities: pd.DataFrame,
    orderbook_summary: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = bucket_probabilities.copy()
    if orderbook_summary.empty or "ticker" not in orderbook_summary.columns:
        return probabilities
    summary = orderbook_summary.drop_duplicates("ticker", keep="last").copy()
    return probabilities.merge(summary, on="ticker", how="left", validate="many_to_one")


def _validate_probability_inputs(bucket_probabilities: pd.DataFrame) -> None:
    missing = [
        column
        for column in ["ticker", "bucket_name", "probability"]
        if column not in bucket_probabilities.columns
    ]
    if missing:
        raise ValueError(f"Bucket probabilities are missing columns: {missing}")
    probabilities = pd.to_numeric(bucket_probabilities["probability"], errors="coerce")
    values = probabilities.to_numpy(dtype=float)
    if not pd.Series(values).map(math.isfinite).all():
        raise ValueError("Bucket probabilities contain non-finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Bucket probabilities must be within [0, 1]")


def _staleness_seconds(row: pd.Series, evaluated_at: datetime) -> float | None:
    provided = _optional_float(row.get("staleness_seconds"))
    if provided is not None:
        return provided
    fetched_at = row.get("fetched_at")
    if fetched_at is None or pd.isna(fetched_at):
        return None
    parsed = pd.to_datetime(fetched_at, errors="coerce")
    if pd.isna(parsed):
        return None
    fetched_dt = parsed.to_pydatetime()
    if fetched_dt.tzinfo is None:
        fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
    evaluated_dt = evaluated_at if evaluated_at.tzinfo is not None else evaluated_at.replace(tzinfo=timezone.utc)
    return max(0.0, float((evaluated_dt.astimezone(timezone.utc) - fetched_dt.astimezone(timezone.utc)).total_seconds()))


def _ceil_to_increment(value: Decimal, increment: Decimal) -> Decimal:
    units = (value / increment).to_integral_value(rounding=ROUND_CEILING)
    return units * increment


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _finite(value: Any) -> bool:
    return _optional_float(value) is not None


def _explicit_false(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return not value
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip().lower() in {"false", "0", "no"}


def _validate_price(value: Any) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"price_dollars must be within [0, 1], got {value!r}")


def _require_positive(value: Any, name: str) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and greater than 0, got {value!r}")


def _require_nonnegative(value: Any, name: str) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")


def _dedupe_reasons(reasons: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        cleaned = str(reason).strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result
