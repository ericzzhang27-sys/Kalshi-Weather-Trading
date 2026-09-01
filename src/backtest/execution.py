from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

"""
Executable price assumptions (spec section 5):

- Buying YES: enter at historical YES ask
- Selling / closing YES: exit at historical YES bid
- If holds to settlement:
    YES settlement = $1 if correct bucket else $0
    NO equivalent = $1 if YES incorrect else $0 (modeled as selling YES)
- Fees separate (see fees.py)
- If required bid/ask missing: do not invent liquidity, skip trade, record reason.
"""

@dataclass(frozen=True)
class ExecutionConfig:
    buy_price_field: str = "yes_ask"   # field for buying YES
    sell_price_field: str = "yes_bid"  # field for selling YES
    allow_midpoint_fallback: bool = False
    fallback_reason_required: bool = True


@dataclass(frozen=True)
class ExecutableQuote:
    execution_timestamp: pd.Timestamp
    entry_price: float
    quote_field: str
    gap_minutes: float
    depth_contracts: float | None


def next_open_quote(signal: pd.Series, market_rows: pd.DataFrame, *, side: str, max_gap_minutes: int = 5) -> ExecutableQuote | None:
    signal_time = pd.to_datetime(signal["signal_timestamp"], utc=True)
    future = market_rows[pd.to_datetime(market_rows["timestamp"], utc=True) > signal_time].copy()
    if future.empty:
        return None
    future["timestamp"] = pd.to_datetime(future["timestamp"], utc=True)
    cutoff = signal_time + pd.Timedelta(minutes=int(max_gap_minutes))
    future = future[future["timestamp"] <= cutoff].sort_values("timestamp", kind="stable")
    for _, row in future.iterrows():
        gap = (row["timestamp"] - signal_time).total_seconds() / 60.0
        if side == "BUY_YES":
            raw_price, field, raw_depth = row.get("yes_ask_open"), "yes_ask_open", row.get("yes_ask_size_open")
        elif side == "BUY_NO":
            yes_bid = _finite_float(row.get("yes_bid_open"))
            raw_price = None if yes_bid is None else 1.0 - yes_bid
            field, raw_depth = "implied_no_ask_from_yes_bid_open", row.get("yes_bid_size_open")
        else:
            raise ValueError(f"Unsupported side: {side}")
        price = _finite_float(raw_price)
        if price is not None and 0.0 < price < 1.0:
            return ExecutableQuote(row["timestamp"], price, field, gap, _finite_float(raw_depth))
    return None


def adverse_fill_price(price: float, *, ticks: int = 0, tick_size: float = 0.01) -> float | None:
    if ticks < 0 or not math.isfinite(tick_size) or tick_size < 0.0:
        raise ValueError("adverse slippage ticks and tick size must be nonnegative")
    adjusted = float(price) + int(ticks) * float(tick_size)
    return adjusted if 0.0 < adjusted < 1.0 else None


def same_close_quote(signal: pd.Series, *, side: str) -> ExecutableQuote | None:
    if side == "BUY_YES":
        raw_price, field = signal.get("yes_ask_close"), "yes_ask_close"
    elif side == "BUY_NO":
        yes_bid = _finite_float(signal.get("yes_bid_close"))
        raw_price = None if yes_bid is None else 1.0 - yes_bid
        field = "implied_no_ask_from_yes_bid_close"
    else:
        raise ValueError(f"Unsupported side: {side}")
    price = _finite_float(raw_price)
    if price is None or not 0.0 < price < 1.0:
        return None
    timestamp = pd.to_datetime(signal["signal_timestamp"], utc=True)
    return ExecutableQuote(timestamp, price, field, 0.0, None)


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None

def resolve_executable_price(
    row: dict,
    side: str,
    config: ExecutionConfig | None = None,
) -> tuple[float | None, str]:
    """
    Resolve executable price for a given side.

    side: "BUY_YES" or "SELL_YES" (or "BUY_NO" which is equivalent to SELL_YES at NO price)
    Returns (price, reason). If price is None, reason explains why skipped.
    Never invent liquidity.
    """
    config = config or ExecutionConfig()
    side_u = side.upper()
    if side_u in ("BUY_YES",):
        price = row.get(config.buy_price_field) or row.get("yes_ask") or row.get("best_yes_ask")
        # Also try yes_ask_close from candles
        if price is None or (isinstance(price,float) and math.isnan(price)):
            price = row.get("yes_ask_close")
        if price is None or (isinstance(price, float) and math.isnan(price)):
            return None, "missing_yes_ask"
        try:
            p = float(price)
            if not math.isfinite(p) or p < 0 or p > 1:
                return None, "invalid_yes_ask"
            return p, ""
        except Exception:
            return None, "invalid_yes_ask"
    elif side_u in ("SELL_YES", "BUY_NO", "SELL"):
        price = row.get(config.sell_price_field) or row.get("yes_bid") or row.get("best_yes_bid")
        if price is None or (isinstance(price,float) and math.isnan(price)):
            price = row.get("yes_bid_close")
        if price is None or (isinstance(price, float) and math.isnan(price)):
            # NO equivalent: buying NO at price = 1 - YES_bid? But spec says edge_sell_yes = yes_bid - model_prob
            # For execution of NO, we still need YES bid? Actually buying NO price = NO ask = 1 - YES bid
            # So if YES bid missing, we cannot execute NO either.
            return None, "missing_yes_bid"
        try:
            p = float(price)
            if not math.isfinite(p) or p < 0 or p > 1:
                return None, "invalid_yes_bid"
            return p, ""
        except Exception:
            return None, "invalid_yes_bid"
    else:
        return None, f"unknown_side_{side}"

def settlement_value(
    market_ticker: str,
    bucket_lower: float | None,
    bucket_upper: float | None,
    actual_high: float,
) -> int:
    """
    YES settlement value $1 if actual_high in bucket, else $0.
    Uses lower-open, upper-closed: lower < actual <= upper.
    """
    v = float(actual_high)
    if not math.isfinite(v):
        raise ValueError(f"actual_high must be finite {actual_high!r}")
    if bucket_lower is not None and v <= float(bucket_lower):
        return 0
    if bucket_upper is not None and v > float(bucket_upper):
        return 0
    return 1

def pnl_for_trade(
    side: str,
    entry_price: float,
    settlement: int,
    contracts: float = 1.0,
    fees_entry: float = 0.0,
    fees_exit: float = 0.0,
) -> tuple[float, float]:
    """
    Returns (gross_pnl, net_pnl) for holding to settlement.

    For BUY_YES:
      gross = settlement*contracts - entry_price*contracts
      net   = gross - fees_entry  (settlement has no fee)

    For SELL_YES (short YES): gross = entry_price*contracts - settlement*contracts
    But spec models NO as buying NO at NO ask (~1 - YES bid). So edge_sell_yes = yes_bid - model_prob.
    For backtest we support BUY_NO as entry at NO ask = 1 - YES_bid, settlement = 1 if YES=0 else 0.
    Simplify: SELL_YES side means we sold YES at bid, if YES wins we owe $1.

    Currently engine only uses BUY_YES and BUY_NO (NO equivalent).
    """
    if side.upper() == "BUY_YES":
        gross = settlement * contracts - entry_price * contracts
        net = gross - fees_entry
        return float(gross), float(net)
    elif side.upper() in ("SELL_YES",):
        # Selling YES at bid: receive entry_price, pay settlement at close
        gross = entry_price * contracts - settlement * contracts
        net = gross - fees_entry  # fee on entry (selling also pays fee at sell price)
        return float(gross), float(net)
    elif side.upper() == "BUY_NO":
        # Buying NO: entry at NO ask = 1 - YES bid. For backtest we pass YES bid as entry_price? Actually caller should convert.
        # If entry_price is NO price (0-1), then gross = settlement_no - entry_price, settlement_no = 1 - settlement_yes
        # Here entry_price is expected to be NO ask price.
        # We'll treat generically: pnl = settlement_no - entry
        # But caller must provide correct settlement_no.
        # To avoid confusion, map BUY_NO to SELL_YES logic with converted prices outside.
        gross = settlement * contracts - entry_price * contracts
        net = gross - fees_entry
        return float(gross), float(net)
    else:
        raise ValueError(f"Unknown side {side}")
