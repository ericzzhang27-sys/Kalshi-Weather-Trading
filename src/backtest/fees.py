from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING

"""
Kalshi transaction fee implementation.

Formula (current as of 2026-01):
  For a YES price p in [0,1] and quantity q:

    gross_cost = p * q
    raw_fee    = fee_rate * q * p * (1 - p)
    trade_fee  = ceil_to_increment(raw_fee, trade_fee_increment)
    total_cost = ceil_to_increment(gross_cost + trade_fee, balance_increment)
    fee_charged = total_cost - gross_cost

Where fee_rate is typically 0.07 (7%) and increments are 0.0001 and 0.01 respectively.

This matches src/trading/edge.py estimate_kalshi_buy_fee but isolated for backtesting
and easily updatable if Kalshi changes fee schedule.

Put fee implementation in a separate function with clear documentation so it can
easily be updated - as required by spec section 5.

Assumptions:
 - Taker fees only (maker fees not modeled).
 - Fees apply on entry (buy). Exit fees (sell) use same formula but price is bid.
 - Settlement has no fee.
 - Contracts pay $1 if YES settles true else $0.
"""

DEFAULT_FEE_RATE = 0.07
DEFAULT_TRADE_INCREMENT = 0.0001
DEFAULT_BALANCE_INCREMENT = 0.01

def _decimal(v) -> Decimal:
    return Decimal(str(v))

def _ceil_to_increment(value: Decimal, inc: Decimal) -> Decimal:
    units = (value / inc).to_integral_value(rounding=ROUND_CEILING)
    return units * inc

def _validate_price(p: float) -> None:
    if not math.isfinite(p) or p < 0 or p > 1:
        raise ValueError(f"price must be in [0,1], got {p!r}")

def kalshi_fee(
    price_dollars: float,
    contracts: int = 1,
    fee_rate: float = DEFAULT_FEE_RATE,
    trade_fee_rounding_increment: float = DEFAULT_TRADE_INCREMENT,
    balance_rounding_increment: float = DEFAULT_BALANCE_INCREMENT,
) -> float:
    """
    Calculate Kalshi taker fee for buying `contracts` at `price_dollars`.

    Returns total fee (not per-contract) for the given quantity.

    Documented so fee schedule can be easily updated:
      - fee_rate: 0.07 = 7% of notional exposure per contract
      - raw_fee = fee_rate * q * p * (1-p)
      - ceiling to trade increment (0.0001) then balance increment (0.01)
    """
    _validate_price(price_dollars)
    if contracts <= 0 or not math.isfinite(contracts) or int(contracts) != contracts:
        raise ValueError(f"contracts must be a positive whole number, got {contracts!r}")
    if not math.isfinite(fee_rate) or fee_rate < 0:
        raise ValueError(f"fee_rate must be >=0, got {fee_rate!r}")
    price = _decimal(price_dollars)
    qty = _decimal(contracts)
    rate = _decimal(fee_rate)
    gross = price * qty
    raw = rate * qty * price * (Decimal("1") - price)
    trade_fee = _ceil_to_increment(raw, _decimal(trade_fee_rounding_increment))
    total = _ceil_to_increment(gross + trade_fee, _decimal(balance_rounding_increment))
    return float(total - gross)


kalshi_taker_fee = kalshi_fee

def kalshi_fee_per_contract(
    price_dollars: float,
    fee_rate: float = DEFAULT_FEE_RATE,
    trade_fee_rounding_increment: float = DEFAULT_TRADE_INCREMENT,
    balance_rounding_increment: float = DEFAULT_BALANCE_INCREMENT,
) -> float:
    """Fee per single contract at price p."""
    return kalshi_fee(price_dollars, contracts=1.0, fee_rate=fee_rate,
                      trade_fee_rounding_increment=trade_fee_rounding_increment,
                      balance_rounding_increment=balance_rounding_increment)

def net_edge_after_fees(
    model_probability: float,
    executable_price: float,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> float:
    """
    Net edge = model_prob - executable_price - fee_per_contract - slippage.
    Slippage handled separately; this is gross edge minus fee.
    """
    _validate_price(model_probability)
    _validate_price(executable_price)
    fee = kalshi_fee_per_contract(executable_price, fee_rate=fee_rate)
    return float(model_probability - executable_price - fee)
