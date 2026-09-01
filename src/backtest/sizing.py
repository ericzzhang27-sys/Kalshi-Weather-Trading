from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

"""
Position sizing (spec section 8).

Methods:
 - fixed contract size (1 contract per signal)
 - fixed dollar exposure ($10 per trade)
 - fractional Kelly

Kelly must NOT be default. Default is fixed-size.
Hard limits for max exposure per market/event/day/bankroll.
"""

@dataclass
class SizingConfig:
    method: Literal["fixed_contracts", "fixed_dollar", "kelly_fractional"] = "fixed_contracts"
    fixed_contracts: float = 1.0
    fixed_dollar: float = 10.0
    kelly_fraction: float = 0.25
    max_contracts_per_market: float = 5.0
    max_contracts_per_order: int = 1
    max_contracts_per_event: float = 10.0
    max_dollars_per_order: float = 5.0
    max_dollars_per_market: float = 20.0
    max_dollars_per_event: float = 30.0
    max_daily_exposure: float = 100.0
    max_total_exposure: float = 50.0
    max_top_book_fraction: float = 0.10
    bankroll: float = 1000.0
    min_price: float = 0.01
    max_price: float = 0.99


def conservative_paper_sizing() -> SizingConfig:
    """Plan-mandated $1,000 bankroll limits for shadow/paper evaluation."""
    return SizingConfig(
        method="kelly_fractional",
        kelly_fraction=0.25,
        max_contracts_per_order=100,
        max_contracts_per_market=100,
        max_dollars_per_order=5.0,
        max_dollars_per_market=5.0,
        max_dollars_per_event=10.0,
        max_daily_exposure=10.0,
        max_total_exposure=50.0,
        max_top_book_fraction=0.10,
        bankroll=1000.0,
    )

def fixed_contract_sizing(contracts: float = 1.0, config: SizingConfig | None = None) -> float:
    cfg = config or SizingConfig()
    return min(float(contracts), cfg.max_contracts_per_market)

def fixed_dollar_sizing(price: float, target_dollar: float = 10.0, config: SizingConfig | None = None) -> float:
    """
    Contracts = dollar exposure / price.
    Example $10 at $0.50 -> 20 contracts, but capped.
    """
    cfg = config or SizingConfig()
    if price <= 0 or not math.isfinite(price):
        return 0.0
    contracts = target_dollar / price
    # Fractional contracts allowed but Kalshi min 1? We'll allow float for sizing calc, but execution rounds?
    return min(contracts, cfg.max_contracts_per_market)

def kelly_fractional_sizing(
    model_prob: float,
    market_price: float,
    fraction: float = 0.25,
    config: SizingConfig | None = None,
) -> float:
    """
    Fractional Kelly sized by NOTIONAL dollar exposure (contracts * price),
    not by a raw contract count.

    For a binary bet bought at price p paying $1 on win, with model win
    probability q, the full-Kelly fraction of bankroll to stake is:

        f* = (q - p) / (1 - p)      (= edge / (1 - p))

    We stake `fraction` (fractional Kelly, e.g. 0.25 = quarter Kelly) of that:

        notional = bankroll * fraction * f*

    and convert to contracts at the market price:

        contracts = notional / market_price

    The result is capped by max_dollars_per_market (notional cap applied
    BEFORE conversion) and max_contracts_per_market.
    """
    cfg = config or SizingConfig()
    if not (0 < market_price < 1) or not (0 < model_prob < 1):
        return 0.0
    edge = model_prob - market_price
    if edge <= 0:
        return 0.0
    kelly_full = edge / (1.0 - market_price)
    notional = cfg.bankroll * fraction * kelly_full
    if not math.isfinite(notional) or notional <= 0:
        return 0.0
    notional = min(notional, cfg.max_dollars_per_market)
    contracts = notional / market_price
    contracts = min(contracts, cfg.max_contracts_per_market)
    return float(max(0.0, contracts))

def size_position(
    model_prob: float,
    market_price: float,
    config: SizingConfig | None = None,
) -> float:
    cfg = config or SizingConfig()
    if cfg.method == "fixed_contracts":
        return float(math.floor(fixed_contract_sizing(cfg.fixed_contracts, cfg)))
    elif cfg.method == "fixed_dollar":
        return float(math.floor(fixed_dollar_sizing(market_price, cfg.fixed_dollar, cfg)))
    elif cfg.method == "kelly_fractional":
        return float(math.floor(kelly_fractional_sizing(model_prob, market_price, cfg.kelly_fraction, cfg)))
    else:
        return fixed_contract_sizing(1.0, cfg)


def requested_contracts(model_probability: float, entry_price: float, config: SizingConfig) -> int:
    raw = int(size_position(model_probability, entry_price, config))
    dollar_cap = math.floor(max(0.0, config.max_dollars_per_order) / entry_price)
    return max(0, min(raw, int(config.max_contracts_per_order), int(config.max_contracts_per_market), dollar_cap))


def cap_contracts_for_portfolio(
    requested: int,
    entry_price: float,
    *,
    available_cash: float,
    market_exposure: float,
    event_exposure: float,
    daily_exposure: float,
    total_exposure: float,
    config: SizingConfig,
) -> int:
    if requested <= 0 or not 0.0 < entry_price < 1.0:
        return 0
    caps = [
        math.floor(max(0.0, available_cash) / entry_price),
        math.floor(max(0.0, config.max_dollars_per_market - market_exposure) / entry_price),
        math.floor(max(0.0, config.max_dollars_per_event - event_exposure) / entry_price),
        math.floor(max(0.0, config.max_daily_exposure - daily_exposure) / entry_price),
        math.floor(max(0.0, config.max_total_exposure - total_exposure) / entry_price),
    ]
    return max(0, min(int(requested), *map(int, caps)))
