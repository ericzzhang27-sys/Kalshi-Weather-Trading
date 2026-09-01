from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .execution import adverse_fill_price, next_open_quote, same_close_quote
from .fees import kalshi_taker_fee
from .sizing import SizingConfig, cap_contracts_for_portfolio, requested_contracts
from .strategies import threshold_signal


@dataclass(frozen=True)
class BacktestConfig:
    threshold: float = 0.05
    allow_buy_no: bool = True
    execution_mode: Literal["next_candle_open", "same_candle_close_diagnostic"] = "next_candle_open"
    max_execution_gap_minutes: int = 5
    fee_rate: float = 0.07
    adverse_slippage_ticks: int = 0
    tick_size: float = 0.01
    one_position_per_market: bool = True
    require_depth_for_multiple_contracts: bool = True
    sizing: SizingConfig = field(default_factory=SizingConfig)


@dataclass
class EngineConfig:
    """Read-compatible adapter for legacy callers; now executes causally."""
    strategy: str = "A"
    threshold: float = 0.05
    strategy_config: object | None = None
    sizing_config: SizingConfig | None = None
    trade_policy: str = "one_position_per_market"
    allow_midpoint_fallback: bool = False
    max_spread: float | None = 0.25
    min_liquidity: float | None = None
    fee_rate: float = 0.07
    execution_delay_minutes: int = 0
    max_volume_fraction: float | None = None
    round_contracts: bool = True
    entry_cutoff_hour_ny: int | None = 16
    min_entry_price: float | None = None
    max_entry_price: float | None = None


def run_backtest(aligned: pd.DataFrame, config: BacktestConfig | EngineConfig | None = None) -> pd.DataFrame:
    if isinstance(config, EngineConfig):
        config = BacktestConfig(
            threshold=config.threshold,
            fee_rate=config.fee_rate,
            one_position_per_market=config.trade_policy != "continuous_rebalancing",
            sizing=config.sizing_config or SizingConfig(),
        )
    cfg = config or BacktestConfig()
    required = {
        "signal_timestamp", "target_date", "event_ticker", "market_ticker",
        "model_probability", "settlement", "settlement_source",
    }
    missing = sorted(required - set(aligned.columns))
    if missing:
        raise ValueError(f"Aligned rows missing required columns: {missing}")
    if aligned.empty:
        return _empty_ledger()
    if set(aligned["settlement_source"].dropna().unique()) != {"kalshi_result"}:
        raise ValueError("Backtest settlement must come exclusively from Kalshi result")

    rows = aligned.copy()
    rows["signal_timestamp"] = pd.to_datetime(rows["signal_timestamp"], utc=True)
    rows["timestamp"] = pd.to_datetime(rows.get("timestamp", rows["signal_timestamp"]), utc=True)
    rows["_settlement_timestamp"] = _settlement_times(rows)
    market_histories = {
        str(market): group.sort_values("timestamp", kind="stable")
        for market, group in rows.groupby(rows["market_ticker"].astype(str), sort=False)
    }
    cash = float(cfg.sizing.bankroll)
    active: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    traded_markets: set[str] = set()

    for _, signal in rows.sort_values(["signal_timestamp", "market_ticker"], kind="stable").iterrows():
        now = signal["signal_timestamp"]
        cash = _release_settled(active, completed, cash, now)
        market = str(signal["market_ticker"])
        if cfg.one_position_per_market and market in traded_markets:
            continue
        side, predicted_edge = threshold_signal(
            signal,
            threshold=float(cfg.threshold),
            allow_buy_no=cfg.allow_buy_no,
            fee_rate=cfg.fee_rate,
            slippage_per_contract=cfg.adverse_slippage_ticks * cfg.tick_size,
        )
        if side is None:
            continue
        market_rows = market_histories[market]
        quote = (
            next_open_quote(signal, market_rows, side=side, max_gap_minutes=cfg.max_execution_gap_minutes)
            if cfg.execution_mode == "next_candle_open"
            else same_close_quote(signal, side=side)
        )
        if quote is None or quote.execution_timestamp >= signal["_settlement_timestamp"]:
            continue
        entry_price = adverse_fill_price(
            quote.entry_price,
            ticks=cfg.adverse_slippage_ticks,
            tick_size=cfg.tick_size,
        )
        if entry_price is None:
            continue
        fair = float(signal["model_probability"])
        lower_probability = float(signal.get("probability_lower", fair))
        upper_probability = float(signal.get("probability_upper", fair))
        side_probability = fair if side == "BUY_YES" else 1.0 - fair
        fee_per_contract = kalshi_taker_fee(entry_price, 1, fee_rate=cfg.fee_rate)
        point_net_edge = side_probability - entry_price - fee_per_contract
        lower_side_probability = lower_probability if side == "BUY_YES" else 1.0 - upper_probability
        lower_net_edge = lower_side_probability - entry_price - fee_per_contract
        if lower_net_edge + 1e-12 < float(cfg.threshold):
            continue
        request = requested_contracts(side_probability, entry_price, cfg.sizing)
        if cfg.require_depth_for_multiple_contracts and quote.depth_contracts is None:
            request = min(request, 1)
        elif quote.depth_contracts is not None:
            demonstrated_depth = int(np.floor(quote.depth_contracts * cfg.sizing.max_top_book_fraction))
            request = min(request, demonstrated_depth)
        exposure = _active_exposures(active, signal)
        contracts = cap_contracts_for_portfolio(
            request, entry_price, available_cash=cash,
            market_exposure=exposure["market"], event_exposure=exposure["event"],
            daily_exposure=exposure["daily"], total_exposure=exposure["total"],
            config=cfg.sizing,
        )
        if contracts <= 0:
            continue
        fee = kalshi_taker_fee(entry_price, contracts, fee_rate=cfg.fee_rate)
        cost = entry_price * contracts + fee
        if cost > cash + 1e-12:
            continue
        cash -= cost
        yes_settlement = int(signal["settlement"])
        side_settlement = yes_settlement if side == "BUY_YES" else 1 - yes_settlement
        active.append({
            "trade_id": len(active) + len(completed) + 1,
            "city": signal.get("city", "NYC"),
            "target_date": str(signal["target_date"]),
            "event_ticker": str(signal["event_ticker"]),
            "market_ticker": market,
            "signal_timestamp": now,
            "execution_timestamp": quote.execution_timestamp,
            "settlement_timestamp": signal["_settlement_timestamp"],
            "prediction_time": signal.get("prediction_time"),
            "probability_age_minutes": signal.get("probability_age_minutes"),
            "model_probability": fair,
            "side": side,
            "entry_price": entry_price,
            "unadjusted_entry_price": quote.entry_price,
            "quote_source": signal.get("quote_source", "kalshi_candlestick"),
            "quote_field": quote.quote_field,
            "execution_gap_minutes": quote.gap_minutes,
            "contracts": int(contracts),
            "gross_cost": entry_price * contracts,
            "fees": fee,
            "slippage_per_contract": entry_price - quote.entry_price,
            "cash_committed": cost,
            "predicted_edge": float(point_net_edge),
            "lower_confidence_edge": float(lower_net_edge),
            "settlement": side_settlement,
            "settlement_source": "kalshi_result",
            "execution_assumption": cfg.execution_mode,
            "evidence_label": (
                "historical_proxy_validated"
                if cfg.execution_mode == "next_candle_open"
                else "diagnostic_noncausal"
            ),
            "supports_profitability_claim": bool(quote.depth_contracts is not None),
            "_payout": float(side_settlement * contracts),
        })
        traded_markets.add(market)

    cash = _release_settled(active, completed, cash, pd.Timestamp.max.tz_localize("UTC"))
    if not completed:
        return _empty_ledger()
    ledger = pd.DataFrame(completed).sort_values(["settlement_timestamp", "trade_id"])
    ledger["gross_pnl"] = ledger["_payout"] - ledger["gross_cost"]
    ledger["net_pnl"] = ledger["_payout"] - ledger["cash_committed"]
    return ledger.drop(columns=["_payout"]).reset_index(drop=True)


def _settlement_times(rows: pd.DataFrame) -> pd.Series:
    if "settlement_timestamp" not in rows:
        raise ValueError("Canonical market rows require settlement_timestamp for capital locking")
    result = pd.to_datetime(rows["settlement_timestamp"], errors="coerce", utc=True)
    if result.isna().any():
        raise ValueError("settlement_timestamp contains missing or invalid values")
    return result


def _release_settled(active, completed, cash: float, now: pd.Timestamp) -> float:
    remaining = []
    for position in active:
        if position["settlement_timestamp"] <= now:
            cash += float(position["_payout"])
            position["cash_after_settlement"] = cash
            completed.append(position)
        else:
            remaining.append(position)
    active[:] = remaining
    return cash


def _active_exposures(active, signal: pd.Series) -> dict[str, float]:
    return {
        "market": sum(float(x["gross_cost"]) for x in active if x["market_ticker"] == str(signal["market_ticker"])),
        "event": sum(float(x["gross_cost"]) for x in active if x["event_ticker"] == str(signal["event_ticker"])),
        "daily": sum(float(x["gross_cost"]) for x in active if x["target_date"] == str(signal["target_date"])),
        "total": sum(float(x["gross_cost"]) for x in active),
    }


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_id", "target_date", "event_ticker", "market_ticker",
        "signal_timestamp", "execution_timestamp", "settlement_timestamp", "side",
        "entry_price", "contracts", "fees", "settlement", "settlement_source",
        "gross_pnl", "net_pnl",
    ])


def save_ledger(ledger: pd.DataFrame, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "trades.csv"
    ledger.to_csv(path, index=False)
    return path
