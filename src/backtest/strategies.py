from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .fees import kalshi_taker_fee

"""
Probability-based trading rules (spec section 6).

All strategies are modular; thresholds not optimized on full dataset.
Use chronological split for selection.

Strategies:
 A - simple edge threshold BUY_YES when edge >= threshold
 B - symmetric YES/NO
 C - confidence constrained
 D - one trade per event (greatest edge bucket)
 E - multiple buckets with caps
"""

@dataclass
class StrategyConfig:
    name: str = "A_simple_threshold"
    threshold: float = 0.05
    min_prob_distance_from_50: float = 0.0  # for C
    max_positions_per_event: int = 1
    max_capital_per_event: float | None = None
    allow_short: bool = False  # for B: allow BUY_NO
    max_spread: float | None = 0.25
    min_liquidity: float | None = None
    # signal policy handled by engine, not strategy

STRATEGY_THRESHOLDS_A = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15]


def threshold_signal(
    row: pd.Series,
    *,
    threshold: float,
    allow_buy_no: bool = True,
    fee_rate: float = 0.07,
    slippage_per_contract: float = 0.0,
) -> tuple[str | None, float]:
    """Compute a conservative cost-aware signal; execution still happens later."""
    try:
        probability = float(row.get("model_probability"))
    except (TypeError, ValueError):
        return None, 0.0
    raw_ask = row.get("yes_ask_close", row.get("yes_ask"))
    raw_bid = row.get("yes_bid_close", row.get("yes_bid"))
    yes_ask = float(raw_ask) if _finite(raw_ask) else None
    yes_bid = float(raw_bid) if _finite(raw_bid) else None
    lower_probability = float(row.get("probability_lower", probability)) if _finite(row.get("probability_lower", probability)) else probability
    upper_probability = float(row.get("probability_upper", probability)) if _finite(row.get("probability_upper", probability)) else probability
    yes_fee = kalshi_taker_fee(yes_ask, 1, fee_rate=fee_rate) if yes_ask is not None else 0.0
    no_price = 1.0 - yes_bid if yes_bid is not None else None
    no_fee = kalshi_taker_fee(no_price, 1, fee_rate=fee_rate) if no_price is not None else 0.0
    yes_edge = lower_probability - yes_ask - yes_fee - slippage_per_contract if yes_ask is not None else float("-inf")
    no_edge = (1.0 - upper_probability) - no_price - no_fee - slippage_per_contract if no_price is not None else float("-inf")
    if yes_edge >= threshold and yes_edge >= no_edge:
        return "BUY_YES", float(yes_edge)
    if allow_buy_no and no_edge >= threshold:
        return "BUY_NO", float(no_edge)
    return None, float(max(yes_edge, no_edge, 0.0))

def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False

def strategy_a_simple_threshold(row: pd.Series, threshold: float = 0.05) -> tuple[str | None, float | None, str]:
    """
    Buy YES when model_prob - ask >= threshold.
    Returns (side, edge, reason)
    """
    mp = row.get("model_probability")
    ask = row.get("yes_ask")
    if not _finite(mp) or not _finite(ask):
        return None, None, "missing_price_or_prob"
    edge = float(mp) - float(ask)
    if edge >= threshold:
        return "BUY_YES", edge, f"edge_{edge:.3f}_ge_{threshold}"
    return None, edge, f"edge_{edge:.3f}_below_{threshold}"

def strategy_b_symmetric(row: pd.Series, threshold: float = 0.05) -> tuple[str | None, float | None, str]:
    """
    Buy YES when P > ask + threshold
    Buy NO  when P < bid - threshold  (take equivalent NO position)
    """
    mp = row.get("model_probability")
    ask = row.get("yes_ask")
    bid = row.get("yes_bid")
    if not _finite(mp):
        return None, None, "missing_prob"
    if _finite(ask):
        edge_yes = float(mp) - float(ask)
        if edge_yes >= threshold:
            return "BUY_YES", edge_yes, f"YES_edge_{edge_yes:.3f}"
    if _finite(bid):
        edge_no = float(bid) - float(mp)  # edge for selling YES / buying NO
        # buying NO price is (1 - bid?) But for backtest we model NO profit as 1 - YES settlement.
        # So edge for NO: (1-mp) - (1 - bid) = bid - mp
        if edge_no >= threshold:
            return "BUY_NO", edge_no, f"NO_edge_{edge_no:.3f}"
    return None, None, "no_edge"

def strategy_c_confidence_constrained(row: pd.Series, threshold: float = 0.05, min_distance: float = 0.10) -> tuple[str | None, float | None, str]:
    """
    Only trade when edge >= threshold and |model_prob - 0.5| >= min_distance
    """
    mp = row.get("model_probability")
    if not _finite(mp):
        return None, None, "missing_prob"
    if abs(float(mp) - 0.5) < min_distance:
        return None, None, f"prob_{mp:.3f}_too_close_to_0.5"
    return strategy_a_simple_threshold(row, threshold)

def strategy_d_one_per_event(df_event: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    At each timestamp, pick bucket with greatest positive executable edge.
    df_event: all rows for one event at one timestamp (or overall, will group by timestamp)
    Returns subset with only best edge per timestamp if edge >= threshold
    """
    if df_event.empty:
        return df_event
    # Need to group by timestamp
    best_rows = []
    for ts, group in df_event.groupby("timestamp", sort=False):
        # Compute edges
        candidates = []
        for _, row in group.iterrows():
            side, edge, reason = strategy_a_simple_threshold(row, threshold)
            if side is not None:
                candidates.append((edge, row))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_edge, best_row = candidates[0]
            best_rows.append(best_row)
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows)

def strategy_e_multiple_buckets(
    df_ts_event: pd.DataFrame,
    threshold: float = 0.05,
    max_positions: int = 3,
    max_capital_per_event: float | None = None,
    fee_rate: float = 0.07,
) -> pd.DataFrame:
    """
    Allow multiple positive-edge buckets but impose caps.
    Also sanity check total probability exposure: if sum of selected YES probabilities >1? Not directly,
    but we cap number of positions and capital.
    """
    candidates = []
    for _, row in df_ts_event.iterrows():
        side, edge, _ = strategy_a_simple_threshold(row, threshold)
        if side is not None:
            candidates.append((edge, row))
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Take up to max_positions
    selected = candidates[:max_positions]
    # If max_capital_per_event set, we could filter by notional, but sizing handled separately
    if not selected:
        return pd.DataFrame()
    return pd.DataFrame([r for _, r in selected])

# ------------------------------------------------------------
# Generic dispatcher
# ------------------------------------------------------------
STRATEGY_FUNCS = {
    "A": strategy_a_simple_threshold,
    "B": strategy_b_symmetric,
    "C": strategy_c_confidence_constrained,
    "D": None,  # special per-event
    "E": None,  # special per-event
}

def apply_strategy(
    joined_df: pd.DataFrame,
    strategy: str = "A",
    threshold: float = 0.05,
    config: StrategyConfig | None = None,
) -> pd.DataFrame:
    """
    Apply strategy to joined dataset, return signals DataFrame with side, predicted_edge.

    For D/E, grouping logic applied outside; this handles per-row strategies A/B/C.
    """
    strategy = strategy.upper()
    signals = []
    for idx, row in joined_df.iterrows():
        if strategy == "A":
            side, edge, reason = strategy_a_simple_threshold(row, threshold)
        elif strategy == "B":
            side, edge, reason = strategy_b_symmetric(row, threshold)
        elif strategy == "C":
            min_dist = config.min_prob_distance_from_50 if config else 0.10
            side, edge, reason = strategy_c_confidence_constrained(row, threshold, min_dist)
        else:
            # For D/E caller should use dedicated group logic
            side, edge, reason = strategy_a_simple_threshold(row, threshold)
        if side is not None:
            signals.append({
                **row.to_dict(),
                "signal_side": side,
                "predicted_edge": edge,
                "signal_reason": reason,
                "strategy": strategy,
                "threshold": threshold,
            })
    if not signals:
        return pd.DataFrame()
    sig_df = pd.DataFrame(signals)
    # For D: filter to best per (target_date, timestamp)
    if strategy == "D":
        # group by target_date+timestamp and keep max edge
        sig_df = sig_df.sort_values(["target_date","timestamp","predicted_edge"], ascending=[True,True,False])
        sig_df = sig_df.groupby(["target_date","timestamp"], as_index=False).first()
    elif strategy == "E":
        max_pos = config.max_positions_per_event if config else 3
        # group and cap
        out = []
        for (tdate, ts), group in sig_df.groupby(["target_date","timestamp"], sort=False):
            group_sorted = group.sort_values("predicted_edge", ascending=False).head(max_pos)
            out.append(group_sorted)
        sig_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return sig_df
