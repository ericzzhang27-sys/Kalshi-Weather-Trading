"""Funded-account challenge simulator.

Replays a backtest ledger (or a bootstrapped resample of its trading days)
against a prop-firm rule set: profit target, max daily loss, max total
drawdown, minimum trading days, and a per-market notional concentration cap.

Rules are documented in config/funded_challenge.yaml. This module is
simulation-only; it never places orders and does not touch live trading code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ChallengeRules:
    starting_balance: float = 25_000.0
    profit_target_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    max_total_drawdown_pct: float = 0.10
    breach_action: str = "fail"            # "fail" | "pause_day"
    drawdown_mode: str = "static_from_initial"  # "static_from_initial" | "trailing_peak"
    min_trading_days: int = 5
    per_market_notional_cap_pct: float = 0.10

    @property
    def profit_target(self) -> float:
        return self.starting_balance * self.profit_target_pct

    @property
    def max_daily_loss(self) -> float:
        return self.starting_balance * self.max_daily_loss_pct

    @property
    def per_market_notional_cap(self) -> float:
        return self.starting_balance * self.per_market_notional_cap_pct


@dataclass
class ChallengeOutcome:
    result: str                 # "passed" | "failed_daily_loss" | "failed_drawdown" | "incomplete"
    days_traded: int
    final_equity: float
    peak_equity: float
    max_drawdown: float         # negative dollars, worst vs the reference level
    worst_day_pnl: float
    trades_taken: int
    trades_skipped: int = 0


def _prepare(ledger: pd.DataFrame) -> pd.DataFrame:
    df = ledger.copy()
    ts = pd.to_datetime(df["signal_timestamp"], utc=True).dt.tz_convert("America/New_York")
    df["day"] = ts.dt.date
    df["notional"] = pd.to_numeric(df["gross_cost"], errors="coerce").fillna(0.0)
    return df.sort_values("signal_timestamp")


def simulate_challenge(ledger: pd.DataFrame, rules: ChallengeRules) -> ChallengeOutcome:
    """Replay a chronological ledger through the challenge rules.

    Trades whose notional exceeds the per-market cap are clipped to it.
    A trade that would push its day's realized P&L beyond the daily-loss
    limit either fails the account ("fail") or is skipped and the day is
    paused ("pause_day").
    """
    df = _prepare(ledger)
    equity = rules.starting_balance
    floor = rules.starting_balance * (1 - rules.max_total_drawdown_pct)
    target_level = rules.starting_balance + rules.profit_target
    peak = equity
    day_pnl: dict[object, float] = {}
    paused_days: set[object] = set()
    worst_day = 0.0
    max_dd = 0.0
    skipped = 0
    taken = 0

    for row in df.itertuples():
        if equity >= target_level:
            break
        if row.day in paused_days:
            skipped += 1
            continue
        pnl = float(row.net_pnl)
        notional = float(row.notional)
        if notional > rules.per_market_notional_cap:
            scale = rules.per_market_notional_cap / notional
            pnl *= scale
        new_day = day_pnl.get(row.day, 0.0) + pnl
        new_equity = equity + pnl

        if new_day <= -rules.max_daily_loss:
            if rules.breach_action == "fail":
                return ChallengeOutcome(
                    result="failed_daily_loss", days_traded=len(day_pnl),
                    final_equity=new_equity, peak_equity=peak,
                    max_drawdown=min(max_dd, new_equity - peak), worst_day_pnl=new_day,
                    trades_taken=taken, trades_skipped=skipped)
            paused_days.add(row.day)
            skipped += 1
            continue

        equity = new_equity
        day_pnl[row.day] = new_day
        taken += 1
        peak = max(peak, equity)
        ref = rules.starting_balance if rules.drawdown_mode == "static_from_initial" else peak
        max_dd = min(max_dd, equity - ref)
        worst_day = min(worst_day, new_day)
        if rules.drawdown_mode == "static_from_initial" and equity <= floor:
            return ChallengeOutcome(
                result="failed_drawdown", days_traded=len(day_pnl),
                final_equity=equity, peak_equity=peak, max_drawdown=max_dd,
                worst_day_pnl=worst_day, trades_taken=taken, trades_skipped=skipped)
        if rules.drawdown_mode == "trailing_peak" and equity <= peak - rules.profit_target:
            return ChallengeOutcome(
                result="failed_drawdown", days_traded=len(day_pnl),
                final_equity=equity, peak_equity=peak, max_drawdown=max_dd,
                worst_day_pnl=worst_day, trades_taken=taken, trades_skipped=skipped)

    if equity >= target_level and len(day_pnl) >= rules.min_trading_days:
        result = "passed"
    elif equity >= target_level:
        result = "incomplete"          # hit target but too few trading days
    else:
        dd_limit = rules.starting_balance * rules.max_total_drawdown_pct
        ref = rules.starting_balance if rules.drawdown_mode == "static_from_initial" else peak
        result = "failed_drawdown" if equity <= ref - dd_limit else "incomplete"
    return ChallengeOutcome(
        result=result, days_traded=len(day_pnl), final_equity=equity,
        peak_equity=peak, max_drawdown=max_dd, worst_day_pnl=worst_day,
        trades_taken=taken, trades_skipped=skipped)


def bootstrap_pass_rate(ledger: pd.DataFrame, rules: ChallengeRules,
                        n_paths: int = 1000, max_days: int = 120,
                        seed: int = 7) -> dict:
    """Day-block bootstrap: sample trading days with replacement to estimate
    the probability of passing the challenge before failing it."""
    df = _prepare(ledger)
    by_day = {day: frame for day, frame in df.groupby("day")}
    days = list(by_day.keys())
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    durations: list[int] = []

    for _ in range(n_paths):
        equity = rules.starting_balance
        floor = rules.starting_balance * (1 - rules.max_total_drawdown_pct)
        target_level = rules.starting_balance + rules.profit_target
        result = None
        n_days = 0
        for i in rng.integers(0, len(days), size=max_days):
            day_frame = by_day[days[i]]
            day_total = _day_pnl_capped(day_frame, rules)
            n_days += 1
            if day_total <= -rules.max_daily_loss:
                result = "failed_daily_loss"
                break
            equity += day_total
            if equity <= floor:
                result = "failed_drawdown"
                break
            if equity >= target_level:
                result = "passed" if n_days >= rules.min_trading_days else "incomplete"
                break
        durations.append(n_days)
        counts[result or "timeout"] = counts.get(result or "timeout", 0) + 1

    return {
        "n_paths": n_paths,
        "pass_rate": counts.get("passed", 0) / n_paths,
        "outcome_counts": counts,
        "median_days_to_resolution": int(np.median(durations)),
    }


def _day_pnl_capped(day_frame: pd.DataFrame, rules: ChallengeRules) -> float:
    """Sum of a day's trade P&L with per-market notional caps applied."""
    total = 0.0
    for r in day_frame.itertuples():
        pnl = float(r.net_pnl)
        notional = float(r.notional)
        if notional > rules.per_market_notional_cap:
            pnl *= rules.per_market_notional_cap / notional
        total += pnl
    return total
