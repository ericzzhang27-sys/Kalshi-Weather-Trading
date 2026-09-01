"""Tests for the funded-challenge simulator (src/backtest/challenge)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.challenge import (  # noqa: E402
    ChallengeRules,
    bootstrap_pass_rate,
    simulate_challenge,
)

RULES = ChallengeRules(starting_balance=10_000.0, min_trading_days=2)


def _ledger(pnls: list[float], days: list[str] | None = None) -> pd.DataFrame:
    n = len(pnls)
    days = days or [f"2025-06-{i + 1:02d}" for i in range(n)]
    ts = [pd.Timestamp(f"{d} 14:00", tz="America/New_York").tz_convert("UTC") for d in days]
    return pd.DataFrame({
        "signal_timestamp": [t.isoformat() for t in ts],
        "net_pnl": pnls,
        "gross_cost": [500.0] * n,
    })


def test_pass_on_steady_profit():
    led = _ledger([300.0] * 6)
    out = simulate_challenge(led, RULES)
    assert out.result == "passed"
    # target is +1000 on 10k -> reached during the 4th winning day
    assert out.final_equity == pytest.approx(11_200.0)


def test_daily_loss_breach_fails():
    led = _ledger([-600.0, -600.0])  # each day beyond the -$500 limit
    out = simulate_challenge(led, RULES)
    assert out.result == "failed_daily_loss"


def test_pause_day_action_skips_rest_of_day():
    rules = ChallengeRules(starting_balance=10_000.0, breach_action="pause_day",
                           min_trading_days=2)
    # Day 1: second trade pushes the day past -$500 -> paused and skipped.
    led = _ledger([-300.0, -250.0, 500.0, 500.0, 500.0],
                  days=["2025-06-01", "2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04"])
    out = simulate_challenge(led, rules)
    assert out.result == "passed"          # paused day skipped, then recovers
    assert out.trades_skipped >= 1


def test_drawdown_fail_static_floor():
    led = _ledger([-400.0] * 5)   # each day under the daily limit, cumulative breach
    out = simulate_challenge(led, RULES)
    assert out.result == "failed_drawdown"


def test_per_market_notional_cap_scales_pnl():
    led = _ledger([100.0], days=["2025-06-01"])   # $500 notional > $100 cap
    rules = ChallengeRules(starting_balance=10_000.0, per_market_notional_cap_pct=0.01)
    out = simulate_challenge(led, rules)
    assert out.final_equity == pytest.approx(10_000.0 + 100.0 * (100.0 / 500.0))


def test_bootstrap_returns_counts():
    rng_days = [f"2025-07-{d:02d}" for d in range(1, 21)]
    pnls = np.random.default_rng(3).normal(150.0, 250.0, size=20).tolist()
    led = _ledger(pnls, days=rng_days)
    res = bootstrap_pass_rate(led, RULES, n_paths=50, max_days=60, seed=1)
    assert 0.0 <= res["pass_rate"] <= 1.0
    assert sum(res["outcome_counts"].values()) == 50
