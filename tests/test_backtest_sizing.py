"""Tests for position sizing (src/backtest/sizing), notably the notional-based
fractional Kelly scheme."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.sizing import (  # noqa: E402
    SizingConfig,
    fixed_contract_sizing,
    fixed_dollar_sizing,
    kelly_fractional_sizing,
)


def test_kelly_notional_matches_formula():
    cfg = SizingConfig(method="kelly_fractional", bankroll=1000.0,
                       kelly_fraction=0.25, max_contracts_per_market=1e9,
                       max_dollars_per_market=1e9)
    q, p = 0.60, 0.40
    expected_notional = 1000.0 * 0.25 * ((q - p) / (1.0 - p))
    got = kelly_fractional_sizing(q, p, config=cfg)
    assert got == pytest.approx(expected_notional / p)


def test_kelly_no_edge_returns_zero():
    cfg = SizingConfig(method="kelly_fractional")
    assert kelly_fractional_sizing(0.30, 0.30, config=cfg) == 0.0
    assert kelly_fractional_sizing(0.20, 0.30, config=cfg) == 0.0
    assert kelly_fractional_sizing(float("nan"), 0.30, config=cfg) == 0.0
    assert kelly_fractional_sizing(0.99, 0.0, config=cfg) == 0.0


def test_kelly_dollar_cap_binds_before_conversion():
    # Huge bankroll -> notional would exceed per-market dollar cap.
    cfg = SizingConfig(method="kelly_fractional", bankroll=1_000_000.0,
                       kelly_fraction=0.5, max_dollars_per_market=20.0,
                       max_contracts_per_market=10_000.0)
    q, p = 0.90, 0.10
    got = kelly_fractional_sizing(q, p, config=cfg)
    assert got == pytest.approx(20.0 / p)          # $20 notional at price p
    assert got * p == pytest.approx(20.0)


def test_kelly_contract_cap_binds():
    cfg = SizingConfig(method="kelly_fractional", bankroll=1000.0,
                       kelly_fraction=0.9, max_contracts_per_market=5.0)
    got = kelly_fractional_sizing(0.95, 0.50, config=cfg)
    assert got == pytest.approx(5.0)


def test_kelly_scales_with_edge_and_price():
    cfg = SizingConfig(method="kelly_fractional", bankroll=1000.0,
                       kelly_fraction=0.25, max_contracts_per_market=1e9,
                       max_dollars_per_market=1e9)
    small_edge = kelly_fractional_sizing(0.52, 0.50, config=cfg)
    big_edge = kelly_fractional_sizing(0.70, 0.50, config=cfg)
    assert big_edge > small_edge > 0.0
    # Same notional at half the price -> twice the contracts
    hi = kelly_fractional_sizing(0.60, 0.40, config=cfg)
    lo = kelly_fractional_sizing(0.60, 0.20, config=cfg)
    assert lo > hi


def test_other_methods_unchanged():
    cfg = SizingConfig(method="fixed_contracts")
    assert fixed_contract_sizing(config=cfg) == 1.0
    cfg2 = SizingConfig(method="fixed_dollar", max_contracts_per_market=100.0)
    assert fixed_dollar_sizing(0.50, config=cfg2) == pytest.approx(20.0)
