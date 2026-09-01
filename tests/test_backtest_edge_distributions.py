import numpy as np
import pandas as pd
import pytest

from src.backtest.edge_distributions import (
    EDGE_LABELS,
    bootstrap_outcome_distributions,
    implied_win_probability,
    pnl_distribution_by_edge,
    pnl_histogram_by_edge,
    wilson_interval,
    winrate_by_edge,
)


def make_ledger(rows):
    return pd.DataFrame(rows)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 < lo < 0.8 < hi < 1.0
    lo0, hi0 = wilson_interval(0, 5)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lon, hin = wilson_interval(1, 0)
    assert np.isnan(lon) and np.isnan(hin)


def test_implied_win_probability_side_aware():
    ledger = make_ledger([
        {"side": "BUY_YES", "model_probability": 0.7},
        {"side": "BUY_NO", "model_probability": 0.3},
        {"side": "BUY_NO", "model_probability": None},
    ])
    implied = implied_win_probability(ledger)
    assert implied.iloc[0] == pytest.approx(0.7)
    assert implied.iloc[1] == pytest.approx(0.7)
    assert np.isnan(implied.iloc[2])


def test_pnl_distribution_by_edge_buckets_and_stats():
    # 20 trades with edge 0.03: winners +$0.60/contract, losers -$0.40/contract
    rows = []
    for i in range(12):
        rows.append({"predicted_edge": 0.03, "net_pnl": 0.60, "contracts": 1.0,
                     "model_probability": 0.6, "side": "BUY_YES"})
    for i in range(8):
        rows.append({"predicted_edge": 0.03, "net_pnl": -0.40, "contracts": 1.0,
                     "model_probability": 0.6, "side": "BUY_YES"})
    ledger = make_ledger(rows)
    table = pnl_distribution_by_edge(ledger)
    row = table[table["edge_bucket"] == "2-5%"].iloc[0]
    assert int(row["n_trades"]) == 20
    assert row["win_rate"] == pytest.approx(0.6)
    assert row["mean_pnl_per_contract"] == pytest.approx((12 * 0.6 - 8 * 0.4) / 20)
    assert row["pnl_q50"] == pytest.approx(0.6)  # majority winners -> median win
    assert row["total_net_pnl"] == pytest.approx(12 * 0.6 - 8 * 0.4)
    assert row["bootstrap_mean_lo"] <= row["mean_pnl_per_contract"] <= row["bootstrap_mean_hi"]
    empty = table[table["edge_bucket"] == "15%+"].iloc[0]
    assert int(empty["n_trades"]) == 0


def test_pnl_distribution_scales_with_contracts():
    ledger = make_ledger([
        {"predicted_edge": 0.06, "net_pnl": 1.2, "contracts": 2.0},
        {"predicted_edge": 0.06, "net_pnl": -0.8, "contracts": 2.0},
    ])
    table = pnl_distribution_by_edge(ledger)
    row = table[table["edge_bucket"] == "5-10%"].iloc[0]
    assert row["max_pnl_per_contract"] == pytest.approx(0.6)
    assert row["min_pnl_per_contract"] == pytest.approx(-0.4)
    assert row["mean_pnl_per_contract"] == pytest.approx((0.6 - 0.4) / 2)


def test_winrate_by_edge_calibration_gap():
    # Implied q=0.65 but realized win rate 0.9 -> positive calibration gap
    rows = ([{"predicted_edge": 0.08, "net_pnl": 0.5, "contracts": 1.0,
              "model_probability": 0.65, "side": "BUY_YES"}] * 9 +
            [{"predicted_edge": 0.08, "net_pnl": -0.5, "contracts": 1.0,
              "model_probability": 0.65, "side": "BUY_YES"}])
    ledger = make_ledger(rows)
    table = winrate_by_edge(ledger)
    row = table[table["edge_bucket"] == "5-10%"].iloc[0]
    assert int(row["n_trades"]) == 10
    assert row["win_rate"] == pytest.approx(0.9)
    assert row["avg_implied_win_prob"] == pytest.approx(0.65)
    assert row["calibration_gap_pp"] == pytest.approx(25.0)
    assert row["wilson_lo"] < 0.9 < row["wilson_hi"]


def test_bootstrap_samples_shapes():
    rng = np.random.default_rng(0)
    ledger = make_ledger({
        "predicted_edge": rng.uniform(0.01, 0.2, 100),
        "net_pnl": rng.choice([0.5, -0.5], 100),
        "contracts": np.ones(100),
        "model_probability": np.full(100, 0.55),
        "side": ["BUY_YES"] * 100,
    })
    out = bootstrap_outcome_distributions(ledger, n_resamples=50, seed=1)
    for key in ("pnl_samples", "winrate_samples"):
        assert not out[key].empty
        assert out[key]["edge_bucket"].nunique() >= 1
    assert len(out["pnl_samples"]) == 50 * out["pnl_samples"]["edge_bucket"].nunique()


def test_histogram_table_structure():
    ledger = make_ledger([
        {"predicted_edge": 0.03, "net_pnl": 0.5, "contracts": 1.0},
        {"predicted_edge": 0.03, "net_pnl": -0.5, "contracts": 1.0},
    ])
    hist = pnl_histogram_by_edge(ledger)
    assert not hist.empty
    assert list(hist.columns) == ["pnl_bin"] + EDGE_LABELS + ["unknown"]
