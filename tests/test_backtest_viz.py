"""Tests for the backtest visualization suite (src/backtest/viz).

Unit tests use minimal structural fixtures (a handful of rows) to exercise
code paths deterministically. Integration tests load the real pipeline
artifacts from outputs/backtests/ and are skipped when those files are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.viz import (  # noqa: E402
    BacktestResult,
    SECTION_ORDER,
    build_all_figures,
    comparison_table,
    create_backtest_report,
    create_comparison_report,
    get_registered_plots,
    run_single_plot,
)
from src.backtest.viz.schema import prepare_trades  # noqa: E402
from src.backtest.viz.stats import (  # noqa: E402
    calibration_table,
    exposure_frame,
    fee_scenarios,
    group_breakdown,
    mae_mfe,
    streak_summary,
    summary_stats,
)


@pytest.fixture(scope="module")
def rich_result() -> BacktestResult:
    """Small structural fixture covering every optional field the suite knows."""
    rng = np.random.default_rng(3)
    n = 60
    days = pd.date_range("2025-01-01", periods=n // 2, freq="D", tz="UTC")
    rows = []
    equity_vals = []
    bankroll = 1000.0
    tickers = ["HIGHNY-25A01-B33", "HIGHNY-25A02-B34", "HIGHNY-25B15-B41"]
    for i in range(n):
        day = days[i % len(days)]
        entry = day + pd.Timedelta(hours=int(7 + 9 * rng.random()))
        exit_ = day + pd.Timedelta(hours=20)
        contracts = float(rng.integers(5, 40))
        price = float(rng.uniform(0.1, 0.8))
        settle = float(rng.integers(0, 2))
        fees = 0.07 * contracts * price * (1 - price)
        net = contracts * (settle - price) - fees
        bankroll += net
        rows.append({
            "trade_id": i,
            "city": "NYC" if i % 3 else "CHI",
            "target_date": day.strftime("%Y-%m-%d"),
            "event_ticker": f"EV{i}",
            "market_ticker": tickers[i % len(tickers)],
            "bucket_label": f"{30 + i % 10}-{31 + i % 10}",
            "signal_timestamp": entry.isoformat(),
            "model_name": "fixture_model",
            "model_probability": float(np.clip(price + rng.normal(0, 0.05), 0.01, 0.99)),
            "entry_bid": round(max(price - 0.02, 0.01), 3),
            "entry_ask": round(price, 3),
            "entry_price": round(price, 3),
            "predicted_edge": float(rng.uniform(0.03, 0.18)),
            "side": "BUY_YES",
            "contracts": contracts,
            "gross_cost": contracts * price,
            "fees": float(fees),
            "exit_timestamp": exit_.isoformat(),
            "settlement": settle,
            "gross_pnl": contracts * (settle - price),
            "net_pnl": float(net),
            "bankroll_after": bankroll,
            "strategy": "fixture",
        })
        equity_vals.append((entry, bankroll))
    ledger = pd.DataFrame(rows)
    eq = pd.Series([v for _, v in sorted(equity_vals)], index=pd.DatetimeIndex(
        [t for t, _ in sorted(equity_vals)]))
    px_rows = []
    for ticker in tickers:
        base = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        for k, ts in enumerate(base):
            px_rows.append({"timestamp": ts, "market_ticker": ticker,
                            "price": float(np.clip(0.5 + np.sin(k / 4) * 0.2, 0.01, 0.99))})
    return BacktestResult(name="fixture", initial_capital=1000.0, equity=eq,
                          trades=ledger, prices=pd.DataFrame(px_rows),
                          meta={"fee_rate": 0.07})


def test_registry_has_plots_for_every_section():
    sections = {spec.section for spec in get_registered_plots()}
    assert set(SECTION_ORDER).issubset(sections | {"overview"})
    assert len(get_registered_plots()) >= 50


def test_summary_stats_keys(rich_result):
    s = summary_stats(rich_result)
    for key in ("total_return", "annualized_return", "sharpe", "sortino", "max_drawdown",
                "annualized_volatility", "win_rate", "profit_factor", "n_trades",
                "avg_trade_pnl", "median_trade_pnl", "avg_holding_days", "turnover",
                "fees_paid", "slippage_paid", "largest_win", "largest_loss"):
        assert key in s, key
    assert s["n_trades"] == 60
    assert s["total_return"] == pytest.approx(s["net_pnl"] / 1000.0)
    assert -1.0 <= s["max_drawdown"] <= 0.0


def test_prepare_trades_derived_columns(rich_result):
    t = rich_result.prepared_trades()
    for col in ("signal_ts", "exit_ts", "holding_days", "win", "direction",
                "slippage_per_contract", "tod_bucket", "edge_bin", "price_bin",
                "days_to_settle", "month_period", "temp_bucket"):
        assert col in t.columns, col


def test_exposure_frame_consistency(rich_result):
    e = exposure_frame(rich_result)
    assert len(e) > 0
    assert (e["n_positions"].min() >= 0)
    assert e["gross_exposure"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert e["exposure_pct"].notna().any()


def test_mae_mfe_with_prices(rich_result):
    t = mae_mfe(rich_result)
    assert t["mae_price"].notna().any()
    assert t["mfe_price"].notna().any()
    assert (t.dropna(subset=["mae_usd"])["mae_usd"] >= 0).all()


def test_calibration_and_scenarios(rich_result):
    cal = calibration_table(rich_result)
    assert not cal.empty and cal["n"].sum() == 60
    fs = fee_scenarios(rich_result)
    assert set(fs["fee_multiplier"]) >= {0.0, 1.0}
    zero_cost = fs[(fs["fee_multiplier"] == 0.0) & (fs["extra_slippage_cents"] == 0.0)]
    with_cost = fs[(fs["fee_multiplier"] == 1.0) & (fs["extra_slippage_cents"] == 0.0)]
    assert float(zero_cost["net_pnl"].iloc[0]) > float(with_cost["net_pnl"].iloc[0])


def test_group_breakdown_and_streaks(rich_result):
    gb = group_breakdown(rich_result.prepared_trades(), "city")
    assert set(gb.index) == {"NYC", "CHI"}
    st = streak_summary(rich_result.prepared_trades()["net_pnl"])
    assert st["max_consecutive_wins"] >= 0 and st["max_consecutive_losses"] >= 0


def test_filter_rebases_and_slices(rich_result):
    full = summary_stats(rich_result)
    sub = rich_result.filter(min_edge=1e9)
    assert len(sub.trades) == 0
    sub2 = rich_result.filter(start="2025-01-05", end="2025-01-20")
    assert 0 < len(sub2.trades) < full["n_trades"]
    stats2 = summary_stats(sub2)
    assert stats2["n_trades"] == len(sub2.trades)


def test_minimal_ledger_without_optional_fields():
    """Only timestamps + pnl -> everything else degrades gracefully."""
    n = 12
    ts = pd.date_range("2025-02-01", periods=n, freq="12h", tz="UTC")
    ledger = pd.DataFrame({
        "signal_timestamp": [t.isoformat() for t in ts],
        "net_pnl": np.linspace(-5, 15, n),
        "gross_pnl": np.linspace(-4, 16, n),
    })
    result = BacktestResult.from_ledger(ledger, name="minimal")
    figures = build_all_figures(result)
    built = sum(len(figs) for figs in figures.values())
    skipped = {sp.name for sp in get_registered_plots()} - {
        name for figs in figures.values() for name in figs}
    assert built >= 20
    assert "equity_curve" in figures["performance"]
    for plot_name in skipped:
        fig = run_single_plot(result, plot_name)
        assert fig is None or len(fig.data) >= 1


def test_create_backtest_report_html(rich_result, tmp_path):
    out = tmp_path / "report.html"
    figures = create_backtest_report(rich_result, output_path=out)
    html = out.read_text(encoding="utf-8")
    assert out.exists() and html.lstrip().startswith("<!DOCTYPE html>")
    for section in SECTION_ORDER[1:]:
        assert section in figures
        assert len(figures[section]) > 0
    assert "Overview" in html and "plotly" in html.lower()


def test_comparison_report(rich_result, tmp_path):
    other = BacktestResult.from_ledger(rich_result.trades.assign(
        strategy="variant"), initial_capital=1000.0, name="variant")
    other.equity = rich_result.equity_series() * 1.05
    table, figs = create_comparison_report(
        {"a": rich_result, "b": other}, output_path=tmp_path / "cmp.html")
    assert list(table["run"]) == ["b", "a"] or set(table["run"]) == {"a", "b"}
    assert {"equity", "drawdown"} <= set(figs)
    assert (tmp_path / "cmp.html").exists()


REAL_LEDGER = REPO_ROOT / "outputs" / "backtests" / "trades.csv"


@pytest.mark.skipif(not REAL_LEDGER.exists(), reason="real pipeline artifact missing")
def test_real_pipeline_artifact_end_to_end(tmp_path):
    result = BacktestResult.from_ledger(pd.read_csv(REAL_LEDGER), name="real_integration")
    figures = build_all_figures(result)
    built_names = [n for figs in figures.values() for n in figs]
    assert len(built_names) >= 45, built_names
    s = summary_stats(result)
    assert s["n_trades"] > 100 and np.isfinite(s["sharpe"])
    create_backtest_report(result, output_path=tmp_path / "real.html")
    assert (tmp_path / "real.html").stat().st_size > 200_000
