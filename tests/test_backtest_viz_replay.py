"""Tests for the daily market replay module (src/backtest/viz/replay)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.viz.replay import (  # noqa: E402
    bucket_title,
    build_daily_replay_figure,
    load_candle_frame,
)

CANDLES_CSV = REPO_ROOT / "data" / "kalshi" / "processed" / "historical_candles_processed.csv"


def _day_candles() -> pd.DataFrame:
    ts = pd.date_range("2025-06-01 11:00", periods=24, freq="h", tz="UTC")
    rows = []
    for ticker in ("HIGHNY-25JUN01-B78.5", "HIGHNY-25JUN01-T87"):
        base = pd.Series(np.linspace(0.2, 0.8, len(ts)), index=ts)
        rows.append(pd.DataFrame({
            "market_ticker": ticker,
            "event_ticker": "HIGHNY-25JUN01",
            "target_date": "2025-06-01",
            "floor_strike": 78.0 if "B" in ticker else np.nan,
            "cap_strike": 79.0 if "B" in ticker else np.nan,
            "volume": 10.0,
            "timestamp": ts,
            "price": base.to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def _day_trades() -> pd.DataFrame:
    ts = pd.DatetimeIndex([pd.Timestamp("2025-06-01 14:00", tz="UTC"),
                           pd.Timestamp("2025-06-01 16:00", tz="UTC")])
    return pd.DataFrame({
        "signal_ts": ts,
        "exit_ts": pd.DatetimeIndex([pd.Timestamp("2025-06-01 20:00", tz="UTC")] * 2),
        "market_ticker": ["HIGHNY-25JUN01-B78.5", "HIGHNY-25JUN01-B78.5"],
        "bucket_label": ["78deg to 79deg"] * 2,
        "side_norm": ["BUY_YES", "SELL_YES"],
        "direction": [1.0, -1.0],
        "entry_price": [0.35, 0.55],
        "exit_price": [0.60, 0.40],
        "contracts": [4.0, 2.0],
        "net_pnl": [0.90, -0.35],
        "predicted_edge": [0.08, 0.05],
    })


def test_build_daily_replay_figure_structure():
    fig = build_daily_replay_figure(_day_candles(), _day_trades())
    assert len(fig.data) > 0
    # one subplot row per market
    assert fig.layout.yaxis2 is not None  # two markets -> at least 2 y axes
    line_traces = [t for t in fig.data if t.mode == "lines"]
    marker_traces = [t for t in fig.data if t.mode.startswith("markers")]
    assert len(line_traces) == 2
    assert len(marker_traces) >= 1  # entry + exit markers present
    entry = next(t for t in marker_traces if t.name == "buy entry")
    sell_entry = next(t for t in marker_traces if t.name == "sell entry")
    exit_ = next(t for t in marker_traces if t.name == "exit")
    assert len(entry.x) == 1 and len(sell_entry.x) == 1 and len(exit_.x) == 2
    # hover text carries timestamp, price, position and P&L details
    assert "position" in entry.text[0]
    assert "P&L" in exit_.text[0]


def test_build_daily_replay_figure_no_data():
    empty = _day_candles().iloc[:0]
    fig = build_daily_replay_figure(empty, _day_trades().iloc[:0])
    assert any("No bucket markets" in (a.text or "") for a in fig.layout.annotations)


def test_bucket_title_variants():
    tr = pd.Series({"bucket_label": "78deg to 79deg"})
    assert "78-79" in bucket_title(tr, "TKR")
    floor_only = pd.Series({"floor_strike": 80.0, "cap_strike": 81.0})
    assert "80-81" in bucket_title(floor_only, "TKR")


def test_position_timeline_realized_pnl_sign():
    from src.backtest.viz.replay import _asof_state, _position_timeline
    t = _day_trades()
    timeline = _position_timeline(t)
    after_exit = timeline[timeline["ts"] >= pd.Timestamp("2025-06-01 20:00", tz="UTC")]
    assert after_exit["rpnl"].iloc[-1] == pytest.approx(0.90 - 0.35)
    assert after_exit["pos"].iloc[-1] == pytest.approx(0.0)     # flat after exits
    mid = timeline[(timeline["ts"] >= pd.Timestamp("2025-06-01 14:00", tz="UTC"))
                   & (timeline["ts"] < pd.Timestamp("2025-06-01 20:00", tz="UTC"))]
    assert mid["pos"].iloc[0] == pytest.approx(4.0)
    c = _day_candles()
    state = _asof_state(c[c["market_ticker"] == "HIGHNY-25JUN01-B78.5"]["timestamp"], timeline)
    assert (state["pos"] == 0.0).any() and (state["pos"] == 4.0).any()


@pytest.mark.skipif(not CANDLES_CSV.exists(), reason="real candle artifact missing")
def test_load_candle_frame_smoke():
    frame = load_candle_frame(CANDLES_CSV)
    assert {"timestamp", "market_ticker", "target_date", "price"} <= set(frame.columns)
    counts = frame.groupby("target_date")["market_ticker"].nunique()
    assert (counts == 6).mean() > 0.9
