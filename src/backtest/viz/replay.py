"""Daily market replay: intraday price paths for all Kalshi temperature
bucket markets on one trading date, with strategy trade entries/exits overlaid.

Built for the backtesting dashboard's "Daily Market Replay" section. All
bucket charts share a single synchronized time axis (plotly subplots with
`shared_xaxes=True`, so zoom/pan on any row propagates to every other row).

Inputs are real pipeline artifacts:
  - `data/kalshi/processed/historical_candles_processed.csv`
        hourly Kalshi candlesticks (price = mid of yes bid/ask closes)
  - the engine ledger (`outputs/backtests/trades.*`) normalized by
    `BacktestResult.prepared_trades()`
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.viz.schema import NY_TZ

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDLES = REPO_ROOT / "data" / "kalshi" / "processed" / "historical_candles_processed.csv"

_CANDLE_USECOLS = [
    "market_ticker", "event_ticker", "target_date",
    "floor_strike", "cap_strike", "timestamp",
    "yes_bid_close", "yes_ask_close", "volume",
]

COLOR_BUY = "#2ca02c"
COLOR_SELL = "#d62728"
COLOR_WIN = "#2ca02c"
COLOR_LOSS = "#d62728"
COLOR_PRICE = "#1f77b4"


def load_candle_frame(candles_path: str | Path | None = None) -> pd.DataFrame:
    """Full candle history -> long frame [timestamp, market_ticker, target_date,
    floor_strike, cap_strike, volume, price].

    Price is the hourly mid of yes bid/ask closes; rows without a two-sided
    quote are dropped. Timestamps remain tz-aware UTC.
    """
    path = Path(candles_path) if candles_path else DEFAULT_CANDLES
    px = pd.read_csv(path, usecols=_CANDLE_USECOLS)
    bid = pd.to_numeric(px["yes_bid_close"], errors="coerce")
    ask = pd.to_numeric(px["yes_ask_close"], errors="coerce")
    px["price"] = (bid + ask) / 2.0
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True, errors="coerce")
    px["volume"] = pd.to_numeric(px["volume"], errors="coerce")
    px["target_date"] = px["target_date"].astype(str)
    px = px.dropna(subset=["timestamp", "price"])
    keep = ["market_ticker", "event_ticker", "target_date", "floor_strike",
            "cap_strike", "volume", "timestamp", "price"]
    return px[keep].reset_index(drop=True)


def bucket_title(row: pd.Series, ticker: str) -> str:
    """Human-readable bucket label like '78-79°F', '< 80°F' or '≥ 87°F'."""
    if "bucket_label" in row.index and pd.notna(row.get("bucket_label")):
        raw = str(row["bucket_label"]).strip()
        if raw.lower() not in ("", "nan", "unknown"):
            m = re.match(r"^(-?\d+(?:\.\d+)?)\s*deg\s*to\s*(-?\d+(?:\.\d+)?)\s*deg$", raw,
                         flags=re.IGNORECASE)
            if m:
                return f"{ticker} · {float(m.group(1)):g}-{float(m.group(2)):g}°F"
            cleaned = raw.replace("deg", "").replace("°", "").strip()
            return f"{ticker} · {cleaned}°F" if cleaned[-1:].isdigit() else f"{ticker} · {cleaned}"
    lo = _num(row.get("floor_strike"))
    hi = _num(row.get("cap_strike"))
    if np.isnan(lo) and np.isnan(hi):
        return ticker
    if np.isnan(lo):
        return f"{ticker} · < {hi:g}°F"
    if np.isnan(hi):
        return f"{ticker} · ≥ {lo:g}°F"
    return f"{ticker} · {lo:g}-{hi:g}°F"


def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out


def _ny_naive(ts: pd.Series) -> pd.Series:
    """UTC timestamps -> naive America/New_York wall-clock times for display."""
    return pd.to_datetime(ts).dt.tz_convert(NY_TZ).dt.tz_localize(None)


def _is_buy(side: str) -> bool:
    return str(side).upper().startswith("BUY")


def _position_timeline(trades_mkt: pd.DataFrame) -> pd.DataFrame:
    """Trade events -> [ts, position_after, realized_pnl_after] in UTC."""
    events: list[tuple[pd.Timestamp, float, float]] = []
    for r in trades_mkt.itertuples():
        signed = float(r.contracts) * float(r.direction)
        events.append((r.signal_ts, signed, 0.0))
        if pd.notna(r.exit_ts):
            events.append((r.exit_ts, -signed, float(r.net_pnl)))
    if not events:
        return pd.DataFrame({"ts": pd.DatetimeIndex([]), "pos": [], "rpnl": []})
    ev = pd.DataFrame(events, columns=["ts", "delta", "pnl"]).sort_values("ts")
    ev = ev.groupby("ts", as_index=False).sum().sort_values("ts")
    ev["pos"] = ev["delta"].cumsum()
    ev["rpnl"] = ev["pnl"].cumsum()
    return ev[["ts", "pos", "rpnl"]]


def _asof_state(candle_ts: pd.Series, timeline: pd.DataFrame) -> pd.DataFrame:
    """Carry position / realized P&L forward to every candle timestamp."""
    out = pd.DataFrame({"ts": candle_ts})
    if timeline.empty:
        out["pos"] = 0.0
        out["rpnl"] = 0.0
        return out
    merged = pd.merge_asof(out.sort_values("ts"), timeline.sort_values("ts"),
                           on="ts", direction="backward")
    merged["pos"] = merged["pos"].fillna(0.0)
    merged["rpnl"] = merged["rpnl"].fillna(0.0)
    return merged


def build_daily_replay_figure(day_candles: pd.DataFrame,
                              day_trades: pd.DataFrame,
                              title: str | None = None) -> go.Figure:
    """Synchronized multi-row figure: one intraday price chart per bucket market.

    Parameters
    ----------
    day_candles:
        Long frame for ONE target date ([market_ticker, timestamp(UTC), price, ...]).
    day_trades:
        Prepared ledger rows for that same target date (needs signal_ts, exit_ts,
        market_ticker, side_norm, direction, entry_price, exit_price, contracts,
        net_pnl).
    """
    tickers = list(dict.fromkeys(day_candles["market_ticker"].astype(str)))
    traded_only = sorted(set(day_trades["market_ticker"].astype(str)) - set(tickers))
    markets = tickers + traded_only
    if not markets:
        fig = go.Figure()
        fig.add_annotation(text="No bucket markets or trades for this date.",
                           showarrow=False, font=dict(size=16, color="#888"))
        return fig

    fig = make_subplots(
        rows=len(markets), cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[_title_for(ticker, day_trades, day_candles) for ticker in markets],
    )

    x_lo = day_candles["timestamp"].min() if len(day_candles) else None
    x_hi = day_candles["timestamp"].max() if len(day_candles) else None

    for i, ticker in enumerate(markets):
        row = i + 1
        show_legends = row == 1
        candles_mkt = day_candles[day_candles["market_ticker"].astype(str) == ticker]
        trades_mkt = day_trades[day_trades["market_ticker"].astype(str) == ticker]
        timeline = _position_timeline(trades_mkt)

        if len(candles_mkt):
            c = candles_mkt.sort_values("timestamp")
            state = _asof_state(c["timestamp"], timeline)
            # Hover: timestamp, market price, current position, realized trade P&L.
            text = [
                f"{t:%Y-%m-%d %H:%M ET}<br>mid: {p * 100:.1f}\u00a2"
                f"<br>position: {pos:+g} ct"
                f"<br>realized trade P&L: ${rpnl:+,.2f}"
                for t, p, pos, rpnl in zip(_ny_naive(state["ts"]), c["price"],
                                           state["pos"], state["rpnl"])
            ]
            fig.add_trace(go.Scatter(
                x=_ny_naive(state["ts"]), y=c["price"].to_numpy() * 100.0,
                mode="lines", name="mid price",
                line=dict(color=COLOR_PRICE, width=2),
                text=text, hovertemplate="%{text}<extra></extra>",
                showlegend=show_legends,
                legendgroup="price",
            ), row=row, col=1)

        if len(trades_mkt):
            pos_at = dict(zip(timeline["ts"], timeline["pos"])) if not timeline.empty else {}

            def _pos_after(ts: pd.Timestamp) -> str:
                known = [t for t in pos_at if t <= ts]
                return f"{pos_at[max(known)]:+g}" if known else "0"

            buys = trades_mkt[[_is_buy(s) for s in trades_mkt["side_norm"]]]
            sells = trades_mkt[~trades_mkt["side_norm"].map(_is_buy)]
            for side_df, symbol, color, label in (
                (buys, "triangle-up", COLOR_BUY, "buy entry"),
                (sells, "triangle-down", COLOR_SELL, "sell entry"),
            ):
                if not len(side_df):
                    continue
                size_txt = [
                    f"{c:g} ct @ {(p if pd.notna(p) else 0) * 100:.1f}\u00a2"
                    for c, p in zip(side_df["contracts"], side_df["entry_price"])
                ]
                edge_txt = [
                    f"<br>edge: {e:.1%}" if pd.notna(e) else ""
                    for e in side_df.get("predicted_edge", pd.Series([np.nan] * len(side_df)))
                ]
                text = [
                    f"{t:%Y-%m-%d %H:%M ET}<br>{lab.upper()} {s}"
                    f"{e}<br>position after: {pa} ct<br>net P&L: ${n:+,.2f}"
                    for t, s, e, n, lab, pa in zip(_ny_naive(side_df["signal_ts"]),
                                                   size_txt, edge_txt,
                                                   side_df["net_pnl"], label,
                                                   [_pos_after(t) for t in side_df["signal_ts"]])
                ]
                fig.add_trace(go.Scatter(
                    x=_ny_naive(side_df["signal_ts"]),
                    y=side_df["entry_price"].to_numpy() * 100.0,
                    mode="markers+text", name=label,
                    marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color="black")),
                    text=text, hovertemplate="%{text}<extra></extra>",
                    showlegend=show_legends, legendgroup=label,
                ), row=row, col=1)

            exited = trades_mkt[pd.notna(trades_mkt["exit_ts"]) & pd.notna(trades_mkt["exit_price"])]
            if len(exited):
                colors = [COLOR_WIN if n > 0 else COLOR_LOSS for n in exited["net_pnl"]]
                text = [
                    f"{t:%Y-%m-%d %H:%M ET}<br>EXIT {c:g} ct @ {p * 100:.1f}\u00a2"
                    f"<br>trade P&L: ${n:+,.2f} {'WIN' if n > 0 else 'LOSS'}"
                    for t, c, p, n in zip(_ny_naive(exited["exit_ts"]),
                                          exited["contracts"], exited["exit_price"],
                                          exited["net_pnl"])
                ]
                fig.add_trace(go.Scatter(
                    x=_ny_naive(exited["exit_ts"]),
                    y=exited["exit_price"].to_numpy() * 100.0,
                    mode="markers", name="exit",
                    marker=dict(symbol="x", size=10, color=colors, line=dict(width=1)),
                    text=text, hovertemplate="%{text}<extra></extra>",
                    showlegend=show_legends, legendgroup="exit",
                ), row=row, col=1)

        fig.update_yaxes(title_text="\u00a2", range=[-2, 102], row=row, col=1)
        fig.update_xaxes(showgrid=True, row=row, col=1)

    if x_lo is not None and x_hi is not None:
        pad = pd.Timedelta(hours=1)
        fig.update_xaxes(range=[_ny_naive(pd.Series([x_lo])).iloc[0] - pad,
                                _ny_naive(pd.Series([x_hi])).iloc[0] + pad])

    height = 210 * len(markets) + 110
    fig.update_layout(
        template="plotly_white",
        title=title or "Daily market replay",
        height=height,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        margin=dict(l=50, r=30, t=70, b=40),
    )
    return fig


def _title_for(ticker: str, day_trades: pd.DataFrame, day_candles: pd.DataFrame) -> str:
    tr = day_trades[day_trades["market_ticker"].astype(str) == ticker]
    if len(tr):
        row = tr.iloc[0]
    else:
        cd = day_candles[day_candles["market_ticker"].astype(str) == ticker]
        row = cd.iloc[0] if len(cd) else pd.Series(dtype=object)
    return bucket_title(row, ticker)
