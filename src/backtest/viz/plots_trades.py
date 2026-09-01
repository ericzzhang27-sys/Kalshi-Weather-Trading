"""Section 4: trade-level analysis visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.viz._plotutil import (COLOR_ACCENT, COLOR_MAIN, COLOR_NEG, COLOR_POS,
                                        add_winrate_line, bar_colors, empty_figure, style)
from src.backtest.viz.registry import register_plot
from src.backtest.viz.schema import TOD_ORDER, BacktestResult
from src.backtest.viz.stats import mae_mfe, streak_summary, streaks, weekday_pnl


@register_plot("trades", title="Trade timeline on equity", requires=("trades",))
def plot_trade_timeline(result: BacktestResult) -> go.Figure:
    eq = result.equity_series()
    t = result.prepared_trades().sort_values("signal_ts")
    fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, name="equity",
                               line=dict(color="#555", width=1.5)))
    wins = t[t["win"]]
    losses = t[~t["win"]]
    size = 6 + 10 * np.sqrt(pd.to_numeric(t["net_pnl"], errors="coerce").abs().fillna(0)
                            / max(t["gross_cost"].max() or 1.0, 1e-9))
    for frame, color, name in ((wins, COLOR_POS, "winning trades"), (losses, COLOR_NEG, "losing trades")):
        if len(frame) == 0:
            continue
        eq_at = eq.asof(pd.DatetimeIndex(frame["signal_ts"]))
        fig.add_trace(go.Scatter(
            x=frame["signal_ts"], y=eq_at.values, mode="markers", name=name,
            marker=dict(color=color, size=size.loc[frame.index].tolist(),
                        symbol="triangle-up" if color == COLOR_POS else "triangle-down", opacity=0.8),
            customdata=np.stack([frame["market_ticker"], frame["net_pnl"],
                                 frame["predicted_edge"], frame["contracts"]], axis=-1),
            hovertemplate=("%{x|%Y-%m-%d %H:%M}<br>%{customdata[0]}<br>P&L $%{customdata[1]:,.2f}"
                           "<br>edge %{customdata[2]:.1%}<extra></extra>"),
        ))
    return style(fig, f"Trade timeline over equity - {result.name}", y_title="Equity ($)")


@register_plot("trades", title="Trades on market price", requires=("prices",))
def plot_trades_on_price(result: BacktestResult, market_ticker: str | None = None) -> go.Figure:
    px = result.prices.copy()
    ticker_col = "market_ticker"
    ts_col = "timestamp"
    px[ts_col] = pd.to_datetime(px[ts_col], utc=True)
    t = result.prepared_trades()
    if market_ticker is None:
        market_ticker = str(t.groupby("market_ticker")["net_pnl"].sum().abs().idxmax())
    px = px[px[ticker_col].astype(str) == str(market_ticker)].sort_values(ts_col)
    trades = t[t["market_ticker"].astype(str) == str(market_ticker)]
    fig = go.Figure(go.Scatter(x=px[ts_col], y=pd.to_numeric(px["price"], errors="coerce"),
                               name="market price", line=dict(color="#666", width=1)))
    for mask, color, symbol, name in ((trades["win"], COLOR_POS, "triangle-up", "wins"),
                                      (~trades["win"] & ~trades["win"].eq(False).all(), COLOR_NEG, "triangle-down", "losses")):
        sub = trades[mask.fillna(False)]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(x=sub["signal_ts"], y=sub["entry_price"], mode="markers",
                                 name=name, marker=dict(color=color, symbol=symbol, size=11),
                                 customdata=np.stack([sub["side_norm"], sub["contracts"], sub["net_pnl"]], axis=-1),
                                 hovertemplate="%{x|%m-%d %H:%M}<br>%{customdata[0]} x%{customdata[1]}"
                                               "<br>P&L $%{customdata[2]:,.2f}<extra></extra>"))
    return style(fig, f"Trade entries on price - {market_ticker}", y_title="Price (c)")


@register_plot("trades", title="Win vs loss distributions", requires=("trades",))
def plot_win_loss_distribution(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    fig = make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5], row_heights=[0.62, 0.38],
                        vertical_spacing=0.14, horizontal_spacing=0.1,
                        specs=[[{"colspan": 2}, None],
                               [{"type": "box"}, {"type": "box"}]],
                        subplot_titles=("P&L distribution: winners vs losers",
                                        "Winner P&L spread", "Loser P&L spread"))
    fig.add_trace(go.Histogram(x=t[t["win"]]["net_pnl"], name="winners", nbinsx=40,
                               marker_color=COLOR_POS, opacity=0.75), row=1, col=1)
    fig.add_trace(go.Histogram(x=t[~t["win"]]["net_pnl"], name="losers", nbinsx=40,
                               marker_color=COLOR_NEG, opacity=0.75), row=1, col=1)
    fig.add_trace(go.Box(x=t[t["win"]]["net_pnl"], name="winners", marker_color=COLOR_POS,
                         boxmean=True), row=2, col=1)
    fig.add_trace(go.Box(x=t[~t["win"]]["net_pnl"], name="losers", marker_color=COLOR_NEG,
                         boxmean=True), row=2, col=2)
    fig.update_xaxes(title_text="Net P&L ($)", row=1, col=1)
    return style(fig, "Winning vs losing trade distributions", height=600)


@register_plot("trades", title="P&L vs holding time", requires=("trades",))
def plot_pnl_vs_holding_time(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    hold = pd.to_numeric(t["holding_days"], errors="coerce").clip(upper=14)
    pnl = pd.to_numeric(t["net_pnl"], errors="coerce")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Per-trade P&L vs holding days", "Average P&L by holding bucket"))
    fig.add_trace(go.Scatter(x=hold, y=pnl, mode="markers", name="trades",
                             marker=dict(color=np.where(pnl >= 0, COLOR_POS, COLOR_NEG), opacity=0.55, size=7),
                             hovertemplate="%{x:.1f}d<br>$%{y:,.2f}<extra></extra>"), row=1, col=1)
    bins = [-0.01, 0.25, 0.5, 1, 2, 3, 100]
    labels = ["<6h", "6-12h", "12-24h", "1-2d", "2-3d", ">3d"]
    bucket = pd.cut(hold, bins=bins, labels=labels)
    avg = pnl.groupby(bucket, observed=True).mean()
    n = pnl.groupby(bucket, observed=True).size()
    fig.add_trace(go.Bar(x=avg.index.astype(str), y=avg.values, marker_color=bar_colors(avg),
                         customdata=n.values, showlegend=False,
                         hovertemplate="%{x}<br>avg $%{y:,.2f}<br>n=%{customdata}<extra></extra>"),
                  row=1, col=2)
    fig.add_hline(y=0, row=1, col=2, line_color="#888", line_width=1)
    avg_hold = float(t["holding_days"].mean())
    return style(fig, f"P&L by holding period (average hold {avg_hold:.2f} days)", height=420)


@register_plot("trades", title="P&L by entry time of day", requires=("trades",))
def plot_pnl_by_tod(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    g = t.groupby("tod_bucket", observed=True)
    agg = g.agg(total_pnl=("net_pnl", "sum"), win_rate=("win", "mean"), n=("net_pnl", "size")) \
           .reindex([b for b in TOD_ORDER if b in set(t["tod_bucket"])])
    fig = go.Figure()
    fig.add_bar(x=agg.index.astype(str), y=agg["total_pnl"], marker_color=bar_colors(agg["total_pnl"]),
                name="total P&L", customdata=np.stack([agg["n"], agg["win_rate"]], axis=-1),
                hovertemplate="%{x}<br>$%{y:,.2f} total<br>n=%{customdata[0]}<br>win %{customdata[1]:.0%}<extra></extra>")
    add_winrate_line(fig, agg.index.astype(str), agg["win_rate"])
    fig.update_layout(xaxis_title="Entry time (America/New_York)")
    return style(fig, "Total P&L and win rate by entry time of day")


@register_plot("trades", title="P&L by weekday", requires=("trades",))
def plot_pnl_by_weekday(result: BacktestResult) -> go.Figure:
    wd = weekday_pnl(result)
    fig = go.Figure()
    fig.add_bar(x=wd.index.astype(str), y=wd["net_pnl"], marker_color=bar_colors(wd["net_pnl"]),
                name="total P&L", customdata=np.stack([wd["n_trades"], wd["win_rate"]], axis=-1),
                hovertemplate="%{x}<br>$%{y:,.2f}<br>n=%{customdata[0]}<extra></extra>")
    add_winrate_line(fig, wd.index.astype(str), wd["win_rate"])
    return style(fig, "Total P&L and win rate by entry weekday")


@register_plot("trades", title="P&L by month", requires=("trades",))
def plot_pnl_by_month(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    monthly = t.set_index("signal_ts")["net_pnl"].resample("ME").agg(["sum", "count"])
    monthly = monthly[monthly["count"] > 0]
    colors = bar_colors(monthly["sum"])
    fig = go.Figure(go.Bar(x=monthly.index.strftime("%Y-%m"), y=monthly["sum"], marker_color=colors,
                           customdata=monthly["count"],
                           hovertemplate="%{x}<br>$%{y:,.2f}<br>%{customdata} trades<extra></extra>"))
    return style(fig, "Monthly net P&L", y_title="$")


@register_plot("trades", title="MAE / MFE analysis", requires=("prices",))
def plot_mae_mfe(result: BacktestResult) -> go.Figure:
    t = mae_mfe(result)
    t = t.dropna(subset=["mae_price", "mfe_price"])
    if t.empty:
        return empty_figure("MAE/MFE unavailable: no market price path between entry and exit",
                            "Maximum Adverse / Favorable Excursion per trade")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("MAE vs MFE (price points)", "Realized P&L vs MFE"))
    outcome = np.where(t["net_pnl"] >= 0, COLOR_POS, COLOR_NEG)
    fig.add_trace(go.Scatter(x=t["mae_price"], y=t["mfe_price"], mode="markers",
                             marker=dict(color=outcome, opacity=0.6, size=7), name="trades",
                             hovertemplate="MAE %{x:.3f}<br>MFE %{y:.3f}<extra></extra>"), row=1, col=1)
    lims = [0, max(t["mae_price"].max(), t["mfe_price"].max()) * 1.05]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="MAE=MFE", showlegend=False,
                             line=dict(dash="dot", color="#999")), row=1, col=1)
    fig.update_xaxes(title_text="Max adverse excursion", row=1, col=1)
    fig.update_yaxes(title_text="Max favorable excursion", row=1, col=1)
    fig.add_trace(go.Scatter(x=t["mfe_price"], y=pd.to_numeric(t["net_pnl"], errors="coerce"),
                             mode="markers", marker=dict(color=COLOR_ACCENT, opacity=0.6, size=7),
                             name="pnl vs mfe",
                             hovertemplate="MFE %{x:.3f}<br>$%{y:,.2f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Max favorable excursion", row=1, col=2)
    fig.update_yaxes(title_text="Net P&L $", row=1, col=2)
    return style(fig, "Maximum Adverse / Favorable Excursion per trade")


@register_plot("trades", title="Consecutive wins / losses", requires=("trades",))
def plot_streaks(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().sort_values("signal_ts")
    s = streaks(pd.Series(pd.to_numeric(t["net_pnl"], errors="coerce").values))
    summary = streak_summary(pd.to_numeric(t["net_pnl"], errors="coerce"))
    fig = go.Figure(go.Bar(x=list(range(len(s))), y=s.values,
                           marker_color=[COLOR_POS if v >= 0 else COLOR_NEG for v in s.values],
                           hovertemplate="trade #%{x}<br>streak %{y:+.0f}<extra></extra>"))
    fig.add_annotation(text=(f"max win streak {summary['max_consecutive_wins']} &nbsp;|&nbsp; "
                             f"max loss streak {summary['max_consecutive_losses']}"),
                       xref="paper", yref="paper", x=0.01, y=1.08, showarrow=False,
                       font=dict(size=13))
    fig.update_layout(xaxis_title="trade #", yaxis_title="streak length (+win/-loss)")
    return style(fig, "Consecutive win/loss runs across trade sequence")


@register_plot("trades", title="Trade detail table", requires=("trades",))
def trades_table_figure(result: BacktestResult, page_size: int = 15) -> go.Figure:
    t = result.prepared_trades()
    cols = ["signal_ts", "market_ticker", "city", "bucket_label", "model_name", "strategy",
            "side_norm", "entry_price", "contracts", "fees", "slippage_usd",
            "predicted_edge", "settlement", "net_pnl", "holding_days"]
    view = t[[c for c in cols if c in t.columns]].copy()
    if "signal_ts" in view:
        view["signal_ts"] = pd.DatetimeIndex(view["signal_ts"]).strftime("%Y-%m-%d %H:%M")
    for c in ("entry_price", "predicted_edge", "settlement"):
        if c in view:
            view[c] = pd.to_numeric(view[c], errors="coerce").map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    for c in ("fees", "slippage_usd", "net_pnl"):
        if c in view:
            view[c] = pd.to_numeric(view[c], errors="coerce").map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
    view.columns = [str(c).replace("_", " ") for c in view.columns]
    fig = go.Figure(go.Table(
        header=dict(values=list(view.columns), fill_color="#263444", font=dict(color="white", size=11),
                    align="left"),
        cells=dict(values=[view[c] for c in view.columns],
                   fill_color=[[ "#f7fbff" if i % 2 == 0 else "white" for i in range(len(view))] for _ in view.columns],
                   font=dict(size=10), align="left", height=22),
        columnwidth=[82] + [70] * (len(view.columns) - 1),
    ))
    return style(fig, f"All {len(view)} trades (newest first in raw ledger order)",
                 height=120 + min(len(view), 30) * 24)


@register_plot("trades", title="Win rate / P&L by characteristic", requires=("trades",))
def plot_perf_by_characteristic(result: BacktestResult, by: str = "city") -> go.Figure:
    """Generic breakdown: swap `by` for any trade column."""
    t = result.prepared_trades()
    if by not in t.columns:
        return go.Figure()
    g = t.groupby(by, observed=True)
    agg = g.agg(n=("net_pnl", "size"), avg_pnl=("net_pnl", "mean"), total_pnl=("net_pnl", "sum"),
                win_rate=("win", "mean")).sort_values("total_pnl", ascending=False).head(20)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=agg.index.astype(str), y=agg["total_pnl"], marker_color=bar_colors(agg["total_pnl"]),
                name="total P&L", customdata=np.stack([agg["n"], agg["avg_pnl"]], axis=-1),
                secondary_y=False,
                hovertemplate="%{x}<br>$%{y:,.2f}<br>n=%{customdata[0]}<br>avg $%{customdata[1]:,.2f}<extra></extra>")
    add_winrate_line(fig, agg.index.astype(str), agg["win_rate"])
    fig.update_yaxes(title_text="Total net P&L $", secondary_y=False)
    fig.update_layout(xaxis_title=str(by))
    return style(fig, f"P&L and win rate by {by}")
