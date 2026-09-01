"""Section 5: strategy edge diagnostics (prediction-market focused)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.viz._plotutil import (COLOR_ACCENT, COLOR_MAIN, COLOR_NEG, COLOR_POS,
                                        add_winrate_line, bar_colors, empty_figure, style)
from src.backtest.viz.registry import register_plot
from src.backtest.viz.schema import EDGE_LABELS, BacktestResult
from src.backtest.viz.stats import calibration_scores, calibration_table, expected_vs_realized


@register_plot("edge", title="Expected edge at entry vs realized P&L", requires=("col:predicted_edge",))
def plot_edge_vs_realized(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    edge = pd.to_numeric(t["predicted_edge"], errors="coerce")
    pnl_pc = pd.to_numeric(t["net_pnl"], errors="coerce") / pd.to_numeric(t["contracts"], errors="coerce").replace(0, np.nan)
    fig = go.Figure(go.Scatter(
        x=edge, y=pnl_pc, mode="markers",
        marker=dict(color=np.where(pnl_pc >= 0, COLOR_POS, COLOR_NEG), opacity=0.55, size=7),
        customdata=t["market_ticker"].astype(str),
        hovertemplate="edge %{x:.1%}<br>realized/contract $%{y:+.3f}<br>%{customdata}<extra></extra>",
    ))
    ok = edge.notna() & pnl_pc.notna()
    if ok.sum() > 10:
        coef = np.polyfit(edge[ok], pnl_pc[ok], 1)
        xs = np.linspace(np.nanmin(edge), np.nanmax(edge), 50)
        fig.add_trace(go.Scatter(x=xs, y=coef[0] * xs + coef[1], mode="lines",
                                 name=f"fit (corr {np.corrcoef(edge[ok], pnl_pc[ok])[0, 1]:.2f})",
                                 line=dict(color=COLOR_MAIN, width=2)))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_yaxes(title="$ per contract")
    fig.update_xaxes(title="Predicted edge at entry", tickformat=".0%")
    return style(fig, "Did bigger predicted edges produce bigger realized profits?")


@register_plot("edge", title="Calibration / reliability plot", requires=("col:model_probability", "col:settlement"))
def plot_calibration(result: BacktestResult, bins: int = 10) -> go.Figure:
    cal = calibration_table(result, bins=bins)
    if cal.empty:
        return go.Figure()
    scores = calibration_scores(result)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.12,
                        subplot_titles=("Reliability curve", "Sample count per bin"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dot", color="#999")), row=1, col=1)
    size = 14 * np.sqrt(cal["n"] / max(cal["n"].max(), 1))
    fig.add_trace(go.Scatter(x=cal["avg_predicted"], y=cal["realized_freq"], mode="markers",
                             name="model", marker=dict(color=COLOR_MAIN, size=size, opacity=0.8,
                                                       line=dict(width=1, color="#333")),
                             customdata=cal["n"],
                             hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}"
                                           "<br>n=%{customdata}<extra></extra>"), row=1, col=1)
    fig.update_xaxes(title="Model probability", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="Realized frequency", range=[0, 1], row=1, col=1)
    labels = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in cal["bin"]]
    fig.add_trace(go.Bar(x=labels, y=cal["n"], marker_color=COLOR_ACCENT, showlegend=False,
                         name="count"), row=1, col=2)
    note = (f"ECE {scores.get('ece', float('nan')):.3%} | "
            f"Brier {scores.get('brier', float('nan')):.3f} | n={scores.get('n', 0)}")
    fig.add_annotation(text=note, xref="paper", yref="paper", x=0.02, y=1.12, showarrow=False,
                       font=dict(size=13))
    fig.update_xaxes(tickangle=45, row=1, col=2)
    return style(fig, "Model probability calibration on traded outcomes")


@register_plot("edge", title="Performance by predicted-edge bucket", requires=("col:predicted_edge",))
def plot_edge_bucket_panel(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    t = t.copy()
    t["edge_bin"] = pd.Categorical(t["edge_bin"], categories=EDGE_LABELS + ["unknown"], ordered=True)
    g = t.groupby("edge_bin", observed=False)
    agg = g.agg(n=("net_pnl", "size"), win_rate=("win", "mean"), avg_pnl=("net_pnl", "mean"),
                total_pnl=("net_pnl", "sum"), avg_edge=("predicted_edge", "mean")).reset_index()
    agg["label"] = agg["edge_bin"].astype(str) + f" (avg {agg['avg_edge'].map(lambda v: f'{v:.1%}' if pd.notna(v) else '-')})"
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.12,
                        subplot_titles=("Trade count", "Win rate", "Avg P&L per trade ($)", "Total P&L ($)"))
    rows_cols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    series = [agg["n"], agg["win_rate"], agg["avg_pnl"], agg["total_pnl"]]
    for (r, c), s, nm in zip(rows_cols, series, ["n", "win rate", "avg pnl", "total pnl"]):
        if nm == "win rate":
            fig.add_trace(go.Bar(x=agg["label"], y=s, marker_color=COLOR_MAIN, showlegend=False,
                                 text=[f"{v:.0%}" for v in s], textposition="outside"), r, c)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=r, col=c)
        else:
            colors = bar_colors(s) if nm != "n" else COLOR_MAIN
            fig.add_trace(go.Bar(x=agg["label"], y=s, marker_color=colors, showlegend=False), r, c)
    for ax in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
        fig.update_layout({ax: dict(tickangle=30)})
    fig.update_annotations(font_size=13)
    return style(fig, "Edge-bucket diagnostics: does more predicted edge mean more realized profit?",
                 height=700)


@register_plot("edge", title="Entry price vs realized outcome", requires=("trades",))
def plot_entry_price_vs_outcome(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().dropna(subset=["entry_price"])
    outcome_known = pd.to_numeric(t["settlement"], errors="coerce")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Settlement rate by entry price decile",
                                        "P&L scatter vs entry price"))
    known = t[outcome_known.isin([0, 1])]
    if len(known) >= 10:
        binned = known.assign(bin=pd.qcut(pd.to_numeric(known["entry_price"], errors="coerce"), q=10, duplicates="drop"))
        rates = binned.groupby("bin", observed=True)["settlement"].mean()
        mids = [iv.mid for iv in rates.index]
        fig.add_trace(go.Bar(x=[f"{m:.2f}" for m in mids], y=rates.values, marker_color=COLOR_MAIN,
                             name="settled YES"), row=1, col=1)
        fig.update_yaxes(tickformat=".0%", title_text="settled YES %", row=1, col=1)
        fig.update_xaxes(title_text="entry price bin midpoint", row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_numeric(t["entry_price"], errors="coerce"),
                             y=pd.to_numeric(t["net_pnl"], errors="coerce"), mode="markers",
                             marker=dict(color=np.where(outcome_known.fillna(-1) == 1, COLOR_POS,
                                                        np.where(outcome_known == 0, COLOR_NEG, "#bbb")),
                                        opacity=0.55, size=7), name="trades",
                             hovertemplate="%{x:.2f}c entry<br>$%{y:+.2f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="entry price", row=1, col=2)
    fig.update_yaxes(title_text="net P&L $", row=1, col=2)
    return style(fig, "Where in the price range does the strategy trade and win?")


@register_plot("edge", title="Cumulative expected value vs realized value", requires=("col:predicted_edge",))
def plot_expected_vs_realized(result: BacktestResult) -> go.Figure:
    ev = expected_vs_realized(result)
    if ev.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev.index, y=ev["expected_cum"], name="expected cum P&L (model)",
                             line=dict(color=COLOR_ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=ev.index, y=ev["realized_cum"], name="realized cum net P&L",
                             line=dict(color=COLOR_MAIN, width=2)))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    gap = float(ev["realized_cum"].iloc[-1] - ev["expected_cum"].iloc[-1])
    fig.add_annotation(text=f"final gap (realized - expected): ${gap:+,.2f}",
                       xref="paper", yref="paper", x=0.99, y=1.1, xanchor="right", showarrow=False,
                       font=dict(size=13, color=COLOR_POS if gap >= 0 else COLOR_NEG))
    return style(fig, "Is the model's promised edge showing up in the account?", y_title="$")


@register_plot("edge", title="Performance by temperature bucket", requires=("trades",))
def plot_perf_by_temp_bucket(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    g = t.groupby("temp_bucket", observed=True)
    agg = g.agg(n=("net_pnl", "size"), total_pnl=("net_pnl", "sum"), win_rate=("win", "mean"))
    def sort_key(label):
        try:
            return float(str(label).split("-")[0].replace(">=", "").replace("<", "").strip())
        except Exception:
            return float("inf")
    agg = agg.sort_index(key=lambda idx: pd.Index([sort_key(v) for v in idx])).tail(25)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=agg.index.astype(str), y=agg["total_pnl"], marker_color=bar_colors(agg["total_pnl"]),
                name="total P&L", customdata=agg["n"], secondary_y=False,
                hovertemplate="%{x}:F bucket<br>$%{y:,.2f}<br>n=%{customdata}<extra></extra>")
    add_winrate_line(fig, agg.index.astype(str), agg["win_rate"])
    fig.update_layout(xaxis_title="temperature bucket (F)")
    fig.update_yaxes(title_text="Total net P&L $", secondary_y=False)
    return style(fig, "Net P&L and win rate by temperature bucket")


@register_plot("edge", title="Performance by time to settlement", requires=("col:days_to_settle",))
def plot_perf_by_time_to_settle(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().copy()
    bins = [-1, 0, 1, 2, 3, 7, 10000]
    labels = ["same day", "1d", "2d", "3d", "4-7d", ">7d"]
    t["ttl_bucket"] = pd.cut(pd.to_numeric(t["days_to_settle"], errors="coerce"), bins=bins, labels=labels)
    g = t.groupby("ttl_bucket", observed=True)
    agg = g.agg(n=("net_pnl", "size"), total_pnl=("net_pnl", "sum"), win_rate=("win", "mean"),
                avg_pnl=("net_pnl", "mean"))
    if agg.empty:
        return empty_figure("No trades with resolvable settlement timing",
                            "Does holding length until settlement change profitability?")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=agg.index.astype(str), y=agg["total_pnl"], marker_color=bar_colors(agg["total_pnl"]),
                name="total P&L", customdata=np.stack([agg["n"], agg["avg_pnl"]], axis=-1),
                secondary_y=False,
                hovertemplate="%{x}<br>$%{y:,.2f} total<br>n=%{customdata[0]}<br>avg $%{customdata[1]:,.2f}<extra></extra>")
    add_winrate_line(fig, agg.index.astype(str), agg["win_rate"])
    fig.update_layout(xaxis_title="time between entry and settlement")
    fig.update_yaxes(title_text="Total net P&L $", secondary_y=False)
    return style(fig, "Does holding length until settlement change profitability?")


@register_plot("edge", title="Performance by entry price range", requires=("trades",))
def plot_perf_by_price_range(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    g = t.groupby("price_bin", observed=True)
    agg = g.agg(n=("net_pnl", "size"), total_pnl=("net_pnl", "sum"), win_rate=("win", "mean"),
                avg_pnl=("net_pnl", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=agg["price_bin"].astype(str), y=agg["total_pnl"],
                marker_color=bar_colors(agg["total_pnl"]), name="total P&L",
                customdata=np.stack([agg["n"], agg["win_rate"], agg["avg_pnl"]], axis=-1),
                secondary_y=False,
                hovertemplate="%{x}<br>$%{y:,.2f}<br>n=%{customdata[0]}<br>win %{customdata[1]:.0%}"
                              "<br>avg $%{customdata[2]:,.2f}<extra></extra>")
    add_winrate_line(fig, agg["price_bin"].astype(str), agg["win_rate"])
    fig.update_layout(xaxis_title="entry price bin")
    fig.update_yaxes(title_text="Total net P&L $", secondary_y=False)
    return style(fig, "Edge by entry price range (cheap tails vs favorites)")


@register_plot("edge", title="Performance by liquidity proxy", requires=("col:spread_entry",))
def plot_perf_by_liquidity(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().copy()
    spread = pd.to_numeric(t["spread_entry"], errors="coerce")
    has_volume = "volume" in t.columns and pd.to_numeric(t["volume"], errors="coerce").notna().any()
    metric_label = "bid-ask spread at entry"
    if has_volume:
        vol = pd.to_numeric(t["volume"], errors="coerce")
        t["liq_bucket"] = pd.qcut(vol.rank(method="first"), q=5,
                                  labels=["Q1 least liquid", "Q2", "Q3", "Q4", "Q5 most liquid"])
    else:
        t["liq_bucket"] = pd.cut(spread, bins=[-0.01, 0.02, 0.05, 0.10, 0.20, 1.0],
                                 labels=["<2c", "2-5c", "5-10c", "10-20c", ">20c"])
        metric_label = "spread bucket"
    g = t.groupby("liq_bucket", observed=True)
    agg = g.agg(n=("net_pnl", "size"), total_pnl=("net_pnl", "sum"), win_rate=("win", "mean"),
                avg_pnl=("net_pnl", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=agg["liq_bucket"].astype(str), y=agg["total_pnl"],
                marker_color=bar_colors(agg["total_pnl"]), name="total P&L",
                customdata=np.stack([agg["n"], agg["avg_pnl"]], axis=-1), secondary_y=False,
                hovertemplate="%{x}<br>$%{y:,.2f}<br>n=%{customdata[0]}"
                              "<br>avg $%{customdata[1]:,.2f}<extra></extra>")
    add_winrate_line(fig, agg["liq_bucket"].astype(str), agg["win_rate"])
    fig.update_layout(xaxis_title=metric_label)
    fig.update_yaxes(title_text="Total net P&L $", secondary_y=False)
    return style(fig, "Does the edge survive where liquidity is thin?")
