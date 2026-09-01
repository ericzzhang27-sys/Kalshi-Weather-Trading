"""Section 2: performance visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.viz._plotutil import (COLOR_ACCENT, COLOR_BENCH, COLOR_MAIN, COLOR_NEG,
                                        COLOR_POS, bar_colors, style)
from src.backtest.viz.registry import register_plot
from src.backtest.viz.schema import BacktestResult, daily_returns
from src.backtest.viz.stats import (ANNUALIZATION, best_worst_periods, drawdown_frame,
                                    drawdown_spells, period_pnl)


@register_plot("performance", title="Equity curve", requires=("trades",))
def plot_equity_curve(result: BacktestResult) -> go.Figure:
    eq = result.equity_series()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=result.name,
                             line=dict(color=COLOR_MAIN, width=2),
                             fill="tozeroy" if False else None,
                             hovertemplate="%{x|%b %d %Y}<br>$%{y:,.2f}<extra></extra>"))
    if result.has_benchmark:
        bench = result.benchmark_series().reindex(eq.index.union(result.benchmark_series().index)).ffill()
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="benchmark",
                                 line=dict(color=COLOR_BENCH, width=1.5, dash="dash")))
    fig.add_hline(y=result.initial_capital, line_width=1, line_dash="dot",
                  line_color="#999", annotation_text="initial capital")
    return style(fig, f"Equity curve - {result.name}", y_title="Equity ($)")


@register_plot("performance", title="Equity vs benchmark (indexed)", requires=("benchmark", "trades"))
def plot_equity_vs_benchmark(result: BacktestResult) -> go.Figure:
    eq = result.equity_series() / result.initial_capital * 100.0
    bench = result.benchmark_series()
    bench = bench / bench.iloc[0] * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="strategy", line=dict(color=COLOR_MAIN, width=2)))
    fig.add_trace(go.Scatter(x=bench.index, y=bench.reindex(eq.index).ffill().values, name="benchmark",
                             line=dict(color=COLOR_BENCH, width=1.5, dash="dash")))
    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color="#bbb")
    return style(fig, "Strategy vs benchmark (start = 100)", y_title="Indexed value")


@register_plot("performance", title="Cumulative gross vs net P&L", requires=("col:gross_pnl",))
def plot_cum_gross_vs_net(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().sort_values("signal_ts")
    x = pd.DatetimeIndex(t["signal_ts"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pd.to_numeric(t["gross_pnl"], errors="coerce").fillna(0).cumsum(),
                             name="gross P&L", line=dict(color=COLOR_POS, width=2), stackgroup=None))
    fig.add_trace(go.Scatter(x=x, y=pd.to_numeric(t["net_pnl"], errors="coerce").fillna(0).cumsum(),
                             name="net P&L", line=dict(color=COLOR_MAIN, width=2)))
    gap = (pd.to_numeric(t["gross_pnl"], errors="coerce").fillna(0)
           - pd.to_numeric(t["net_pnl"], errors="coerce").fillna(0)).clip(lower=0).cumsum()
    fig.add_trace(go.Scatter(x=x, y=gap, name="execution costs", line=dict(color=COLOR_ACCENT, width=1.5, dash="dot")))
    return style(fig, "Cumulative gross P&L vs net P&L", y_title="$")


@register_plot("performance", title="Daily / weekly P&L", requires=("trades",))
def plot_periodic_pnl(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().set_index("signal_ts")
    daily = t["net_pnl"].resample("D").sum()
    weekly = t["net_pnl"].resample("W").sum()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Daily P&L", "Weekly P&L"))
    fig.add_trace(go.Bar(x=daily.index, y=daily.values, marker_color=bar_colors(daily),
                         showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=weekly.index, y=weekly.values, marker_color=bar_colors(weekly),
                         showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="$", row=2, col=1)
    fig.update_xaxes(row=2, col=1, title_text="")
    return style(fig, "P&L by period", height=560)


@register_plot("performance", title="Returns distribution", requires=("trades",))
def plot_returns_histogram(result: BacktestResult) -> go.Figure:
    rets = daily_returns(result.equity_series())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets.values, nbinsx=50, name="daily returns",
                               marker_color=COLOR_MAIN, opacity=0.85,
                               histnorm="probability density"))
    mu, sigma = float(rets.mean()), float(rets.std())
    xs = np.linspace(rets.min(), rets.max(), 200)
    fig.add_trace(go.Scatter(x=xs, y=(1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2)),
                             mode="lines", name="normal fit", line=dict(color=COLOR_ACCENT)))
    fig.add_vline(x=0, line_dash="dot", line_color="#888")
    fig.update_layout(bargap=0.02)
    return style(fig, f"Daily returns distribution (mean={mu:.3%}, sd={sigma:.3%})",
                 y_title="Density", x_title="Daily return", height=400)


@register_plot("performance", title=f"Rolling returns ({30}d)", requires=("trades",))
def plot_rolling_returns(result: BacktestResult, window_days: int = 30) -> go.Figure:
    eq = result.equity_series().resample("D").last()
    roll = eq.pct_change(window_days)
    fig = go.Figure(go.Scatter(x=roll.index, y=roll.values, name=f"{window_days}d rolling return",
                               line=dict(color=COLOR_MAIN, width=2), fill="tozeroy"))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.update_yaxes(tickformat=".0%")
    return style(fig, f"Rolling {window_days}-day returns", y_title="Return")


@register_plot("performance", title=f"Rolling Sharpe ({30}d)", requires=("trades",))
def plot_rolling_sharpe(result: BacktestResult, window_days: int = 30) -> go.Figure:
    rets = daily_returns(result.equity_series())
    mean = rets.rolling(window_days).mean()
    std = rets.rolling(window_days).std()
    sharpe = mean / std.replace(0, np.nan) * np.sqrt(ANNUALIZATION)
    fig = go.Figure(go.Scatter(x=sharpe.index, y=sharpe.values, name=f"{window_days}d rolling Sharpe",
                               line=dict(color=COLOR_MAIN, width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    return style(fig, f"Rolling {window_days}-day Sharpe (annualized, {ANNUALIZATION:.0f}d)",
                 y_title="Sharpe")


@register_plot("performance", title=f"Rolling volatility ({30}d)", requires=("trades",))
def plot_rolling_volatility(result: BacktestResult, window_days: int = 30) -> go.Figure:
    rets = daily_returns(result.equity_series())
    vol = rets.rolling(window_days).std() * np.sqrt(ANNUALIZATION)
    fig = go.Figure(go.Scatter(x=vol.index, y=vol.values, name=f"{window_days}d rolling vol",
                               line=dict(color=COLOR_ACCENT, width=2), fill="tozeroy"))
    return style(fig, f"Rolling {window_days}-day annualized volatility", y_title="Volatility")


@register_plot("performance", title="Drawdown curve", requires=("trades",))
def plot_drawdown_curve(result: BacktestResult) -> go.Figure:
    dd = drawdown_frame(result.equity_series())
    fig = go.Figure(go.Scatter(x=dd.index, y=dd["drawdown_pct"], name="drawdown",
                               line=dict(color=COLOR_NEG, width=1.5), fill="tozeroy"))
    trough_idx = dd["drawdown_pct"].idxmin()
    depth = dd.loc[trough_idx, "drawdown_pct"]
    fig.add_annotation(x=trough_idx, y=depth, text=f"max {depth:.1%}", showarrow=True, arrowhead=2,
                       ax=40, ay=-20, font=dict(color=COLOR_NEG))
    fig.update_yaxes(tickformat=".0%", autorange=True)
    return style(fig, "Drawdown from running peak", y_title="Drawdown")


@register_plot("performance", title="Underwater duration analysis", requires=("trades",))
def plot_underwater_duration(result: BacktestResult) -> go.Figure:
    dd = drawdown_frame(result.equity_series())
    spells = drawdown_spells(result.equity_series()).head(5)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                        row_heights=[0.65, 0.35],
                        subplot_titles=("Underwater curve", "Longest drawdown spells (days underwater)"))
    fig.add_trace(go.Scatter(x=dd.index, y=dd["drawdown_pct"], fill="tozeroy",
                             line=dict(color=COLOR_NEG, width=1), showlegend=False,
                             hovertemplate="%{x|%b %d %Y}<br>%{y:.1%}<extra></extra>"), row=1, col=1)
    if len(spells):
        labels = [f"{r.start:%Y-%m-%d} ({r.depth_pct:.0%} deep)" for r in spells.itertuples()]
        fig.add_trace(go.Bar(y=labels[::-1], x=spells["days"][::-1], orientation="h",
                             marker_color=COLOR_NEG, showlegend=False,
                             hovertemplate="%{y}<br>%{x:.0f} days<extra></extra>"), row=2, col=1)
        fig.update_xaxes(title_text="days", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    return style(fig, "Drawdown depth and time-to-recovery", height=600)


@register_plot("performance", title="Best / worst periods", requires=("trades",))
def plot_best_worst_periods(result: BacktestResult, k: int = 5) -> go.Figure:
    panels = []
    for freq, label in (("D", "Day"), ("W", "Week"), ("ME", "Month")):
        bw = best_worst_periods(result, freq=freq, k=k)
        panels.append((label, bw))
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=[p[0] for p in panels])
    for i, (label, bw) in enumerate(panels, start=1):
        combined = pd.concat([bw["best"], bw["worst"]]).sort_values()
        colors = bar_colors(combined.values)
        names = [ts.strftime("%Y-%m-%d") for ts in combined.index]
        fig.add_trace(go.Bar(x=names, y=combined.values, marker_color=colors, showlegend=False,
                             hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"), row=1, col=i)
        fig.update_xaxes(tickangle=45, row=1, col=i)
    fig.update_yaxes(title_text="Net P&L ($)", row=1, col=1)
    return style(fig, f"Best / worst {k} periods", height=430)


@register_plot("performance", title="Monthly P&L heatmap", requires=("trades",))
def plot_monthly_heatmap(result: BacktestResult) -> go.Figure:
    pnl = period_pnl(result, freq="ME")
    idx = pd.PeriodIndex(pd.DatetimeIndex(pnl.index), freq="M")
    frame = pd.DataFrame({"year": idx.year, "month": idx.month, "pnl": pnl.values})
    pivot = frame.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[int(m) - 1] for m in pivot.columns]
    fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index.astype(str),
                               colorscale="RdYlGn", zmid=0,
                               hovertemplate="%{y} %{x}<br>$%{z:,.2f}<extra></extra>",
                               colorbar=dict(title="P&L $")))
    return style(fig, "Monthly net P&L heatmap", y_title="Year", height=320)
