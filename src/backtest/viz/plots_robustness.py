"""Section 7: robustness and stability visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.viz._plotutil import COLOR_ACCENT, COLOR_MAIN, COLOR_NEG, COLOR_POS, add_winrate_line, bar_colors, style
from src.backtest.viz.registry import register_plot
from src.backtest.viz.schema import BacktestResult
from src.backtest.viz.stats import ANNUALIZATION, daily_returns


@register_plot("robustness", title="Rolling stability panel", requires=("trades",))
def plot_rolling_stability(result: BacktestResult, window_days: int = 30) -> go.Figure:
    eq = result.equity_series()
    rets = daily_returns(eq)
    roll_ret = eq.resample("D").last().pct_change(window_days)
    std = rets.rolling(window_days).std().replace(0, np.nan)
    roll_sharpe = rets.rolling(window_days).mean() / std * np.sqrt(ANNUALIZATION)
    roll_vol = rets.rolling(window_days).std() * np.sqrt(ANNUALIZATION)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"Rolling {window_days}d return", f"Rolling {window_days}d Sharpe",
                                        f"Rolling {window_days}d volatility"))
    fig.add_trace(go.Scatter(x=roll_ret.index, y=roll_ret.values, name="return",
                             line=dict(color=COLOR_MAIN, width=2), fill="tozeroy"), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="sharpe",
                             line=dict(color=COLOR_ACCENT, width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888", row=2, col=1)
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, name="volatility",
                             line=dict(color=COLOR_NEG, width=2)), row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    return style(fig, "Is performance deteriorating? Rolling metrics over time", height=720)


@register_plot("robustness", title="Quarterly stability table", requires=("trades",))
def plot_subperiod_performance(result: BacktestResult, freq: str = "QE") -> go.Figure:
    t = result.prepared_trades().set_index("signal_ts")
    pnl = t["net_pnl"].resample(freq).sum()
    trades = t["net_pnl"].resample(freq).size()
    win_rate = t["win"].resample(freq).mean()
    eq_q_end = result.equity_series().resample(freq).last()
    ret = eq_q_end.pct_change()
    try:
        labels = pd.PeriodIndex(pnl.index, freq="Q").astype(str)
    except Exception:
        labels = [ts.strftime("%Y-%m-%d") for ts in pnl.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in ret.fillna(pnl)]
    fig.add_bar(x=list(labels), y=pnl.values, marker_color=bar_colors(pnl), name="net P&L $",
                secondary_y=False, customdata=np.stack([trades.values,
                                                        win_rate.reindex(pnl.index).values], axis=-1),
                hovertemplate="%{x}<br>$%{y:,.2f}<br>%{customdata[0]} trades"
                              "<br>win %{customdata[1]:.0%}<extra></extra>")
    fig.add_scatter(x=list(labels), y=ret.values, mode="lines+markers", name="period return %",
                    marker_color=COLOR_ACCENT, secondary_y=True)
    fig.update_layout(xaxis_tickangle=-45)
    fig.update_yaxes(title_text="Net P&L $", secondary_y=False)
    fig.update_yaxes(title_text="Return %", tickformat=".0%", secondary_y=True)
    return style(fig, "Sub-period consistency of results")


@register_plot("robustness", title="Breakdown grid (city / market / model / side)", requires=("trades",))
def plot_breakdown_grid(result: BacktestResult, by_cols: tuple[str, ...] = ("city", "model_name", "strategy", "side_norm")) -> go.Figure:
    cols = [c for c in by_cols if c in result.prepared_trades().columns][:4]
    if not cols:
        return go.Figure()
    rows = 1 if len(cols) <= 2 else 2
    ncol = len(cols) if len(cols) <= 2 else 2
    fig = make_subplots(rows=rows, cols=ncol, horizontal_spacing=0.12, vertical_spacing=0.18,
                        subplot_titles=[f"by {c}" for c in cols])
    for i, col in enumerate(cols):
        t = result.prepared_trades()
        g = t.groupby(col, observed=True)
        agg = g["net_pnl"].sum().sort_values(ascending=False).head(10)
        r, c = divmod(i, ncol)
        fig.add_trace(go.Bar(x=agg.index.astype(str), y=agg.values,
                             marker_color=[COLOR_POS if v >= 0 else COLOR_NEG for v in agg.values],
                             showlegend=False, name=str(col),
                             hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>"),
                      row=r + 1, col=c + 1)
        fig.update_xaxes(tickangle=30, row=r + 1, col=c + 1)
    fig.update_yaxes(title_text="Net P&L $")
    return style(fig, "Robustness breakdowns", height=420 * rows)


@register_plot("robustness", title="Direction split (BUY YES vs BUY NO)", requires=("col:side_norm",))
def plot_direction_split(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    sides = sorted(t["side_norm"].dropna().unique())
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.14,
                        subplot_titles=("P&L distribution by direction", "Totals by direction"))
    for side in sides:
        sub = pd.to_numeric(t[t["side_norm"] == side]["net_pnl"], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=sub, nbinsx=35, name=str(side), opacity=0.7,
                                   histnorm="probability density"), row=1, col=1)
    agg = t.groupby("side_norm", observed=True).agg(total_pnl=("net_pnl", "sum"),
                                                    win_rate=("win", "mean"),
                                                    n=("net_pnl", "size")).reset_index()
    fig.add_trace(go.Bar(x=agg["side_norm"].astype(str), y=agg["total_pnl"],
                         marker_color=bar_colors(agg["total_pnl"]), showlegend=False,
                         customdata=np.stack([agg["n"], agg["win_rate"]], axis=-1),
                         hovertemplate="%{x}<br>$%{y:,.2f}<br>n=%{customdata[0]}"
                                       "<br>win %{customdata[1]:.0%}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="net P&L per trade ($)", row=1, col=1)
    fig.update_yaxes(title_text="density", row=1, col=1)
    fig.update_xaxes(title_text="side", row=1, col=2)
    return style(fig, "Long/short (YES/NO) balance and contribution")


@register_plot("robustness", title="Concentration of profits (top-N dependence)", requires=("trades",))
def plot_profit_concentration(result: BacktestResult) -> go.Figure:
    """How dependent is total profit on the best few days/trades?"""
    t = result.prepared_trades().assign(date=t_signal_date(result))
    daily = t.groupby("date")["net_pnl"].sum().sort_values(ascending=False)
    total = float(daily.sum())
    ks = list(range(0, min(len(daily), 20) + 1))
    remaining = [total - float(daily.iloc[:k].sum()) for k in ks]
    top_trades = pd.to_numeric(t["net_pnl"], errors="coerce").sort_values(ascending=False)
    remaining_trades = [float(top_trades.sum()) - float(top_trades.iloc[:k].sum()) for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=remaining, mode="lines+markers", name="after removing top days",
                             line=dict(color=COLOR_MAIN, width=2)))
    fig.add_trace(go.Scatter(x=ks, y=remaining_trades, mode="lines+markers",
                             name="after removing top trades", line=dict(color=COLOR_ACCENT, width=2)))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(xaxis_title="# of best periods/trades removed",
                      yaxis_title="remaining net P&L ($)")
    return style(fig, "Fragility check: remove the lucky streak and what is left?")


def t_signal_date(result: BacktestResult) -> pd.Series:
    return result.prepared_trades()["signal_date"]
