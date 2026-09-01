"""Section 3: exposure and risk visualizations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.viz._plotutil import (COLOR_ACCENT, COLOR_MAIN, COLOR_NEG, COLOR_POS,
                                        bar_colors, empty_figure, style)
from src.backtest.viz.registry import register_plot
from src.backtest.viz.schema import BacktestResult
from src.backtest.viz.stats import concentration_series, exposure_frame, group_breakdown, risk_contribution_by_group


def _expo(result: BacktestResult) -> pd.DataFrame:
    return exposure_frame(result)


@register_plot("risk", title="Gross exposure", requires=("trades",))
def plot_gross_exposure(result: BacktestResult) -> go.Figure:
    e = _expo(result)
    fig = go.Figure(go.Scatter(x=e.index, y=e["gross_exposure"], name="gross exposure",
                               line=dict(color=COLOR_MAIN, width=2), fill="tozeroy"))
    if result.has_trades:
        cap = result.meta.get("max_daily_exposure")
        if cap:
            fig.add_hline(y=cap, line_dash="dash", line_color=COLOR_NEG,
                          annotation_text=f"cap ${cap:,.0f}")
    return style(fig, "Gross dollar exposure over time", y_title="$")


@register_plot("risk", title="Net exposure", requires=("trades",))
def plot_net_exposure(result: BacktestResult) -> go.Figure:
    e = _expo(result)
    fig = go.Figure(go.Scatter(x=e.index, y=e["net_exposure"], name="net exposure",
                               line=dict(color=COLOR_ACCENT, width=2), fill="tozeroy"))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    return style(fig, "Net (signed YES-positive / NO-negative) exposure over time", y_title="$")


@register_plot("risk", title="Capital deployed vs cash", requires=("trades",))
def plot_capital_deployed(result: BacktestResult) -> go.Figure:
    e = _expo(result)
    eq = result.equity_series().reindex(e.index).ffill()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e.index, y=eq.values, name="equity", line=dict(color="#444", width=1.5)))
    fig.add_trace(go.Scatter(x=e.index, y=e["capital_deployed"], name="capital deployed",
                             line=dict(color=COLOR_MAIN, width=2), fill="tozeroy"))
    cash = result.cash_series()
    if cash is not None:
        cash = cash.reindex(e.index).ffill()
        fig.add_trace(go.Scatter(x=cash.index, y=cash.values, name="cash",
                                 line=dict(color=COLOR_POS, width=1.5, dash="dot")))
    return style(fig, "Capital deployment over time", y_title="$")


@register_plot("risk", title="Exposure as % of bankroll", requires=("trades",))
def plot_exposure_pct(result: BacktestResult) -> go.Figure:
    e = _expo(result)
    fig = go.Figure(go.Scatter(x=e.index, y=e["exposure_pct"], name="exposure / equity",
                               line=dict(color=COLOR_NEG, width=2), fill="tozeroy"))
    fig.update_yaxes(tickformat=".0%")
    return style(fig, "Gross exposure as % of bankroll", y_title="% of equity")


@register_plot("risk", title="Simultaneous open positions", requires=("trades",))
def plot_n_positions(result: BacktestResult) -> go.Figure:
    e = _expo(result)
    fig = go.Figure(go.Scatter(x=e.index, y=e["n_positions"], name="open positions",
                               line=dict(color=COLOR_MAIN, width=1.5),
                               line_shape="hv", fill="tozeroy"))
    avg = float(e["n_positions"].mean())
    fig.add_hline(y=avg, line_dash="dot", line_color="#888",
                  annotation_text=f"avg {avg:.1f}")
    return style(fig, "Number of simultaneous open positions", y_title="count")


@register_plot("risk", title="Exposure by market/city/strategy", requires=("trades",))
def plot_exposure_by_group(result: BacktestResult, by: str = "market_ticker") -> go.Figure:
    t = result.prepared_trades()
    notional = pd.to_numeric(t["notional"], errors="coerce").fillna(pd.to_numeric(t["gross_cost"], errors="coerce"))
    grouped = notional.groupby(t[by], observed=True).sum().sort_values(ascending=False).head(15)
    fig = go.Figure(go.Bar(x=grouped.index.astype(str), y=grouped.values,
                           marker_color=COLOR_MAIN,
                           hovertemplate="%{x}<br>$%{y:,.2f} total entered<extra></extra>"))
    return style(fig, f"Total capital entered by {by}", y_title="$ entered")


@register_plot("risk", title="Position concentration (HHI)", requires=("trades",))
def plot_position_concentration(result: BacktestResult) -> go.Figure:
    hhi = concentration_series(result)
    n_markets = result.prepared_trades()["market_ticker"].nunique()
    fair = 1.0 / max(n_markets, 1)
    fig = go.Figure(go.Scatter(x=hhi.index, y=hhi.values, name="HHI",
                               line=dict(color=COLOR_ACCENT, width=2), fill="tozeroy"))
    fig.add_hline(y=fair, line_dash="dot", line_color="#888",
                  annotation_text=f"equal-weight across {n_markets} markets ({fair:.3f})")
    return style(fig, "Position concentration (Herfindahl index of open notionals)",
                 y_title="HHI (higher = more concentrated)")


@register_plot("risk", title="P&L contribution by group", requires=("trades",))
def plot_pnl_contribution(result: BacktestResult, by: str = "city") -> go.Figure:
    breakdown = group_breakdown(result.prepared_trades(), by)
    fig = go.Figure(go.Bar(x=breakdown.index.astype(str), y=breakdown["net_pnl"],
                           marker_color=bar_colors(breakdown["net_pnl"]),
                           customdata=np.stack([breakdown["n_trades"], breakdown["win_rate"]], axis=-1),
                           hovertemplate=("%{x}<br>net P&L $%{y:,.2f}<br>%{customdata[0]} trades"
                                          "<br>win rate %{customdata[1]:.1%}<extra></extra>")))
    return style(fig, f"Net P&L contribution by {by}", y_title="$")


@register_plot("risk", title="Risk contribution by group", requires=("trades",))
def plot_risk_contribution(result: BacktestResult, by: str = "city") -> go.Figure:
    rc = risk_contribution_by_group(result, by)
    if rc.empty or rc["vol_contribution"].isna().all():
        return empty_figure("Risk contribution needs multiple groups and daily P&L variance",
                            f"Risk (volatility) contribution vs P&L by {by}")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=rc[by].astype(str), y=rc["vol_contribution"], name="vol contribution ($/day)",
                marker_color=COLOR_MAIN, secondary_y=False,
                hovertemplate="%{x}<br>$%{y:.2f}/day<extra></extra>")
    fig.add_scatter(x=rc[by].astype(str), y=rc["net_pnl"], name="net P&L ($)",
                    mode="markers+lines", marker=dict(color=COLOR_ACCENT, size=9), secondary_y=True)
    fig.update_yaxes(title_text="$ vol per day", secondary_y=False)
    fig.update_yaxes(title_text="net P&L $", secondary_y=True)
    return style(fig, f"Risk (volatility) contribution vs P&L by {by}")


@register_plot("risk", title="Exposure dashboard panel", requires=("trades",))
def plot_exposure_panel(result: BacktestResult) -> go.Figure:
    """One-glance risk panel: gross/net exposure, % bankroll, position count."""
    e = _expo(result)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("Gross vs net exposure ($)", "Exposure as % of bankroll",
                                        "Open positions"))
    fig.add_trace(go.Scatter(x=e.index, y=e["gross_exposure"], name="gross",
                             line=dict(color=COLOR_MAIN, width=2), fill="tozeroy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=e.index, y=e["net_exposure"], name="net",
                             line=dict(color=COLOR_ACCENT, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=e.index, y=e["exposure_pct"], name="% bankroll",
                             line=dict(color=COLOR_NEG, width=1.8), fill="tozeroy",
                             showlegend=False), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.add_trace(go.Scatter(x=e.index, y=e["n_positions"], name="positions",
                             line=dict(color=COLOR_POS, width=1.5), line_shape="hv",
                             fill="tozeroy", showlegend=False), row=3, col=1)
    return style(fig, "Risk & exposure panel", height=640)
