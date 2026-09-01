"""Section 6: execution cost diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest.viz._plotutil import (COLOR_ACCENT, COLOR_MAIN, COLOR_NEG, COLOR_POS,
                                        add_winrate_line, bar_colors, empty_figure, style)
from src.backtest.viz.registry import register_plot
from src.backtest.viz.schema import BacktestResult
from src.backtest.viz.stats import fee_scenarios


@register_plot("execution", title="Gross vs net equity (cost drag)", requires=("col:gross_pnl",))
def plot_gross_vs_net_equity(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().sort_values("signal_ts")
    x = pd.DatetimeIndex(t["signal_ts"])
    net_eq = result.initial_capital + pd.to_numeric(t["net_pnl"], errors="coerce").fillna(0).cumsum()
    gross_eq = result.initial_capital + pd.to_numeric(t["gross_pnl"], errors="coerce").fillna(0).cumsum()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.1, subplot_titles=("Equity: gross vs net of costs",
                                                              "Cumulative cost drag ($)",))
    fig.add_trace(go.Scatter(x=x, y=gross_eq.values, name="gross (no costs)",
                             line=dict(color=COLOR_POS, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=net_eq.values, name="net (fees + slippage)",
                             line=dict(color=COLOR_MAIN, width=2)), row=1, col=1)
    fig.add_hline(y=result.initial_capital, line_width=1, line_dash="dot", line_color="#999", row=1, col=1)
    drag = gross_eq - net_eq
    fig.add_trace(go.Scatter(x=x, y=drag.values, name="cost drag",
                             line=dict(color=COLOR_NEG, width=1.8), fill="tozeroy"), row=2, col=1)
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="$ paid", row=2, col=1)
    return style(fig, "What did fees and slippage take away?", height=600)


@register_plot("execution", title="Slippage distribution", requires=("col:slippage_usd",))
def plot_slippage_distribution(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    slip_cents = pd.to_numeric(t["slippage_per_contract"], errors="coerce") * 100.0
    slip_usd = pd.to_numeric(t["slippage_usd"], errors="coerce")
    known = slip_cents.dropna()
    valid_usd = slip_usd.dropna()
    if known.empty and valid_usd.empty:
        return empty_figure("Slippage unavailable: ledger lacks entry bid/ask quotes",
                            "Execution slippage (entry price vs prevailing mid)")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Slippage per contract (cents)", "Slippage per trade ($)"))
    if len(known):
        fig.add_trace(go.Histogram(x=known, nbinsx=40, marker_color=COLOR_ACCENT,
                                   showlegend=False), row=1, col=1)
        med = float(known.median())
        fig.add_vline(x=med, line_dash="dash", line_color=COLOR_NEG, annotation_text=f"median {med:.2f}c",
                      row=1, col=1)
    fig.update_xaxes(title_text="cents", row=1, col=1)
    valid_usd = slip_usd.dropna()
    if len(valid_usd):
        fig.add_trace(go.Histogram(x=valid_usd, nbinsx=40, marker_color=COLOR_MAIN,
                                   showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="$ per trade", row=1, col=2)
    total = float(slip_usd.fillna(0).sum())
    fig.add_annotation(text=f"total slippage paid: ${total:,.2f}", xref="paper", yref="paper",
                       x=0.5, y=1.12, showarrow=False, font=dict(size=13))
    return style(fig, "Execution slippage (entry price vs prevailing mid)")


@register_plot("execution", title="Fees and costs over time", requires=("trades",))
def plot_costs_over_time(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().set_index("signal_ts")
    fees_daily = pd.to_numeric(t["fees"], errors="coerce").fillna(0).resample("W").sum()
    slip_daily = pd.to_numeric(t["slippage_usd"], errors="coerce").fillna(0).resample("W").sum()
    fees_cum = pd.to_numeric(t["fees"], errors="coerce").fillna(0).sort_index().cumsum()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55],
                        vertical_spacing=0.12,
                        subplot_titles=("Weekly fees vs slippage ($)", "Cumulative fees paid ($)"))
    fig.add_trace(go.Bar(x=fees_daily.index, y=fees_daily.values, name="fees / wk",
                         marker_color=COLOR_MAIN, offsetgroup="a"), row=1, col=1)
    fig.add_trace(go.Bar(x=slip_daily.index, y=slip_daily.values, name="slippage / wk",
                         marker_color=COLOR_ACCENT, offsetgroup="b"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.DatetimeIndex(fees_cum.index), y=fees_cum.values, name="cumulative fees",
                             line=dict(color=COLOR_NEG, width=2), fill="tozeroy"), row=2, col=1)
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="$", row=2, col=1)
    total_fees = float(fees_cum.iloc[-1]) if len(fees_cum) else 0.0
    total_slip = float(pd.to_numeric(t["slippage_usd"], errors="coerce").fillna(0).sum())
    fig.add_annotation(text=f"total fees ${total_fees:,.2f} + slippage ${total_slip:,.2f} "
                            f"= ${total_fees + total_slip:,.2f}",
                       xref="paper", yref="paper", x=0.01, y=1.13, showarrow=False, font=dict(size=13))
    return style(fig, "Execution costs over time", height=620)


@register_plot("execution", title="P&L waterfall: gross to net", requires=("trades",))
def plot_cost_waterfall(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    gross = float(pd.to_numeric(t["gross_pnl"], errors="coerce").fillna(0).sum())
    fees = float(pd.to_numeric(t["fees"], errors="coerce").fillna(0).sum())
    slip = float(pd.to_numeric(t["slippage_usd"], errors="coerce").fillna(0).sum())
    fig = go.Figure(go.Waterfall(
        x=["Gross P&L", "- Fees", "- Slippage", "= Net P&L"],
        measure=["absolute", "relative", "relative", "total"],
        y=[gross, -fees, -slip, 0],
        text=[f"${gross:,.2f}", f"-${fees:,.2f}", f"-${slip:,.2f}",
              f"${gross - fees - slip:,.2f}"],
        connector=dict(line=dict(color="#aaa")),
        increasing=dict(marker_color=COLOR_POS), decreasing=dict(marker_color=COLOR_NEG),
        totals=dict(marker_color=COLOR_MAIN),
    ))
    pct_of_gross = (fees + slip) / abs(gross) * 100 if gross else np.nan
    sub = f"costs consumed {pct_of_gross:.0f}% of gross P&L" if np.isfinite(pct_of_gross) else ""
    return style(fig, f"P&L waterfall: gross -> net ({sub})", y_title="$")


@register_plot("execution", title="Turnover over time", requires=("trades",))
def plot_turnover_over_time(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades()
    notional = pd.to_numeric(t["notional"], errors="coerce").fillna(
        pd.to_numeric(t["gross_cost"], errors="coerce"))
    monthly_notional = notional.groupby(t["month_period"]).sum()
    monthly_trades = t.groupby("month_period").size()
    eq_mean = result.equity_series().resample("ME").mean()
    eq_by_month = pd.Series(eq_mean.values,
                            index=pd.PeriodIndex(eq_mean.index, freq="M").astype(str)).reindex(monthly_notional.index).ffill()
    turnover = monthly_notional / eq_by_month.replace(0, np.nan)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=list(turnover.index.astype(str)), y=turnover.values, marker_color=COLOR_MAIN,
                name="notional / avg equity", secondary_y=False,
                hovertemplate="%{x}<br>turnover %{y:.2f}x<extra></extra>")
    fig.add_scatter(x=list(turnover.index.astype(str)), y=monthly_trades.reindex(turnover.index).values,
                    mode="lines+markers", name="# trades", marker_color=COLOR_ACCENT, secondary_y=True)
    fig.update_yaxes(title_text="turnover (x equity)", secondary_y=False)
    fig.update_yaxes(title_text="trades / month", secondary_y=True)
    fig.update_layout(xaxis_tickangle=-45)
    return style(fig, "Monthly turnover")


@register_plot("execution", title="Trade size vs slippage", requires=("col:slippage_per_contract",))
def plot_size_vs_slippage(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().dropna(subset=["slippage_per_contract"])
    contracts = pd.to_numeric(t["contracts"], errors="coerce")
    slip_c = pd.to_numeric(t["slippage_per_contract"], errors="coerce") * 100.0
    if len(t) == 0:
        return empty_figure("Slippage unavailable: ledger lacks entry bid/ask quotes",
                            "Trade size vs slippage")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Contracts vs slippage per contract", "Avg slippage by size bucket"))
    fig.add_trace(go.Scatter(x=contracts, y=slip_c, mode="markers",
                             marker=dict(color=COLOR_ACCENT, opacity=0.5, size=7),
                             hovertemplate="%{x:.0f} contracts<br>%{y:.2f}c<extra></extra>",
                             showlegend=False), row=1, col=1)
    bins = [0, 10, 25, 50, 100, 250, 1000, 1e9]
    labels = ["<10", "10-25", "25-50", "50-100", "100-250", "250-1k", ">1k"]
    size_bin = pd.cut(contracts, bins=bins, labels=labels)
    avg = slip_c.groupby(size_bin, observed=True).mean()
    n = slip_c.groupby(size_bin, observed=True).size()
    fig.add_trace(go.Bar(x=avg.index.astype(str), y=avg.values, marker_color=COLOR_MAIN,
                         customdata=n.values, showlegend=False,
                         hovertemplate="%{x}<br>avg %{y:.2f}c<br>n=%{customdata}<extra></extra>"),
                  row=1, col=2)
    corr = float(np.corrcoef(contracts[contracts.notna() & slip_c.notna()],
                             slip_c[contracts.notna() & slip_c.notna()])[0, 1]) \
        if (contracts.notna() & slip_c.notna()).sum() > 10 else np.nan
    note = f"corr(contracts, slippage c/contract) = {corr:+.2f}" if np.isfinite(corr) else ""
    return style(fig, f"Do larger trades pay more slippage? {note}")


@register_plot("execution", title="Performance by trade size", requires=("trades",))
def plot_perf_by_trade_size(result: BacktestResult) -> go.Figure:
    t = result.prepared_trades().copy()
    notional = pd.to_numeric(t["notional"], errors="coerce").fillna(
        pd.to_numeric(t["gross_cost"], errors="coerce"))
    bins = [-1, 50, 150, 400, 800, 1e9]
    labels = ["<$50", "$50-150", "$150-400", "$400-800", ">$800"]
    t["size_bucket"] = pd.cut(notional, bins=bins, labels=labels)
    g = t.groupby("size_bucket", observed=True)
    agg = g.agg(n=("net_pnl", "size"), total_pnl=("net_pnl", "sum"), win_rate=("win", "mean"),
                avg_pnl=("net_pnl", "mean")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=agg["size_bucket"].astype(str), y=agg["total_pnl"],
                marker_color=bar_colors(agg["total_pnl"]), name="total P&L $",
                customdata=np.stack([agg["n"], agg["win_rate"]], axis=-1), secondary_y=False,
                hovertemplate="%{x}<br>$%{y:,.2f}<br>n=%{customdata[0]}"
                              "<br>win %{customdata[1]:.0%}<extra></extra>")
    add_winrate_line(fig, agg["size_bucket"].astype(str), agg["win_rate"])
    fig.update_layout(xaxis_title="trade size (notional at entry)")
    fig.update_yaxes(title_text="Total net P&L $", secondary_y=False)
    return style(fig, "Does performance depend on position size?")


@register_plot("execution", title="Fee/slippage scenario sensitivity", requires=("trades",))
def plot_fee_sensitivity(result: BacktestResult) -> go.Figure:
    scenarios = fee_scenarios(result)
    fig = go.Figure()
    for cents in sorted(scenarios["extra_slippage_cents"].unique()):
        sub = scenarios[scenarios["extra_slippage_cents"] == cents]
        fig.add_trace(go.Scatter(
            x=sub["fee_multiplier"], y=sub["net_pnl"], mode="lines+markers",
            name=f"+{cents:g}c extra slippage/side",
            line=dict(width=2), customdata=sub["effective_fee_rate"],
            hovertemplate="fee x%{x:g} (%{customdata:.1%})<br>+{c}c slippage<br>$%{y:,.2f}".replace(
                "{c}", f"{cents:g}") + "<extra></extra>",
        ))
    zero = scenarios[(scenarios["fee_multiplier"] == 1.0) & (scenarios["extra_slippage_cents"] == 0.0)]
    if len(zero):
        base = float(zero["net_pnl"].iloc[0])
        fig.add_hline(y=base, line_dash="dot", line_color="#888",
                      annotation_text=f"current assumptions ${base:,.2f}")
    fig.update_layout(xaxis_title="fee rate multiplier (1x = current 7% Kalshi schedule)",
                      xaxis_tickformat=".0%")
    fig.update_yaxes(title="Net P&L under scenario ($)")
    return style(fig, "Would the strategy survive harsher fee / slippage regimes?")
