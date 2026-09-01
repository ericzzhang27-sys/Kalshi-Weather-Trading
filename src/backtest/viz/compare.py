"""Comparison mode: load multiple BacktestResults side by side."""
from __future__ import annotations

import html as html_mod
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.backtest.viz.schema import BacktestResult
from src.backtest.viz.stats import ANNUALIZATION, drawdown_frame, exposure_frame, summary_stats

logger = logging.getLogger(__name__)


def comparison_table(results: dict[str, BacktestResult] | list[BacktestResult]) -> pd.DataFrame:
    """Sortable head-to-head metrics for each run (one row per backtest)."""
    if isinstance(results, list):
        results = {r.name: r for r in results}
    rows = {}
    for name, result in results.items():
        try:
            s = summary_stats(result)
        except Exception as exc:
            logger.warning("Could not summarize %s: %s", name, exc)
            continue
        expo = exposure_frame(result)
        rows[name] = {
            "run": name,
            "total_return": s["total_return"],
            "net_pnl": s["net_pnl"],
            "annualized_return": s["annualized_return"],
            "sharpe": s["sharpe"],
            "sortino": s["sortino"],
            "max_drawdown": s["max_drawdown"],
            "volatility": s["annualized_volatility"],
            "n_trades": s["n_trades"],
            "win_rate": s["win_rate"],
            "profit_factor": s["profit_factor"],
            "turnover": s["turnover"],
            "avg_gross_exposure_pct": s["avg_gross_exposure_pct"],
            "fees_paid": s["fees_paid"],
            "slippage_paid": s["slippage_paid"],
            "execution_costs": s["execution_costs"],
        }
    df = pd.DataFrame(rows).T.reset_index(drop=True)
    if not df.empty:
        df = df.sort_values("total_return", ascending=False)
    return df


def plot_compare_equity(results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        eq = result.equity_series() / result.initial_capital * 100.0
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name,
                                 line=dict(width=2),
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>" + html_mod.escape(name) + "</extra>"))
    fig.add_hline(y=100, line_dash="dot", line_color="#999")
    return _style(fig, "Equity curves comparison (indexed to 100)", "Indexed value")


def plot_compare_drawdown(results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        dd = drawdown_frame(result.equity_series())["drawdown_pct"]
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, line=dict(width=1.6)))
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, "Drawdown comparison", "Drawdown")


def plot_compare_rolling_sharpe(results: dict[str, BacktestResult], window_days: int = 30) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        rets = result.equity_series().resample("D").last().pct_change().dropna()
        sharpe = rets.rolling(window_days).mean() / rets.rolling(window_days).std().replace(0, np.nan) \
            * np.sqrt(ANNUALIZATION)
        fig.add_trace(go.Scatter(x=sharpe.index, y=sharpe.values, name=name, line=dict(width=1.8)))
    fig.add_hline(y=0, line_dash="dot", line_color="#999")
    return _style(fig, f"Rolling {window_days}d Sharpe comparison", "Sharpe")


def plot_compare_exposure(results: dict[str, BacktestResult]) -> go.Figure:
    fig = go.Figure()
    for name, result in results.items():
        e = exposure_frame(result)
        if len(e) == 0:
            continue
        pct = e["exposure_pct"].replace([np.inf, -np.inf], np.nan)
        fig.add_trace(go.Scatter(x=e.index, y=pct.values, name=name,
                                 line=dict(width=1.5), opacity=0.85))
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, "Gross exposure as % of bankroll", "% of equity")


def comparison_table_figure(table: pd.DataFrame) -> go.Figure:
    view = table.copy()
    for c in ("total_return", "annualized_return", "max_drawdown", "volatility",
              "win_rate", "avg_gross_exposure_pct"):
        if c in view:
            view[c] = pd.to_numeric(view[c], errors="coerce").map(
                lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    for c in ("sharpe", "sortino", "profit_factor", "turnover"):
        if c in view:
            view[c] = pd.to_numeric(view[c], errors="coerce").map(
                lambda v: f"{v:.2f}" if pd.notna(v) and abs(v) < 9000 else ("inf" if pd.notna(v) else "-"))
    money_cols = ["net_pnl", "fees_paid", "slippage_paid", "execution_costs"]
    for c in money_cols:
        if c in view:
            view[c] = pd.to_numeric(view[c], errors="coerce").map(
                lambda v: f"${v:,.0f}" if pd.notna(v) else "-")
    if "n_trades" in view:
        view["n_trades"] = pd.to_numeric(view["n_trades"], errors="coerce").map(lambda v: f"{v:,.0f}")
    header_vals = [c.replace("_", " ") for c in view.columns]
    fig = go.Figure(go.Table(
        header=dict(values=header_vals, fill_color="#263444", font=dict(color="white", size=12)),
        cells=dict(values=[view[c] for c in view.columns],
                   font=dict(size=11.5), height=26, align="left"),
    ))
    fig.update_layout(title="Strategy comparison table", height=60 + max(len(view), 1) * 30 + 80)
    return fig


def create_comparison_report(
    results: dict[str, BacktestResult] | list[BacktestResult],
    output_path: str | Path | None = None,
    title: str = "Backtest comparison report",
) -> tuple[pd.DataFrame, dict[str, go.Figure]]:
    """Side-by-side dashboard for multiple runs; returns (table, figures)."""
    if isinstance(results, list):
        results = {r.name: r for r in results}
    if len(results) < 1:
        raise ValueError("comparison report needs at least one result")
    table = comparison_table(results)
    figures = {
        "table": comparison_table_figure(table),
        "equity": plot_compare_equity(results),
        "drawdown": plot_compare_drawdown(results),
        "rolling_sharpe": plot_compare_rolling_sharpe(results),
        "exposure": plot_compare_exposure(results),
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        css = ("body{font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;margin:24px auto;"
               "max-width:1280px;} h1{font-size:24px;} .blk{margin-top:28px;}")
        parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>", f"<title>{html_mod.escape(title)}</title>",
                 f"<style>{css}</style></head><body>", f"<h1>{html_mod.escape(title)}</h1>",
                 f"<div style='color:#667;font-size:13px'>generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M UTC}"
                 f" &nbsp;|&nbsp; {len(results)} runs</div>"]
        include_js = True
        for name, fig in figures.items():
            parts.append(f"<div class='blk' id='{name}'>"
                         + fig.to_html(full_html=False, include_plotlyjs=include_js,
                                       config={"displaylogo": False})
                         + "</div>")
            include_js = False
        parts.append("</body></html>")
        out.write_text("\n".join(parts), encoding="utf-8")
        logger.info("Comparison report written to %s", out)
    return table, figures


def _style(fig: go.Figure, title: str, y_title: str | None = None, height: int = 430) -> go.Figure:
    layout = dict(template="plotly_white", hovermode="x unified", height=height,
                  margin=dict(l=50, r=30, t=55, b=40), title=title,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
    if y_title:
        layout["yaxis_title"] = y_title
    fig.update_layout(**layout)
    return fig
