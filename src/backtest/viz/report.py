"""Full-report builder: `create_backtest_report(result)` renders every registered
diagnostic into one standalone interactive HTML dashboard."""
from __future__ import annotations

import html as html_mod
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.backtest.viz.registry import SECTION_ORDER, build_all_figures, get_registered_plots, run_plot
from src.backtest.viz.schema import BacktestResult
from src.backtest.viz.stats import summary_stats

logger = logging.getLogger(__name__)

SECTION_TITLES = {
    "overview": "Overview",
    "performance": "Performance",
    "risk": "Exposure & Risk",
    "trades": "Trade Analysis",
    "edge": "Strategy Edge Diagnostics",
    "execution": "Execution Diagnostics",
    "robustness": "Robustness & Stability",
}

_CARD_GREEN = "#1a7f37"
_CARD_RED = "#c62828"


def _fmt(value, kind: str = "auto") -> str:
    if value is None or (isinstance(value, float) and not pd.notna(value)):
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind in ("money",):
        return f"${v:,.2f}"
    if kind == "money_signed":
        sign = "+" if v >= 0 else ""
        return f"{sign}${v:,.2f}"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "pct1":
        return f"{v:.2%}"
    if kind == "ratio":
        return f"{v:.2f}" if pd.notna(v) and abs(v) < 9000 else ("inf" if v > 0 else "-")
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:,.4g}"


def _card(label: str, value: str, color: str | None = None, sub: str = "") -> str:
    style_attr = f"color:{color};" if color else ""
    sub_html = f'<div class="card-sub">{html_mod.escape(sub)}</div>' if sub else ""
    return (f'<div class="card"><div class="card-label">{html_mod.escape(label)}</div>'
            f'<div class="card-value" style="{style_attr}">{value}</div>{sub_html}</div>')


def overview_cards_html(result: BacktestResult) -> str:
    s = summary_stats(result)
    green, red = _CARD_GREEN, _CARD_RED
    pnl_color = green if s["total_pnl"] >= 0 else red
    pf = s["profit_factor"]
    cards = [
        _card("Total P&L", _fmt(s["total_pnl"], "money_signed"), pnl_color,
              f"return {_fmt(s['total_return'], 'pct')}"),
        _card("Annualized return", _fmt(s["annualized_return"], "pct"),
              green if s["annualized_return"] >= 0 else red),
        _card("Sharpe ratio", _fmt(s["sharpe"], "ratio")),
        _card("Sortino ratio", _fmt(s["sortino"], "ratio")),
        _card("Max drawdown", _fmt(s["max_drawdown"], "pct1"), red),
        _card("Ann. volatility", _fmt(s["annualized_volatility"], "pct")),
        _card("Win rate", _fmt(s["win_rate"], "pct"), green),
        _card("Profit factor", _fmt(pf, "ratio"), green if (pd.notna(pf) and pf >= 1) else red),
        _card("Trades", _fmt(s["n_trades"], "int")),
        _card("Avg trade P&L", _fmt(s["avg_trade_pnl"], "money_signed"),
              green if s["avg_trade_pnl"] >= 0 else red),
        _card("Median trade P&L", _fmt(s["median_trade_pnl"], "money_signed"),
              green if s["median_trade_pnl"] >= 0 else red),
        _card("Avg holding period", f"{s['avg_holding_days']:.2f} d"),
        _card("Turnover", f"{s['turnover']:.1f}x"),
        _card("Fees + slippage", _fmt(s["execution_costs"], "money"), red,
              f"fees ${s['fees_paid']:,.0f} / slip ${s['slippage_paid']:,.0f}"),
        _card("Largest win / loss",
              f"<span style='color:{green}'>{_fmt(s['largest_win'], 'money_signed')}</span> "
              f"/ <span style='color:{red}'>{_fmt(s['largest_loss'], 'money_signed')}</span>"),
    ]
    return '<div class="cards">' + "".join(cards) + "</div>"


def _fig_to_html(fig: go.Figure, plotly_js_included: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=not plotly_js_included,
                       config={"displaylogo": False, "responsive": True})


_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       margin: 24px auto; max-width: 1280px; background: #fff; color: #222; }
h1 { font-size: 26px; margin-bottom: 2px; }
h2 { font-size: 20px; border-bottom: 2px solid #e3e6ea; padding-bottom: 6px; margin-top: 44px; }
.meta { color: #667; font-size: 13px; margin-bottom: 18px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
         gap: 10px; margin: 16px 0; }
.card { background: #f6f8fa; border: 1px solid #e3e6ea; border-radius: 8px; padding: 10px 12px; }
.card-label { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #586069; }
.card-value { font-size: 19px; font-weight: 600; margin-top: 3px; white-space: nowrap; }
.card-sub { font-size: 11px; color: #8792a2; margin-top: 2px; }
.toc { background:#f6f8fa; padding:12px 18px; border-radius:8px; font-size:14px; display:inline-block;}
.toc a { color:#1f77b4; text-decoration:none; margin-right:14px; }
.plot-block { margin-top: 26px; }
.skipped { color:#98a1ad; font-size:12.5px; font-style:italic; margin: 6px 0; }
footer { margin-top: 60px; color: #99a; font-size: 11.5px; }
"""


def build_report_figures(result: BacktestResult) -> dict[str, dict[str, go.Figure]]:
    """Run the full registry for a result; returns section -> {plot_name: figure}."""
    return build_all_figures(result)


def create_backtest_report(
    result: BacktestResult,
    output_path: str | Path | None = None,
    title: str | None = None,
    include_sections: list[str] | None = None,
) -> dict[str, dict[str, go.Figure]]:
    """Generate the complete interactive HTML backtest report.

    Returns the figures grouped by section; when `output_path` is given also
    writes a standalone HTML file (plotly.js loaded from CDN).
    """
    figures = build_report_figures(result)
    title = title or f"Backtest report - {result.name}"

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html_mod.escape(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{html_mod.escape(title)}</h1>",
        f"<div class='meta'>generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M UTC} &nbsp;|&nbsp; "
        f"initial capital ${result.initial_capital:,.0f}</div>",
    ]

    toc = "".join(f"<a href='#sec-{sec}'>{SECTION_TITLES.get(sec, sec)}</a>"
                  for sec in SECTION_ORDER if sec in figures)
    parts.append(f"<nav class='toc'>{toc}</nav>")
    parts.append("<h2 id='sec-overview'>Overview</h2>")
    parts.append(overview_cards_html(result))

    plotlyjs_written = False
    for section in SECTION_ORDER:
        if include_sections and section not in include_sections:
            continue
        if section == "overview":
            continue
        figs = figures.get(section)
        parts.append(f"<h2 id='sec-{section}'>{SECTION_TITLES.get(section, section)}</h2>")
        if not figs:
            parts.append("<p class='skipped'>No plots available for this section with the provided data.</p>")
            continue
        for name, fig in figs.items():
            parts.append(f"<div class='plot-block' id='plot-{name}'>")
            parts.append(_fig_to_html(fig, plotlyjs_written))
            plotlyjs_written = True
            parts.append("</div>")
        skipped = [spec.title for spec in get_registered_plots(section)
                   if spec.name not in figs]
        if skipped:
            parts.append("<p class='skipped'>Skipped (missing optional data): "
                         + ", ".join(html_mod.escape(s) for s in skipped) + "</p>")
    parts.append(f"<footer>Generated by src.backtest.viz for '{result.name}'</footer>")
    parts.append("</body></html>")
    html_doc = "\n".join(parts)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_doc, encoding="utf-8")
        logger.info("Backtest report written to %s", out)
    return figures


def render_overview_cards(result: BacktestResult) -> str:
    """Standalone HTML fragment of the summary cards (for embedding elsewhere)."""
    return overview_cards_html(result)


def run_single_plot(result: BacktestResult, plot_name: str):
    """Convenience: build one registered plot by name (used by dashboards/tests)."""
    spec = next((sp for sp in get_registered_plots() if sp.name == plot_name), None)
    if spec is None:
        raise KeyError(f"No registered plot named '{plot_name}'. "
                       f"Available: {[sp.name for sp in get_registered_plots()]}")
    return run_plot(spec, result)
