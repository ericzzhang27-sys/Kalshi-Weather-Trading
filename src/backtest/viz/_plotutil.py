"""Shared styling and small helpers for viz plot modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

COLOR_POS = "#2ca02c"
COLOR_NEG = "#d62728"
COLOR_MAIN = "#1f77b4"
COLOR_BENCH = "#7f7f7f"
COLOR_ACCENT = "#ff7f0e"
PAPER_BG = "#fafafa"

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=50, r=30, t=55, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def style(fig: go.Figure, title: str | None = None, y_title: str | None = None,
          x_title: str | None = None, height: int = 420) -> go.Figure:
    layout = dict(LAYOUT_DEFAULTS)
    layout["height"] = height
    if title:
        layout["title"] = title
    if y_title:
        layout["yaxis_title"] = y_title
    if x_title:
        layout["xaxis_title"] = x_title
    fig.update_layout(**layout)
    return fig


def empty_figure(message: str, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="#888"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if title:
        fig.update_layout(title=title)
    return style(fig)


def bar_colors(values: pd.Series | np.ndarray) -> list[str]:
    return [COLOR_POS if (pd.notna(v) and v >= 0) else COLOR_NEG for v in values]


def add_winrate_line(fig: go.Figure, x, win_rate: pd.Series, name: str = "win rate") -> go.Figure:
    fig.add_trace(go.Scatter(
        x=x, y=win_rate, name=name, yaxis="y2",
        mode="lines+markers", line=dict(color=COLOR_MAIN, width=2),
        marker=dict(size=6), hovertemplate="%{x}<br>win rate: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(yaxis2=dict(title=name, overlaying="y", side="right", range=[0, 1],
                                  tickformat=".0%"))
    return fig


def agg_by_column(t: pd.DataFrame, col: str) -> pd.DataFrame:
    """Standard per-group aggregates used across breakdown plots."""
    g = t.groupby(col, observed=True)
    out = g.agg(n_trades=("net_pnl", "size"), total_pnl=("net_pnl", "sum"),
                avg_pnl=("net_pnl", "mean"), win_rate=("win", "mean"))
    return out.reset_index()
