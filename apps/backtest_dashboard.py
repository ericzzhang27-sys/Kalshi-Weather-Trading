"""Interactive backtest diagnostics dashboard (Streamlit).

Run from the repo root::

    streamlit run apps/backtest_dashboard.py

Loads one or more real engine ledgers from `outputs/backtests/`, attaches the
real candle history for price overlays, and exposes every registered plot with
live filtering by date range, edge, price, side, and trade characteristics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.viz import (  # noqa: E402
    BacktestResult,
    SECTION_ORDER,
    build_all_figures,
    comparison_table_figure,
    get_registered_plots,
)
from src.backtest.viz.report import render_overview_cards  # noqa: E402
from src.backtest.viz.compare import (  # noqa: E402
    create_comparison_report,
    plot_compare_drawdown,
    plot_compare_equity,
    plot_compare_rolling_sharpe,
)
from src.backtest.viz.demo import DEFAULT_CANDLES, DEFAULT_LEDGERS, load_candle_prices  # noqa: E402
from src.backtest.viz.replay import (  # noqa: E402
    build_daily_replay_figure,
    load_candle_frame,
)

SECTION_TITLES = {
    "overview": "Overview",
    "performance": "Performance",
    "risk": "Exposure & Risk",
    "trades": "Trade Analysis",
    "edge": "Strategy Edge",
    "execution": "Execution Costs",
    "robustness": "Robustness & Stability",
}

st.set_page_config(page_title="Backtest diagnostics", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Loading ledger...")
def load_ledger(path_str: str, mtime: float = 0.0) -> pd.DataFrame:
    path = Path(path_str)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner="Loading candle history...")
def load_prices(tickers: tuple[str, ...], candles_mtime: float = 0.0) -> pd.DataFrame | None:
    return load_candle_prices(DEFAULT_CANDLES, market_tickers=list(tickers))


@st.cache_data(show_spinner="Building figures...", hash_funcs={BacktestResult: lambda r: id(r)})
def build_section_figures(result: BacktestResult) -> dict:
    return build_all_figures(result)


@st.cache_data(show_spinner="Loading candle history for replay...")
def load_replay_candles(candles_mtime: float = 0.0) -> pd.DataFrame | None:
    if not DEFAULT_CANDLES.exists():
        return None
    return load_candle_frame(DEFAULT_CANDLES)


def available_ledgers() -> dict[str, str]:
    found = {}
    backtest_dir = REPO_ROOT / "outputs" / "backtests"
    if backtest_dir.exists():
        for name, default_path in DEFAULT_LEDGERS.items():
            if Path(default_path).exists():
                found[name] = str(default_path)
        for extra in sorted(backtest_dir.glob("trades*.parquet")) + sorted(backtest_dir.glob("trades*.csv")):
            key = f"custom: {extra.name}"
            if str(extra) not in found.values() and "high_risk" not in extra.name and extra.name != "trades.csv":
                found[key] = str(extra)
    return found


def _replay_date_options(full_results: dict[str, BacktestResult],
                         candles: pd.DataFrame | None) -> list[str]:
    dates: set[str] = set()
    for result in full_results.values():
        if result.has_trades and "target_date" in result.trades.columns:
            dates |= set(result.trades["target_date"].astype(str))
    if candles is not None:
        dates |= set(candles["target_date"].astype(str))
    return sorted(dates)


def render_daily_replay(primary_full: BacktestResult, candles: pd.DataFrame | None) -> None:
    st.header("Daily market replay")
    st.caption("Intraday price path of every Kalshi temperature bucket market for one "
               "trading date, with strategy entries/exits overlaid. All charts share "
               "one synchronized time axis - zoom or pan any row to move all of them.")
    if candles is None:
        st.warning("Candle history not found; replay needs "
                   "data/kalshi/processed/historical_candles_processed.csv.")
        return

    prep = primary_full.prepared_trades() if primary_full.has_trades else None
    date_options = _replay_date_options({primary_full.name: primary_full}, candles)
    if not date_options:
        st.info("No dated markets available for replay.")
        return

    if "replay_date" not in st.session_state:
        trade_dates = sorted(set(prep["target_date"].astype(str))) if prep is not None else []
        st.session_state["replay_date"] = trade_dates[-1] if trade_dates else date_options[0]
    current = st.session_state["replay_date"]
    idx = date_options.index(current) if current in date_options else 0

    nav_prev, nav_pick, nav_next, nav_info = st.columns([1, 2, 1, 3])
    with nav_prev:
        if st.button("\u2190 Previous day", disabled=idx <= 0, use_container_width=True):
            st.session_state["replay_date"] = date_options[idx - 1]
            st.rerun()
    with nav_pick:
        picked = st.date_input(
            "Trading date",
            value=pd.Timestamp(current).date(),
            min_value=pd.Timestamp(date_options[0]).date(),
            max_value=pd.Timestamp(date_options[-1]).date(),
            key="replay_picker",
        )
    with nav_next:
        if st.button("Next day \u2192", disabled=idx >= len(date_options) - 1,
                     use_container_width=True):
            st.session_state["replay_date"] = date_options[idx + 1]
            st.rerun()
    with nav_info:
        st.caption(f"Date {idx + 1} of {len(date_options)}")

    picked_str = str(picked)
    if picked_str != current:
        st.session_state["replay_date"] = picked_str
        st.rerun()

    day_candles = candles[candles["target_date"] == picked_str]
    day_trades = (prep[prep["target_date"].astype(str) == picked_str]
                  if prep is not None else pd.DataFrame())

    m_left, m_mid, m_right, m_last = st.columns(4)
    n_markets = day_candles["market_ticker"].nunique() if len(day_candles) else 0
    m_left.metric("Bucket markets", n_markets)
    m_mid.metric("Trades", int(len(day_trades)))
    day_pnl = float(day_trades["net_pnl"].sum()) if len(day_trades) else 0.0
    m_right.metric("Day net P&L", f"${day_pnl:+,.2f}")
    m_last.metric("Contracts", float(day_trades["contracts"].sum()) if len(day_trades) else 0.0)

    if not len(day_trades) and not len(day_candles):
        st.info(f"No markets or trades on {picked_str}. Use the navigation above.")
        return

    fig = build_daily_replay_figure(
        day_candles, day_trades,
        title=f"Market replay - {primary_full.name} - {picked_str}",
    )
    st.plotly_chart(fig, use_container_width=True, key="daily-replay-chart")


def main() -> None:
    st.title("Backtest visualization & diagnostics")
    choices = available_ledgers()
    if not choices:
        st.error("No ledgers found in outputs/backtests/. Run `python scripts/run_kalshi_backtest.py` first.")
        st.stop()

    with st.sidebar:
        st.header("Runs")
        selected = st.multiselect("Backtest runs", list(choices), default=list(choices)[:1])
        st.divider()
        st.header("Filters")
        start = st.date_input("From", value=None)
        end = st.date_input("To", value=None)
        min_edge_pct = st.slider("Min predicted edge (%)", 0.0, 30.0, 0.0, 0.5)
        max_price_c = st.slider("Max entry price (c)", 1, 99, 99)
        win_only = st.checkbox("Winning trades only")
        loss_only = st.checkbox("Losing trades only")

    if not selected:
        st.info("Select at least one run in the sidebar.")
        st.stop()

    results: dict[str, BacktestResult] = {}
    full_results: dict[str, BacktestResult] = {}
    for name in selected:
        ledger_path = Path(choices[name])
        ledger = load_ledger(str(ledger_path), file_mtime(ledger_path))
        result = BacktestResult.from_ledger(ledger, initial_capital=1000.0, name=name)
        prices = load_prices(tuple(result.trades["market_ticker"].astype(str).unique()),
                             file_mtime(DEFAULT_CANDLES))
        if prices is not None:
            result.prices = prices
        full_results[name] = result
        kwargs = {}
        if start:
            kwargs["start"] = pd.Timestamp(start, tz="UTC")
        if end:
            kwargs["end"] = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        if min_edge_pct > 0:
            kwargs["min_edge"] = min_edge_pct / 100.0
        if max_price_c < 99:
            kwargs["max_price"] = max_price_c / 100.0
        if win_only:
            kwargs["win_only"] = True
        if loss_only:
            kwargs["loss_only"] = True
        results[name] = result.filter(**kwargs)

    primary_name = next(iter(results))
    primary = results[primary_name]
    if not primary.has_trades:
        st.warning("Filters removed every trade for the primary run.")
        st.stop()

    tab_names = [SECTION_TITLES[s] for s in SECTION_ORDER] + ["Daily Market Replay", "Compare runs"]
    tabs = st.tabs(tab_names)

    figures_by_section = build_section_figures(primary)
    for tab, section in zip(tabs[:-2], SECTION_ORDER):
        with tab:
            if section == "overview":
                st.markdown(render_overview_cards(primary), unsafe_allow_html=True)
                continue
            figs = figures_by_section.get(section, {})
            if not figs:
                st.caption("No plots available for this section with the current filters.")
            for name, fig in figs.items():
                spec_title = next((sp.title for sp in get_registered_plots(section) if sp.name == name), name)
                with st.expander(spec_title, expanded=True):
                    st.plotly_chart(fig, use_container_width=True, key=f"{section}-{name}")
            skipped = [sp.title for sp in get_registered_plots(section) if sp.name not in figs]
            if skipped:
                st.caption("Skipped (missing optional data): " + ", ".join(skipped))

    with tabs[-2]:
        render_daily_replay(full_results[primary_name], load_replay_candles(file_mtime(DEFAULT_CANDLES)))

    with tabs[-1]:
        st.header("Comparison mode")
        if len(results) < 2:
            st.caption("Select two or more runs in the sidebar to compare equity curves, "
                       "drawdowns, Sharpe, turnover, costs and win rates side by side.")
        table, _ = create_comparison_report(results, output_path=None)
        st.plotly_chart(comparison_table_figure(table), use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.plotly_chart(plot_compare_equity(results), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_compare_drawdown(results), use_container_width=True)
        with col_c:
            st.plotly_chart(plot_compare_rolling_sharpe(results), use_container_width=True)
        csv = table.to_csv(index=False)
        st.download_button("Download comparison CSV", csv, "comparison_table.csv", "text/csv")

    with st.expander("Trade detail table (sortable)", expanded=False):
        t = primary.prepared_trades()
        cols = ["signal_ts", "market_ticker", "bucket_label", "side_norm", "entry_price",
                "contracts", "fees", "slippage_usd", "predicted_edge", "settlement",
                "net_pnl", "holding_days"]
        view = t[[c for c in cols if c in t.columns]].copy()
        view["signal_ts"] = pd.DatetimeIndex(view["signal_ts"]).strftime("%Y-%m-%d %H:%M")
        st.dataframe(view, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
