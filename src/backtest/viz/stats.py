"""Metric computations for the backtest visualization suite.

All functions take a `BacktestResult` (or its prepared frames) and return
plain pandas/numpy structures so they are independently reusable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.viz.schema import BacktestResult, daily_returns

ANNUALIZATION = 365.0  # weather markets settle and trade every calendar day


# ------------------------------------------------------------------ core metrics
def drawdown_frame(equity: pd.Series) -> pd.DataFrame:
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = np.where(peak > 0, dd / peak, 0.0)
    return pd.DataFrame({"equity": equity, "peak": peak, "drawdown": dd, "drawdown_pct": dd_pct})


def drawdown_spells(equity: pd.Series) -> pd.DataFrame:
    """Each underwater spell: start, end (or None if still open), days, max depth %."""
    dd = drawdown_frame(equity)
    under = dd["drawdown"] < 0
    spells = []
    start = None
    trough = 0.0
    for ts, is_under in under.items():
        if is_under and start is None:
            start = ts
            trough = dd.loc[ts, "drawdown_pct"]
        elif is_under:
            trough = min(trough, float(dd.loc[ts, "drawdown_pct"]))
        elif start is not None:
            spells.append({"start": start, "end": ts, "days": (ts - start).total_seconds() / 86400.0,
                           "depth_pct": trough})
            start = None
    if start is not None:
        spells.append({"start": start, "end": pd.NaT,
                       "days": (equity.index[-1] - start).total_seconds() / 86400.0,
                       "depth_pct": trough})
    return pd.DataFrame(spells).sort_values("days", ascending=False) if spells else pd.DataFrame(
        columns=["start", "end", "days", "depth_pct"])


def summary_stats(result: BacktestResult, benchmark: bool = False) -> dict[str, float]:
    """Headline statistics powering the overview cards."""
    eq = result.equity_series()
    t = result.prepared_trades()
    rets = daily_returns(eq)
    total_pnl = float(eq.iloc[-1] - result.initial_capital)
    total_return = float(eq.iloc[-1] / result.initial_capital - 1.0)
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / (365.0 * 86400.0), 1e-9)
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    vol = float(rets.std()) * np.sqrt(ANNUALIZATION) if len(rets) > 1 else np.nan
    sharpe = float(rets.mean() / rets.std() * np.sqrt(ANNUALIZATION)) if len(rets) > 1 and rets.std() > 0 else np.nan
    downside = rets[rets < 0]
    ddev = float(np.sqrt((downside ** 2).mean())) if len(downside) else np.nan
    sortino = float(rets.mean() / ddev * np.sqrt(ANNUALIZATION)) if ddev and ddev > 0 and len(rets) > 1 else np.nan
    dd = drawdown_frame(eq)["drawdown_pct"]
    max_dd = float(dd.min()) if len(dd) else np.nan

    n_trades = int(len(t))
    wins = t[t["net_pnl"] > 0]["net_pnl"]
    losses = t[t["net_pnl"] < 0]["net_pnl"]
    win_rate = len(wins) / n_trades if n_trades else np.nan
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf if gross_win > 0 else np.nan
    fees_paid = float(pd.to_numeric(t["fees"], errors="coerce").fillna(0).sum())
    slip_paid = float(pd.to_numeric(t["slippage_usd"], errors="coerce").fillna(0).sum())
    equity_base = float(eq.mean())
    if not np.isfinite(equity_base) or equity_base <= 0:
        equity_base = float(result.initial_capital)
    turnover = float(pd.to_numeric(t["notional"], errors="coerce").fillna(
        pd.to_numeric(t["gross_cost"], errors="coerce")).sum() / equity_base) if n_trades else np.nan
    avg_hold = float(t["holding_days"].mean()) if n_trades else np.nan

    stats = {
        "initial_capital": result.initial_capital,
        "final_equity": float(eq.iloc[-1]),
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annualized_return": float(ann_return),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "annualized_volatility": vol,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_trade_pnl": float(t["net_pnl"].mean()) if n_trades else np.nan,
        "median_trade_pnl": float(t["net_pnl"].median()) if n_trades else np.nan,
        "avg_holding_days": avg_hold,
        "turnover": turnover,
        "fees_paid": fees_paid,
        "slippage_paid": slip_paid,
        "execution_costs": fees_paid + slip_paid,
        "largest_win": float(t["net_pnl"].max()) if n_trades else np.nan,
        "largest_loss": float(t["net_pnl"].min()) if n_trades else np.nan,
        "gross_pnl": float(pd.to_numeric(t["gross_pnl"], errors="coerce").fillna(0).sum()),
        "net_pnl": total_pnl,
        "period_days": (eq.index[-1] - eq.index[0]).days,
        "avg_gross_exposure_pct": np.nan,
        "pct_profitable_days": float((daily_returns(eq) > 0).mean()) if len(rets) else np.nan,
    }
    expo = exposure_frame(result)
    if len(expo):
        stats["avg_gross_exposure_pct"] = float(expo["exposure_pct"].replace([np.inf, -np.inf], np.nan).mean())
    return stats


def rolling_metrics(result: BacktestResult, window_days: int = 30) -> pd.DataFrame:
    eq = result.equity_series()
    rets = daily_returns(eq)
    roll_ret = eq.resample("D").last().pct_change(window_days)
    roll_vol = rets.rolling(window_days).std() * np.sqrt(ANNUALIZATION)
    mean = rets.rolling(window_days).mean()
    std = rets.rolling(window_days).std()
    roll_sharpe = (mean / std * np.sqrt(ANNUALIZATION)) if isinstance(mean, pd.Series) else mean
    out = pd.DataFrame({
        "rolling_return": roll_ret,
        "rolling_sharpe": roll_sharpe,
        "rolling_volatility": roll_vol,
    })
    return out.dropna(how="all")


def streaks(net_pnl: pd.Series) -> pd.Series:
    """Signed consecutive streak length at each trade (+win run / -loss run)."""
    sign = np.sign(net_pnl.values)
    out = np.zeros(len(sign), dtype=float)
    run = 0.0
    for i, s in enumerate(sign):
        if s == 0:
            run = 0.0
        elif s > 0:
            run = run + 1 if run > 0 else 1
        else:
            run = run - 1 if run < 0 else -1
        out[i] = run
    return pd.Series(out, index=net_pnl.index, name="streak")


def streak_summary(net_pnl: pd.Series) -> dict[str, float]:
    s = streaks(net_pnl)
    return {
        "max_consecutive_wins": int(s.max()) if len(s) else 0,
        "max_consecutive_losses": int(-s.min()) if len(s) else 0,
        "current_streak": int(s.iloc[-1]) if len(s) else 0,
    }


def best_worst_periods(result: BacktestResult, freq: str = "D", k: int = 5) -> dict[str, pd.Series]:
    pnl = period_pnl(result, freq)
    return {"best": pnl.nlargest(k), "worst": pnl.nsmallest(k)}


def period_pnl(result: BacktestResult, freq: str = "D") -> pd.Series:
    t = result.prepared_trades()
    if t.empty:
        return pd.Series(dtype=float)
    per = t.set_index("signal_ts")["net_pnl"].resample(freq).sum()
    label = {"D": "day", "W": "week", "ME": "month"}.get(freq, freq)
    per.name = f"pnl_per_{label}"
    return per


def weekday_pnl(result: BacktestResult) -> pd.DataFrame:
    t = result.prepared_trades()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = t.groupby("weekday_ny")
    out = g.agg(n_trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
                avg_pnl=("net_pnl", "mean"), win_rate=("win", "mean")).reindex(order)
    return out.dropna(how="all")


# ------------------------------------------------------------------- exposure
def exposure_frame(result: BacktestResult, sample_points: int | None = None) -> pd.DataFrame:
    """Time series of gross/net exposure, position count, capital deployed.

    Derived from trade holding intervals when `positions` snapshots absent.
    """
    def build() -> pd.DataFrame:
        if result.has_positions:
            p = result.positions.copy()
            ts_col = next(c for c in ("timestamp", "time", "signal_ts", "ts") if c in p.columns)
            p[ts_col] = pd.DatetimeIndex(pd.to_datetime(p[ts_col], utc=True))
            value_col = next(c for c in ("value", "notional", "gross_value", "market_value") if c in p.columns)
            signed = value_col if "signed_value" in p.columns else None
            grid = p.set_index(ts_col).sort_index()
            gross = grid[value_col].abs().groupby(level=0).sum().resample("h").ffill()
            net = (grid[signed] if signed is not None else grid[value_col]).groupby(level=0).sum().resample("h").ffill()
            count = grid.groupby(level=0).size().resample("h").sum().ffill()
        elif result.has_trades:
            t = result.prepared_trades()
            events = []
            for ts_in, ts_out, notional, direction in zip(
                    t["signal_ts"], t["exit_ts"],
                    pd.to_numeric(t["notional"], errors="coerce").fillna(
                        pd.to_numeric(t["gross_cost"], errors="coerce")),
                    t["direction"]):
                if pd.isna(notional):
                    continue
                events.append((ts_in, notional, direction))
                events.append((ts_out, -notional, direction))
            if not events:
                return pd.DataFrame(columns=["gross_exposure", "net_exposure", "n_positions",
                                             "cash", "equity", "exposure_pct"])
            ev = pd.DataFrame(events, columns=["t", "delta_notional", "delta_signed"]).sort_values(
                "t", kind="stable")
            gross = ev["delta_notional"].cumsum()
            net = (ev["delta_notional"] * ev["delta_signed"]).cumsum()
            count = np.sign(ev["delta_notional"]).astype(int).cumsum()
            frame = pd.DataFrame(
                {"gross_exposure": gross.to_numpy(), "net_exposure": net.to_numpy(),
                 "n_positions": count.to_numpy()},
                index=pd.DatetimeIndex(ev["t"]))
            frame = frame.groupby(level=0).last().sort_index()
            hourly = frame.resample("h").ffill()
            last = frame.iloc[[-1]]
            hourly = pd.concat([hourly, last]) if hourly.index[-1] != last.index[0] else hourly
            gross, net, count = hourly["gross_exposure"], hourly["net_exposure"], hourly["n_positions"]
        else:
            return pd.DataFrame(columns=["gross_exposure", "net_exposure", "n_positions",
                                         "cash", "equity", "exposure_pct"])
        idx = gross.index
        eq = result.equity_series()
        eq_at = eq.reindex(idx.union(eq.index)).ffill().reindex(idx)
        cash = result.cash_series()
        cash_at = cash.reindex(idx.union(cash.index)).ffill().reindex(idx) if cash is not None else eq_at - gross
        deployed = gross
        bankroll = eq_at.replace(0, np.nan)
        out = pd.DataFrame({
            "gross_exposure": gross,
            "net_exposure": net,
            "n_positions": count,
            "cash": cash_at,
            "equity": eq_at,
            "capital_deployed": deployed,
            "exposure_pct": gross / bankroll,
        })
        return out[out.index >= result.equity_series().index.min()]
    return result._cached("exposure", build)


def concentration_series(result: BacktestResult, max_points: int = 400) -> pd.Series:
    """Herfindahl index of open-notional share by market over time (0=single..1=concentrated)."""
    t = result.prepared_trades()
    if t.empty:
        return pd.Series(dtype=float)
    span_end = max(t["exit_ts"].max(), t["signal_ts"].max())
    grid = pd.date_range(t["signal_ts"].min(), span_end, periods=min(max_points, max(len(t), 2)))
    entries = t["signal_ts"].values.astype("datetime64[ns]")
    exits = t["exit_ts"].values.astype("datetime64[ns]")
    notionals = pd.to_numeric(t["notional"], errors="coerce").fillna(
        pd.to_numeric(t["gross_cost"], errors="coerce")).values.astype(float)
    markets = t["market_ticker"].astype(str).values
    values = []
    for ts in grid:
        naive = np.datetime64(ts.tz_convert("UTC").tz_localize(None))
        open_mask = (entries <= naive) & (exits > naive)
        if not open_mask.any():
            values.append(np.nan)
            continue
        totals = pd.Series(notionals[open_mask]).groupby(markets[open_mask]).sum()
        shares = totals / totals.sum()
        values.append(float((shares ** 2).sum()))
    return pd.Series(values, index=grid, name="hhi")


def group_breakdown(trades: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Aggregate P&L/win-rate/counts by one or more grouping columns."""
    cols = [by] if isinstance(by, str) else list(by)
    missing = [c for c in cols if c not in trades.columns]
    for c in missing:
        trades = trades.copy()
        trades[c] = "unknown"
    grouped = trades.groupby(cols, observed=True)
    out = grouped.apply(lambda g: pd.Series({
        "n_trades": len(g),
        "net_pnl": g["net_pnl"].sum(),
        "gross_pnl": g["gross_pnl"].sum(),
        "avg_pnl": g["net_pnl"].mean(),
        "median_pnl": g["net_pnl"].median(),
        "win_rate": g["win"].mean(),
        "fees": g["fees"].sum(),
        "slippage": g["slippage_usd"].sum(),
        "notional": g["notional"].sum(),
        "avg_edge": g["predicted_edge"].mean(),
        "avg_holding_days": g["holding_days"].mean(),
    }), include_groups=False)
    out["pnl_share"] = out["net_pnl"] / out["net_pnl"].sum() if out["net_pnl"].sum() else np.nan
    return out.sort_values("net_pnl", ascending=False)


def risk_contribution_by_group(result: BacktestResult, by: str = "city") -> pd.DataFrame:
    """Euler-style risk contribution of each group to total daily-P&L volatility."""
    t = result.prepared_trades()
    if t.empty or by not in t.columns:
        return pd.DataFrame()
    daily = t.assign(date=t["signal_date"]).pivot_table(index="date", columns=by,
                                                        values="net_pnl", aggfunc="sum").fillna(0.0)
    total = daily.sum(axis=1)
    var_p = float(total.var(ddof=1)) if len(total) > 1 else np.nan
    rows = []
    for col in daily.columns:
        cov = float(daily[col].cov(total)) if len(daily) > 2 else np.nan
        contrib = cov / var_p if var_p and var_p > 0 and np.isfinite(cov) else np.nan
        rows.append({by: col, "vol_contribution": contrib,
                     "standalone_vol": float(daily[col].std()) if len(daily) > 1 else np.nan,
                     "net_pnl": float(daily[col].sum()), "n_trades": int((daily[col] != 0).sum())})
    out = pd.DataFrame(rows)
    total_contrib = out["vol_contribution"].sum(skipna=True)
    out["vol_share"] = out["vol_contribution"] / total_contrib if total_contrib else np.nan
    return out.sort_values("vol_contribution", ascending=False)


# ------------------------------------------------------------------ MAE / MFE
def mae_mfe(result: BacktestResult, price_col: str = "price") -> pd.DataFrame:
    """Attach MAE/MFE (in $ and in price points) per trade using intratrade market prices."""
    t = result.prepared_trades().copy()
    if not result.has_prices:
        t["mae_usd"] = np.nan; t["mfe_usd"] = np.nan
        t["mae_price"] = np.nan; t["mfe_price"] = np.nan
        return t
    px = result.prices.copy()
    ts_col = "timestamp" if "timestamp" in px.columns else px.columns[0]
    ticker_col = "market_ticker" if "market_ticker" in px.columns else px.columns[1]
    px[ts_col] = pd.to_datetime(px[ts_col], utc=True, errors="coerce")
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ticker, g in px.dropna(subset=[ts_col]).groupby(ticker_col):
        g = g.sort_values(ts_col)
        lookup[str(ticker)] = (g[ts_col].values.astype("datetime64[ns]"),
                               pd.to_numeric(g[price_col], errors="coerce").values)
    mae_p, mfe_p = [], []
    for row in t.itertuples():
        key = str(getattr(row, "market_ticker"))
        entry, exit_ = getattr(row, "signal_ts"), getattr(row, "exit_ts")
        contracts = getattr(row, "contracts") or np.nan
        entry_price = getattr(row, "entry_price")
        direction = getattr(row, "direction")
        if key not in lookup or pd.isna(entry_price) or pd.isna(entry):
            mae_p.append(np.nan); mfe_p.append(np.nan); continue
        times, values = lookup[key]
        entry_naive = np.datetime64(entry.tz_convert("UTC").tz_localize(None))
        exit_naive = np.datetime64(exit_.tz_convert("UTC").tz_localize(None)) if pd.notna(exit_) else None
        lo = np.searchsorted(times, entry_naive)
        hi = np.searchsorted(times, exit_naive, side="right") if exit_naive is not None else len(times)
        window = values[max(lo - 1, 0):min(hi + 1, len(values))] if hi > lo else values[max(lo - 1, 0):lo + 1]
        if len(window) == 0 or np.all(np.isnan(window)):
            mae_p.append(np.nan); mfe_p.append(np.nan); continue
        excursion = direction * (window - entry_price)
        mae_p.append(float(max(-np.nanmin(excursion), 0.0)))
        mfe_p.append(float(max(np.nanmax(excursion), 0.0)))
    t["mae_price"] = mae_p
    t["mfe_price"] = mfe_p
    t["mae_usd"] = t["mae_price"] * pd.to_numeric(t["contracts"], errors="coerce")
    t["mfe_usd"] = t["mfe_price"] * pd.to_numeric(t["contracts"], errors="coerce")
    return t


# ---------------------------------------------------------------- calibration
def calibration_table(result: BacktestResult, bins: int = 10) -> pd.DataFrame:
    """Reliability data from model_probability vs binary settlement outcome on trades."""
    t = result.prepared_trades()
    prob_col = "model_probability"
    outcome_col = "settlement"
    if t.empty or prob_col not in t.columns or outcome_col not in t.columns:
        return pd.DataFrame()
    df = t[[prob_col, outcome_col]].copy()
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df.dropna()
    df = df[df[outcome_col].isin([0, 1])]
    if df.empty:
        return pd.DataFrame()
    edges = np.linspace(0, 1, bins + 1)
    df["bin"] = pd.cut(df[prob_col], bins=edges, include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        n=(outcome_col, "size"),
        avg_predicted=(prob_col, "mean"),
        realized_freq=(outcome_col, "mean"),
    ).reset_index()
    out["calibration_error"] = out["avg_predicted"] - out["realized_freq"]
    out["ece_share"] = out["n"] / out["n"].sum() * out["calibration_error"].abs()
    return out


def calibration_scores(result: BacktestResult) -> dict[str, float]:
    cal = calibration_table(result)
    if cal.empty:
        return {}
    t = result.prepared_trades()
    df = t[["model_probability", "settlement"]].dropna()
    df = df[pd.to_numeric(df["settlement"], errors="coerce").isin([0, 1])]
    p = pd.to_numeric(df["model_probability"], errors="coerce").clip(1e-6, 1 - 1e-6)
    y = pd.to_numeric(df["settlement"], errors="coerce")
    brier = float(((p - y) ** 2).mean())
    logloss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    return {"ece": float(cal["ece_share"].sum()), "brier": brier, "logloss": logloss, "n": int(cal["n"].sum())}


# ------------------------------------------------------- execution scenarios
def fee_scenarios(result: BacktestResult,
                  fee_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0),
                  extra_slippage_cents: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)) -> pd.DataFrame:
    """Net P&L under alternative fee/slippage assumptions.

    Kalshi fees scale linearly with the fee rate, so a multiplier on recorded
    fees is exact. Extra slippage charges `cents` per contract per side.
    """
    t = result.prepared_trades()
    base_fee_rate = float(result.meta.get("fee_rate", 0.07))
    rows = []
    gross = float(pd.to_numeric(t["gross_pnl"], errors="coerce").fillna(0).sum())
    for fm in fee_multipliers:
        for cents in extra_slippage_cents:
            fees = float(pd.to_numeric(t["fees"], errors="coerce").fillna(0).sum()) * fm
            slip_extra = float((pd.to_numeric(t["contracts"], errors="coerce").fillna(0) * cents / 100.0).sum())
            rows.append({"fee_multiplier": fm, "effective_fee_rate": base_fee_rate * fm,
                         "extra_slippage_cents": cents, "gross_pnl": gross,
                         "fees": fees, "extra_slippage": slip_extra,
                         "net_pnl": gross - fees - slip_extra})
    out = pd.DataFrame(rows)
    out["survives_costs"] = out["net_pnl"] > 0
    return out


def expected_vs_realized(result: BacktestResult) -> pd.DataFrame:
    """Cumulative expected P&L (model edge priced at execution cost) vs realized net P&L."""
    t = result.prepared_trades().sort_values("signal_ts")
    contracts = pd.to_numeric(t["contracts"], errors="coerce").fillna(0)
    edge = pd.to_numeric(t["predicted_edge"], errors="coerce").fillna(0)
    expected = (edge * contracts - pd.to_numeric(t["fees"], errors="coerce").fillna(0))
    out = pd.DataFrame({
        "expected_cum": expected.cumsum().values,
        "realized_cum": pd.to_numeric(t["net_pnl"], errors="coerce").fillna(0).cumsum().values,
    }, index=pd.DatetimeIndex(t["signal_ts"]))
    return out
