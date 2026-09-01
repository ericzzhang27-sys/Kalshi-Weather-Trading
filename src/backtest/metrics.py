from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .engine import BacktestConfig, run_backtest

"""
Metrics (spec section 14) and breakdown analyses (section 15) + data integrity + calibration.

For each strategy/model: trades, win rate, P&L, return, avg return, max drawdown, Sharpe, Sortino, profit factor, etc.
"""

def _to_float(v) -> float | None:
    try:
        if v is None or (isinstance(v,float) and np.isnan(v)): return None
        f=float(v)
        if not math.isfinite(f): return None
        return f
    except Exception: return None


def ledger_metrics(
    ledger: pd.DataFrame,
    *,
    starting_cash: float = 1000.0,
    evaluation_dates: list[object] | tuple[object, ...] | None = None,
) -> dict[str, Any]:
    base = compute_ledger_metrics(ledger)
    sharpe = sortino = calmar = None
    max_drawdown_pct = 0.0
    annualized_return = 0.0
    cagr = 0.0
    total_return = 0.0
    elapsed_calendar_days = 0
    if not ledger.empty:
        dates = pd.to_datetime(ledger["target_date"], errors="raise")
        daily = ledger.assign(_date=dates.dt.date).groupby("_date")["net_pnl"].sum()
        if evaluation_dates is not None:
            calendar = pd.Index(
                sorted(set(pd.to_datetime(pd.Series(evaluation_dates), errors="raise").dt.date)),
                name="_date",
            )
            daily = daily.reindex(calendar, fill_value=0.0)
        daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
        downside = np.minimum(daily.to_numpy(float), 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
        sharpe = float(np.sqrt(252.0) * daily.mean() / daily_std) if daily_std > 0 else None
        sortino = (
            float(np.sqrt(252.0) * daily.mean() / downside_deviation)
            if downside_deviation > 0 else None
        )
        equity = starting_cash + daily.cumsum()
        drawdown_pct = equity / equity.cummax().clip(lower=starting_cash) - 1.0
        max_drawdown_pct = float(drawdown_pct.min())
        annualized_return = float((daily.sum() / starting_cash) * (252.0 / len(daily)))
        total_return = float(equity.iloc[-1] / starting_cash - 1.0)
        calendar_dates = pd.to_datetime(pd.Series(daily.index), errors="raise")
        elapsed_calendar_days = int((calendar_dates.max() - calendar_dates.min()).days + 1)
        if equity.iloc[-1] > 0 and elapsed_calendar_days > 0:
            cagr = float(
                (equity.iloc[-1] / starting_cash)
                ** (365.2425 / elapsed_calendar_days)
                - 1.0
            )
        calmar = (
            float(annualized_return / abs(max_drawdown_pct))
            if max_drawdown_pct < 0 else None
        )
    return {
        "n_trades": int(base["n_trades"]),
        "net_pnl": float(base["net_pnl"]),
        "win_rate": float(base["win_rate"]) if np.isfinite(base["win_rate"]) else 0.0,
        "max_drawdown": float(base["max_drawdown"]),
        "max_drawdown_pct": max_drawdown_pct,
        "annualized_return": annualized_return,
        "cagr": cagr,
        "total_return": total_return,
        "elapsed_calendar_days": elapsed_calendar_days,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "profit_factor": float(base["profit_factor"]) if np.isfinite(base["profit_factor"]) else None,
        "n_event_days": int(pd.to_datetime(ledger["target_date"], errors="raise").dt.date.nunique()) if not ledger.empty else 0,
        "n_evaluation_days": int(len(set(pd.to_datetime(pd.Series(evaluation_dates), errors="raise").dt.date))) if evaluation_dates is not None else (
            int(pd.to_datetime(ledger["target_date"], errors="raise").dt.date.nunique()) if not ledger.empty else 0
        ),
        "supports_profitability_claim": bool(
            not ledger.empty
            and "supports_profitability_claim" in ledger
            and ledger["supports_profitability_claim"].all()
        ),
    }


def select_threshold_on_validation(
    aligned: pd.DataFrame,
    thresholds,
    *,
    validation_start: str = "2024-01-01",
    test_start: str = "2025-01-01",
    base_config: BacktestConfig | None = None,
) -> tuple[float, pd.DataFrame]:
    dates = pd.to_datetime(aligned["target_date"], errors="raise")
    validation = aligned[(dates >= pd.Timestamp(validation_start)) & (dates < pd.Timestamp(test_start))]
    if validation.empty:
        raise ValueError("No validation rows are available for threshold selection")
    cfg = base_config or BacktestConfig()
    rows = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        candidate = BacktestConfig(
            threshold=threshold,
            allow_buy_no=cfg.allow_buy_no,
            execution_mode=cfg.execution_mode,
            max_execution_gap_minutes=cfg.max_execution_gap_minutes,
            fee_rate=cfg.fee_rate,
            adverse_slippage_ticks=cfg.adverse_slippage_ticks,
            tick_size=cfg.tick_size,
            one_position_per_market=cfg.one_position_per_market,
            require_depth_for_multiple_contracts=cfg.require_depth_for_multiple_contracts,
            sizing=cfg.sizing,
        )
        rows.append({
            "threshold": threshold,
            "selection_split": "validation",
            **ledger_metrics(run_backtest(validation, candidate)),
        })
    table = pd.DataFrame(rows)
    selected = table.sort_values(
        ["net_pnl", "n_trades", "threshold"], ascending=[False, True, False]
    ).iloc[0]
    table["selected"] = table["threshold"].eq(float(selected["threshold"]))
    return float(selected["threshold"]), table


def untouched_test_rows(aligned: pd.DataFrame, *, test_start: str = "2025-01-01") -> pd.DataFrame:
    dates = pd.to_datetime(aligned["target_date"], errors="raise")
    return aligned[dates >= pd.Timestamp(test_start)].copy()


def season_aware_date_block_uncertainty(
    ledger: pd.DataFrame,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Resample complete trading dates within meteorological seasons."""
    if ledger.empty:
        return {"method": "season_stratified_date_block_bootstrap", "n_resamples": 0}
    dates = pd.to_datetime(ledger["target_date"], errors="raise")
    daily = ledger.assign(_date=dates.dt.date).groupby("_date")["net_pnl"].sum().reset_index()
    month = pd.to_datetime(daily["_date"]).dt.month
    daily["_season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"], default="fall",
    )
    rng = np.random.default_rng(seed)
    totals = np.empty(n_resamples, dtype=float)
    groups = [group["net_pnl"].to_numpy(float) for _, group in daily.groupby("_season")]
    for index in range(n_resamples):
        totals[index] = sum(float(rng.choice(values, size=len(values), replace=True).sum()) for values in groups)
    return {
        "method": "season_stratified_date_block_bootstrap",
        "n_dates": int(len(daily)),
        "n_resamples": int(n_resamples),
        "net_pnl_mean": float(totals.mean()),
        "net_pnl_ci_2_5": float(np.quantile(totals, 0.025)),
        "net_pnl_ci_97_5": float(np.quantile(totals, 0.975)),
    }

def compute_ledger_metrics(ledger: pd.DataFrame, bankroll_initial: float = 1000.0) -> dict[str, Any]:
    if ledger.empty:
        return {
            "n_trades": 0, "win_rate": np.nan, "gross_pnl": 0.0, "net_pnl": 0.0,
            "return_on_capital": 0.0, "avg_return_per_trade": 0.0, "median_return_per_trade": 0.0,
            "avg_predicted_edge": np.nan, "realized_edge": np.nan,
            "max_drawdown": 0.0, "sharpe": np.nan, "sortino": np.nan, "profit_factor": np.nan,
            "pct_profitable_days": np.nan, "daily_vol": np.nan,
            "largest_win": np.nan, "largest_loss": np.nan, "capital_utilization": 0.0, "trades_per_day": 0.0,
        }
    ledger = ledger.copy()
    n = len(ledger)
    wins = (ledger["net_pnl"] > 0).sum()
    win_rate = wins / n if n else np.nan
    gross = ledger["gross_pnl"].sum()
    net = ledger["net_pnl"].sum()
    avg_ret = ledger["net_pnl"].mean()
    med_ret = ledger["net_pnl"].median()
    avg_edge = ledger["predicted_edge"].mean() if "predicted_edge" in ledger.columns else np.nan
    # Realized edge = avg(net_pnl per contract?) For BUY_YES, realized edge = settlement - entry_price - fee? Approx net_pnl / contracts
    if "contracts" in ledger.columns and (ledger["contracts"]>0).any():
        realized = (ledger["net_pnl"] / ledger["contracts"]).mean()
    else:
        realized = np.nan
    # Drawdown
    cum = ledger["net_pnl"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak)
    max_dd = dd.min()  # negative
    # Sharpe: mean/daily vol? Use per-trade sharpe annualized? Spec says document assumptions.
    # We'll compute per-trade Sharpe = mean(net_pnl)/std(net_pnl) * sqrt(252) if std>0, else nan. But better daily.
    # Compute daily pnl
    ledger["signal_date"] = pd.to_datetime(ledger["signal_timestamp"], utc=True).dt.date.astype(str)
    daily = ledger.groupby("signal_date")["net_pnl"].sum()
    daily_vol = daily.std()
    sharpe = (daily.mean() / daily_vol * np.sqrt(252)) if daily_vol and daily_vol !=0 and len(daily)>1 else np.nan
    # Sortino: downside deviation
    downside = daily[daily < 0]
    sortino = (daily.mean() / downside.std() * np.sqrt(252)) if len(downside)>1 and downside.std()!=0 else np.nan
    # Profit factor = gross wins / gross losses
    gross_wins = ledger[ledger["net_pnl"]>0]["net_pnl"].sum()
    gross_losses = -ledger[ledger["net_pnl"]<0]["net_pnl"].sum()
    profit_factor = gross_wins / gross_losses if gross_losses>0 else (np.inf if gross_wins>0 else np.nan)
    pct_profitable_days = (daily>0).mean() if len(daily)>0 else np.nan
    largest_win = ledger["net_pnl"].max()
    largest_loss = ledger["net_pnl"].min()
    capital_util = ledger["gross_cost"].sum() / bankroll_initial if bankroll_initial else 0.0
    n_days = daily.shape[0] if len(daily)>0 else 1
    trades_per_day = n / n_days

    # Annualized return prioritized per user request
    annual_return = float((net / bankroll_initial) * (365 / n_days) if bankroll_initial and n_days else np.nan)
    return {
        "n_trades": int(n),
        "win_rate": float(win_rate) if np.isfinite(win_rate) else np.nan,
        "gross_pnl": float(gross),
        "net_pnl": float(net),
        "return_on_capital": float(net / bankroll_initial) if bankroll_initial else np.nan,
        "annual_return": float(annual_return) if np.isfinite(annual_return) else np.nan,
        "avg_return_per_trade": float(avg_ret),
        "median_return_per_trade": float(med_ret),
        "avg_predicted_edge": float(avg_edge) if np.isfinite(avg_edge) else np.nan,
        "realized_edge": float(realized) if np.isfinite(realized) else np.nan,
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else (float("inf") if profit_factor==np.inf else np.nan),
        "pct_profitable_days": float(pct_profitable_days) if np.isfinite(pct_profitable_days) else np.nan,
        "daily_vol": float(daily_vol) if np.isfinite(daily_vol) else np.nan,
        "largest_win": float(largest_win),
        "largest_loss": float(largest_loss),
        "capital_utilization": float(capital_util),
        "trades_per_day": float(trades_per_day),
        "n_days": int(n_days),
    }

def breakdown_analysis(ledger: pd.DataFrame, aligned_df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """
    Generate results by model, city, bucket, prob range, edge, time-of-day, month, etc.
    Returns dict of DataFrames.
    """
    results = {}
    if ledger.empty:
        return results
    # Time-of-day analysis (America/New_York)
    try:
        import zoneinfo
        ny = zoneinfo.ZoneInfo("America/New_York")
        ledger["signal_timestamp_ny"] = pd.to_datetime(ledger["signal_timestamp"], utc=True).dt.tz_convert(ny)
        ledger["hour_ny"] = ledger["signal_timestamp_ny"].dt.hour
        # Buckets per spec: before 8, 8-10, 10-12, 12-2, 2-4, 4-6, after 6
        def tod_bucket(h):
            if h < 8: return "before 8 AM"
            elif h < 10: return "8-10 AM"
            elif h < 12: return "10 AM-12 PM"
            elif h < 14: return "12-2 PM"
            elif h < 16: return "2-4 PM"
            elif h < 18: return "4-6 PM"
            else: return "after 6 PM"
        ledger["tod_bucket"] = ledger["hour_ny"].apply(tod_bucket)
        results["by_tod"] = ledger.groupby("tod_bucket").agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum"), win_rate=("net_pnl", lambda x: (x>0).mean()), avg_edge=("predicted_edge","mean")).reset_index()
        results["by_hour"] = ledger.groupby("hour_ny").agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum"), win_rate=("net_pnl", lambda x: (x>0).mean())).reset_index()
    except Exception as e:
        results["by_tod"] = pd.DataFrame()

    # By model
    if "model_name" in ledger.columns:
        results["by_model"] = ledger.groupby("model_name").agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum"), win_rate=("net_pnl", lambda x:(x>0).mean()), avg_edge=("predicted_edge","mean")).reset_index()
    # By city
    if "city" in ledger.columns:
        results["by_city"] = ledger.groupby("city").agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum"), win_rate=("net_pnl", lambda x:(x>0).mean())).reset_index()
    # By bucket label
    if "bucket_label" in ledger.columns:
        results["by_bucket"] = ledger.groupby("bucket_label").agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum"), win_rate=("net_pnl", lambda x:(x>0).mean())).reset_index()
    # By predicted edge buckets: 0-2%,2-5%,5-10%,10-15%,15%+
    def edge_bucket(e):
        if e is None or not np.isfinite(e): return "unknown"
        e_pct = e*100
        if e_pct < 2: return "0-2%"
        elif e_pct < 5: return "2-5%"
        elif e_pct <10: return "5-10%"
        elif e_pct <15: return "10-15%"
        else: return "15%+"
    ledger["edge_bucket"] = ledger["predicted_edge"].apply(edge_bucket)
    results["by_edge"] = ledger.groupby("edge_bucket").agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum"), win_rate=("net_pnl", lambda x:(x>0).mean()), avg_pred_edge=("predicted_edge","mean"), realized_win_rate=("settlement","mean"), avg_pnl=("net_pnl","mean"), total_pnl=("net_pnl","sum")).reset_index()
    # By month / season
    ledger["target_month"] = pd.to_datetime(ledger["target_date"]).dt.month
    results["by_month"] = ledger.groupby("target_month").agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum")).reset_index()
    # Probability calibration: by model_probability deciles
    try:
        ledger["prob_bin"] = pd.cut(ledger["model_probability"], bins=np.arange(0,1.01,0.1))
        results["by_prob"] = ledger.groupby("prob_bin", observed=True).agg(n=("net_pnl","size"), win_rate=("settlement","mean"), avg_prob=("model_probability","mean")).reset_index()
    except Exception:
        pass

    # Spread size, liquidity if available in aligned join? Use ledger fields? Spread from entry?
    if "entry_ask" in ledger.columns and "entry_bid" in ledger.columns:
        ledger["spread_at_entry"] = ledger["entry_ask"] - ledger["entry_bid"]
        results["by_spread"] = ledger.groupby(pd.cut(ledger["spread_at_entry"], bins=[0,0.02,0.05,0.1,0.25,1.0])).agg(n_trades=("net_pnl","size"), net_pnl=("net_pnl","sum")).reset_index()

    return results

def calibration_analysis(aligned_df: pd.DataFrame, ledger: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    For model probabilities: multiclass bucket log loss, Brier score, calibration error, reliability curves etc.
    Compare model vs Kalshi implied (bid/ask/midpoint etc).
    """
    if aligned_df is None or aligned_df.empty:
        return {}
    # Brier score: mean (prob - settlement)^2 per bucket
    # Need settlement per aligned row (0/1)
    if "settlement" not in aligned_df.columns or "model_probability" not in aligned_df.columns:
        return {}
    df = aligned_df.dropna(subset=["model_probability","settlement"])
    if df.empty:
        return {}
    df["brier"] = (df["model_probability"] - df["settlement"])**2
    overall_brier = df["brier"].mean()
    # Log loss
    eps = 1e-15
    df["clipped_prob"] = df["model_probability"].clip(eps, 1-eps)
    df["logloss"] = -(df["settlement"]*np.log(df["clipped_prob"]) + (1-df["settlement"])*np.log(1-df["clipped_prob"]))
    overall_logloss = df["logloss"].mean()
    # Calibration error: bin prob vs realized freq
    bins = np.arange(0,1.01,0.1)
    df["prob_bin"] = pd.cut(df["model_probability"], bins=bins, include_lowest=True)
    cal_table = df.groupby("prob_bin", observed=True).agg(count=("settlement","size"), avg_prob=("model_probability","mean"), realized_freq=("settlement","mean")).reset_index()
    cal_table["cal_error"] = (cal_table["avg_prob"] - cal_table["realized_freq"]).abs()
    ece = (cal_table["cal_error"] * cal_table["count"] / cal_table["count"].sum()).sum()  # expected calibration error

    # Kalshi comparison: midpoint vs model
    results = {"brier": float(overall_brier), "logloss": float(overall_logloss), "ece": float(ece), "cal_table": cal_table}
    # If Kalshi prices exist, compute Kalshi Brier
    for col in ["midpoint", "yes_last", "yes_ask", "yes_bid"]:
        if col in df.columns and df[col].notna().any():
            sub = df.dropna(subset=[col])
            if not sub.empty:
                results[f"kalshi_brier_{col}"] = float(((sub[col] - sub["settlement"])**2).mean())
                # calibration by time of day etc. would be similar
    return results

def out_of_sample_split(ledger: pd.DataFrame, validation_start: str | None = None, test_start: str | None = None) -> dict[str, dict[str, Any]]:
    """
    Chronological split. If dates not provided, split by 60/20/20.
    """
    if ledger.empty:
        return {}
    ledger["signal_date_dt"] = pd.to_datetime(ledger["signal_timestamp"], utc=True)
    ledger_sorted = ledger.sort_values("signal_date_dt")
    n = len(ledger_sorted)
    if validation_start and test_start:
        v_dt = pd.to_datetime(validation_start, utc=True)
        t_dt = pd.to_datetime(test_start, utc=True)
        train = ledger_sorted[ledger_sorted["signal_date_dt"] < v_dt]
        val = ledger_sorted[(ledger_sorted["signal_date_dt"] >= v_dt) & (ledger_sorted["signal_date_dt"] < t_dt)]
        test = ledger_sorted[ledger_sorted["signal_date_dt"] >= t_dt]
    else:
        # 60/20/20 split
        train = ledger_sorted.iloc[:int(n*0.6)]
        val = ledger_sorted.iloc[int(n*0.6):int(n*0.8)]
        test = ledger_sorted.iloc[int(n*0.8):]
    return {
        "train": compute_ledger_metrics(train),
        "validation": compute_ledger_metrics(val),
        "test": compute_ledger_metrics(test),
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
    }

def robustness_checks(ledger: pd.DataFrame, aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether profitability disappears under alternative assumptions (spec 17).

    Checks:
     - using ask instead of midpoint (already default)
     - including fees vs gross
     - increasing required edge
     - restricting to liquid markets (volume)
     - restricting to tighter spreads
     - delaying execution by 1min, 5min (computed via engine)
     - limiting to one trade per event
     - removing highest-profit days/trades
     - spread larger than edge concentration
    """
    checks = []
    if ledger.empty:
        return pd.DataFrame()
    base_net = ledger["net_pnl"].sum()
    base_gross = ledger["gross_pnl"].sum()
    checks.append({"check": "base_net", "net_pnl": base_net, "gross_pnl": base_gross, "n_trades": len(ledger)})
    # Gross vs net (fees impact)
    checks.append({"check": "gross_only_(fees_removed)", "net_pnl": base_gross, "gross_pnl": base_gross, "n_trades": len(ledger)})
    # Remove highest profit days
    ledger["signal_date"] = pd.to_datetime(ledger["signal_timestamp"], utc=True).dt.date.astype(str)
    daily = ledger.groupby("signal_date")["net_pnl"].sum().sort_values(ascending=False)
    if len(daily) >= 2:
        top_day = daily.index[0]
        pnl_without_top_day = ledger[ledger["signal_date"] != top_day]["net_pnl"].sum()
        checks.append({"check": "remove_top_profit_day", "net_pnl": pnl_without_top_day, "gross_pnl": ledger[ledger["signal_date"] != top_day]["gross_pnl"].sum(), "n_trades": len(ledger[ledger["signal_date"] != top_day])})
        # top 3 days
        top3 = daily.index[:3]
        pnl_without_top3 = ledger[~ledger["signal_date"].isin(top3)]["net_pnl"].sum()
        checks.append({"check": "remove_top3_profit_days", "net_pnl": pnl_without_top3, "gross_pnl": 0, "n_trades": len(ledger[~ledger["signal_date"].isin(top3)])})
    # Remove highest-profit individual trades
    top_trades = ledger.sort_values("net_pnl", ascending=False).head(5)
    pnl_without_top_trades = ledger.drop(top_trades.index)["net_pnl"].sum()
    checks.append({"check": "remove_top5_trades", "net_pnl": pnl_without_top_trades, "gross_pnl": ledger.drop(top_trades.index)["gross_pnl"].sum(), "n_trades": len(ledger)-5})
    # Tighter spread restriction
    if "entry_ask" in ledger.columns and "entry_bid" in ledger.columns:
        ledger["spread"] = ledger["entry_ask"] - ledger["entry_bid"]
        tight = ledger[ledger["spread"] <= 0.10]
        checks.append({"check": "tight_spread_<=0.10", "net_pnl": tight["net_pnl"].sum(), "gross_pnl": tight["gross_pnl"].sum(), "n_trades": len(tight)})
        tight2 = ledger[ledger["spread"] <= 0.05]
        checks.append({"check": "tight_spread_<=0.05", "net_pnl": tight2["net_pnl"].sum(), "gross_pnl": tight2["gross_pnl"].sum(), "n_trades": len(tight2)})
        # Spread larger than edge
        ledger["spread_larger_than_edge"] = ledger["spread"] > ledger["predicted_edge"]
        pct = ledger["spread_larger_than_edge"].mean() if len(ledger) else np.nan
        checks.append({"check": "pct_spread_larger_than_edge", "net_pnl": float(pct), "gross_pnl": 0, "n_trades": len(ledger)})
    # One trade per event already handled, but count
    per_event = ledger.drop_duplicates(subset=["event_ticker"], keep="first")
    checks.append({"check": "one_trade_per_event_(dedup)", "net_pnl": per_event["net_pnl"].sum(), "gross_pnl": per_event["gross_pnl"].sum(), "n_trades": len(per_event)})

    # Increasing threshold simulation: would require re-running engine, but we can filter ledger by predicted_edge
    for thr in [0.05, 0.075, 0.10, 0.15]:
        sub = ledger[ledger["predicted_edge"] >= thr]
        checks.append({"check": f"edge_filter_>={thr*100:.1f}%", "net_pnl": sub["net_pnl"].sum(), "gross_pnl": sub["gross_pnl"].sum(), "n_trades": len(sub)})

    return pd.DataFrame(checks)

def data_quality_report(canonical_df: pd.DataFrame, aligned_df: pd.DataFrame, prob_df: pd.DataFrame) -> dict[str, Any]:
    """
    Automatically detect data integrity issues (spec 18).
    """
    report: dict[str, Any] = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat()}
    # Duplicate markets
    if not canonical_df.empty:
        dup = canonical_df.duplicated(subset=["market_ticker","timestamp"], keep=False).sum()
        report["duplicate_candles"] = int(dup)
        # Impossible bid > ask
        if "yes_bid" in canonical_df.columns and "yes_ask" in canonical_df.columns:
            bad = canonical_df[(canonical_df["yes_bid"].notna()) & (canonical_df["yes_ask"].notna()) & (canonical_df["yes_bid"] > canonical_df["yes_ask"])]
            report["bid_gt_ask_rows"] = int(len(bad))
        # Malformed bucket boundaries
        if "bucket_lower" in canonical_df.columns and "bucket_upper" in canonical_df.columns:
            bad_bounds = canonical_df[(canonical_df["bucket_lower"].notna()) & (canonical_df["bucket_upper"].notna()) & (canonical_df["bucket_lower"] >= canonical_df["bucket_upper"])]
            report["malformed_bucket_bounds"] = int(len(bad_bounds))
    else:
        report["duplicate_candles"] = 0
        report["bid_gt_ask_rows"] = 0

    # Probability sums
    if not prob_df.empty and "model_probability" in prob_df.columns:
        # Group by prediction_time and check sum ~=1
        if "prediction_time" in prob_df.columns:
            sums = prob_df.groupby("prediction_time")["model_probability"].sum()
            # For some files, buckets per row_id sum to 1; use row_id
            if "row_id" in prob_df.columns:
                sums = prob_df.groupby("row_id")["model_probability"].sum()
            report["prob_sum_mean"] = float(sums.mean()) if len(sums) else np.nan
            report["prob_sum_max_deviation"] = float((sums-1).abs().max()) if len(sums) else np.nan
            report["prob_sum_bad_rows"] = int(((sums-1).abs() > 0.001).sum())
        # Timezone mismatches: prediction_time after settlement? Check prediction_time vs actual settlement date?
        report["prob_missing"] = int(prob_df["model_probability"].isna().sum())
    # Leakage checks: prediction_time after target settlement?
    if not aligned_df.empty and "prediction_time" in aligned_df.columns and "timestamp" in aligned_df.columns:
        # Ensure prediction_time <= timestamp (timestamp-safe)
        bad_leak = aligned_df[pd.to_datetime(aligned_df["prediction_time"], utc=True) > pd.to_datetime(aligned_df["timestamp"], utc=True)]
        report["leakage_prediction_after_candle"] = int(len(bad_leak))
    else:
        report["leakage_prediction_after_candle"] = 0

    # Missing Kalshi prices
    if not canonical_df.empty:
        missing_bid = int(canonical_df["yes_bid"].isna().sum()) if "yes_bid" in canonical_df.columns else 0
        missing_ask = int(canonical_df["yes_ask"].isna().sum()) if "yes_ask" in canonical_df.columns else 0
        report["missing_yes_bid"] = missing_bid
        report["missing_yes_ask"] = missing_ask
    return report
