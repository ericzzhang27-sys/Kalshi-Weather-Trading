from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve  # noqa: F401
from sklearn.metrics import brier_score_loss, log_loss


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def logloss_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Create reliability bins per spec §13.
    Default bins: 0-1%,1-2.5%,2.5-5%,5-10%,10-20%,20-40%,40-60%,60-80%,80-100%
    """
    if bins is None:
        bins = [
            (0.0, 0.01),
            (0.01, 0.025),
            (0.025, 0.05),
            (0.05, 0.10),
            (0.10, 0.20),
            (0.20, 0.40),
            (0.40, 0.60),
            (0.60, 0.80),
            (0.80, 1.0),
        ]
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []
    for lo, hi in bins:
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            mean_pred = float(y_prob[mask].mean())
            emp = float(y_true[mask].mean())
            brier = float(np.mean((y_prob[mask] - y_true[mask]) ** 2))
        else:
            mean_pred = float("nan")
            emp = float("nan")
            brier = float("nan")
        rows.append({
            "bin_lower": lo,
            "bin_upper": hi,
            "bin_label": f"{lo*100:.1f}-{hi*100:.1f}%",
            "count": cnt,
            "mean_predicted": mean_pred,
            "empirical_frequency": emp,
            "brier": brier,
            "calibration_gap": mean_pred - emp if cnt else float("nan"),
        })
    return pd.DataFrame(rows)


def reliability_by_group(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    group_col: str,
    bins: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    out = []
    for grp, sub in df.groupby(group_col, observed=False):
        tbl = reliability_bins(sub[y_true_col].values, sub[y_prob_col].values, bins=bins)
        tbl["group"] = str(grp)
        tbl["group_col"] = group_col
        out.append(tbl)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def calibration_by_time_of_day(df: pd.DataFrame, y_true_col: str, y_prob_col: str) -> pd.DataFrame:
    """
    Spec §14: test before 12, 12-14,14-16,16-17,17-18,18-19,after 19
    """
    pred_time = pd.to_datetime(df["prediction_time"])
    hour = pred_time.dt.hour + pred_time.dt.minute / 60.0
    def bucket(h):
        if h < 12: return "before 12 PM"
        if h < 14: return "12–2 PM"
        if h < 16: return "2–4 PM"
        if h < 17: return "4–5 PM"
        if h < 18: return "5–6 PM"
        if h < 19: return "6–7 PM"
        return "after 7 PM"
    groups = hour.apply(bucket)
    tmp = df.copy()
    tmp["_tod_bucket"] = groups
    # For each bucket, compute reliability for 0-10% region etc?
    # Return per bucket brier/logloss and overall calibration
    rows = []
    for g, sub in tmp.groupby("_tod_bucket"):
        y_true = sub[y_true_col].values
        y_prob = sub[y_prob_col].values
        tbl = reliability_bins(y_true, y_prob)
        # low prob tail calibration
        low_mask = y_prob < 0.05
        low_actual = float(y_true[low_mask].mean()) if low_mask.sum() else float("nan")
        rows.append({
            "time_bucket": g,
            "count": len(sub),
            "brier": brier_score(y_true, y_prob),
            "log_loss": logloss_score(y_true, y_prob),
            "empirical_rate": float(y_true.mean()),
            "mean_pred": float(y_prob.mean()),
            "low_prob_rate_actual_lt5": low_actual,
            "low_prob_count_lt5": int(low_mask.sum()),
        })
    return pd.DataFrame(rows)


def test_by_temperature_state(df: pd.DataFrame, y_true_col: str, y_prob_col: str) -> dict[str, pd.DataFrame]:
    """
    Spec §15: distance below current max, minutes since max, trajectory
    """
    out: dict[str, pd.DataFrame] = {}
    # distance below current max = current_max - current_temp = - (current_temp_minus_max)
    if "current_temp_minus_max_so_far" in df.columns:
        gap = -df["current_temp_minus_max_so_far"]  # positive when below
        # bins: 0, 0-0.5,0.5-1,1-2,>2
        def gap_bucket(v):
            if v <= 0: return "at max"
            if v <= 0.5: return "0–0.5°F below"
            if v <= 1.0: return "0.5–1°F below"
            if v <= 2.0: return "1–2°F below"
            return ">2°F below"
        cats = gap.apply(gap_bucket)
        tmp = df.copy()
        tmp["_gap_bucket"] = cats
        rows = []
        for g, sub in tmp.groupby("_gap_bucket"):
            rows.append({
                "gap_bucket": g,
                "count": len(sub),
                "brier": brier_score(sub[y_true_col].values, sub[y_prob_col].values),
                "log_loss": logloss_score(sub[y_true_col].values, sub[y_prob_col].values),
                "emp_rate": float(sub[y_true_col].mean()),
                "mean_pred": float(sub[y_prob_col].mean()),
            })
        out["by_gap"] = pd.DataFrame(rows)
    else:
        out["by_gap"] = pd.DataFrame()

    if "minutes_since_max_temp_so_far" in df.columns:
        mins = df["minutes_since_max_temp_so_far"]
        def min_bucket(v):
            if pd.isna(v): return "missing"
            if v <= 10: return "0–10 min"
            if v <= 30: return "10–30 min"
            if v <= 60: return "30–60 min"
            if v <= 120: return "60–120 min"
            return ">120 min"
        cats2 = mins.apply(min_bucket)
        tmp = df.copy()
        tmp["_min_bucket"] = cats2
        rows = []
        for g, sub in tmp.groupby("_min_bucket"):
            rows.append({
                "minutes_bucket": g,
                "count": len(sub),
                "brier": brier_score(sub[y_true_col].values, sub[y_prob_col].values),
                "log_loss": logloss_score(sub[y_true_col].values, sub[y_prob_col].values),
                "emp_rate": float(sub[y_true_col].mean()),
                "mean_pred": float(sub[y_prob_col].mean()),
            })
        out["by_minutes"] = pd.DataFrame(rows)
    else:
        out["by_minutes"] = pd.DataFrame()

    # trajectory
    if "temp_change_60m" in df.columns:
        ch = df["temp_change_60m"]
        def traj_bucket(v):
            if pd.isna(v): return "missing"
            if v > 2: return "strongly rising"
            if v > 0.5: return "rising"
            if v > -0.5: return "flat"
            if v > -2: return "falling"
            return "strongly falling"
        cats3 = ch.apply(traj_bucket)
        tmp = df.copy()
        tmp["_traj_bucket"] = cats3
        rows = []
        for g, sub in tmp.groupby("_traj_bucket"):
            rows.append({
                "trajectory": g,
                "count": len(sub),
                "brier": brier_score(sub[y_true_col].values, sub[y_prob_col].values),
                "log_loss": logloss_score(sub[y_true_col].values, sub[y_prob_col].values),
                "emp_rate": float(sub[y_true_col].mean()),
                "mean_pred": float(sub[y_prob_col].mean()),
            })
        out["by_trajectory"] = pd.DataFrame(rows)
    else:
        out["by_trajectory"] = pd.DataFrame()
    return out


def forecast_gap_testing(df: pd.DataFrame, y_true_col: str, y_prob_col: str) -> pd.DataFrame:
    """
    Spec §16: bucket by forecast_gap
    """
    if "forecast_gap" not in df.columns and "forecast_high_minus_current_max" in df.columns:
        gap = df["forecast_high_minus_current_max"]
    else:
        gap = df.get("forecast_gap", pd.Series([np.nan]*len(df)))
    def bucket(v):
        if pd.isna(v): return "missing"
        if v <= -1: return "<= -1°F"
        if v <= 0: return "-1 to 0°F"
        if v <= 1: return "0 to +1°F"
        if v <= 2: return "+1 to +2°F"
        return "> +2°F"
    cats = gap.apply(bucket)
    tmp = df.copy()
    tmp["_f_gap_bucket"] = cats
    rows = []
    for g, sub in tmp.groupby("_f_gap_bucket"):
        rows.append({
            "forecast_gap_bucket": g,
            "count": len(sub),
            "brier": brier_score(sub[y_true_col].values, sub[y_prob_col].values),
            "log_loss": logloss_score(sub[y_true_col].values, sub[y_prob_col].values),
            "emp_rate": float(sub[y_true_col].mean()),
            "mean_pred": float(sub[y_prob_col].mean()),
        })
    return pd.DataFrame(rows)


def tail_failure_report(df: pd.DataFrame, y_true_col: str, y_prob_col: str, threshold: float = 0.05, top_n: int = 50) -> pd.DataFrame:
    """
    Spec §17: rows where predicted < threshold but actual==1
    """
    mask = (df[y_prob_col] < threshold) & (df[y_true_col] == 1)
    failures = df[mask].copy()
    # sort by most confident failures (lowest prob first)
    failures = failures.sort_values(y_prob_col).head(top_n)
    # select display columns
    cols = []
    for c in [
        "target_date",
        "prediction_time",
        "current_temp",
        "current_max_so_far",
        "final_daily_high",
        "forecast_high",
        "minutes_since_max_temp_so_far",
        "current_temp_minus_max_so_far",
        "temp_change_5m",
        "temp_change_15m",
        "temp_change_30m",
        "temp_change_60m",
        "temp_change_120m",
        "temp_change_180m",
        "relative_humidity",
        "cloud_cover",
        "wind_speed",
        "dew_point",
        "precipitation",
        "surface_pressure",
        "forecast_gap",
        y_prob_col,
        y_true_col,
    ]:
        if c in failures.columns:
            cols.append(c)
    # add temperature history not stored; could add temp_change_60m etc.
    if cols:
        failures = failures[cols]
    failures["threshold"] = threshold
    return failures


def low_prob_calibration_detail(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Pay extra attention to low-prob tail.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    # P(Y=1 | p <0.05)
    for thr in [0.05, 0.10]:
        mask = y_prob < thr
        cnt = int(mask.sum())
        if cnt:
            emp = float(y_true[mask].mean())
            mean_pred = float(y_prob[mask].mean())
        else:
            emp = float("nan")
            mean_pred = float("nan")
        yield thr, cnt, emp, mean_pred


def calibration_error_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    # ECE and MCE
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1 else (y_prob >= lo) & (y_prob <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        gap = abs(acc - conf)
        ece += (cnt / total) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}
