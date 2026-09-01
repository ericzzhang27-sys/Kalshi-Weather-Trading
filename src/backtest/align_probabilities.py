from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TIMEZONE = "America/New_York"
REQUIRED_MARKET_COLUMNS = {
    "timestamp", "target_date", "city", "market_ticker", "event_ticker",
    "bucket_lower", "bucket_upper", "result", "settlement_timestamp",
}


def load_model_probabilities(path: Path | str | None = None, prefer: str = "auto") -> pd.DataFrame:
    candidates = ([Path(path)] if path is not None else []) + [
        REPO_ROOT / "outputs/ngboost_bucket_probabilities_calibrated.csv",
        REPO_ROOT / "outputs/final_bucket_probability_predictions.csv",
        REPO_ROOT / "outputs/ngboost_bucket_probs_v0.csv",
    ]
    source = next((candidate for candidate in candidates if candidate.exists()), None)
    if source is None:
        raise FileNotFoundError(f"No model probability file found among {candidates}")
    return standardize_model_probabilities(pd.read_csv(source), source=str(source))


def standardize_model_probabilities(probabilities: pd.DataFrame, source: str = "") -> pd.DataFrame:
    result = probabilities.copy()
    aliases = {
        "probability": "model_probability", "location": "city", "date": "target_date",
        "bucket_lower_temp": "bucket_lower", "bucket_upper_temp": "bucket_upper",
        "prediction_timestamp": "prediction_time",
    }
    for old, new in aliases.items():
        if new not in result and old in result:
            result[new] = result[old]
    required = {"prediction_time", "target_date", "city", "bucket_lower", "bucket_upper", "model_probability"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"Model probabilities missing required columns: {missing}")
    raw_prediction = result["prediction_time"].astype(str).str.strip()
    clock_only = raw_prediction.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?")
    if clock_only.any():
        if "prediction_timestamp" not in result:
            raise ValueError("Clock-only prediction_time requires prediction_timestamp")
        result.loc[clock_only, "prediction_time"] = result.loc[clock_only, "prediction_timestamp"]
    prediction_time = pd.to_datetime(result["prediction_time"], errors="raise")
    if prediction_time.dt.tz is None:
        prediction_time = prediction_time.dt.tz_localize(
            LOCAL_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
        )
    result["prediction_time"] = prediction_time.dt.tz_convert("UTC")
    blocked_dst = int(result["prediction_time"].isna().sum())
    if blocked_dst:
        result = result[result["prediction_time"].notna()].copy()
    result.attrs["blocked_dst_prediction_rows"] = blocked_dst
    result["target_date"] = pd.to_datetime(result["target_date"], errors="raise").dt.date.astype(str)
    result["city"] = result["city"].astype(str).str.upper()
    result["model_probability"] = pd.to_numeric(result["model_probability"], errors="raise")
    invalid = ~np.isfinite(result["model_probability"]) | ~result["model_probability"].between(0.0, 1.0)
    if invalid.any():
        raise ValueError(f"Invalid model probabilities: {int(invalid.sum())} rows")
    if "model_name" not in result:
        result["model_name"] = Path(source).stem if source else "unknown_model"
    result["bucket_key"] = _bucket_key(result)
    keys = ["prediction_time", "target_date", "city", "bucket_key"]
    if result.duplicated(keys).any():
        raise ValueError("Model probabilities contain duplicate prediction/bucket keys")
    return result


_standardize_prob_df = standardize_model_probabilities


def standardize_canonical_markets(markets: pd.DataFrame) -> pd.DataFrame:
    result = markets.copy()
    missing = sorted(REQUIRED_MARKET_COLUMNS - set(result.columns))
    if missing:
        raise ValueError(f"Canonical markets missing required columns: {missing}")
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise", utc=True)
    result["settlement_timestamp"] = pd.to_datetime(
        result["settlement_timestamp"], errors="raise", utc=True
    )
    result["target_date"] = pd.to_datetime(result["target_date"], errors="raise").dt.date.astype(str)
    result["city"] = result["city"].astype(str).str.upper()
    result["result"] = result["result"].astype(str).str.lower()
    unresolved = ~result["result"].isin(["yes", "no"])
    if unresolved.any():
        raise ValueError(f"Markets contain unresolved Kalshi results: {int(unresolved.sum())} rows")
    if (result.groupby("market_ticker")["result"].nunique(dropna=False) != 1).any():
        raise ValueError("Kalshi result is inconsistent within a market ticker")
    if result.duplicated(["market_ticker", "timestamp"]).any():
        raise ValueError("Canonical markets contain duplicate ticker/timestamp rows")
    for canonical, fallbacks in {
        "yes_bid_open": ["yes_bid_open"],
        "yes_ask_open": ["yes_ask_open"],
        "yes_bid_close": ["yes_bid_close", "yes_bid"],
        "yes_ask_close": ["yes_ask_close", "yes_ask"],
    }.items():
        source = next((name for name in fallbacks if name in result), None)
        result[canonical] = pd.to_numeric(result[source], errors="coerce") if source else np.nan
    result["bucket_key"] = _bucket_key(result)
    return result


def align_probabilities_with_markets(
    prob_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    city: str = "NYC",
    *,
    max_probability_age_minutes: int = 90,
) -> pd.DataFrame:
    """Pure backward as-of match with exact buckets and Kalshi-only truth."""
    probs = standardize_model_probabilities(prob_df)
    markets = standardize_canonical_markets(canonical_df)
    city_key = city.upper()
    probs = probs[probs["city"] == city_key].copy()
    markets = markets[markets["city"] == city_key].copy()
    if probs.empty or markets.empty:
        return pd.DataFrame()
    local_date = markets["timestamp"].dt.tz_convert(LOCAL_TIMEZONE).dt.date.astype(str)
    markets = markets[local_date == markets["target_date"]].copy()
    output: list[pd.DataFrame] = []
    grouped_probs = {
        key: frame.sort_values("prediction_time")
        for key, frame in probs.groupby(["target_date", "city", "bucket_key"], sort=False)
    }
    for key, market_group in markets.groupby(["target_date", "city", "bucket_key"], sort=False):
        probability_group = grouped_probs.get(key)
        if probability_group is None:
            continue
        merged = pd.merge_asof(
            market_group.sort_values("timestamp"),
            probability_group.sort_values("prediction_time"),
            left_on="timestamp", right_on="prediction_time", direction="backward",
            allow_exact_matches=True, suffixes=("", "_prob"),
        )
        merged = merged[merged["model_probability"].notna()].copy()
        merged["probability_age_minutes"] = (
            merged["timestamp"] - merged["prediction_time"]
        ).dt.total_seconds() / 60.0
        merged = merged[merged["probability_age_minutes"].between(0.0, float(max_probability_age_minutes))]
        output.append(merged)
    if not output:
        return pd.DataFrame()
    joined = pd.concat(output, ignore_index=True)
    joined["signal_timestamp"] = joined["timestamp"]
    joined["quote_source"] = "kalshi_candlestick"
    joined["settlement"] = joined["result"].map({"yes": 1, "no": 0}).astype(int)
    joined["settlement_source"] = "kalshi_result"
    keep = [
        "signal_timestamp", "timestamp", "prediction_time", "probability_age_minutes",
        "settlement_timestamp", "target_date", "city", "series_ticker", "event_ticker",
        "market_ticker", "bucket_lower", "bucket_upper", "bucket_label", "bucket_key",
        "model_probability", "model_name", "quote_source", "yes_bid_open", "yes_ask_open",
        "yes_bid_close", "yes_ask_close", "result", "settlement", "settlement_source",
    ]
    for optional in ["yes_bid_size_open", "yes_ask_size_open", "volume", "open_interest"]:
        if optional in joined:
            keep.append(optional)
    for optional in [
        "forecast_high", "actual_high", "final_daily_high", "current_max_so_far",
        "mu", "sigma", "distribution_type", "df", "skew", "p_increase",
        "baseline_probability", "remaining_increase", "coverage_80_hit", "coverage_90_hit",
        "weather_cdf_below_truth", "weather_cdf_at_truth",
        "forecast_issue_time_utc", "observation_time_utc", "model_version",
    ]:
        candidate = optional if optional in joined else f"{optional}_prob"
        if candidate in joined and candidate not in keep:
            if candidate != optional:
                joined[optional] = joined[candidate]
            keep.append(optional)
    return joined[[column for column in keep if column in joined]].sort_values(
        ["signal_timestamp", "market_ticker"], kind="stable"
    ).reset_index(drop=True)


def _bucket_key(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index, dtype="string")

    def value(item: Any) -> str:
        if item is None or (isinstance(item, float) and math.isnan(item)):
            return "open"
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError("Bucket bounds must be finite or missing for open tails")
        return f"{numeric:.6f}"
    return df["bucket_lower"].map(value) + "|" + df["bucket_upper"].map(value)
