from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.trading.config import SettlementSettings


PRE_PEAK_FORECAST = "PRE_PEAK_FORECAST"
PEAK_WINDOW_CAUTION = "PEAK_WINDOW_CAUTION"
POST_PEAK_NO_TRADE = "POST_PEAK_NO_TRADE"
VERIFIED_SETTLEMENT_ONLY = "VERIFIED_SETTLEMENT_ONLY"
STALE_WEATHER_NO_TRADE = "STALE_WEATHER_NO_TRADE"

SETTLEMENT_STATE_COLUMNS = [
    "evaluated_at",
    "event_ticker",
    "target_date",
    "settlement_status",
    "probability_mode",
    "settlement_trading_allowed",
    "settlement_reason",
    "prediction_time",
    "current_temp",
    "current_temp_time",
    "observed_high",
    "observed_high_time",
    "minutes_since_observed_high",
    "current_temp_drop_from_high",
    "forecast_remaining_high",
    "forecast_remaining_high_time",
    "daily_forecast_high",
    "verified_high",
    "verified_high_time",
    "weather_no_trade_reasons",
]


@dataclass(frozen=True)
class SettlementState:
    evaluated_at: str
    event_ticker: str
    target_date: str
    settlement_status: str
    probability_mode: str
    settlement_trading_allowed: bool
    settlement_reason: str
    prediction_time: str
    current_temp: float | None = None
    current_temp_time: str = ""
    observed_high: float | None = None
    observed_high_time: str = ""
    minutes_since_observed_high: float | None = None
    current_temp_drop_from_high: float | None = None
    forecast_remaining_high: float | None = None
    forecast_remaining_high_time: str = ""
    daily_forecast_high: float | None = None
    verified_high: float | None = None
    verified_high_time: str = ""
    weather_no_trade_reasons: str = ""


def evaluate_settlement_state(
    *,
    weather: Any | None,
    feature_rows: pd.DataFrame,
    settings: SettlementSettings,
    prediction_time: datetime | pd.Timestamp | str | None = None,
    evaluated_at: datetime | pd.Timestamp | str | None = None,
    event_ticker: str | None = None,
    target_date: date | str | None = None,
) -> SettlementState:
    """
    Classify whether live model probabilities may be used for trading.

    The model still scores the weather distribution. This layer decides whether
    those probabilities are tradable, diagnostic-only, or replaced by verified
    settlement data.
    """
    frames = _weather_frames(weather)
    features = feature_rows.copy() if isinstance(feature_rows, pd.DataFrame) else pd.DataFrame()
    row = features.iloc[0] if not features.empty else pd.Series(dtype=object)
    evaluated_ts = _timestamp(evaluated_at) or pd.Timestamp.now()
    prediction_ts = (
        _timestamp(prediction_time)
        or _timestamp(row.get("prediction_time"))
        or _timestamp(getattr(weather, "prediction_time", None))
        or evaluated_ts
    )
    target_day = _target_date(
        explicit=target_date,
        feature_value=row.get("target_date"),
        weather_value=getattr(weather, "target_date", None),
    )
    event = str(event_ticker or row.get("event_ticker", "") or "")

    observations = frames["hourly_observations"]
    forecasts = frames["hourly_forecasts"]
    daily = frames["daily_forecast"]
    diagnostics = frames["weather_diagnostics"]
    no_trade_reasons = _weather_no_trade_reasons(diagnostics)

    observed_high = _optional_float(row.get("max_temp_so_far"))
    observed_high_time = _timestamp(row.get("max_temp_so_far_source_time"))
    if observed_high is None:
        observed_high, observed_high_time = _observed_high_from_weather(
            observations,
            target_day=target_day,
            prediction_time=prediction_ts,
        )
    current_temp = _optional_float(row.get("current_temp"))
    current_temp_time = _timestamp(row.get("current_temp_source_time"))
    if current_temp is None or current_temp_time is None:
        current_temp, current_temp_time = _current_temp_from_weather(
            observations,
            prediction_time=prediction_ts,
        )

    minutes_since_high = _optional_float(row.get("minutes_since_max_temp_so_far"))
    if minutes_since_high is None and observed_high_time is not None:
        minutes_since_high = max(
            0.0,
            (prediction_ts - observed_high_time).total_seconds() / 60.0,
        )
    temp_drop = None
    if current_temp is not None and observed_high is not None:
        temp_drop = float(observed_high) - float(current_temp)

    daily_forecast_high = _daily_forecast_high(daily, target_day=target_day)
    if daily_forecast_high is None:
        daily_forecast_high = _optional_float(row.get("forecast_high"))
    forecast_remaining_high, forecast_remaining_time = _forecast_remaining_high(
        forecasts,
        target_day=target_day,
        prediction_time=prediction_ts,
    )
    verified_high, verified_high_time = _verified_high_from_observations(
        observations,
        target_day=target_day,
        prediction_time=prediction_ts,
    )

    base = {
        "evaluated_at": _iso(evaluated_ts),
        "event_ticker": event,
        "target_date": "" if target_day is None else target_day.isoformat(),
        "prediction_time": _iso(prediction_ts),
        "current_temp": current_temp,
        "current_temp_time": _iso(current_temp_time),
        "observed_high": observed_high,
        "observed_high_time": _iso(observed_high_time),
        "minutes_since_observed_high": minutes_since_high,
        "current_temp_drop_from_high": temp_drop,
        "forecast_remaining_high": forecast_remaining_high,
        "forecast_remaining_high_time": _iso(forecast_remaining_time),
        "daily_forecast_high": daily_forecast_high,
        "verified_high": verified_high,
        "verified_high_time": _iso(verified_high_time),
        "weather_no_trade_reasons": ";".join(no_trade_reasons),
    }

    if features.empty:
        return _state(
            base,
            STALE_WEATHER_NO_TRADE,
            "diagnostic_no_trade",
            False,
            "missing_live_feature_rows",
        )

    fatal_weather_reasons = [
        reason
        for reason in no_trade_reasons
        if reason
        not in {
            "unverified_observed_high_window",
            "missing_forecast_issue_time",
        }
    ]
    if fatal_weather_reasons:
        return _state(
            base,
            STALE_WEATHER_NO_TRADE,
            "diagnostic_no_trade",
            False,
            ";".join(fatal_weather_reasons),
        )

    prediction_hour = int(prediction_ts.hour)
    if (
        verified_high is not None
        and prediction_hour >= settings.verified_settlement_min_hour
    ):
        return _state(
            base,
            VERIFIED_SETTLEMENT_ONLY,
            "verified_settlement",
            True,
            "nws_24h_max_temperature_available",
        )

    if observed_high is None:
        return _state(
            base,
            STALE_WEATHER_NO_TRADE,
            "diagnostic_no_trade",
            False,
            "missing_observed_high",
        )
    if current_temp is None:
        return _state(
            base,
            STALE_WEATHER_NO_TRADE,
            "diagnostic_no_trade",
            False,
            "missing_current_temp",
        )

    unverified_high = "unverified_observed_high_window" in set(no_trade_reasons)
    if settings.block_unverified_observed_high and unverified_high:
        status = (
            POST_PEAK_NO_TRADE
            if prediction_hour >= settings.typical_peak_hour or (temp_drop or 0.0) > 0.0
            else STALE_WEATHER_NO_TRADE
        )
        return _state(
            base,
            status,
            "diagnostic_no_trade",
            False,
            "unverified_observed_high_window",
        )

    if _post_peak_conditions_met(
        prediction_hour=prediction_hour,
        observed_high=observed_high,
        temp_drop=temp_drop,
        minutes_since_high=minutes_since_high,
        forecast_remaining_high=forecast_remaining_high,
        settings=settings,
    ):
        return _state(
            base,
            POST_PEAK_NO_TRADE,
            "diagnostic_no_trade",
            False,
            "post_peak_temperature_path_no_verified_settlement",
        )

    if prediction_hour >= settings.typical_peak_hour:
        return _state(
            base,
            PEAK_WINDOW_CAUTION,
            "ngboost_peak_window_caution",
            True,
            "within_or_after_typical_peak_window",
        )

    return _state(
        base,
        PRE_PEAK_FORECAST,
        "ngboost_forecast",
        True,
        "before_typical_peak_window",
    )


def settlement_state_frame(state: SettlementState) -> pd.DataFrame:
    return pd.DataFrame([asdict(state)]).reindex(columns=SETTLEMENT_STATE_COLUMNS)


def save_settlement_state(state: SettlementState | pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = state if isinstance(state, pd.DataFrame) else settlement_state_frame(state)
    frame.reindex(columns=SETTLEMENT_STATE_COLUMNS).to_csv(path, index=False)


def apply_settlement_state_to_probabilities(
    probabilities: pd.DataFrame,
    mapping: pd.DataFrame,
    state: SettlementState,
    settings: SettlementSettings,
) -> pd.DataFrame:
    if probabilities.empty:
        return probabilities
    result = _ensure_bucket_bounds(probabilities, mapping)
    result["settlement_status"] = state.settlement_status
    result["settlement_reason"] = state.settlement_reason
    result["settlement_trading_allowed"] = bool(state.settlement_trading_allowed)
    result["probability_mode"] = state.probability_mode
    result["settlement_observed_high"] = state.observed_high
    result["settlement_verified_high"] = state.verified_high

    if state.settlement_status == VERIFIED_SETTLEMENT_ONLY:
        return _apply_verified_settlement_distribution(result, state, settings)

    if not state.settlement_trading_allowed:
        result["probability_signal_status"] = "NO_TRADE"
        result["probability_signal_reason"] = _append_reason(
            result.get("probability_signal_reason"),
            state.settlement_reason or state.settlement_status,
        )
    return result


def _state(
    base: dict[str, Any],
    status: str,
    probability_mode: str,
    trading_allowed: bool,
    reason: str,
) -> SettlementState:
    return SettlementState(
        **base,
        settlement_status=status,
        probability_mode=probability_mode,
        settlement_trading_allowed=trading_allowed,
        settlement_reason=reason,
    )


def _weather_frames(weather: Any | None) -> dict[str, pd.DataFrame]:
    if weather is None:
        return {
            "hourly_observations": pd.DataFrame(),
            "hourly_forecasts": pd.DataFrame(),
            "daily_forecast": pd.DataFrame(),
            "weather_diagnostics": pd.DataFrame(),
        }
    if isinstance(weather, pd.DataFrame):
        return {
            "hourly_observations": _source_role_frame(weather, "hourly_observations"),
            "hourly_forecasts": _source_role_frame(weather, "hourly_forecasts"),
            "daily_forecast": _source_role_frame(weather, "daily_forecast"),
            "weather_diagnostics": _source_role_frame(weather, "weather_diagnostics"),
        }
    return {
        "hourly_observations": getattr(weather, "hourly_observations", pd.DataFrame()),
        "hourly_forecasts": getattr(weather, "hourly_forecasts", pd.DataFrame()),
        "daily_forecast": getattr(weather, "daily_forecast", pd.DataFrame()),
        "weather_diagnostics": getattr(weather, "diagnostics", pd.DataFrame()),
    }


def _source_role_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    if frame.empty or "source_role" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["source_role"].astype(str) == role].copy()


def _weather_no_trade_reasons(diagnostics: pd.DataFrame) -> list[str]:
    if diagnostics.empty or "status" not in diagnostics.columns:
        return []
    rows = diagnostics[diagnostics["status"].astype(str) == "NO_TRADE"].copy()
    if rows.empty:
        return []
    if "no_trade_reason" in rows.columns:
        reasons = rows["no_trade_reason"].dropna().astype(str).tolist()
    else:
        reasons = []
    if "diagnostic_name" in rows.columns:
        empty_reason = (
            rows["no_trade_reason"].fillna("").astype(str).str.strip() == ""
            if "no_trade_reason" in rows.columns
            else pd.Series(True, index=rows.index)
        )
        reasons.extend(
            rows.loc[empty_reason, "diagnostic_name"]
            .dropna()
            .astype(str)
            .tolist()
        )
    return _dedupe(reason for reason in reasons if str(reason).strip())


def _observed_high_from_weather(
    observations: pd.DataFrame,
    *,
    target_day: date | None,
    prediction_time: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    if observations.empty or "timestamp" not in observations.columns:
        return None, None
    frame = observations.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame[frame["timestamp"].notna() & (frame["timestamp"] <= prediction_time)]
    if target_day is not None and "date" in frame.columns:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce").dt.date == target_day]
    if frame.empty:
        return None, None
    if "observed_high_so_far" in frame.columns:
        frame["observed_high_so_far"] = pd.to_numeric(
            frame["observed_high_so_far"],
            errors="coerce",
        )
        rows = frame[frame["observed_high_so_far"].notna()]
        if not rows.empty:
            latest = rows.sort_values("timestamp").iloc[-1]
            source_time = _timestamp(latest.get("observed_high_so_far_source_time")) or _timestamp(
                latest.get("timestamp")
            )
            return float(latest["observed_high_so_far"]), source_time
    if "temperature_2m" not in frame.columns:
        return None, None
    frame["temperature_2m"] = pd.to_numeric(frame["temperature_2m"], errors="coerce")
    rows = frame.dropna(subset=["temperature_2m"])
    if rows.empty:
        return None, None
    max_index = rows["temperature_2m"].idxmax()
    row = rows.loc[max_index]
    return float(row["temperature_2m"]), _timestamp(row.get("timestamp"))


def _current_temp_from_weather(
    observations: pd.DataFrame,
    *,
    prediction_time: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    if observations.empty or not {"timestamp", "temperature_2m"}.issubset(observations.columns):
        return None, None
    frame = observations.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["temperature_2m"] = pd.to_numeric(frame["temperature_2m"], errors="coerce")
    frame = frame[
        frame["timestamp"].notna()
        & (frame["timestamp"] <= prediction_time)
        & frame["temperature_2m"].notna()
    ].sort_values("timestamp")
    if frame.empty:
        return None, None
    latest = frame.iloc[-1]
    return float(latest["temperature_2m"]), _timestamp(latest.get("timestamp"))


def _daily_forecast_high(daily: pd.DataFrame, *, target_day: date | None) -> float | None:
    if daily.empty or "forecast_high" not in daily.columns:
        return None
    frame = daily.copy()
    if target_day is not None and "date" in frame.columns:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce").dt.date == target_day]
    if frame.empty:
        return None
    values = pd.to_numeric(frame["forecast_high"], errors="coerce").dropna()
    return None if values.empty else float(values.iloc[-1])


def _forecast_remaining_high(
    forecasts: pd.DataFrame,
    *,
    target_day: date | None,
    prediction_time: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    if forecasts.empty or not {"timestamp", "temperature_2m"}.issubset(forecasts.columns):
        return None, None
    frame = forecasts.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["temperature_2m"] = pd.to_numeric(frame["temperature_2m"], errors="coerce")
    frame = frame[
        frame["timestamp"].notna()
        & (frame["timestamp"] >= prediction_time)
        & frame["temperature_2m"].notna()
    ]
    if target_day is not None:
        frame = frame[frame["timestamp"].dt.date == target_day]
    if frame.empty:
        return None, None
    max_index = frame["temperature_2m"].idxmax()
    row = frame.loc[max_index]
    return float(row["temperature_2m"]), _timestamp(row.get("timestamp"))


def _verified_high_from_observations(
    observations: pd.DataFrame,
    *,
    target_day: date | None,
    prediction_time: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    if observations.empty or not {"timestamp", "nws_24h_max_temp"}.issubset(observations.columns):
        return None, None
    frame = observations.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["nws_24h_max_temp"] = pd.to_numeric(frame["nws_24h_max_temp"], errors="coerce")
    frame = frame[
        frame["timestamp"].notna()
        & (frame["timestamp"] <= prediction_time)
        & frame["nws_24h_max_temp"].notna()
    ]
    if target_day is not None:
        window_start = pd.Timestamp(target_day)
        window_end = window_start + pd.Timedelta(days=1, hours=6)
        frame = frame[(frame["timestamp"] >= window_start) & (frame["timestamp"] <= window_end)]
    if frame.empty:
        return None, None
    max_index = frame["nws_24h_max_temp"].idxmax()
    row = frame.loc[max_index]
    return float(row["nws_24h_max_temp"]), _timestamp(row.get("timestamp"))


def _post_peak_conditions_met(
    *,
    prediction_hour: int,
    observed_high: float,
    temp_drop: float | None,
    minutes_since_high: float | None,
    forecast_remaining_high: float | None,
    settings: SettlementSettings,
) -> bool:
    if prediction_hour < settings.peak_window_end_hour:
        return False
    if temp_drop is None or temp_drop < settings.post_peak_temp_drop_f:
        return False
    if minutes_since_high is None or minutes_since_high < settings.min_minutes_since_high:
        return False
    if (
        forecast_remaining_high is not None
        and forecast_remaining_high > observed_high + settings.forecast_remaining_margin_f
    ):
        return False
    return True


def _apply_verified_settlement_distribution(
    probabilities: pd.DataFrame,
    state: SettlementState,
    settings: SettlementSettings,
) -> pd.DataFrame:
    verified_high = _optional_float(state.verified_high)
    if verified_high is None:
        probabilities["probability_signal_status"] = "NO_TRADE"
        probabilities["probability_signal_reason"] = _append_reason(
            probabilities.get("probability_signal_reason"),
            "missing_verified_high",
        )
        return probabilities

    result = probabilities.copy()
    if "unconstrained_probability" not in result.columns:
        result["unconstrained_probability"] = result["probability"]
    result["probability_constraint"] = "verified_settlement_state"
    result["probability"] = pd.to_numeric(result["probability"], errors="coerce").fillna(0.0)

    group_key = "row_id" if "row_id" in result.columns else None
    groups = result.groupby(group_key, sort=False) if group_key else [(None, result)]
    matched_any = False
    for _, group in groups:
        winner_index = _winning_bucket_index(group, verified_high)
        group_index = group.index
        if winner_index is None:
            continue
        matched_any = True
        tail = min(max(float(settings.settlement_tail_probability), 0.0), 0.5)
        losing_probability = tail / (len(group_index) - 1) if len(group_index) > 1 else 0.0
        result.loc[group_index, "probability"] = losing_probability
        result.loc[winner_index, "probability"] = 1.0 - tail if len(group_index) > 1 else 1.0

    if matched_any:
        result["probability_signal_status"] = "OK"
        result["probability_signal_reason"] = "verified_settlement_only"
    else:
        result["probability_signal_status"] = "NO_TRADE"
        result["probability_signal_reason"] = _append_reason(
            result.get("probability_signal_reason"),
            "verified_high_not_mapped_to_bucket",
        )
    return result


def _winning_bucket_index(group: pd.DataFrame, temperature: float) -> Any | None:
    if not {"bucket_lower_temp", "bucket_upper_temp"}.issubset(group.columns):
        return None
    for index, row in group.iterrows():
        lower = _optional_float(row.get("bucket_lower_temp"))
        upper = _optional_float(row.get("bucket_upper_temp"))
        lower_ok = lower is None or temperature > lower - 1e-9
        upper_ok = upper is None or temperature <= upper + 1e-9
        if lower_ok and upper_ok:
            return index
    return None


def _ensure_bucket_bounds(probabilities: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    result = probabilities.copy()
    if {
        "bucket_lower_temp",
        "bucket_upper_temp",
    }.issubset(result.columns) or mapping.empty or "ticker" not in result.columns:
        return result
    join_columns = [
        column
        for column in ["ticker", "bucket_lower_temp", "bucket_upper_temp"]
        if column in mapping.columns
    ]
    if len(join_columns) < 3:
        return result
    return result.merge(
        mapping[join_columns].drop_duplicates("ticker"),
        on="ticker",
        how="left",
        validate="many_to_one",
    )


def _append_reason(existing: Any, reason: str) -> Any:
    if isinstance(existing, pd.Series):
        return existing.map(lambda value: ";".join(_dedupe([value, reason])))
    return ";".join(_dedupe([existing, reason]))


def _target_date(
    *,
    explicit: date | str | None,
    feature_value: Any,
    weather_value: Any,
) -> date | None:
    for value in [explicit, feature_value, weather_value]:
        if value is None or value == "":
            continue
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return parsed.date()
    return None


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def _iso(value: Any) -> str:
    timestamp = _timestamp(value)
    return "" if timestamp is None else timestamp.isoformat()


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result
