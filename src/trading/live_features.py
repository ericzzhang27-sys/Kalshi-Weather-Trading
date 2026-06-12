from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import build_feature_matrix, load_feature_list
from src.trading.contract_mapping import ContractMappingResult
from src.trading.live_weather import LiveWeatherSnapshot


LIVE_FEATURE_METADATA_COLUMNS = [
    "row_id",
    "date",
    "target_date",
    "prediction_time",
    "prediction_timestamp",
    "location",
    "event_ticker",
    "forecast_high",
    "bucket_count",
    "mapping_status",
    "weather_status",
    "live_feature_status",
    "no_trade_reason",
]


def build_live_feature_rows(
    weather: LiveWeatherSnapshot,
    mapping: ContractMappingResult,
    feature_list_path: str | Path,
) -> pd.DataFrame:
    """
    Build one model-compatible live feature row for the mapped event.
    """
    feature_columns = load_feature_list(feature_list_path)
    if not mapping.validation.valid:
        raise ValueError(f"Contract mapping is not valid: {mapping.validation.no_trade_reason}")

    hourly_observations = _weather_feature_frame(weather.hourly_observations)
    hourly_forecasts = _weather_feature_frame(weather.hourly_forecasts)
    daily_forecast = _weather_feature_frame(weather.daily_forecast)
    base_rows = _base_prediction_rows(weather, mapping, daily_forecast=daily_forecast)
    inputs: dict[str, pd.DataFrame | list[str]] = {
        "rows": base_rows,
        "hourly": hourly_observations,
        "hourly_forecasts": hourly_forecasts,
        "forecasts": daily_forecast,
        "daily": pd.DataFrame(),
        "notes": [
            "Live Day 2 row built from Open-Meteo-compatible current/forecast data.",
            "Forecast issue/run timestamp may be missing; diagnostics decide trading eligibility.",
        ],
    }
    engineered = build_feature_matrix(inputs)
    if engineered.empty:
        raise ValueError("Live feature builder produced no rows; critical weather fields are missing")
    engineered = _apply_observed_high_so_far_override(engineered, weather)

    missing_features = [column for column in feature_columns if column not in engineered.columns]
    if missing_features:
        raise ValueError(f"Live feature row missing required model features: {missing_features}")

    result = engineered.copy()
    for column in feature_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    weather_status, weather_reason = _weather_status(weather)
    result["event_ticker"] = mapping.event_ticker
    result["bucket_count"] = mapping.validation.bucket_count
    result["mapping_status"] = mapping.validation.status
    result["weather_status"] = weather_status
    result["live_feature_status"] = "NO_TRADE" if weather_status == "NO_TRADE" else "SCOREABLE_SHADOW"
    result["no_trade_reason"] = weather_reason

    ordered_columns = [
        column for column in LIVE_FEATURE_METADATA_COLUMNS if column in result.columns
    ] + feature_columns
    final_rows = result.loc[:, ordered_columns].copy()
    final_rows.attrs["freshness"] = build_live_feature_freshness(
        engineered,
        weather=weather,
        feature_columns=feature_columns,
    )
    return final_rows


def build_live_feature_freshness(
    engineered_rows: pd.DataFrame,
    *,
    weather: LiveWeatherSnapshot,
    feature_columns: list[str],
) -> pd.DataFrame:
    if engineered_rows.empty:
        return pd.DataFrame(columns=_freshness_columns())

    row = engineered_rows.iloc[0]
    prediction_time = pd.Timestamp(row["prediction_time"]).to_pydatetime()
    records: list[dict[str, Any]] = []
    for feature in feature_columns:
        source_role, source_time = _feature_source(feature, row, weather)
        age_minutes = None
        if source_time is not None and not pd.isna(source_time):
            age_minutes = (
                prediction_time - pd.Timestamp(source_time).to_pydatetime()
            ).total_seconds() / 60.0
        value = row.get(feature)
        records.append(
            {
                "row_id": row.get("row_id", ""),
                "feature": feature,
                "source_role": source_role,
                "source_time": "" if source_time is None or pd.isna(source_time) else pd.Timestamp(source_time).isoformat(),
                "prediction_time": pd.Timestamp(prediction_time).isoformat(),
                "age_minutes": age_minutes,
                "is_missing": bool(pd.isna(value)),
                "is_infinite": bool(_is_infinite(value)),
            }
        )
    return pd.DataFrame.from_records(records, columns=_freshness_columns())


def save_live_feature_outputs(
    feature_rows: pd.DataFrame,
    feature_rows_path: str | Path,
    freshness_path: str | Path,
) -> None:
    rows_path = Path(feature_rows_path)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    feature_rows.to_csv(rows_path, index=False)

    freshness = feature_rows.attrs.get("freshness")
    freshness_frame = freshness if isinstance(freshness, pd.DataFrame) else pd.DataFrame(columns=_freshness_columns())
    freshness_output = Path(freshness_path)
    freshness_output.parent.mkdir(parents=True, exist_ok=True)
    freshness_frame.to_csv(freshness_output, index=False)


def _base_prediction_rows(
    weather: LiveWeatherSnapshot,
    mapping: ContractMappingResult,
    *,
    daily_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_ts = pd.Timestamp(weather.target_date)
    daily = _weather_feature_frame(
        weather.daily_forecast if daily_forecast is None else daily_forecast
    )
    if daily.empty or "forecast_high" not in daily.columns:
        raise ValueError("Live weather snapshot is missing daily forecast_high")
    if "date" not in daily.columns and "target_date" in daily.columns:
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["target_date"], errors="coerce").dt.normalize()
    target_rows = daily[daily["date"] == target_ts]
    if target_rows.empty:
        raise ValueError(f"Live weather snapshot has no daily forecast for {weather.target_date}")
    forecast_high = pd.to_numeric(target_rows.iloc[0]["forecast_high"], errors="coerce")
    if pd.isna(forecast_high):
        raise ValueError("Live daily forecast_high is missing")

    prediction_time = pd.Timestamp(weather.prediction_time)
    return pd.DataFrame(
        [
            {
                "row_id": f"live:{mapping.event_ticker}:{prediction_time.isoformat()}",
                "date": target_ts.strftime("%Y-%m-%d"),
                "target_date": target_ts,
                "prediction_time": prediction_time,
                "prediction_timestamp": prediction_time,
                "location": weather.location,
                "forecast_high": float(forecast_high),
                "forecast_source": str(target_rows.iloc[0].get("forecast_source", "")),
            }
        ]
    )


def _weather_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "target_date" not in result.columns:
        if "date" in result.columns:
            result["target_date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
        elif "timestamp" in result.columns:
            result["target_date"] = pd.to_datetime(
                result["timestamp"],
                errors="coerce",
            ).dt.normalize()
    if "target_date" in result.columns:
        result["target_date"] = pd.to_datetime(
            result["target_date"],
            errors="coerce",
        ).dt.normalize()
    return result


def _weather_status(weather: LiveWeatherSnapshot) -> tuple[str, str]:
    if weather.diagnostics.empty:
        return "OK", ""
    no_trade = weather.diagnostics[weather.diagnostics["status"] == "NO_TRADE"]
    if no_trade.empty:
        return "OK", ""
    reasons = [
        str(reason)
        for reason in no_trade["no_trade_reason"].dropna().unique()
        if str(reason)
    ]
    return "NO_TRADE", ";".join(reasons)


def _apply_observed_high_so_far_override(
    engineered: pd.DataFrame,
    weather: LiveWeatherSnapshot,
) -> pd.DataFrame:
    observations = weather.hourly_observations
    if (
        engineered.empty
        or observations.empty
        or "observed_high_so_far" not in observations.columns
        or "prediction_time" not in engineered.columns
    ):
        return engineered

    result = engineered.copy()
    obs = observations.copy()
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], errors="coerce")
    obs["observed_high_so_far"] = pd.to_numeric(obs["observed_high_so_far"], errors="coerce")
    if "observed_high_so_far_source_time" in obs.columns:
        obs["observed_high_so_far_source_time"] = pd.to_datetime(
            obs["observed_high_so_far_source_time"],
            errors="coerce",
        )
    else:
        obs["observed_high_so_far_source_time"] = obs["timestamp"]
    obs = obs.dropna(subset=["timestamp", "observed_high_so_far"]).sort_values("timestamp")
    if obs.empty:
        return result

    for row_index, row in result.iterrows():
        prediction_time = pd.Timestamp(row["prediction_time"])
        usable = obs[obs["timestamp"] <= prediction_time]
        if usable.empty:
            continue
        latest = usable.iloc[-1]
        high = float(latest["observed_high_so_far"])
        source_time = latest["observed_high_so_far_source_time"]
        result.at[row_index, "max_temp_so_far"] = high
        result.at[row_index, "max_temp_so_far_source_time"] = source_time
        if "current_temp" in result.columns:
            result.at[row_index, "current_temp_minus_max_so_far"] = result.at[row_index, "current_temp"] - high
        if "forecast_high" in result.columns:
            result.at[row_index, "max_so_far_minus_forecast_high"] = high - result.at[row_index, "forecast_high"]
        if "forecast_max_so_far" in result.columns:
            result.at[row_index, "max_temp_error_so_far"] = high - result.at[row_index, "forecast_max_so_far"]
        if pd.notna(source_time):
            source_ts = pd.Timestamp(source_time)
            result.at[row_index, "minutes_since_max_temp_so_far"] = (
                prediction_time - source_ts
            ).total_seconds() / 60.0
            result.at[row_index, "hour_of_max_temp_so_far"] = float(source_ts.hour)
    return result


def _feature_source(
    feature: str,
    row: pd.Series,
    weather: LiveWeatherSnapshot,
) -> tuple[str, Any | None]:
    if feature in {
        "day_of_year_sin",
        "hour_sin",
        "hour_cos",
        "month",
        "season",
        "forecast_horizon_hours",
        "minutes_until_typical_peak",
    }:
        return "prediction_clock", row.get("prediction_time")
    if feature in {
        "forecast_current_temp_gap_per_hour_to_peak",
        "needed_warming_rate_minus_recent_rate",
    }:
        return "daily_forecast_and_hourly_observations", row.get("current_temp_source_time")
    if feature in {
        "current_temp",
        "dew_point",
        "cloud_cover_now",
        "wind_speed",
        "precipitation_now",
        "temp_minus_dew_point",
        "wind_dir_sin",
        "wind_dir_cos",
        "temp_change_60m",
        "temp_change_120m",
        "temp_change_180m",
        "temp_change_240m",
        "temp_change_300m",
        "temp_acceleration_60m",
        "temp_change_60m_minus_3h_avg_rate",
    }:
        return "hourly_observations", row.get("current_temp_source_time")
    if feature in {
        "max_temp_so_far",
        "current_temp_minus_max_so_far",
        "minutes_since_max_temp_so_far",
        "hour_of_max_temp_so_far",
        "max_so_far_minus_forecast_high",
        "max_temp_error_so_far",
        "num_new_highs_last_3h",
        "area_under_temp_curve_so_far",
        "near_boundary_duration_so_far",
    }:
        return "hourly_observations", row.get("max_temp_so_far_source_time")
    if feature in {
        "forecast_temp_current_hour",
        "current_temp_minus_forecast_temp",
        "mean_temp_error_so_far",
    }:
        return "hourly_forecasts", row.get("forecast_temp_source_valid_time")
    if feature in {
        "forecast_max_so_far",
        "max_so_far_minus_forecast_max_so_far",
    }:
        return "hourly_forecasts", row.get("forecast_max_so_far_source_valid_time")
    if feature == "forecast_high":
        return "daily_forecast", weather.fetched_at
    return "unknown", None


def _freshness_columns() -> list[str]:
    return [
        "row_id",
        "feature",
        "source_role",
        "source_time",
        "prediction_time",
        "age_minutes",
        "is_missing",
        "is_infinite",
    ]


def _is_infinite(value: Any) -> bool:
    try:
        return bool(np.isinf(float(value)))
    except (TypeError, ValueError):
        return False
