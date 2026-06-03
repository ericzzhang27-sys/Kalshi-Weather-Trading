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

    base_rows = _base_prediction_rows(weather, mapping)
    inputs: dict[str, pd.DataFrame | list[str]] = {
        "rows": base_rows,
        "hourly": weather.hourly_observations,
        "hourly_forecasts": weather.hourly_forecasts,
        "forecasts": weather.daily_forecast,
        "daily": pd.DataFrame(),
        "notes": [
            "Live Day 2 row built from Open-Meteo-compatible current/forecast data.",
            "Forecast issue/run timestamp may be missing; diagnostics decide trading eligibility.",
        ],
    }
    engineered = build_feature_matrix(inputs)
    if engineered.empty:
        raise ValueError("Live feature builder produced no rows; critical weather fields are missing")

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
) -> pd.DataFrame:
    target_ts = pd.Timestamp(weather.target_date)
    daily = weather.daily_forecast
    if daily.empty or "forecast_high" not in daily.columns:
        raise ValueError("Live weather snapshot is missing daily forecast_high")
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
