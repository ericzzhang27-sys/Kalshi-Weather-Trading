from __future__ import annotations

import math

import numpy as np
import pandas as pd


NYC_LAT = 40.7789
NYC_LON = -73.9692


def _solar_features_for_timestamp(timestamp: pd.Timestamp) -> tuple[float, float]:
    """Approximate minutes to sunset and solar elevation for Central Park."""
    if pd.isna(timestamp):
        return float("nan"), float("nan")
    ts = pd.Timestamp(timestamp)
    day = ts.dayofyear
    minute = ts.hour * 60 + ts.minute + ts.second / 60.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (minute / 60.0 - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    latitude = math.radians(NYC_LAT)
    cos_hour_angle = (
        math.cos(math.radians(90.833)) / (math.cos(latitude) * math.cos(declination))
        - math.tan(latitude) * math.tan(declination)
    )
    cos_hour_angle = max(-1.0, min(1.0, cos_hour_angle))
    sunset_minutes = (
        720.0
        - 4.0 * NYC_LON
        - equation_of_time
        + 4.0 * math.degrees(math.acos(cos_hour_angle))
    )
    # UTC offset is encoded by local wall time; NYC is UTC-5 standard / UTC-4 DST.
    offset_hours = ts.tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward").utcoffset().total_seconds() / 3600
    sunset_local_minutes = sunset_minutes + offset_hours * 60.0
    true_solar_minutes = (minute + equation_of_time + 4.0 * NYC_LON - 60.0 * offset_hours) % 1440
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)
    sine_elevation = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sine_elevation))))
    return float(sunset_local_minutes - minute), float(elevation)


def add_hurdle_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic time/solar features without inventing unavailable data."""
    result = df.copy()
    prediction_time = pd.to_datetime(result["prediction_time"], errors="coerce")
    minute = prediction_time.dt.hour * 60 + prediction_time.dt.minute + prediction_time.dt.second / 60.0
    day = prediction_time.dt.dayofyear.astype(float)
    result["minute_of_day_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    result["minute_of_day_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    result["day_of_year_sin"] = np.sin(2.0 * np.pi * day / 366.0)
    result["day_of_year_cos"] = np.cos(2.0 * np.pi * day / 366.0)
    solar = [_solar_features_for_timestamp(ts) for ts in prediction_time]
    result["minutes_to_sunset"] = [item[0] for item in solar]
    result["solar_elevation"] = [item[1] for item in solar]
    result["minutes_until_typical_peak"] = 15.0 * 60.0 - minute
    return result


def hurdle_feature_groups() -> dict[str, list[str]]:
    return {
        "time_solar": [
            "minute_of_day_sin",
            "minute_of_day_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "minutes_to_sunset",
            "solar_elevation",
            "minutes_until_typical_peak",
        ],
        "current_state": [
            "current_temp",
            "current_max_so_far",
            "current_temp_minus_max_so_far",
            "minutes_since_max_temp_so_far",
            "hour_of_max_temp_so_far",
            "observation_age_minutes",
        ],
        "momentum": [
            "temp_change_5m",
            "temp_change_15m",
            "temp_change_30m",
            "temp_change_60m",
            "temp_slope_15m",
            "temp_slope_30m",
            "temp_slope_60m",
            "max_change_last_30m",
            "max_change_last_60m",
            "temp_change_120m",
            "temp_change_180m",
            "temp_acceleration_60m",
        ],
        "forecast": [
            "forecast_high",
            "forecast_gap",
            "forecast_high_minus_current_temp",
            "forecast_revision_1h",
            "forecast_revision_3h",
            "forecast_age_minutes",
        ],
        "atmospheric": [
            "dew_point",
            "relative_humidity",
            "cloud_cover",
            "wind_speed",
            "wind_dir_sin",
            "wind_dir_cos",
            "precipitation",
            "surface_pressure",
            "temp_minus_dew_point",
        ],
    }


def progressive_feature_sets() -> dict[str, list[str]]:
    groups = hurdle_feature_groups()
    result = {
        "A_time_only": groups["time_solar"],
        "B_time_plus_current": groups["time_solar"] + groups["current_state"],
    }
    result["C_plus_momentum"] = result["B_time_plus_current"] + groups["momentum"]
    result["D_plus_forecast"] = result["C_plus_momentum"] + groups["forecast"]
    result["E_plus_atmos"] = result["D_plus_forecast"] + groups["atmospheric"]
    return {name: list(dict.fromkeys(columns)) for name, columns in result.items()}


def select_hurdle_features(
    df: pd.DataFrame,
    level: str = "E_plus_atmos",
    available_only: bool = True,
) -> list[str]:
    feature_sets = progressive_feature_sets()
    if level not in feature_sets:
        raise ValueError(f"Unknown feature level {level!r}; choose from {sorted(feature_sets)}")
    columns = feature_sets[level]
    if available_only:
        columns = [column for column in columns if column in df.columns and not df[column].isna().all()]
    return columns
