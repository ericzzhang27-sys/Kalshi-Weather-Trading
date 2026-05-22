from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .weather_data import (
        CSV_METADATA_ROWS,
        _print_profile,
        _validate_time_column,
        standardize_openmeteo_columns,
    )
except ImportError:
    from weather_data import (
        CSV_METADATA_ROWS,
        _print_profile,
        _validate_time_column,
        standardize_openmeteo_columns,
    )


def _read_openmeteo_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), skiprows=CSV_METADATA_ROWS)


def load_hourly_forecasts(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["date"] = df["time"].dt.date
    return df


def load_daily_forecasts(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise").dt.date
    df["date"] = df["time"]
    df = standardize_openmeteo_columns(
        df,
        {
            "temperature_2m_max (°F)": "forecast_high",
            "temperature_2m_min (°F)": "forecast_low",
        },
    )
    return df


def validate_hourly_forecasts(df: pd.DataFrame) -> None:
    _validate_time_column(df)

    if df["time"].duplicated().any():
        duplicate_count = int(df["time"].duplicated().sum())
        raise ValueError(f"Hourly forecast data has {duplicate_count} duplicate timestamps")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("Hourly forecast timestamps must be monotonic increasing")

    _print_profile(df, "Hourly forecast weather")


def validate_daily_forecasts(df: pd.DataFrame) -> None:
    if "date" not in df.columns:
        raise ValueError("Daily forecast data must include a 'date' column")

    if df["date"].duplicated().any():
        duplicate_count = int(df["date"].duplicated().sum())
        raise ValueError(f"Daily forecast data has {duplicate_count} duplicate dates")

    if "forecast_high" not in df.columns:
        raise ValueError("Daily forecast data must include 'forecast_high'")

    _print_profile(df, "Daily forecast weather")
