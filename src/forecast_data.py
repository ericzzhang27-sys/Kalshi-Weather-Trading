from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .weather_data import (
        CSV_METADATA_ROWS,
        HIGH_MISSING_THRESHOLD_PERCENT,
        _candidate_matches,
        _coerce_numeric_columns,
        _drop_duplicate_keys,
        _existing_columns,
        _find_column,
        _normalise_for_match,
        _print_profile,
        _validate_time_column,
        standardize_openmeteo_columns,
    )
except ImportError:
    from weather_data import (
        CSV_METADATA_ROWS,
        HIGH_MISSING_THRESHOLD_PERCENT,
        _candidate_matches,
        _coerce_numeric_columns,
        _drop_duplicate_keys,
        _existing_columns,
        _find_column,
        _normalise_for_match,
        _print_profile,
        _validate_time_column,
        standardize_openmeteo_columns,
    )


FORECAST_SOURCE = "open_meteo_historical_forecast"


def _read_openmeteo_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), skiprows=CSV_METADATA_ROWS)


def identify_forecast_high_column(df: pd.DataFrame) -> str | None:
    return _find_column(
        df,
        [
            "forecast_high",
            "temperature_2m_max",
            "daily_forecast_high",
            "forecast_daily_high",
            "high_temperature",
        ],
    )


def _standardize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_openmeteo_columns(
        df,
        {
            "weather_code": ["weather_code", "weather_code (wmo code)"],
            "temperature_2m": ["temperature_2m"],
            "temperature_2m_min": ["temperature_2m_min", "forecast_low"],
            "relative_humidity_2m": ["relative_humidity_2m"],
            "dew_point_2m": ["dew_point_2m"],
            "precipitation": ["precipitation"],
            "precipitation_sum": ["precipitation_sum"],
            "precipitation_hours": ["precipitation_hours"],
            "precipitation_probability": ["precipitation_probability"],
            "precipitation_probability_max": ["precipitation_probability_max"],
            "rain": ["rain"],
            "rain_sum": ["rain_sum"],
            "snowfall": ["snowfall"],
            "snowfall_sum": ["snowfall_sum"],
            "surface_pressure": ["surface_pressure"],
            "surface_pressure_mean": ["surface_pressure_mean"],
            "cloud_cover": ["cloud_cover"],
            "cloud_cover_mean": ["cloud_cover_mean"],
            "cloud_cover_low": ["cloud_cover_low"],
            "cloud_cover_mid": ["cloud_cover_mid"],
            "cloud_cover_high": ["cloud_cover_high"],
            "wind_speed_10m": ["wind_speed_10m"],
            "wind_speed_10m_max": ["wind_speed_10m_max"],
            "wind_speed_10m_mean": ["wind_speed_10m_mean"],
            "wind_direction_10m": ["wind_direction_10m"],
            "wind_direction_10m_dominant": ["wind_direction_10m_dominant"],
            "wind_gusts_10m": ["wind_gusts_10m"],
            "wind_gusts_10m_max": ["wind_gusts_10m_max"],
            "daylight_duration": ["daylight_duration"],
            "sunshine_duration": ["sunshine_duration"],
            "shortwave_radiation": ["shortwave_radiation"],
            "shortwave_radiation_sum": ["shortwave_radiation_sum"],
            "direct_radiation": ["direct_radiation"],
            "diffuse_radiation": ["diffuse_radiation"],
            "is_day": ["is_day"],
            "sunrise": ["sunrise"],
            "sunset": ["sunset"],
        },
    )


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
            "forecast_high": ["temperature_2m_max"],
            "forecast_low": ["temperature_2m_min"],
        },
    )
    return df


def standardize_daily_forecasts(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    date_column = _find_column(clean, ["date", "time", "timestamp", "datetime"])
    if date_column is None:
        raise ValueError("Could not identify a daily forecast target date column")
    if date_column != "date":
        clean = clean.rename(columns={date_column: "date"})

    clean["date"] = pd.to_datetime(clean["date"], errors="raise").dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    forecast_high_column = identify_forecast_high_column(clean)
    if forecast_high_column is None:
        raise ValueError(
            "Could not identify forecast daily high column. "
            f"Available columns: {list(clean.columns)}"
        )
    if forecast_high_column != "forecast_high":
        clean = clean.rename(columns={forecast_high_column: "forecast_high"})

    clean["forecast_source"] = FORECAST_SOURCE
    clean = _standardize_forecast_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "forecast_high",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_hours",
            "precipitation_probability_max",
            "rain_sum",
            "snowfall_sum",
            "wind_gusts_10m_max",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "weather_code",
            "cloud_cover_mean",
            "dew_point_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
        ],
    )
    clean = _drop_duplicate_keys(clean, ["date", "location"])
    clean = clean.sort_values(["location", "date"]).reset_index(drop=True)

    columns = _existing_columns(
        clean,
        [
            "date",
            "location",
            "forecast_high",
            "forecast_source",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_hours",
            "precipitation_probability_max",
            "rain_sum",
            "snowfall_sum",
            "wind_gusts_10m_max",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "weather_code",
            "cloud_cover_mean",
            "dew_point_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
            "daylight_duration",
            "sunshine_duration",
            "sunrise",
            "sunset",
        ],
    )
    return clean.loc[:, columns]


def standardize_hourly_forecasts(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    timestamp_column = _find_column(clean, ["timestamp", "time", "datetime", "date"])
    if timestamp_column is None:
        raise ValueError("Could not identify an hourly forecast timestamp column")
    if timestamp_column != "timestamp":
        clean = clean.rename(columns={timestamp_column: "timestamp"})

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="raise")
    clean["date"] = clean["timestamp"].dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    clean["forecast_source"] = FORECAST_SOURCE
    clean = _standardize_forecast_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "precipitation_probability",
            "rain",
            "snowfall",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "is_day",
        ],
    )
    clean = _drop_duplicate_keys(clean, ["timestamp", "location"])
    clean = clean.sort_values(["location", "timestamp"]).reset_index(drop=True)

    columns = _existing_columns(
        clean,
        [
            "timestamp",
            "date",
            "location",
            "forecast_source",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "precipitation_probability",
            "rain",
            "snowfall",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "is_day",
        ],
    )
    return clean.loc[:, columns]


def _temperature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        normalised = _normalise_for_match(column)
        if (
            "temperature" in normalised
            or "dewpoint" in normalised
            or column in {"forecast_high", "forecast_low"}
        ):
            columns.append(str(column))
    return columns


def _append_missing_key_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
    columns: list[str],
) -> None:
    for column in columns:
        if column not in df.columns:
            warnings.append(f"{dataset_name}: missing required column '{column}'")
            continue

        missing_count = int(df[column].isna().sum())
        if missing_count:
            warnings.append(f"{dataset_name}: {missing_count} missing values in '{column}'")


def _append_duplicate_key_warning(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
    keys: list[str],
) -> None:
    if not all(key in df.columns for key in keys):
        return

    duplicate_count = int(df.duplicated(subset=keys).sum())
    if duplicate_count:
        warnings.append(
            f"{dataset_name}: {duplicate_count} duplicate rows by {', '.join(keys)}"
        )


def _append_range_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    for column in _temperature_columns(df):
        values = pd.to_numeric(df[column], errors="coerce")
        bad_count = int(((values < -100) | (values > 140)).sum())
        if bad_count:
            warnings.append(
                f"{dataset_name}: {bad_count} impossible Fahrenheit temperature values in '{column}'"
            )

    for column in df.columns:
        normalised = _normalise_for_match(column)
        values = pd.to_numeric(df[column], errors="coerce")
        if (
            ("precipitation" in normalised or "rain" in normalised or "snowfall" in normalised)
            and int((values < 0).sum())
        ):
            warnings.append(f"{dataset_name}: negative precipitation/rain/snowfall in '{column}'")
        if (
            ("windspeed" in normalised or "windgusts" in normalised)
            and int((values < 0).sum())
        ):
            warnings.append(f"{dataset_name}: negative wind speed/gust values in '{column}'")


def _append_high_missing_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    if df.empty:
        warnings.append(f"{dataset_name}: dataframe is empty")
        return

    missing_percentages = df.isna().mean().mul(100)
    for column, missing_percent in missing_percentages.items():
        if missing_percent >= HIGH_MISSING_THRESHOLD_PERCENT and missing_percent > 0:
            missing_count = int(df[column].isna().sum())
            warnings.append(
                f"{dataset_name}: '{column}' is {missing_percent:.2f}% missing "
                f"({missing_count} rows); do not silently treat this as complete"
            )


def _append_forecast_source_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    if "forecast_source" not in df.columns:
        warnings.append(f"{dataset_name}: missing forecast_source")
    else:
        missing_source = int(df["forecast_source"].isna().sum())
        if missing_source:
            warnings.append(f"{dataset_name}: {missing_source} rows missing forecast_source")

        unique_sources = sorted(str(value) for value in df["forecast_source"].dropna().unique())
        if len(unique_sources) != 1:
            warnings.append(
                f"{dataset_name}: ambiguous forecast_source values {unique_sources}"
            )

    as_of_candidates = [
        "forecast_created_at",
        "forecast_issue_time",
        "forecast_reference_time",
        "model_run_time",
        "run_timestamp",
        "as_of",
        "issued_at",
    ]
    has_as_of_column = any(
        any(_candidate_matches(str(column), candidate) for candidate in as_of_candidates)
        for column in df.columns
    )
    if not has_as_of_column:
        warnings.append(
            f"{dataset_name}: forecast creation/model-run timestamp is missing; "
            "true as-of-time forecast availability is not fully verifiable"
        )


def validate_forecast_values(df: pd.DataFrame, granularity: str) -> list[str]:
    warnings: list[str] = []
    dataset_name = f"{granularity} forecast"
    granularity_normalised = granularity.strip().lower()

    if granularity_normalised.startswith("daily"):
        _append_missing_key_warnings(
            warnings,
            df,
            dataset_name,
            ["date", "location", "forecast_high"],
        )
        _append_duplicate_key_warning(warnings, df, dataset_name, ["date", "location"])
    elif granularity_normalised.startswith("hour"):
        _append_missing_key_warnings(warnings, df, dataset_name, ["timestamp", "location"])
        _append_duplicate_key_warning(warnings, df, dataset_name, ["timestamp", "location"])
    else:
        warnings.append(f"Unknown forecast granularity '{granularity}'")

    _append_range_warnings(warnings, df, dataset_name)
    _append_high_missing_warnings(warnings, df, dataset_name)
    _append_forecast_source_warnings(warnings, df, dataset_name)
    return warnings


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
