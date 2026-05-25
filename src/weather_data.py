from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


CSV_METADATA_ROWS = 3
TOP_MISSING_COLUMNS = 10
HIGH_MISSING_THRESHOLD_PERCENT = 25.0


def _normalise_for_match(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.replace("Adeg", "").replace("A", "")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _base_openmeteo_name(value: str) -> str:
    return str(value).split("(", 1)[0].strip()


def _candidate_matches(column: str, candidate: str) -> bool:
    column_norm = _normalise_for_match(column)
    candidate_norm = _normalise_for_match(candidate)

    if column_norm == candidate_norm:
        return True

    candidate_base_norm = _normalise_for_match(_base_openmeteo_name(candidate))
    column_base_norm = _normalise_for_match(_base_openmeteo_name(column))
    return bool(candidate_base_norm) and column_base_norm == candidate_base_norm


def _iter_rename_targets(rename_map: dict[str, str | Iterable[str]]) -> Iterable[tuple[str, list[str]]]:
    for source_or_target, target_or_sources in rename_map.items():
        if isinstance(target_or_sources, str):
            yield target_or_sources, [source_or_target]
        else:
            yield source_or_target, list(target_or_sources)


def standardize_openmeteo_columns(
    df: pd.DataFrame,
    rename_map: dict[str, str | Iterable[str]],
) -> pd.DataFrame:
    """Rename Open-Meteo columns while tolerating unit suffix differences."""
    rename_lookup: dict[str, str] = {}

    for target, candidates in _iter_rename_targets(rename_map):
        if target in df.columns:
            continue

        for candidate in candidates:
            exact_matches = [column for column in df.columns if column == candidate]
            if exact_matches:
                rename_lookup[exact_matches[0]] = target
                break

            fuzzy_matches = [
                column
                for column in df.columns
                if column not in rename_lookup and _candidate_matches(column, candidate)
            ]
            if len(fuzzy_matches) == 1:
                rename_lookup[fuzzy_matches[0]] = target
                break

    return df.rename(columns=rename_lookup)


def _read_openmeteo_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), skiprows=CSV_METADATA_ROWS)


def _missing_percentages(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().mul(100).sort_values(ascending=False)


def _print_missing_summary(df: pd.DataFrame) -> None:
    missing = _missing_percentages(df).head(TOP_MISSING_COLUMNS)
    missing = missing[missing > 0]

    print("Top missing-value percentages:")
    if missing.empty:
        print("  none")
        return

    for column, percent in missing.items():
        print(f"  {column}: {percent:.2f}%")


def _print_profile(df: pd.DataFrame, label: str, date_column: str = "date") -> None:
    print(f"\n{label}")
    print(f"Shape: {df.shape}")

    if date_column in df.columns and not df.empty:
        print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
    elif "time" in df.columns and not df.empty:
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")

    print(f"Columns: {list(df.columns)}")
    _print_missing_summary(df)


def _validate_time_column(df: pd.DataFrame) -> None:
    if "time" not in df.columns:
        raise ValueError("Expected Open-Meteo CSV to include a 'time' column")


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        exact_matches = [
            column for column in df.columns if str(column).strip().lower() == candidate.lower()
        ]
        if exact_matches:
            return str(exact_matches[0])

        fuzzy_matches = [
            str(column) for column in df.columns if _candidate_matches(str(column), candidate)
        ]
        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0]

    return None


def identify_actual_high_column(df: pd.DataFrame) -> str | None:
    return _find_column(
        df,
        [
            "actual_high",
            "actual_daily_high",
            "temperature_2m_max",
            "daily_high",
            "high_temperature",
        ],
    )


def _standardize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_openmeteo_columns(
        df,
        {
            "weather_code": ["weather_code", "weather_code (wmo code)"],
            "temperature_2m": ["temperature_2m"],
            "temperature_2m_min": ["temperature_2m_min", "actual_daily_low"],
            "temperature_2m_mean": ["temperature_2m_mean"],
            "relative_humidity_2m": ["relative_humidity_2m"],
            "dew_point_2m": ["dew_point_2m"],
            "apparent_temperature": ["apparent_temperature"],
            "precipitation": ["precipitation"],
            "precipitation_sum": ["precipitation_sum"],
            "precipitation_hours": ["precipitation_hours"],
            "rain": ["rain"],
            "rain_sum": ["rain_sum"],
            "snowfall": ["snowfall"],
            "snowfall_sum": ["snowfall_sum"],
            "pressure_msl": ["pressure_msl"],
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
            "et0_fao_evapotranspiration": ["et0_fao_evapotranspiration"],
            "is_day": ["is_day"],
            "sunrise": ["sunrise"],
            "sunset": ["sunset"],
        },
    )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _drop_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.drop_duplicates()
    if all(key in df.columns for key in keys):
        df = df.drop_duplicates(subset=keys, keep="first")
    return df


def _existing_columns(df: pd.DataFrame, ordered_columns: Iterable[str]) -> list[str]:
    columns: list[str] = []
    for column in ordered_columns:
        if column in df.columns and column not in columns:
            columns.append(column)
    return columns


def load_hourly_weather(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["date"] = df["time"].dt.date
    return df


def load_daily_weather(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise").dt.date
    df["date"] = df["time"]
    df = standardize_openmeteo_columns(
        df,
        {
            "actual_daily_high": ["temperature_2m_max", "actual_high"],
            "actual_daily_low": ["temperature_2m_min", "actual_low"],
        },
    )
    return df


def standardize_daily_weather(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    date_column = _find_column(clean, ["date", "time", "timestamp", "datetime"])
    if date_column is None:
        raise ValueError("Could not identify a daily weather date/time column")
    if date_column != "date":
        clean = clean.rename(columns={date_column: "date"})

    clean["date"] = pd.to_datetime(clean["date"], errors="raise").dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    actual_high_column = identify_actual_high_column(clean)
    if actual_high_column is None:
        raise ValueError(
            "Could not identify actual daily high column. "
            f"Available columns: {list(clean.columns)}"
        )
    if actual_high_column != "actual_high":
        clean = clean.rename(columns={actual_high_column: "actual_high"})

    clean = _standardize_weather_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "actual_high",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "weather_code",
            "daylight_duration",
            "sunshine_duration",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ],
    )
    clean = _drop_duplicate_keys(clean, ["date", "location"])
    clean = clean.sort_values(["location", "date"]).reset_index(drop=True)

    columns = _existing_columns(
        clean,
        [
            "date",
            "location",
            "actual_high",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "weather_code",
            "temperature_2m_min",
            "temperature_2m_mean",
            "daylight_duration",
            "sunshine_duration",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
            "precipitation_hours",
            "wind_direction_10m_dominant",
            "cloud_cover_mean",
            "dew_point_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
            "surface_pressure_mean",
            "sunrise",
            "sunset",
        ],
    )
    return clean.loc[:, columns]


def standardize_hourly_weather(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    timestamp_column = _find_column(clean, ["timestamp", "time", "datetime", "date"])
    if timestamp_column is None:
        raise ValueError("Could not identify an hourly weather timestamp column")
    if timestamp_column != "timestamp":
        clean = clean.rename(columns={timestamp_column: "timestamp"})

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="raise")
    clean["date"] = clean["timestamp"].dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    clean = _standardize_weather_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "weather_code",
            "pressure_msl",
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
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "weather_code",
            "pressure_msl",
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


def _append_missing_key_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
    columns: Iterable[str],
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


def _temperature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        normalised = _normalise_for_match(column)
        if (
            "temperature" in normalised
            or "dewpoint" in normalised
            or column in {"actual_high", "actual_daily_high"}
        ):
            columns.append(str(column))
    return columns


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


def validate_weather_values(df: pd.DataFrame, granularity: str) -> list[str]:
    warnings: list[str] = []
    dataset_name = f"{granularity} weather"
    granularity_normalised = granularity.strip().lower()

    if granularity_normalised.startswith("daily"):
        _append_missing_key_warnings(warnings, df, dataset_name, ["date", "location", "actual_high"])
        _append_duplicate_key_warning(warnings, df, dataset_name, ["date", "location"])
    elif granularity_normalised.startswith("hour"):
        _append_missing_key_warnings(warnings, df, dataset_name, ["timestamp", "location"])
        _append_duplicate_key_warning(warnings, df, dataset_name, ["timestamp", "location"])
    else:
        warnings.append(f"Unknown weather granularity '{granularity}'")

    _append_range_warnings(warnings, df, dataset_name)
    _append_high_missing_warnings(warnings, df, dataset_name)
    return warnings


def validate_hourly_weather(df: pd.DataFrame) -> None:
    _validate_time_column(df)

    if df["time"].duplicated().any():
        duplicate_count = int(df["time"].duplicated().sum())
        raise ValueError(f"Hourly weather data has {duplicate_count} duplicate timestamps")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("Hourly weather timestamps must be monotonic increasing")

    _print_profile(df, "Hourly actual weather")


def validate_daily_weather(df: pd.DataFrame) -> None:
    if "date" not in df.columns:
        raise ValueError("Daily weather data must include a 'date' column")

    if df["date"].duplicated().any():
        duplicate_count = int(df["date"].duplicated().sum())
        raise ValueError(f"Daily weather data has {duplicate_count} duplicate dates")

    if "actual_daily_high" not in df.columns:
        raise ValueError("Daily weather data must include 'actual_daily_high'")

    if "actual_daily_low" in df.columns:
        comparable = df[["actual_daily_high", "actual_daily_low"]].dropna()
        bad_rows = comparable["actual_daily_high"] < comparable["actual_daily_low"]
        if bad_rows.any():
            raise ValueError("Daily weather data has actual_daily_high below actual_daily_low")

    _print_profile(df, "Daily actual weather")
