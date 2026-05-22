from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


CSV_METADATA_ROWS = 3
TOP_MISSING_COLUMNS = 10


def _normalise_for_match(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.replace("Â", "")
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
    """Rename Open-Meteo columns while tolerating small unit suffix differences."""
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
            "temperature_2m_max (°F)": "actual_daily_high",
            "temperature_2m_min (°F)": "actual_daily_low",
        },
    )
    return df


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
