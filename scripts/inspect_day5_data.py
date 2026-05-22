from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.forecast_data import (  # noqa: E402
    load_daily_forecasts,
    load_hourly_forecasts,
    validate_daily_forecasts,
    validate_hourly_forecasts,
)
from src.weather_data import (  # noqa: E402
    load_daily_weather,
    load_hourly_weather,
    validate_daily_weather,
    validate_hourly_weather,
)


MISSING_WARNING_THRESHOLD = 25.0


DATA_PATHS = {
    "hourly actual weather": [
        Path("data/raw/nyc_hourly_weather.csv"),
        Path("data/raw/hourly_raw_nyc_openmeteo.csv"),
    ],
    "daily actual weather": [
        Path("data/raw/nyc_daily_weather.csv"),
        Path("data/raw/daily_raw_nyc_openmeteo.csv"),
    ],
    "hourly forecasts": [
        Path("data/forecasts/nyc_historical_forecasts_hourly.csv"),
        Path("data/forecasts/hourly_forecasts_nyc_openmeteo.csv"),
    ],
    "daily forecasts": [
        Path("data/forecasts/nyc_historical_forecasts_daily.csv"),
        Path("data/forecasts/daily_forecasts_nyc_openmeteo.csv"),
    ],
}


def _resolve_data_path(label: str, candidates: list[Path]) -> Path:
    for index, relative_path in enumerate(candidates):
        path = REPO_ROOT / relative_path
        if path.exists():
            if index > 0:
                print(f"Using fallback {label} file: {relative_path}")
            return path

    candidate_text = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{candidate_text}")


def _missing_percentages(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().mul(100).sort_values(ascending=False)


def _print_missing_percentages(df: pd.DataFrame) -> None:
    missing = _missing_percentages(df)

    print("\nMerged missing-value percentages:")
    for column, percent in missing.items():
        if percent > 0:
            print(f"  {column}: {percent:.2f}%")

    if (missing > 0).sum() == 0:
        print("  none")


def _validate_no_missing(df: pd.DataFrame, column: str) -> None:
    missing_count = int(df[column].isna().sum())
    if missing_count:
        raise ValueError(f"Merged daily data has {missing_count} missing values in '{column}'")


def _find_column_by_base_name(df: pd.DataFrame, base_name: str) -> str | None:
    if base_name in df.columns:
        return base_name

    matches = [
        column
        for column in df.columns
        if column.split("(", 1)[0].strip() == base_name
    ]
    if len(matches) == 1:
        return matches[0]

    return None


def _warn_if_many_missing(df: pd.DataFrame, column: str, label: str) -> None:
    matched_column = _find_column_by_base_name(df, column)
    if matched_column is None:
        return

    missing_percent = float(df[matched_column].isna().mean() * 100)
    if missing_percent >= MISSING_WARNING_THRESHOLD:
        print(
            f"WARNING: {label} is {missing_percent:.2f}% missing. "
            "Do not rely on precipitation probability as a core feature for "
            "the full 2022-2026 dataset."
        )


def main() -> None:
    print("Day 5 Data Inspection Report")
    print("=" * 28)
    # This dataset is intentionally proxy data for the prototype. It is not a
    # guaranteed point-in-time trading forecast or official Kalshi settlement feed.
    print(
        "Note: this prototype uses Open-Meteo forecast history and Open-Meteo "
        "actual weather as proxy data, not exact Kalshi trader-visible forecasts "
        "or official settlement observations."
    )

    hourly_weather_path = _resolve_data_path(
        "hourly actual weather",
        DATA_PATHS["hourly actual weather"],
    )
    daily_weather_path = _resolve_data_path(
        "daily actual weather",
        DATA_PATHS["daily actual weather"],
    )
    hourly_forecasts_path = _resolve_data_path(
        "hourly forecasts",
        DATA_PATHS["hourly forecasts"],
    )
    daily_forecasts_path = _resolve_data_path(
        "daily forecasts",
        DATA_PATHS["daily forecasts"],
    )

    hourly_weather = load_hourly_weather(hourly_weather_path)
    daily_weather = load_daily_weather(daily_weather_path)
    hourly_forecasts = load_hourly_forecasts(hourly_forecasts_path)
    daily_forecasts = load_daily_forecasts(daily_forecasts_path)

    validate_hourly_weather(hourly_weather)
    validate_daily_weather(daily_weather)
    validate_hourly_forecasts(hourly_forecasts)
    validate_daily_forecasts(daily_forecasts)

    merged = daily_forecasts.merge(
        daily_weather,
        on="date",
        how="inner",
        suffixes=("_forecast", "_actual"),
    )
    merged["error"] = merged["actual_daily_high"] - merged["forecast_high"]

    if merged["date"].duplicated().any():
        duplicate_count = int(merged["date"].duplicated().sum())
        raise ValueError(f"Merged daily data has {duplicate_count} duplicate dates")

    for column in ["forecast_high", "actual_daily_high", "error"]:
        _validate_no_missing(merged, column)

    print("\nMerged Daily Target Data")
    print(f"Merged rows: {len(merged)}")
    print(f"Merged date range: {merged['date'].min()} to {merged['date'].max()}")
    print("\nFirst 20 target rows:")
    print(
        merged[["date", "forecast_high", "actual_daily_high", "error"]]
        .head(20)
        .to_string(index=False)
    )

    print("\nError summary:")
    print(merged["error"].describe().to_string())
    _print_missing_percentages(merged)

    _warn_if_many_missing(merged, "precipitation_probability_max", "precipitation_probability_max")
    _warn_if_many_missing(hourly_forecasts, "precipitation_probability (%)", "precipitation_probability")

    print("\nSUCCESS: Day 5 data checks passed and target error was constructed.")


if __name__ == "__main__":
    main()
