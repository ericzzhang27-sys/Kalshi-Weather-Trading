from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import (  # noqa: E402
    create_data_inventory,
    infer_datetime_column,
    load_csv,
    read_openmeteo_metadata,
    write_verification_report,
)
from src.forecast_data import (  # noqa: E402
    identify_forecast_high_column,
    standardize_daily_forecasts,
    standardize_hourly_forecasts,
    validate_forecast_values,
)
from src.weather_data import (  # noqa: E402
    identify_actual_high_column,
    standardize_daily_weather,
    standardize_hourly_weather,
    validate_weather_values,
)


OUTPUTS_DIR = REPO_ROOT / "outputs" / "data_audit"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DATA_PATH_CANDIDATES = {
    "daily_actual": [
        REPO_ROOT / "data" / "raw" / "daily_raw_nyc_openmeteo.csv",
    ],
    "hourly_actual": [
        REPO_ROOT / "data" / "raw" / "hourly_raw_nyc_openmeteo.csv",
    ],
    "daily_forecast": [
        REPO_ROOT / "data" / "raw" / "daily_forecasts_nyc_openmeteo.csv",
        REPO_ROOT / "data" / "forecasts" / "daily_forecasts_nyc_openmeteo.csv",
    ],
    "hourly_forecast": [
        REPO_ROOT / "data" / "raw" / "hourly_forecasts_nyc_openmeteo.csv",
        REPO_ROOT / "data" / "forecasts" / "hourly_forecasts_nyc_openmeteo.csv",
    ],
}


def _resolve_data_path(label: str, candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked}")


def _date_range(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or df.empty:
        return "not available"
    values = pd.to_datetime(df[column], errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min().date()} to {values.max().date()}"


def _timestamp_range(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or df.empty:
        return "not available"
    values = pd.to_datetime(df[column], errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min()} to {values.max()}"


def _describe_series(series: pd.Series) -> dict[str, float | int]:
    description = series.describe(percentiles=[0.25, 0.5, 0.75])
    ordered_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    result: dict[str, float | int] = {}
    for key in ordered_keys:
        value = description.get(key)
        if pd.isna(value):
            result[key] = float("nan")
        elif key == "count":
            result[key] = int(value)
        else:
            result[key] = float(value)
    return result


def _missing_sections(named_frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, df in named_frames.items():
        if df.empty:
            continue
        missing_counts = df.isna().sum()
        missing_percents = df.isna().mean().mul(100)
        for column, missing_count in missing_counts.items():
            if int(missing_count) == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "missing_count": int(missing_count),
                    "missing_percent": f"{missing_percents[column]:.2f}",
                }
            )
    return rows


def _duplicate_sections(named_keys: dict[str, tuple[pd.DataFrame, list[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, (df, keys) in named_keys.items():
        if not all(key in df.columns for key in keys):
            rows.append(
                {
                    "dataset": dataset,
                    "duplicate_key": ", ".join(keys),
                    "duplicate_count": "not checked; key missing",
                }
            )
            continue

        duplicate_count = int(df.duplicated(subset=keys).sum())
        rows.append(
            {
                "dataset": dataset,
                "duplicate_key": ", ".join(keys),
                "duplicate_count": duplicate_count,
            }
        )
    return rows


def _unit_from_column(column: str) -> str | None:
    if "(" not in column or ")" not in column:
        return None
    return column.split("(", 1)[1].split(")", 1)[0].strip()


def _columns_matching(columns: list[str], tokens: list[str]) -> list[str]:
    matches = []
    for column in columns:
        column_lower = column.lower()
        if any(token in column_lower for token in tokens):
            matches.append(column)
    return matches


def _unit_summary(raw_frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    all_columns: list[str] = []
    for frame in raw_frames.values():
        all_columns.extend(str(column) for column in frame.columns)

    temperature_columns = _columns_matching(all_columns, ["temperature", "dew_point"])
    wind_columns = _columns_matching(all_columns, ["wind_speed", "wind_gusts"])
    precipitation_columns = _columns_matching(all_columns, ["precipitation", "rain", "snowfall"])

    def unit_text(columns: list[str]) -> str:
        if not columns:
            return "No matching columns found."
        units = sorted({unit for column in columns if (unit := _unit_from_column(column))})
        unique_columns = sorted(set(columns))
        if units:
            return f"Units visible in column names: {', '.join(units)}. Columns: {', '.join(unique_columns)}"
        return (
            "No unit metadata was visible in the matching column names. "
            f"Columns: {', '.join(unique_columns)}"
        )

    return {
        "temperature": unit_text(temperature_columns),
        "wind": unit_text(wind_columns),
        "precipitation": unit_text(precipitation_columns),
        "conversion_policy": "No unit conversion was applied during Day 6 cleaning.",
    }


def _looks_celsius_like(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return False
    mostly_celsius_range = float(values.between(-20, 45).mean()) >= 0.9
    return mostly_celsius_range and float(values.max()) <= 60


def _append_alignment_warnings(
    warnings: list[str],
    merged: pd.DataFrame,
    daily_actual: pd.DataFrame,
    daily_forecast: pd.DataFrame,
) -> None:
    minimum_input_rows = min(len(daily_actual), len(daily_forecast))
    if minimum_input_rows and len(merged) < minimum_input_rows * 0.9:
        warnings.append(
            "Merged daily rows are much smaller than one or both inputs; "
            "inspect date/location coverage before modeling"
        )

    if not merged.empty:
        extreme_errors = int((merged["forecast_error"].abs() > 40).sum())
        if extreme_errors:
            warnings.append(
                f"{extreme_errors} merged rows have abs(forecast_error) > 40 F"
            )

    if _looks_celsius_like(daily_actual["actual_high"]):
        warnings.append("actual_high values look Celsius-like while Fahrenheit is expected")
    if _looks_celsius_like(daily_forecast["forecast_high"]):
        warnings.append("forecast_high values look Celsius-like while Fahrenheit is expected")


def _metadata_summary(paths: dict[str, Path], location_filled: bool) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for label, path in paths.items():
        file_metadata = read_openmeteo_metadata(path)
        if file_metadata:
            metadata[label] = ", ".join(
                f"{key}={value}" for key, value in file_metadata.items()
            )

    if location_filled:
        metadata["location_column"] = (
            "No row-level location/station column was present in the CSV data; "
            "the cleaning step filled location='NYC'."
        )
    return metadata


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        label: _resolve_data_path(label, candidates)
        for label, candidates in DATA_PATH_CANDIDATES.items()
    }

    raw_frames = {label: load_csv(path) for label, path in paths.items()}
    inventory = create_data_inventory(
        list(paths.values()),
        OUTPUTS_DIR / "data_inventory.csv",
    )

    actual_high_column = identify_actual_high_column(raw_frames["daily_actual"])
    forecast_high_column = identify_forecast_high_column(raw_frames["daily_forecast"])
    if actual_high_column is None:
        raise ValueError("Could not identify the actual high column in daily actual data")
    if forecast_high_column is None:
        raise ValueError("Could not identify the forecast high column in daily forecast data")

    daily_clean = standardize_daily_weather(raw_frames["daily_actual"], location="NYC")
    hourly_clean = standardize_hourly_weather(raw_frames["hourly_actual"], location="NYC")
    forecasts_clean = standardize_daily_forecasts(raw_frames["daily_forecast"], location="NYC")
    hourly_forecasts_clean = standardize_hourly_forecasts(
        raw_frames["hourly_forecast"],
        location="NYC",
    )

    daily_output = PROCESSED_DIR / "daily_clean.csv"
    hourly_output = PROCESSED_DIR / "hourly_clean.csv"
    forecasts_output = PROCESSED_DIR / "forecasts_clean.csv"
    hourly_forecasts_output = PROCESSED_DIR / "hourly_forecasts_clean.csv"
    preview_output = PROCESSED_DIR / "modeling_base_preview.csv"

    daily_clean.to_csv(daily_output, index=False)
    hourly_clean.to_csv(hourly_output, index=False)
    forecasts_clean.to_csv(forecasts_output, index=False)
    hourly_forecasts_clean.to_csv(hourly_forecasts_output, index=False)

    warnings: list[str] = []
    warnings.extend(validate_weather_values(daily_clean, "daily"))
    warnings.extend(validate_weather_values(hourly_clean, "hourly"))
    warnings.extend(validate_forecast_values(forecasts_clean, "daily"))
    warnings.extend(validate_forecast_values(hourly_forecasts_clean, "hourly"))

    merged = daily_clean.merge(
        forecasts_clean,
        on=["date", "location"],
        how="inner",
        suffixes=("_actual", "_forecast"),
    )
    merged["forecast_error"] = merged["actual_high"] - merged["forecast_high"]
    modeling_preview_columns = [
        "date",
        "location",
        "actual_high",
        "forecast_high",
        "forecast_error",
        "forecast_source",
    ]
    merged.loc[:, modeling_preview_columns].to_csv(preview_output, index=False)

    _append_alignment_warnings(warnings, merged, daily_clean, forecasts_clean)
    warnings.append(
        "Forecast data is an Open-Meteo historical forecast proxy, not confirmed official NWS archived forecast data"
    )
    if "location" not in raw_frames["daily_actual"].columns or "location" not in raw_frames["daily_forecast"].columns:
        warnings.append("Location was filled as 'NYC' because row-level location/station information is missing")

    known_risks = [
        "Forecast rows do not include an as-of/model-run timestamp, so point-in-time forecast availability cannot be fully verified.",
        "Open-Meteo forecast history is not proven to match official NWS forecasts or Kalshi trader-visible forecasts.",
        "Actual and forecast files include coordinate metadata, but no official station identifier.",
        "High-missingness forecast precipitation-probability columns should not be used as complete features without a deliberate missing-data plan.",
    ]

    cleaned_outputs = [
        str(OUTPUTS_DIR / "data_inventory.csv"),
        str(OUTPUTS_DIR / "data_verification_report.md"),
        str(hourly_output),
        str(daily_output),
        str(forecasts_output),
        str(hourly_forecasts_output),
        str(preview_output),
    ]

    alignment = {
        "actual_daily_date_range": _date_range(daily_clean, "date"),
        "forecast_daily_date_range": _date_range(forecasts_clean, "date"),
        "actual_hourly_timestamp_range": _timestamp_range(hourly_clean, "timestamp"),
        "forecast_hourly_timestamp_range": _timestamp_range(hourly_forecasts_clean, "timestamp"),
        "daily_actual_rows": len(daily_clean),
        "daily_forecast_rows": len(forecasts_clean),
        "overlap_rows": len(merged),
        "merged_preview_rows": len(merged),
    }

    summary = {
        "key_columns": {
            "actual_daily_date_column": infer_datetime_column(raw_frames["daily_actual"]),
            "actual_high_column_used": actual_high_column,
            "forecast_daily_date_column": infer_datetime_column(raw_frames["daily_forecast"]),
            "forecast_high_column_used": forecast_high_column,
            "hourly_actual_time_column": infer_datetime_column(raw_frames["hourly_actual"]),
            "hourly_forecast_time_column": infer_datetime_column(raw_frames["hourly_forecast"]),
            "forecast_source": "open_meteo_historical_forecast",
        },
        "unit_summary": _unit_summary(raw_frames),
        "missing_sections": _missing_sections(
            {
                "daily_clean": daily_clean,
                "hourly_clean": hourly_clean,
                "forecasts_clean": forecasts_clean,
                "hourly_forecasts_clean": hourly_forecasts_clean,
            }
        ),
        "duplicates": _duplicate_sections(
            {
                "daily_clean": (daily_clean, ["date", "location"]),
                "hourly_clean": (hourly_clean, ["timestamp", "location"]),
                "forecasts_clean": (forecasts_clean, ["date", "location"]),
                "hourly_forecasts_clean": (
                    hourly_forecasts_clean,
                    ["timestamp", "location"],
                ),
            }
        ),
        "alignment": alignment,
        "forecast_error_summary": _describe_series(merged["forecast_error"]),
        "known_risks": known_risks,
        "cleaned_outputs": cleaned_outputs,
        "metadata": _metadata_summary(paths, location_filled=True),
    }

    write_verification_report(
        OUTPUTS_DIR / "data_verification_report.md",
        inventory,
        warnings,
        summary,
    )

    print("Day 6 data verification complete.")
    print(f"Actual daily date range: {alignment['actual_daily_date_range']}")
    print(f"Forecast daily date range: {alignment['forecast_daily_date_range']}")
    print(f"Daily actual rows: {alignment['daily_actual_rows']}")
    print(f"Daily forecast rows: {alignment['daily_forecast_rows']}")
    print(f"Merged rows: {alignment['merged_preview_rows']}")
    print(f"Actual high column used: {actual_high_column}")
    print(f"Forecast high column used: {forecast_high_column}")
    print(f"Report: {OUTPUTS_DIR / 'data_verification_report.md'}")


if __name__ == "__main__":
    main()
