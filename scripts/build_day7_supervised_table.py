from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.supervised_table import (  # noqa: E402
    PREDICTION_TIMES,
    expand_targets_to_prediction_times,
    validate_supervised_rows,
)
from src.target_builder import (  # noqa: E402
    build_daily_forecast_error_targets,
    validate_daily_targets,
)


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"

DAILY_ACTUAL_INPUT = PROCESSED_DIR / "daily_clean.csv"
DAILY_FORECAST_INPUT = PROCESSED_DIR / "forecasts_clean.csv"
INPUT_PROFILE_PATHS = [
    DAILY_ACTUAL_INPUT,
    DAILY_FORECAST_INPUT,
    PROCESSED_DIR / "hourly_clean.csv",
    PROCESSED_DIR / "hourly_forecasts_clean.csv",
    PROCESSED_DIR / "modeling_base_preview.csv",
]

DAILY_TARGET_OUTPUT = PROCESSED_DIR / "daily_forecast_error_targets.csv"
SUPERVISED_OUTPUT = PROCESSED_DIR / "supervised_forecast_error_rows.csv"
TARGET_SUMMARY_OUTPUT = OUTPUTS_DIR / "target_summary.csv"

SUMMARY_COLUMNS = [
    "location",
    "prediction_time",
    "n_rows",
    "mean_error",
    "median_error",
    "std_error",
    "min_error",
    "p05_error",
    "p25_error",
    "p75_error",
    "p95_error",
    "max_error",
    "missing_actual_high",
    "missing_forecast_high",
    "missing_forecast_error",
]
FLOAT_SUMMARY_COLUMNS = [
    "mean_error",
    "median_error",
    "std_error",
    "min_error",
    "p05_error",
    "p25_error",
    "p75_error",
    "p95_error",
    "max_error",
]


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Day 7 input file is missing: {path}")
    return pd.read_csv(path)


def _date_range_text(df: pd.DataFrame, column: str = "date") -> str:
    if column not in df.columns or df.empty:
        return "not available"
    values = pd.to_datetime(df[column], errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min().date()} to {values.max().date()}"


def _timestamp_range_text(df: pd.DataFrame, column: str = "timestamp") -> str:
    if column not in df.columns or df.empty:
        return "not available"
    values = pd.to_datetime(df[column], errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min()} to {values.max()}"


def _print_key_missing_values(df: pd.DataFrame) -> None:
    key_columns = [
        "date",
        "timestamp",
        "location",
        "actual_high",
        "forecast_high",
        "forecast_error",
        "forecast_source",
    ]
    present_columns = [column for column in key_columns if column in df.columns]
    if not present_columns:
        print("  key missing values: no standard key columns found")
        return

    print("  key missing values:")
    for column in present_columns:
        print(f"    {column}: {int(df[column].isna().sum())}")


def _print_duplicate_counts(df: pd.DataFrame) -> None:
    keys_to_check = [
        ["date", "location"],
        ["timestamp", "location"],
        ["date", "location", "prediction_time"],
    ]
    printed = False
    for keys in keys_to_check:
        if all(key in df.columns for key in keys):
            duplicate_count = int(df.duplicated(subset=keys).sum())
            print(f"  duplicate count by {keys}: {duplicate_count}")
            printed = True
    if not printed:
        print("  duplicate count: not checked; no standard key set found")


def _print_input_profile(path: Path) -> None:
    print(f"\nInput profile: {path.relative_to(REPO_ROOT)}")
    if not path.exists():
        print("  missing")
        return

    df = pd.read_csv(path)
    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)}")
    if "date" in df.columns:
        print(f"  date range: {_date_range_text(df)}")
    if "timestamp" in df.columns:
        print(f"  timestamp range: {_timestamp_range_text(df)}")
    _print_key_missing_values(df)
    _print_duplicate_counts(df)


def _numeric_stat(series: pd.Series, stat: str) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    if stat == "mean":
        return float(values.mean())
    if stat == "median":
        return float(values.median())
    if stat == "std":
        return float(values.std())
    if stat == "min":
        return float(values.min())
    if stat == "max":
        return float(values.max())
    raise ValueError(f"Unknown stat: {stat}")


def _quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(q))


def _summary_row(
    df: pd.DataFrame,
    location: str,
    prediction_time: str,
) -> dict[str, Any]:
    errors = df["forecast_error"]
    return {
        "location": location,
        "prediction_time": prediction_time,
        "n_rows": int(len(df)),
        "mean_error": _numeric_stat(errors, "mean"),
        "median_error": _numeric_stat(errors, "median"),
        "std_error": _numeric_stat(errors, "std"),
        "min_error": _numeric_stat(errors, "min"),
        "p05_error": _quantile(errors, 0.05),
        "p25_error": _quantile(errors, 0.25),
        "p75_error": _quantile(errors, 0.75),
        "p95_error": _quantile(errors, 0.95),
        "max_error": _numeric_stat(errors, "max"),
        "missing_actual_high": int(df["actual_high"].isna().sum()),
        "missing_forecast_high": int(df["forecast_high"].isna().sum()),
        "missing_forecast_error": int(df["forecast_error"].isna().sum()),
    }


def build_target_summary(supervised_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [_summary_row(supervised_df, "ALL", "ALL")]

    for location, group in supervised_df.groupby("location", sort=True):
        rows.append(_summary_row(group, str(location), "ALL"))

    for prediction_time, group in supervised_df.groupby("prediction_time", sort=True):
        rows.append(_summary_row(group, "ALL", str(prediction_time)))

    for (location, prediction_time), group in supervised_df.groupby(
        ["location", "prediction_time"],
        sort=True,
    ):
        rows.append(_summary_row(group, str(location), str(prediction_time)))

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _forecast_source_warnings(forecasts_df: pd.DataFrame) -> list[str]:
    warning_messages: list[str] = []
    if "forecast_source" in forecasts_df.columns:
        sources = sorted(str(value) for value in forecasts_df["forecast_source"].dropna().unique())
        if any("open_meteo" in source.lower() for source in sources):
            warning_messages.append(
                "Forecast data is an Open-Meteo historical forecast proxy, not "
                "confirmed official NWS archived forecast data."
            )
    else:
        warning_messages.append("Forecast data has no forecast_source column.")

    as_of_candidates = {
        "forecast_created_at",
        "forecast_issue_time",
        "forecast_reference_time",
        "model_run_time",
        "run_timestamp",
        "as_of",
        "issued_at",
    }
    normalized_columns = {str(column).strip().lower() for column in forecasts_df.columns}
    if not normalized_columns.intersection(as_of_candidates):
        warning_messages.append(
            "Forecast rows do not include an as-of/model-run timestamp; Day 7 uses "
            "the cleaned daily forecast_high as the target baseline."
        )

    return warning_messages


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Day 7 input inspection")
    for path in INPUT_PROFILE_PATHS:
        _print_input_profile(path)

    daily_actual = _read_required_csv(DAILY_ACTUAL_INPUT)
    daily_forecasts = _read_required_csv(DAILY_FORECAST_INPUT)

    captured_warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        daily_targets = build_daily_forecast_error_targets(daily_actual, daily_forecasts)
        validate_daily_targets(daily_targets)

        supervised_rows = expand_targets_to_prediction_times(
            daily_targets,
            prediction_times=PREDICTION_TIMES,
        )
        validate_supervised_rows(supervised_rows)

    captured_warning_messages.extend(str(item.message) for item in caught)
    captured_warning_messages.extend(_forecast_source_warnings(daily_forecasts))

    daily_targets.to_csv(DAILY_TARGET_OUTPUT, index=False)
    supervised_rows.to_csv(SUPERVISED_OUTPUT, index=False)

    target_summary = build_target_summary(supervised_rows)
    target_summary.loc[:, FLOAT_SUMMARY_COLUMNS] = target_summary.loc[
        :,
        FLOAT_SUMMARY_COLUMNS,
    ].round(6)
    target_summary.to_csv(TARGET_SUMMARY_OUTPUT, index=False)

    target_errors = pd.to_numeric(daily_targets["forecast_error"], errors="coerce")
    locations = sorted(str(value) for value in daily_targets["location"].dropna().unique())

    print("\nDay 7 supervised forecast-error table complete.")
    print(f"Daily target rows: {len(daily_targets):,}")
    print(f"Supervised rows: {len(supervised_rows):,}")
    print(f"Date range: {_date_range_text(daily_targets)}")
    print(f"Locations: {', '.join(locations)}")
    print(f"Prediction times: {', '.join(PREDICTION_TIMES)}")
    print(
        "Target forecast_error mean/std/min/max: "
        f"{target_errors.mean():.3f} / {target_errors.std():.3f} / "
        f"{target_errors.min():.3f} / {target_errors.max():.3f}"
    )
    print(f"Daily targets: {DAILY_TARGET_OUTPUT}")
    print(f"Supervised rows: {SUPERVISED_OUTPUT}")
    print(f"Target summary: {TARGET_SUMMARY_OUTPUT}")

    if captured_warning_messages:
        print("\nWarnings:")
        for message in dict.fromkeys(captured_warning_messages):
            print(f"- {message}")


if __name__ == "__main__":
    main()
