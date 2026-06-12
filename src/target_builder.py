from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .forecast_data import identify_forecast_high_column
    from .weather_data import _find_column, identify_actual_high_column
except ImportError:
    from forecast_data import identify_forecast_high_column
    from weather_data import _find_column, identify_actual_high_column


DEFAULT_IMPLICIT_LOCATION = "single_location"
FORECAST_ERROR_DECIMALS = 10
DATE_CANDIDATES = ["date", "target_date", "valid_date", "time", "timestamp", "datetime"]
LOCATION_CANDIDATES = ["location", "station", "station_id", "city", "market", "market_city"]
ACTUAL_SOURCE_CANDIDATES = ["actual_source", "observation_source", "source"]
FORECAST_SOURCE_CANDIDATES = [
    "forecast_source",
    "forecast_data_source",
    "model_source",
    "source",
]
OPTIONAL_ACTUAL_AUDIT_COLUMNS = [
    "official_daily_high_f",
    "actual_source",
    "source_file",
    "source_station",
    "source_station_name",
]
OPTIONAL_FORECAST_AUDIT_COLUMNS = [
    "forecast_source",
    "forecast_issue_time",
    "openmeteo_forecast_high_f",
    "nws_forecast_high_f",
    "forecast_fallback_reason",
    "ndfd_valid_time_utc",
    "ndfd_lead_hours",
    "ndfd_grid_distance_km",
]
TARGET_COLUMNS = ["date", "location", "actual_high", "forecast_high", "forecast_error"]
PREDICTION_TARGET_COLUMNS = [
    "date",
    "location",
    "prediction_time",
    "prediction_timestamp",
    "actual_high",
    "forecast_high",
    "forecast_error",
]


def _rename_first_match(
    df: pd.DataFrame,
    candidates: Iterable[str],
    target: str,
) -> pd.DataFrame:
    column = _find_column(df, candidates)
    if column is not None and column != target:
        df = df.rename(columns={column: target})
    return df


def _standardize_date_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = _rename_first_match(df, DATE_CANDIDATES, "date")
    if "date" not in df.columns:
        raise ValueError(f"{label} must include a daily date column")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    return df


def _standardize_location_column(df: pd.DataFrame) -> pd.DataFrame:
    return _rename_first_match(df, LOCATION_CANDIDATES, "location")


def _fill_missing_location_values(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "location" not in df.columns:
        return df

    locations = df["location"]
    non_missing = locations.dropna().astype(str).unique()
    if locations.isna().any():
        if len(non_missing) > 1:
            raise ValueError(f"{label} has missing location values and multiple locations")
        fill_value = non_missing[0] if len(non_missing) == 1 else DEFAULT_IMPLICIT_LOCATION
        df["location"] = locations.fillna(fill_value)

    df["location"] = df["location"].astype(str)
    return df


def _single_location_value(df: pd.DataFrame, label: str) -> str:
    if "location" not in df.columns:
        return DEFAULT_IMPLICIT_LOCATION

    locations = df["location"].dropna().astype(str).unique()
    if len(locations) == 0:
        return DEFAULT_IMPLICIT_LOCATION
    if len(locations) == 1:
        return str(locations[0])
    raise ValueError(
        f"{label} has multiple locations, so a missing counterpart location column "
        "cannot be filled safely"
    )


def _align_location_columns(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_df = _standardize_location_column(actual_df)
    forecast_df = _standardize_location_column(forecast_df)

    actual_df = _fill_missing_location_values(actual_df, "daily_df")
    forecast_df = _fill_missing_location_values(forecast_df, "forecasts_df")

    actual_has_location = "location" in actual_df.columns
    forecast_has_location = "location" in forecast_df.columns

    if not actual_has_location and not forecast_has_location:
        actual_df["location"] = DEFAULT_IMPLICIT_LOCATION
        forecast_df["location"] = DEFAULT_IMPLICIT_LOCATION
    elif not actual_has_location:
        actual_df["location"] = _single_location_value(forecast_df, "forecasts_df")
    elif not forecast_has_location:
        forecast_df["location"] = _single_location_value(actual_df, "daily_df")

    actual_df["location"] = actual_df["location"].astype(str)
    forecast_df["location"] = forecast_df["location"].astype(str)
    return actual_df, forecast_df


def _prepare_daily_actuals(daily_df: pd.DataFrame) -> pd.DataFrame:
    actual_df = daily_df.copy()
    actual_df = _standardize_date_column(actual_df, "daily_df")

    actual_high_column = identify_actual_high_column(actual_df)
    if actual_high_column is None:
        raise ValueError(
            "Could not identify actual daily high column. "
            f"Available columns: {list(actual_df.columns)}"
        )
    if actual_high_column == "official_daily_high_f":
        actual_df["actual_high"] = pd.to_numeric(actual_df[actual_high_column], errors="coerce")
    elif actual_high_column != "actual_high":
        actual_df = actual_df.rename(columns={actual_high_column: "actual_high"})

    actual_df = _rename_first_match(actual_df, ACTUAL_SOURCE_CANDIDATES, "actual_source")
    actual_df["actual_high"] = pd.to_numeric(actual_df["actual_high"], errors="coerce")
    return actual_df


def _prepare_daily_forecasts(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    forecast_df = forecasts_df.copy()
    forecast_df = _standardize_date_column(forecast_df, "forecasts_df")

    forecast_high_column = identify_forecast_high_column(forecast_df)
    if forecast_high_column is None:
        raise ValueError(
            "Could not identify forecast daily high column. "
            f"Available columns: {list(forecast_df.columns)}"
        )
    if forecast_high_column != "forecast_high":
        forecast_df = forecast_df.rename(columns={forecast_high_column: "forecast_high"})

    forecast_df = _rename_first_match(
        forecast_df,
        FORECAST_SOURCE_CANDIDATES,
        "forecast_source",
    )
    forecast_df["forecast_high"] = pd.to_numeric(
        forecast_df["forecast_high"],
        errors="coerce",
    )
    return forecast_df


def _normalise_prediction_time(value: object) -> str:
    parts = str(value).strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Prediction time must be HH:MM, got {value!r}")
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}"


def _prepare_prediction_forecasts(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    forecast_df = _prepare_daily_forecasts(forecasts_df)
    if "prediction_time" not in forecast_df.columns and "prediction_timestamp" not in forecast_df.columns:
        raise ValueError("Prediction-time forecasts must include prediction_time or prediction_timestamp")

    if "prediction_timestamp" in forecast_df.columns:
        forecast_df["prediction_timestamp"] = pd.to_datetime(
            forecast_df["prediction_timestamp"],
            errors="raise",
        )
    if "prediction_time" not in forecast_df.columns:
        forecast_df["prediction_time"] = forecast_df["prediction_timestamp"].dt.strftime("%H:%M")

    forecast_df["prediction_time"] = forecast_df["prediction_time"].map(_normalise_prediction_time)
    if "prediction_timestamp" not in forecast_df.columns:
        forecast_df["prediction_timestamp"] = pd.to_datetime(
            forecast_df["date"].dt.strftime("%Y-%m-%d") + " " + forecast_df["prediction_time"],
            errors="raise",
        )

    if "forecast_issue_time" in forecast_df.columns:
        forecast_df["forecast_issue_time"] = pd.to_datetime(
            forecast_df["forecast_issue_time"],
            errors="coerce",
            utc=True,
        )
    return forecast_df


def _require_unique_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicate_count = int(df.duplicated(subset=keys).sum())
    if duplicate_count:
        raise ValueError(
            f"{label} has {duplicate_count} duplicate rows by {', '.join(keys)}"
        )


def _available_ordered_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def build_daily_forecast_error_targets(
    daily_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join actual daily highs to daily forecast highs by date/location.
    Compute forecast_error = actual_high - forecast_high.
    Return one row per date/location.
    """
    actual_df = _prepare_daily_actuals(daily_df)
    forecast_df = _prepare_daily_forecasts(forecasts_df)
    actual_df, forecast_df = _align_location_columns(actual_df, forecast_df)

    keys = ["date", "location"]
    _require_unique_keys(actual_df, keys, "daily_df")
    _require_unique_keys(forecast_df, keys, "forecasts_df")

    actual_columns = _available_ordered_columns(
        actual_df,
        ["date", "location", "actual_high", *OPTIONAL_ACTUAL_AUDIT_COLUMNS],
    )
    forecast_columns = _available_ordered_columns(
        forecast_df,
        ["date", "location", "forecast_high", *OPTIONAL_FORECAST_AUDIT_COLUMNS],
    )

    merged = actual_df.loc[:, actual_columns].merge(
        forecast_df.loc[:, forecast_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    minimum_input_rows = min(len(actual_df), len(forecast_df))
    if minimum_input_rows and len(merged) < minimum_input_rows:
        warnings.warn(
            "Daily target join dropped rows because actual and forecast date/location "
            "coverage does not fully overlap.",
            stacklevel=2,
        )

    merged["forecast_error"] = (
        merged["actual_high"] - merged["forecast_high"]
    ).round(FORECAST_ERROR_DECIMALS)

    ordered_columns = TARGET_COLUMNS + [
        column
        for column in [*OPTIONAL_ACTUAL_AUDIT_COLUMNS, *OPTIONAL_FORECAST_AUDIT_COLUMNS]
        if column in merged.columns
    ]
    result = merged.loc[:, ordered_columns].sort_values(["location", "date"])
    return result.reset_index(drop=True)


def build_prediction_forecast_error_rows(
    daily_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join daily actual highs to prediction-time forecast highs by date/location.
    Compute forecast_error = actual_high - forecast_high for each prediction row.
    """
    actual_df = _prepare_daily_actuals(daily_df)
    forecast_df = _prepare_prediction_forecasts(forecasts_df)
    actual_df, forecast_df = _align_location_columns(actual_df, forecast_df)

    _require_unique_keys(actual_df, ["date", "location"], "daily_df")
    _require_unique_keys(
        forecast_df,
        ["date", "location", "prediction_time"],
        "forecasts_df",
    )

    actual_columns = _available_ordered_columns(
        actual_df,
        ["date", "location", "actual_high", *OPTIONAL_ACTUAL_AUDIT_COLUMNS],
    )
    forecast_columns = _available_ordered_columns(
        forecast_df,
        [
            "date",
            "location",
            "prediction_time",
            "prediction_timestamp",
            "forecast_high",
            *OPTIONAL_FORECAST_AUDIT_COLUMNS,
        ],
    )

    merged = actual_df.loc[:, actual_columns].merge(
        forecast_df.loc[:, forecast_columns],
        on=["date", "location"],
        how="inner",
        validate="one_to_many",
    )

    if len(actual_df) and merged[["date", "location"]].drop_duplicates().shape[0] < len(actual_df):
        warnings.warn(
            "Prediction target join dropped dates because actual and forecast coverage "
            "does not fully overlap.",
            stacklevel=2,
        )

    merged["forecast_error"] = (
        merged["actual_high"] - merged["forecast_high"]
    ).round(FORECAST_ERROR_DECIMALS)

    ordered_columns = PREDICTION_TARGET_COLUMNS + [
        column
        for column in [*OPTIONAL_ACTUAL_AUDIT_COLUMNS, *OPTIONAL_FORECAST_AUDIT_COLUMNS]
        if column in merged.columns
    ]
    result = merged.loc[:, ordered_columns].sort_values(
        ["location", "date", "prediction_timestamp"]
    )
    return result.reset_index(drop=True)


def validate_daily_targets(df: pd.DataFrame) -> None:
    """
    Validate one daily target row per date/location and verify forecast_error math.
    Raise ValueError with clear messages if validation fails.
    """
    missing_columns = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Daily targets missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Daily targets are empty")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Daily targets contain unparseable date values")

    for column in ["location", "actual_high", "forecast_high", "forecast_error"]:
        missing_count = int(df[column].isna().sum())
        if missing_count:
            raise ValueError(f"Daily targets contain {missing_count} missing {column} values")

    duplicate_count = int(df.duplicated(subset=["date", "location"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Daily targets contain {duplicate_count} duplicate date/location rows"
        )

    expected_error = pd.to_numeric(df["actual_high"], errors="coerce") - pd.to_numeric(
        df["forecast_high"],
        errors="coerce",
    )
    actual_error = pd.to_numeric(df["forecast_error"], errors="coerce")
    if not np.allclose(actual_error, expected_error, rtol=0.0, atol=1e-9):
        bad_count = int((~np.isclose(actual_error, expected_error, rtol=0.0, atol=1e-9)).sum())
        raise ValueError(
            f"Daily targets have {bad_count} rows where forecast_error != "
            "actual_high - forecast_high"
        )

    abs_errors = actual_error.abs()
    extreme_count = int((abs_errors > 40).sum())
    if extreme_count:
        warnings.warn(
            f"{extreme_count} daily forecast_error values exceed 40 F in absolute value.",
            stacklevel=2,
        )

    mean_error = float(actual_error.mean())
    if abs(mean_error) > 15:
        warnings.warn(
            f"Daily forecast_error mean is far from zero: {mean_error:.2f} F.",
            stacklevel=2,
        )

    zero_share = float((abs_errors <= 1e-9).mean())
    if zero_share == 1.0:
        warnings.warn(
            "All daily forecast_error values are exactly zero.",
            stacklevel=2,
        )
    elif zero_share >= 0.95:
        warnings.warn(
            f"{zero_share:.1%} of daily forecast_error values are exactly zero.",
            stacklevel=2,
        )
