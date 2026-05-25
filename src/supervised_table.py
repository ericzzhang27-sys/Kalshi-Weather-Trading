from __future__ import annotations

import numpy as np
import pandas as pd


PREDICTION_TIMES = ["09:00", "11:00", "13:00", "15:00"]
TARGET_COLUMN = "forecast_error"
AUDIT_ONLY_COLUMNS = ["actual_high"]
BASELINE_FEATURE_COLUMNS = ["forecast_high"]
SUPERVISED_REQUIRED_COLUMNS = [
    "date",
    "location",
    "prediction_time",
    "prediction_timestamp",
    "actual_high",
    "forecast_high",
    "forecast_error",
]


def define_prediction_times() -> list[str]:
    """
    Return the prediction times used for Day 7.
    """
    return list(PREDICTION_TIMES)


def expand_targets_to_prediction_times(
    target_df: pd.DataFrame,
    prediction_times: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cross-join each daily target row with the prediction times.
    Return one row per date/location/prediction_time.
    """
    times = prediction_times if prediction_times is not None else define_prediction_times()
    if not times:
        raise ValueError("At least one prediction time is required")

    targets = target_df.copy()
    targets["date"] = pd.to_datetime(targets["date"], errors="raise").dt.normalize()

    time_df = pd.DataFrame({"prediction_time": [str(value) for value in times]})
    expanded = targets.assign(_day7_cross_join_key=1).merge(
        time_df.assign(_day7_cross_join_key=1),
        on="_day7_cross_join_key",
        how="inner",
    )
    expanded = expanded.drop(columns="_day7_cross_join_key")
    expanded = add_prediction_timestamp(expanded)

    leading_columns = [
        column for column in SUPERVISED_REQUIRED_COLUMNS if column in expanded.columns
    ]
    remaining_columns = [
        column for column in expanded.columns if column not in leading_columns
    ]
    expanded = expanded.loc[:, leading_columns + remaining_columns]
    return expanded.sort_values(["location", "date", "prediction_timestamp"]).reset_index(
        drop=True
    )


def add_prediction_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine date and prediction_time into prediction_timestamp.
    """
    result = df.copy()
    if "date" not in result.columns:
        raise ValueError("Cannot build prediction_timestamp without a date column")
    if "prediction_time" not in result.columns:
        raise ValueError("Cannot build prediction_timestamp without prediction_time")

    dates = pd.to_datetime(result["date"], errors="raise").dt.strftime("%Y-%m-%d")
    times = result["prediction_time"].astype(str).str.strip()
    result["prediction_timestamp"] = pd.to_datetime(
        dates + " " + times,
        errors="raise",
    )
    return result


def validate_supervised_rows(df: pd.DataFrame) -> None:
    """
    Validate the supervised Day 7 skeleton table.
    Raise ValueError with clear messages if validation fails.
    """
    missing_columns = [
        column for column in SUPERVISED_REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Supervised rows missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Supervised rows are empty")

    actual_times = sorted(df["prediction_time"].dropna().astype(str).unique())
    expected_times = sorted(PREDICTION_TIMES)
    if actual_times != expected_times:
        raise ValueError(
            "Supervised rows contain unexpected prediction_time values: "
            f"expected {expected_times}, got {actual_times}"
        )

    for column in ["date", "location", "prediction_time", "forecast_error"]:
        missing_count = int(df[column].isna().sum())
        if missing_count:
            raise ValueError(f"Supervised rows contain {missing_count} missing {column} values")

    duplicate_count = int(
        df.duplicated(subset=["date", "location", "prediction_time"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "Supervised rows contain "
            f"{duplicate_count} duplicate date/location/prediction_time rows"
        )

    parsed_timestamps = pd.to_datetime(df["prediction_timestamp"], errors="coerce")
    if parsed_timestamps.isna().any():
        raise ValueError("Supervised rows contain unparseable prediction_timestamp values")

    row_counts = df.groupby(["date", "location"]).size()
    expected_count = len(PREDICTION_TIMES)
    bad_counts = row_counts[row_counts != expected_count]
    if not bad_counts.empty:
        raise ValueError(
            "Every date/location must have exactly "
            f"{expected_count} prediction-time rows; found {bad_counts.to_dict()}"
        )

    expected_error = pd.to_numeric(df["actual_high"], errors="coerce") - pd.to_numeric(
        df["forecast_high"],
        errors="coerce",
    )
    actual_error = pd.to_numeric(df["forecast_error"], errors="coerce")
    if actual_error.isna().any():
        raise ValueError("Supervised rows contain non-numeric forecast_error values")
    if not np.allclose(actual_error, expected_error, rtol=0.0, atol=1e-9):
        bad_count = int((~np.isclose(actual_error, expected_error, rtol=0.0, atol=1e-9)).sum())
        raise ValueError(
            f"Supervised rows have {bad_count} rows where forecast_error != "
            "actual_high - forecast_high"
        )
