from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"

DEFAULT_INPUT_PATHS = {
    "rows": PROCESSED_DIR / "supervised_forecast_error_rows.csv",
    "modeling_base": PROCESSED_DIR / "modeling_base.csv",
    "hourly": PROCESSED_DIR / "hourly_clean.csv",
    "hourly_forecasts": PROCESSED_DIR / "hourly_forecasts_clean.csv",
    "daily": PROCESSED_DIR / "daily_clean.csv",
    "forecasts": PROCESSED_DIR / "forecasts_clean.csv",
}

TARGET_COLUMN = "forecast_error"
TYPICAL_PEAK_HOUR = 15
TEMP_CHANGE_WINDOWS_MINUTES = [60, 120, 180, 240, 300]

ISSUE_TIME_CANDIDATES = [
    "issue_time",
    "issued_at",
    "forecast_issue_time",
    "forecast_created_at",
    "forecast_reference_time",
    "model_run_time",
    "run_time",
    "run_timestamp",
    "reference_time",
    "as_of",
]

VALID_TIME_CANDIDATES = [
    "valid_time",
    "forecast_valid_time",
    "timestamp",
]

TEMPERATURE_CANDIDATES = [
    "temperature_2m",
    "temperature",
    "temp",
    "temp_f",
]

DEW_POINT_CANDIDATES = [
    "dew_point_2m",
    "dew_point",
]

CLOUD_COVER_CANDIDATES = [
    "cloud_cover",
    "cloudcover",
]

WIND_SPEED_CANDIDATES = [
    "wind_speed_10m",
    "wind_speed",
]

WIND_DIRECTION_CANDIDATES = [
    "wind_direction_10m",
    "wind_direction",
]

PRECIPITATION_CANDIDATES = [
    "precipitation",
    "rain",
]

PRECIP_PROB_CANDIDATES = [
    "precipitation_probability",
    "precip_probability",
    "precip_prob",
]

CRITICAL_COLUMNS = [
    TARGET_COLUMN,
    "forecast_high",
    "actual_high",
    "prediction_time",
    "target_date",
    "current_temp",
    "max_temp_so_far",
]

EXCLUDED_FEATURE_COLUMNS = {
    "station",
    "location",
    "date",
    "target_date",
    "prediction_time",
    "prediction_clock_time",
    "prediction_timestamp",
    "actual_high",
    TARGET_COLUMN,
    "target",
    "final_high",
    "official_high",
    "observed_daily_high",
    "daily_high",
    "actual_max_temp",
    "max_temp_today",
    "forecast_source",
}

LEAKAGE_NAME_FRAGMENTS = [
    "actual_high",
    "official_high",
    "final_high",
    "observed_daily_high",
    "daily_high",
    "actual_max_temp",
    "max_temp_today",
]

METADATA_NAME_FRAGMENTS = [
    "source_time",
    "issue_time",
    "valid_time",
    "timestamp",
    "created_at",
    "reference_time",
    "run_time",
    "as_of",
]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _read_csv_if_exists(path: Path, required: bool = False) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Required input file is missing: {path}")
    return _empty_frame()


def _append_note(df: pd.DataFrame, note: str) -> None:
    notes = list(df.attrs.get("feature_notes", []))
    if note not in notes:
        notes.append(note)
    df.attrs["feature_notes"] = notes


def _merge_notes(target: pd.DataFrame, *sources: pd.DataFrame) -> None:
    notes = list(target.attrs.get("feature_notes", []))
    for source in sources:
        for note in source.attrs.get("feature_notes", []):
            if note not in notes:
                notes.append(note)
    target.attrs["feature_notes"] = notes


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return str(normalized[candidate.lower()])
    return None


def _normalize_location_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "location" not in result.columns and "station" in result.columns:
        result["location"] = result["station"]
    if "location" in result.columns:
        result["location"] = result["location"].astype(str)
    return result


def _standardize_rows(df: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_location_column(df)

    if "target_date" in result.columns:
        result["target_date"] = pd.to_datetime(result["target_date"], errors="raise").dt.normalize()
    elif "date" in result.columns:
        result["target_date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    else:
        raise ValueError("Supervised rows must contain either target_date or date")

    if "date" not in result.columns:
        result["date"] = result["target_date"].dt.strftime("%Y-%m-%d")

    if "prediction_timestamp" in result.columns:
        prediction_ts = pd.to_datetime(result["prediction_timestamp"], errors="raise")
        if "prediction_time" in result.columns:
            raw_prediction_time = result["prediction_time"].astype(str)
            time_only = raw_prediction_time.str.match(
                r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$",
            ).all()
            if time_only:
                result["prediction_clock_time"] = raw_prediction_time
        result["prediction_time"] = prediction_ts
    elif "prediction_time" in result.columns:
        raw_prediction_time = result["prediction_time"].astype(str)
        time_only = raw_prediction_time.str.match(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$").all()
        if time_only:
            result["prediction_clock_time"] = result["prediction_time"].astype(str)
            dates = result["target_date"].dt.strftime("%Y-%m-%d")
            result["prediction_time"] = pd.to_datetime(
                dates + " " + result["prediction_clock_time"],
                errors="raise",
            )
        else:
            parsed_prediction = pd.to_datetime(result["prediction_time"], errors="coerce")
            if parsed_prediction.isna().any():
                raise ValueError("prediction_time contains unparseable values")
            result["prediction_time"] = parsed_prediction
    else:
        raise ValueError("Supervised rows must contain prediction_time or prediction_timestamp")

    result["prediction_timestamp"] = result["prediction_time"]

    for column in ["actual_high", "forecast_high", TARGET_COLUMN]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def _standardize_time_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = _normalize_location_column(df)

    valid_col = _first_existing(result, VALID_TIME_CANDIDATES)
    if valid_col is not None:
        result[valid_col] = pd.to_datetime(result[valid_col], errors="raise")
        if valid_col != "timestamp" and "timestamp" not in result.columns:
            result["timestamp"] = result[valid_col]

    if "date" in result.columns:
        result["target_date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    elif "timestamp" in result.columns:
        result["target_date"] = pd.to_datetime(result["timestamp"], errors="raise").dt.normalize()

    for issue_col in ISSUE_TIME_CANDIDATES:
        if issue_col in result.columns:
            result[issue_col] = pd.to_datetime(result[issue_col], errors="raise")

    numeric_candidates = (
        TEMPERATURE_CANDIDATES
        + DEW_POINT_CANDIDATES
        + CLOUD_COVER_CANDIDATES
        + WIND_SPEED_CANDIDATES
        + WIND_DIRECTION_CANDIDATES
        + PRECIPITATION_CANDIDATES
        + PRECIP_PROB_CANDIDATES
        + ["forecast_high"]
    )
    for column in dict.fromkeys(numeric_candidates):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "timestamp" not in result.columns and table_name in {"hourly", "hourly_forecasts"}:
        raise ValueError(f"{table_name} must contain a forecast/observation valid timestamp")

    return result


def load_inputs(
    rows_path: str | Path | None = None,
    hourly_path: str | Path | None = None,
    hourly_forecasts_path: str | Path | None = None,
    daily_path: str | Path | None = None,
    forecasts_path: str | Path | None = None,
) -> dict[str, pd.DataFrame | list[str]]:
    """
    Load Day 7 supervised rows and cleaned weather inputs.

    Datetime columns are standardized as naive local pandas timestamps. The Day 7
    clock-only prediction_time column is retained as prediction_clock_time, while
    prediction_time is standardized to the full prediction timestamp.
    """
    rows_candidate = Path(rows_path) if rows_path is not None else DEFAULT_INPUT_PATHS["rows"]
    if rows_path is None and not rows_candidate.exists():
        rows_candidate = DEFAULT_INPUT_PATHS["modeling_base"]

    hourly_candidate = Path(hourly_path) if hourly_path is not None else DEFAULT_INPUT_PATHS["hourly"]
    hourly_forecasts_candidate = (
        Path(hourly_forecasts_path)
        if hourly_forecasts_path is not None
        else DEFAULT_INPUT_PATHS["hourly_forecasts"]
    )
    daily_candidate = Path(daily_path) if daily_path is not None else DEFAULT_INPUT_PATHS["daily"]
    forecasts_candidate = (
        Path(forecasts_path) if forecasts_path is not None else DEFAULT_INPUT_PATHS["forecasts"]
    )

    rows = _standardize_rows(_read_csv_if_exists(rows_candidate, required=True))
    hourly = _standardize_time_table(_read_csv_if_exists(hourly_candidate), "hourly")
    hourly_forecasts = _standardize_time_table(
        _read_csv_if_exists(hourly_forecasts_candidate),
        "hourly_forecasts",
    )
    daily = _standardize_time_table(_read_csv_if_exists(daily_candidate), "daily")
    forecasts = _standardize_time_table(_read_csv_if_exists(forecasts_candidate), "forecasts")

    notes: list[str] = []
    if not hourly_forecasts.empty and _first_existing(hourly_forecasts, ISSUE_TIME_CANDIDATES) is None:
        notes.append(
            "hourly_forecasts_clean.csv has forecast valid timestamps but no issue/run/reference "
            "timestamp; forecast issuance safety cannot be directly verified."
        )
    if not forecasts.empty and _first_existing(forecasts, ISSUE_TIME_CANDIDATES) is None:
        notes.append(
            "forecasts_clean.csv has daily forecast highs but no issue/run/reference timestamp; "
            "forecast_high is treated as the Day 7 baseline known at prediction time."
        )

    return {
        "rows": rows,
        "hourly": hourly,
        "hourly_forecasts": hourly_forecasts,
        "daily": daily,
        "forecasts": forecasts,
        "notes": notes,
    }


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic calendar and clock features.

    forecast_horizon_hours is measured from prediction_time to the typical daily
    peak temperature time, defined as 3 PM local time on target_date.
    """
    result = df.copy()
    prediction_time = pd.to_datetime(result["prediction_time"], errors="raise")
    target_date = pd.to_datetime(result["target_date"], errors="raise").dt.normalize()

    day_of_year = prediction_time.dt.dayofyear.astype(float)
    result["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    result["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)

    hour_fraction = (
        prediction_time.dt.hour.astype(float)
        + prediction_time.dt.minute.astype(float) / 60.0
        + prediction_time.dt.second.astype(float) / 3600.0
    )
    result["hour_sin"] = np.sin(2.0 * np.pi * hour_fraction / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour_fraction / 24.0)
    result["month"] = prediction_time.dt.month.astype("int64")

    month = result["month"]
    season = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        [0, 1, 2, 3],
        default=np.nan,
    )
    result["season"] = season.astype(float)

    peak_time = target_date + pd.to_timedelta(TYPICAL_PEAK_HOUR, unit="h")
    result["forecast_horizon_hours"] = (peak_time - prediction_time).dt.total_seconds() / 3600.0
    _merge_notes(result, df)
    _append_note(
        result,
        "forecast_horizon_hours and minutes_until_typical_peak use 3 PM local time "
        "on target_date as the typical peak temperature time.",
    )
    return result


def _group_key(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value
    return value


def _latest_at_or_before(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    by: list[str],
    left_time_col: str,
    right_time_col: str,
    value_cols: list[str],
    source_time_name: str | None = None,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(index=left.index)
    for column in value_cols:
        if column in right.columns and pd.api.types.is_datetime64_any_dtype(right[column]):
            result[column] = pd.NaT
        else:
            result[column] = np.nan
    if source_time_name is not None:
        result[source_time_name] = pd.NaT

    if left.empty or right.empty:
        return result

    usable_right = right.dropna(subset=[right_time_col]).copy()
    if usable_right.empty:
        return result

    groups: dict[tuple[Any, ...], pd.DataFrame] = {}
    for key, group in usable_right.sort_values(right_time_col).groupby(by, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        groups[tuple(_group_key(item) for item in key)] = group.reset_index(drop=True)

    for left_key, left_group in left.groupby(by, dropna=False):
        if not isinstance(left_key, tuple):
            left_key = (left_key,)
        group = groups.get(tuple(_group_key(item) for item in left_key))
        if group is None or group.empty:
            continue

        right_times = pd.to_datetime(group[right_time_col], errors="coerce")
        right_ns = right_times.astype("int64").to_numpy()
        left_times = pd.to_datetime(left_group[left_time_col], errors="coerce")

        for row_index, left_ts in left_times.items():
            if pd.isna(left_ts):
                continue
            pos = int(np.searchsorted(right_ns, left_ts.value, side="right") - 1)
            if pos < 0:
                continue
            source_time = right_times.iloc[pos]
            if tolerance is not None and left_ts - source_time > tolerance:
                continue
            for column in value_cols:
                result.at[row_index, column] = group.at[pos, column]
            if source_time_name is not None:
                result.at[row_index, source_time_name] = source_time

    return result


def _observation_interval_minutes(hourly: pd.DataFrame) -> float | None:
    if hourly.empty or "timestamp" not in hourly.columns or "location" not in hourly.columns:
        return None

    diffs = (
        hourly.sort_values(["location", "timestamp"])
        .groupby("location")["timestamp"]
        .diff()
        .dropna()
    )
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return None
    return float(positive_diffs.median().total_seconds() / 60.0)


def _build_cumulative_max_table(
    hourly: pd.DataFrame,
    temp_col: str,
    output_col: str,
    source_col: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    required_cols = ["location", "target_date", "timestamp", temp_col]
    working = hourly.dropna(subset=["location", "target_date", "timestamp"]).loc[:, required_cols].copy()
    for _, group in working.sort_values("timestamp").groupby(["location", "target_date"], dropna=False):
        max_values: list[float] = []
        source_times: list[pd.Timestamp | pd.NaT] = []
        current_max = -np.inf
        current_source: pd.Timestamp | pd.NaT = pd.NaT
        for _, row in group.iterrows():
            value = row[temp_col]
            if pd.notna(value) and float(value) >= current_max:
                current_max = float(value)
                current_source = row["timestamp"]
            max_values.append(np.nan if current_max == -np.inf else current_max)
            source_times.append(current_source)
        enriched = group.loc[:, ["location", "target_date", "timestamp"]].copy()
        enriched[output_col] = max_values
        enriched[source_col] = source_times
        pieces.append(enriched)

    if not pieces:
        return pd.DataFrame(columns=["location", "target_date", "timestamp", output_col, source_col])
    return pd.concat(pieces, ignore_index=True)


def add_observed_weather_features(rows: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    _merge_notes(result, rows)

    if hourly.empty:
        for column in [
            "current_temp",
            "max_temp_so_far",
            "temp_change_30m",
            *[f"temp_change_{minutes}m" for minutes in TEMP_CHANGE_WINDOWS_MINUTES],
            "temp_acceleration_60m",
            "temp_change_60m_minus_3h_avg_rate",
        ]:
            result[column] = np.nan
        _append_note(result, "Observed weather features skipped because hourly_clean.csv is missing.")
        return result

    temp_col = _first_existing(hourly, TEMPERATURE_CANDIDATES)
    if temp_col is None:
        result["current_temp"] = np.nan
        result["max_temp_so_far"] = np.nan
        for column in [
            "temp_change_30m",
            *[f"temp_change_{minutes}m" for minutes in TEMP_CHANGE_WINDOWS_MINUTES],
            "temp_acceleration_60m",
            "temp_change_60m_minus_3h_avg_rate",
        ]:
            result[column] = np.nan
        _append_note(result, "Observed temperature features skipped; no temperature column found.")
        return result

    dew_col = _first_existing(hourly, DEW_POINT_CANDIDATES)
    cloud_col = _first_existing(hourly, CLOUD_COVER_CANDIDATES)
    wind_speed_col = _first_existing(hourly, WIND_SPEED_CANDIDATES)
    wind_dir_col = _first_existing(hourly, WIND_DIRECTION_CANDIDATES)
    precip_col = _first_existing(hourly, PRECIPITATION_CANDIDATES)

    value_mapping = {
        temp_col: "current_temp",
    }
    optional_mapping = {
        dew_col: "dew_point",
        cloud_col: "cloud_cover_now",
        wind_speed_col: "wind_speed",
        wind_dir_col: "wind_direction_degrees",
        precip_col: "precipitation_now",
    }
    for source, target in optional_mapping.items():
        if source is not None:
            value_mapping[source] = target
        else:
            result[target] = np.nan
            _append_note(result, f"{target} skipped; no source column found in hourly_clean.csv.")

    right = hourly.loc[:, ["location", "timestamp", *value_mapping.keys()]].rename(columns=value_mapping)
    current = _latest_at_or_before(
        result,
        right,
        by=["location"],
        left_time_col="prediction_time",
        right_time_col="timestamp",
        value_cols=list(value_mapping.values()),
        source_time_name="current_temp_source_time",
        tolerance=pd.Timedelta(hours=3),
    )
    for column in current.columns:
        result[column] = current[column]

    if "dew_point" in result.columns:
        result["temp_minus_dew_point"] = result["current_temp"] - result["dew_point"]

    if "wind_direction_degrees" in result.columns:
        direction_radians = np.deg2rad(pd.to_numeric(result["wind_direction_degrees"], errors="coerce"))
        result["wind_dir_sin"] = np.sin(direction_radians)
        result["wind_dir_cos"] = np.cos(direction_radians)
        result = result.drop(columns=["wind_direction_degrees"])
    else:
        result["wind_dir_sin"] = np.nan
        result["wind_dir_cos"] = np.nan

    cumulative = _build_cumulative_max_table(
        hourly,
        temp_col=temp_col,
        output_col="max_temp_so_far",
        source_col="max_temp_so_far_source_time",
    )
    max_so_far = _latest_at_or_before(
        result,
        cumulative,
        by=["location", "target_date"],
        left_time_col="prediction_time",
        right_time_col="timestamp",
        value_cols=["max_temp_so_far", "max_temp_so_far_source_time"],
        tolerance=pd.Timedelta(hours=24),
    )
    result["max_temp_so_far"] = max_so_far["max_temp_so_far"]
    result["max_temp_so_far_source_time"] = max_so_far["max_temp_so_far_source_time"]

    interval_minutes = _observation_interval_minutes(hourly)
    if interval_minutes is None or interval_minutes > 45:
        result["temp_change_30m"] = np.nan
        _append_note(
            result,
            "temp_change_30m skipped; hourly_clean.csv cadence is hourly, so 30-minute "
            "temperature change would fake unavailable precision.",
        )
    else:
        result["_temp_30m_lookup_time"] = result["prediction_time"] - pd.Timedelta(minutes=30)
        lookback_30m = _latest_at_or_before(
            result,
            right.loc[:, ["location", "timestamp", "current_temp"]],
            by=["location"],
            left_time_col="_temp_30m_lookup_time",
            right_time_col="timestamp",
            value_cols=["current_temp"],
            tolerance=pd.Timedelta(minutes=20),
        )
        result["temp_change_30m"] = result["current_temp"] - lookback_30m["current_temp"]
        result = result.drop(columns=["_temp_30m_lookup_time"])

    for minutes in TEMP_CHANGE_WINDOWS_MINUTES:
        lookup_col = f"_temp_{minutes}m_lookup_time"
        result[lookup_col] = result["prediction_time"] - pd.Timedelta(minutes=minutes)
        lookback = _latest_at_or_before(
            result,
            right.loc[:, ["location", "timestamp", "current_temp"]],
            by=["location"],
            left_time_col=lookup_col,
            right_time_col="timestamp",
            value_cols=["current_temp"],
            tolerance=pd.Timedelta(minutes=45),
        )
        result[f"temp_change_{minutes}m"] = result["current_temp"] - lookback["current_temp"]
        result = result.drop(columns=[lookup_col])

    result["temp_acceleration_60m"] = (
        2.0 * result["temp_change_60m"] - result["temp_change_120m"]
    )
    result["temp_change_60m_minus_3h_avg_rate"] = (
        result["temp_change_60m"] - result["temp_change_180m"] / 3.0
    )

    return result


def _select_latest_forecast_rows_with_issue(
    rows: pd.DataFrame,
    hourly_forecasts: pd.DataFrame,
    *,
    issue_col: str,
    valid_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    output_cols = [
        "forecast_temp_current_hour",
        "forecast_temp_source_issue_time",
        "forecast_temp_source_valid_time",
        "forecast_max_so_far",
        "forecast_max_so_far_source_valid_time",
        "forecast_source_issue_time",
    ]
    result = pd.DataFrame(index=rows.index)
    for column in output_cols:
        result[column] = pd.NaT if column.endswith("_time") else np.nan

    for row_index, row in rows.iterrows():
        candidates = hourly_forecasts[
            (hourly_forecasts["location"] == row["location"])
            & (hourly_forecasts["target_date"] == row["target_date"])
            & (hourly_forecasts[issue_col] <= row["prediction_time"])
        ]
        if candidates.empty:
            continue
        latest_issue = candidates[issue_col].max()
        run = candidates[candidates[issue_col] == latest_issue].sort_values(valid_col)
        current_candidates = run[run[valid_col] <= row["prediction_time"]]
        if not current_candidates.empty:
            current = current_candidates.iloc[-1]
            if "temperature_2m" in value_cols:
                result.at[row_index, "forecast_temp_current_hour"] = current["temperature_2m"]
            result.at[row_index, "forecast_temp_source_issue_time"] = latest_issue
            result.at[row_index, "forecast_temp_source_valid_time"] = current[valid_col]

        so_far = run[run[valid_col] <= row["prediction_time"]]
        if not so_far.empty and "temperature_2m" in value_cols:
            max_pos = pd.to_numeric(so_far["temperature_2m"], errors="coerce").idxmax()
            result.at[row_index, "forecast_max_so_far"] = so_far.loc[max_pos, "temperature_2m"]
            result.at[row_index, "forecast_max_so_far_source_valid_time"] = so_far.loc[max_pos, valid_col]
            result.at[row_index, "forecast_source_issue_time"] = latest_issue

    return result


def _next_3h_from_issued_forecasts(
    rows: pd.DataFrame,
    hourly_forecasts: pd.DataFrame,
    *,
    issue_col: str,
    valid_col: str,
    cloud_col: str | None,
    precip_prob_col: str | None,
) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    result["cloud_cover_next_3h"] = np.nan
    result["precip_probability_next_3h"] = np.nan
    result["next_3h_forecast_source_issue_time"] = pd.NaT

    if cloud_col is None and precip_prob_col is None:
        return result

    for row_index, row in rows.iterrows():
        candidates = hourly_forecasts[
            (hourly_forecasts["location"] == row["location"])
            & (hourly_forecasts["target_date"] == row["target_date"])
            & (hourly_forecasts[issue_col] <= row["prediction_time"])
        ]
        if candidates.empty:
            continue
        latest_issue = candidates[issue_col].max()
        run = candidates[candidates[issue_col] == latest_issue]
        window = run[
            (run[valid_col] > row["prediction_time"])
            & (run[valid_col] <= row["prediction_time"] + pd.Timedelta(hours=3))
        ]
        if window.empty:
            continue
        if cloud_col is not None:
            result.at[row_index, "cloud_cover_next_3h"] = pd.to_numeric(
                window[cloud_col],
                errors="coerce",
            ).mean()
        if precip_prob_col is not None:
            result.at[row_index, "precip_probability_next_3h"] = pd.to_numeric(
                window[precip_prob_col],
                errors="coerce",
            ).mean()
        result.at[row_index, "next_3h_forecast_source_issue_time"] = latest_issue

    return result


def add_forecast_relative_features(
    rows: pd.DataFrame,
    hourly_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    result = rows.copy()
    _merge_notes(result, rows)

    if "forecast_high" in result.columns:
        result["forecast_high"] = pd.to_numeric(result["forecast_high"], errors="coerce")

    for column in [
        "forecast_temp_current_hour",
        "current_temp_minus_forecast_temp",
        "forecast_max_so_far",
        "max_so_far_minus_forecast_max_so_far",
        "cloud_cover_next_3h",
        "precip_probability_next_3h",
    ]:
        if column not in result.columns:
            result[column] = np.nan

    if hourly_forecasts.empty:
        _append_note(result, "Forecast-relative hourly features skipped because hourly_forecasts_clean.csv is missing.")
        return result

    temp_col = _first_existing(hourly_forecasts, TEMPERATURE_CANDIDATES)
    valid_col = _first_existing(hourly_forecasts, VALID_TIME_CANDIDATES)
    issue_col = _first_existing(hourly_forecasts, ISSUE_TIME_CANDIDATES)
    cloud_col = _first_existing(hourly_forecasts, CLOUD_COVER_CANDIDATES)
    precip_prob_col = _first_existing(hourly_forecasts, PRECIP_PROB_CANDIDATES)

    if temp_col is None or valid_col is None:
        _append_note(
            result,
            "Forecast-relative temperature features skipped; hourly forecast temperature "
            "or valid timestamp column is missing.",
        )
        return result

    forecast_work = hourly_forecasts.copy()
    if temp_col != "temperature_2m":
        forecast_work["temperature_2m"] = forecast_work[temp_col]
    if valid_col != "timestamp":
        forecast_work["timestamp"] = forecast_work[valid_col]
        valid_col = "timestamp"

    if issue_col is not None:
        selected = _select_latest_forecast_rows_with_issue(
            result,
            forecast_work,
            issue_col=issue_col,
            valid_col=valid_col,
            value_cols=["temperature_2m"],
        )
        for column in selected.columns:
            result[column] = selected[column]

        next_3h = _next_3h_from_issued_forecasts(
            result,
            forecast_work,
            issue_col=issue_col,
            valid_col=valid_col,
            cloud_col=cloud_col,
            precip_prob_col=precip_prob_col,
        )
        for column in next_3h.columns:
            result[column] = next_3h[column]
    else:
        current_forecast = _latest_at_or_before(
            result,
            forecast_work.loc[:, ["location", "target_date", "timestamp", "temperature_2m"]],
            by=["location", "target_date"],
            left_time_col="prediction_time",
            right_time_col="timestamp",
            value_cols=["temperature_2m"],
            source_time_name="forecast_temp_source_valid_time",
            tolerance=pd.Timedelta(hours=3),
        )
        result["forecast_temp_current_hour"] = current_forecast["temperature_2m"]
        result["forecast_temp_source_valid_time"] = current_forecast[
            "forecast_temp_source_valid_time"
        ]

        cumulative_forecast = _build_cumulative_max_table(
            forecast_work,
            temp_col="temperature_2m",
            output_col="forecast_max_so_far",
            source_col="forecast_max_so_far_source_valid_time",
        )
        forecast_max = _latest_at_or_before(
            result,
            cumulative_forecast,
            by=["location", "target_date"],
            left_time_col="prediction_time",
            right_time_col="timestamp",
            value_cols=["forecast_max_so_far", "forecast_max_so_far_source_valid_time"],
            tolerance=pd.Timedelta(hours=24),
        )
        result["forecast_max_so_far"] = forecast_max["forecast_max_so_far"]
        result["forecast_max_so_far_source_valid_time"] = forecast_max[
            "forecast_max_so_far_source_valid_time"
        ]
        _append_note(
            result,
            "cloud_cover_next_3h and precip_probability_next_3h skipped because hourly "
            "forecast issue/run timestamps are unavailable; future valid times cannot be "
            "proven to come from a run issued at or before prediction_time.",
        )

    result["current_temp_minus_forecast_temp"] = (
        result["current_temp"] - result["forecast_temp_current_hour"]
    )
    result["max_so_far_minus_forecast_max_so_far"] = (
        result["max_temp_so_far"] - result["forecast_max_so_far"]
    )

    if precip_prob_col is None:
        _append_note(
            result,
            "precip_probability_next_3h unavailable or partially unavailable because the "
            "hourly forecast precipitation probability column is missing or sparse.",
        )

    return result


def add_solar_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    prediction_time = pd.to_datetime(result["prediction_time"], errors="raise")
    target_date = pd.to_datetime(result["target_date"], errors="raise").dt.normalize()
    peak_time = target_date + pd.to_timedelta(TYPICAL_PEAK_HOUR, unit="h")
    result["minutes_until_typical_peak"] = (
        peak_time - prediction_time
    ).dt.total_seconds() / 60.0
    _merge_notes(result, df)
    _append_note(
        result,
        "minutes_until_sunset skipped; no extra solar dependency was added for Day 8.",
    )
    return result


def add_forecast_update_features(rows: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    _merge_notes(result, rows)

    if forecasts.empty:
        _append_note(result, "Forecast update features skipped because forecasts_clean.csv is missing.")
        return result

    issue_col = _first_existing(forecasts, ISSUE_TIME_CANDIDATES)
    if issue_col is None:
        _append_note(
            result,
            "recent_forecast_revision, forecast_spread, and model_disagreement skipped; "
            "forecasts_clean.csv has no repeated issue/run timestamp.",
        )
        return result

    if "forecast_high" not in forecasts.columns:
        _append_note(
            result,
            "Forecast update features skipped; forecasts_clean.csv has no forecast_high column.",
        )
        return result

    revisions = pd.DataFrame(index=result.index)
    revisions["recent_forecast_revision"] = np.nan
    revisions["forecast_spread"] = np.nan
    revisions["model_disagreement"] = np.nan

    for row_index, row in result.iterrows():
        candidates = forecasts[
            (forecasts["location"] == row["location"])
            & (forecasts["target_date"] == row["target_date"])
            & (forecasts[issue_col] <= row["prediction_time"])
        ].sort_values(issue_col)
        if candidates.empty:
            continue

        highs = pd.to_numeric(candidates["forecast_high"], errors="coerce")
        if len(highs.dropna()) >= 2:
            revisions.at[row_index, "recent_forecast_revision"] = highs.iloc[-1] - highs.iloc[-2]
        if "forecast_source" in candidates.columns and candidates["forecast_source"].nunique() > 1:
            latest_issue = candidates[issue_col].max()
            latest = candidates[candidates[issue_col] == latest_issue]
            latest_highs = pd.to_numeric(latest["forecast_high"], errors="coerce")
            revisions.at[row_index, "forecast_spread"] = latest_highs.max() - latest_highs.min()
            revisions.at[row_index, "model_disagreement"] = latest_highs.std()

    for column in revisions.columns:
        result[column] = revisions[column]
    return result


def handle_missing_features(
    df: pd.DataFrame,
    missingness_output_path: str | Path | None = None,
) -> pd.DataFrame:
    result = df.copy()
    _merge_notes(result, df)

    missingness = pd.DataFrame(
        {
            "feature": list(result.columns),
            "missing_count": [int(result[column].isna().sum()) for column in result.columns],
            "missing_fraction": [
                float(result[column].isna().mean()) if len(result) else 0.0
                for column in result.columns
            ],
        }
    ).sort_values(["missing_fraction", "feature"], ascending=[False, True])

    if missingness_output_path is not None:
        output_path = Path(missingness_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        missingness.to_csv(output_path, index=False)

    present_critical = [column for column in CRITICAL_COLUMNS if column in result.columns]
    missing_critical = [column for column in CRITICAL_COLUMNS if column not in result.columns]
    if missing_critical:
        _append_note(result, f"Critical columns absent before dropping rows: {missing_critical}.")

    before = len(result)
    result = result.dropna(subset=present_critical).reset_index(drop=True)
    dropped = before - len(result)
    result.attrs["missingness_report"] = missingness
    result.attrs["dropped_critical_rows"] = int(dropped)
    result.attrs["critical_columns"] = present_critical
    if before:
        result.attrs["dropped_critical_fraction"] = float(dropped / before)
    else:
        result.attrs["dropped_critical_fraction"] = 0.0

    if dropped:
        _append_note(result, f"Dropped {dropped} rows missing critical Day 8 fields.")

    return result


def build_feature_matrix(
    inputs: dict[str, pd.DataFrame | list[str]],
    missingness_output_path: str | Path | None = None,
) -> pd.DataFrame:
    rows = inputs["rows"]
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("inputs['rows'] must be a DataFrame")

    result = rows.copy()
    result.attrs["feature_notes"] = list(inputs.get("notes", []))

    result = add_time_features(result)

    hourly = inputs.get("hourly", _empty_frame())
    result = add_observed_weather_features(
        result,
        hourly if isinstance(hourly, pd.DataFrame) else _empty_frame(),
    )

    hourly_forecasts = inputs.get("hourly_forecasts", _empty_frame())
    result = add_forecast_relative_features(
        result,
        hourly_forecasts if isinstance(hourly_forecasts, pd.DataFrame) else _empty_frame(),
    )

    result = add_solar_time_features(result)

    forecasts = inputs.get("forecasts", _empty_frame())
    result = add_forecast_update_features(
        result,
        forecasts if isinstance(forecasts, pd.DataFrame) else _empty_frame(),
    )

    result = handle_missing_features(result, missingness_output_path=missingness_output_path)
    sort_columns = [
        column
        for column in ["location", "target_date", "prediction_time"]
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns).reset_index(drop=True)
    return result


def _is_metadata_column(column: str) -> bool:
    lower = column.lower()
    return any(fragment in lower for fragment in METADATA_NAME_FRAGMENTS)


def _is_leakage_column(column: str) -> bool:
    lower = column.lower()
    return any(fragment == lower or fragment in lower for fragment in LEAKAGE_NAME_FRAGMENTS)


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_columns: list[str] = []
    excluded_columns: list[str] = []

    for column in df.columns:
        lower = column.lower()
        dtype = df[column].dtype
        should_exclude = (
            lower in EXCLUDED_FEATURE_COLUMNS
            or _is_metadata_column(lower)
            or _is_leakage_column(lower)
            or df[column].isna().all()
            or pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )
        if should_exclude:
            excluded_columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            feature_columns.append(column)
        else:
            excluded_columns.append(column)

    return feature_columns, excluded_columns


def write_feature_columns(df: pd.DataFrame, output_path: str | Path) -> dict[str, Any]:
    feature_columns, excluded_columns = select_feature_columns(df)
    spec: dict[str, Any] = {
        "target": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "notes": {
            "target_definition": "forecast_error = actual_high - forecast_high",
            "timestamp_rule": "All features must be available at or before prediction_time.",
            "forecast_horizon_hours": (
                "Hours from prediction_time until 3 PM local time on target_date."
            ),
            "feature_notes": list(df.attrs.get("feature_notes", [])),
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec
