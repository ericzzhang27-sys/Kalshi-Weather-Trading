from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import validate_feature_columns_exist  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "day8_features"

MODELING_ROWS_PATH = PROCESSED_DIR / "modeling_rows_v1.csv"
FEATURE_COLUMNS_PATH = OUTPUTS_DIR / "feature_columns.json"
FINAL_FEATURE_LIST_PATH = REPO_ROOT / "outputs" / "final_feature_list.json"
HOURLY_PATH = PROCESSED_DIR / "hourly_clean.csv"
HOURLY_FORECASTS_PATH = PROCESSED_DIR / "hourly_forecasts_clean.csv"
DAILY_PATH = PROCESSED_DIR / "daily_clean.csv"
FORECASTS_PATH = PROCESSED_DIR / "forecasts_clean.csv"
SUPERVISED_PATH = PROCESSED_DIR / "supervised_forecast_error_rows.csv"

REPORT_PATH = OUTPUTS_DIR / "feature_integrity_report.md"
CHECKS_CSV_PATH = OUTPUTS_DIR / "feature_integrity_checks.csv"

NUMERIC_TOLERANCE = 1e-8
DAILY_HIGH_TOLERANCE = 0.11
TEMP_CHANGE_WINDOWS_MINUTES = [60, 120, 180, 240, 300]


FEATURE_PROVENANCE = {
    "forecast_high": "forecasts_clean.forecast_high joined by date/location",
    "day_of_year_sin": "sin(2*pi*prediction_time.dayofyear/366)",
    "day_of_year_cos": "cos(2*pi*prediction_time.dayofyear/366)",
    "hour_sin": "sin(2*pi*fractional_prediction_hour/24)",
    "hour_cos": "cos(2*pi*fractional_prediction_hour/24)",
    "month": "prediction_time.month",
    "season": "encoded from prediction_time.month",
    "forecast_horizon_hours": "hours from prediction_time to 15:00 on target_date",
    "current_temp": "hourly_clean.temperature_2m at prediction_time",
    "dew_point": "hourly_clean.dew_point_2m at prediction_time",
    "cloud_cover_now": "hourly_clean.cloud_cover at prediction_time",
    "wind_speed": "hourly_clean.wind_speed_10m at prediction_time",
    "precipitation_now": "hourly_clean.precipitation at prediction_time",
    "temp_minus_dew_point": "current_temp - dew_point",
    "wind_dir_sin": "sin(hourly_clean.wind_direction_10m in radians)",
    "wind_dir_cos": "cos(hourly_clean.wind_direction_10m in radians)",
    "max_temp_so_far": "cumulative max of hourly_clean.temperature_2m within target_date",
    "temp_change_60m": "current_temp - observed temp 60 minutes earlier",
    "temp_change_120m": "current_temp - observed temp 120 minutes earlier",
    "temp_change_180m": "current_temp - observed temp 180 minutes earlier",
    "temp_change_240m": "current_temp - observed temp 240 minutes earlier",
    "temp_change_300m": "current_temp - observed temp 300 minutes earlier",
    "temp_acceleration_60m": "2*temp_change_60m - temp_change_120m",
    "temp_change_60m_minus_3h_avg_rate": "temp_change_60m - temp_change_180m/3",
    "forecast_temp_current_hour": "NDFD daily-high forecast_high when hourly NWS forecast temperature is unavailable",
    "current_temp_minus_forecast_temp": "current_temp - forecast_temp_current_hour",
    "forecast_max_so_far": "NDFD daily-high forecast_high when hourly NWS forecast temperature is unavailable",
    "max_so_far_minus_forecast_max_so_far": "max_temp_so_far - forecast_max_so_far",
    "current_temp_minus_max_so_far": "current_temp - max_temp_so_far",
    "minutes_since_max_temp_so_far": "prediction_time - latest timestamp of max_temp_so_far",
    "hour_of_max_temp_so_far": "hour of latest timestamp of max_temp_so_far",
    "max_so_far_minus_forecast_high": "max_temp_so_far - forecast_high",
    "mean_temp_error_so_far": "mean observed temp so far minus NDFD daily-high forecast when hourly NWS forecast temperature is unavailable",
    "max_temp_error_so_far": "max_temp_so_far - forecast_max_so_far",
    "num_new_highs_last_3h": "strict new observed highs in trailing (t - 3h, t] window",
    "temp_range_so_far": "max_temp_so_far - min_temp_so_far within target_date",
    "area_under_temp_curve_so_far": "cumulative hourly trapezoid integral of observed temp",
    "near_boundary_duration_so_far": "count so far where abs(temp - round(temp)) <= 0.5",
    "minutes_until_typical_peak": "minutes from prediction_time to 15:00 on target_date",
    "forecast_current_temp_gap_per_hour_to_peak": (
        "(forecast_high - current_temp) / hours from prediction_time to 15:00 on target_date"
    ),
    "needed_warming_rate_minus_recent_rate": (
        "forecast_current_temp_gap_per_hour_to_peak - temp_change_180m / 3"
    ),
}


def _read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    for column in parse_dates or []:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _normalize_target_date(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "target_date" in result.columns:
        result["target_date"] = pd.to_datetime(result["target_date"], errors="coerce").dt.normalize()
    elif "date" in result.columns:
        result["target_date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    return result


def _dedupe_time_keys(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in key_columns if column not in df.columns]
    if df.empty or missing:
        return df
    return (
        df.sort_values(key_columns)
        .drop_duplicates(key_columns, keep="last")
        .reset_index(drop=True)
    )


def _is_openmeteo_forecast_frame(df: pd.DataFrame) -> bool:
    if df.empty or "forecast_source" not in df.columns:
        return False
    sources = df["forecast_source"].dropna().astype(str).str.lower().unique()
    return len(sources) > 0 and all("open_meteo" in source for source in sources)


def _group_key(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value
    return value


def _latest_at_or_before_expected(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    by: list[str],
    left_time_col: str,
    right_time_col: str,
    value_cols: list[str],
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(index=left.index)
    for column in value_cols:
        if column in right.columns and pd.api.types.is_datetime64_any_dtype(right[column]):
            result[column] = pd.NaT
        else:
            result[column] = np.nan

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
        valid_right = right_times.notna()
        group = group.loc[valid_right].copy()
        right_times = right_times.loc[valid_right].astype("datetime64[ns]")
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

    return result


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    affected_rows: int = 0,
    columns: list[str] | None = None,
    details: str = "",
) -> None:
    checks.append(
        {
            "check": name,
            "status": status,
            "affected_rows": int(affected_rows),
            "columns": ", ".join(columns or []),
            "details": details,
        },
    )


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["status"] == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def _mismatch_count(actual: pd.Series, expected: pd.Series, tolerance: float = NUMERIC_TOLERANCE) -> int:
    actual_num = pd.to_numeric(actual, errors="coerce")
    expected_num = pd.to_numeric(expected, errors="coerce")
    both_missing = actual_num.isna() & expected_num.isna()
    close = np.isclose(
        actual_num.fillna(0.0),
        expected_num.fillna(0.0),
        rtol=0.0,
        atol=tolerance,
    )
    return int((~both_missing & ~close).sum())


def _datetime_mismatch_count(actual: pd.Series, expected: pd.Series) -> int:
    actual_dt = pd.to_datetime(actual, errors="coerce")
    expected_dt = pd.to_datetime(expected, errors="coerce")
    both_missing = actual_dt.isna() & expected_dt.isna()
    equal = actual_dt.eq(expected_dt)
    return int((~both_missing & ~equal).sum())


def _feature_check(
    checks: list[dict[str, Any]],
    feature: str,
    actual: pd.Series,
    expected: pd.Series,
    tolerance: float = NUMERIC_TOLERANCE,
) -> None:
    mismatches = _mismatch_count(actual, expected, tolerance=tolerance)
    _add_check(
        checks,
        f"Feature formula: {feature}",
        "PASS" if mismatches == 0 else "FAIL",
        mismatches,
        [feature],
        FEATURE_PROVENANCE.get(feature, "feature formula"),
    )


def _build_observed_expected(hourly: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in hourly.sort_values("timestamp").groupby(["location", "target_date"], dropna=False):
        current_max = -np.inf
        current_min = np.inf
        max_source = pd.NaT
        near_count = 0
        area = 0.0
        temp_sum = 0.0
        observed_count = 0
        prev_time: pd.Timestamp | None = None
        prev_temp: float | None = None

        rows: list[dict[str, Any]] = []
        new_high_flags: list[int] = []
        timestamps: list[pd.Timestamp] = []

        for _, row in group.iterrows():
            timestamp = row["timestamp"]
            temp = row["temperature_2m"]
            temp_float = float(temp) if pd.notna(temp) else np.nan

            if prev_time is not None and prev_temp is not None and pd.notna(temp_float):
                elapsed_hours = (timestamp - prev_time).total_seconds() / 3600.0
                if elapsed_hours >= 0:
                    area += ((prev_temp + temp_float) / 2.0) * elapsed_hours

            strict_new_high = 0
            if pd.notna(temp_float):
                observed_count += 1
                temp_sum += temp_float
                if temp_float > current_max:
                    strict_new_high = 1
                if temp_float >= current_max:
                    current_max = temp_float
                    max_source = timestamp
                if temp_float < current_min:
                    current_min = temp_float
                if abs(temp_float - round(temp_float)) <= 0.5:
                    near_count += 1

            timestamps.append(timestamp)
            new_high_flags.append(strict_new_high)

            max_value = np.nan if current_max == -np.inf else current_max
            min_value = np.nan if current_min == np.inf else current_min
            rows.append(
                {
                    "location": row["location"],
                    "target_date": row["target_date"],
                    "prediction_time": timestamp,
                    "expected_current_temp": temp_float,
                    "expected_dew_point": row["dew_point_2m"],
                    "expected_cloud_cover_now": row["cloud_cover"],
                    "expected_wind_speed": row["wind_speed_10m"],
                    "expected_precipitation_now": row["precipitation"],
                    "expected_wind_dir_sin": np.sin(np.deg2rad(row["wind_direction_10m"])),
                    "expected_wind_dir_cos": np.cos(np.deg2rad(row["wind_direction_10m"])),
                    "expected_max_temp_so_far": max_value,
                    "expected_max_temp_so_far_source_time": max_source,
                    "expected_current_temp_minus_max_so_far": temp_float - max_value,
                    "expected_minutes_since_max_temp_so_far": (
                        timestamp - max_source
                    ).total_seconds()
                    / 60.0,
                    "expected_hour_of_max_temp_so_far": float(max_source.hour),
                    "expected_temp_range_so_far": max_value - min_value,
                    "expected_area_under_temp_curve_so_far": area,
                    "expected_near_boundary_duration_so_far": near_count,
                    "expected_mean_temp_so_far": temp_sum / observed_count if observed_count else np.nan,
                },
            )

            prev_time = timestamp
            prev_temp = temp_float if pd.notna(temp_float) else None

        for i, row in enumerate(rows):
            window_start = row["prediction_time"] - pd.Timedelta(hours=3)
            count = sum(
                flag
                for timestamp, flag in zip(timestamps, new_high_flags, strict=False)
                if timestamp > window_start and timestamp <= row["prediction_time"]
            )
            row["expected_num_new_highs_last_3h"] = float(count)

        pieces.append(pd.DataFrame(rows))

    return pd.concat(pieces, ignore_index=True)


def _build_forecast_expected(hourly_forecasts: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in hourly_forecasts.sort_values("timestamp").groupby(
        ["location", "target_date"],
        dropna=False,
    ):
        current_max = -np.inf
        max_source = pd.NaT
        rows: list[dict[str, Any]] = []
        for _, row in group.iterrows():
            timestamp = row["timestamp"]
            temp = row["temperature_2m"]
            temp_float = float(temp) if pd.notna(temp) else np.nan
            if pd.notna(temp_float) and temp_float >= current_max:
                current_max = temp_float
                max_source = timestamp
            max_value = np.nan if current_max == -np.inf else current_max
            rows.append(
                {
                    "location": row["location"],
                    "target_date": row["target_date"],
                    "prediction_time": timestamp,
                    "expected_forecast_temp_current_hour": temp_float,
                    "expected_forecast_temp_source_valid_time": timestamp,
                    "expected_forecast_max_so_far": max_value,
                    "expected_forecast_max_so_far_source_valid_time": max_source,
                },
            )
        pieces.append(pd.DataFrame(rows))
    return pd.concat(pieces, ignore_index=True)


def _add_temp_change_expectations(expected: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    result = expected.copy()
    lookup = hourly.loc[:, ["location", "timestamp", "temperature_2m"]].rename(
        columns={"temperature_2m": "lookup_temp"},
    )
    for minutes in TEMP_CHANGE_WINDOWS_MINUTES:
        lookup_col = f"_lookup_{minutes}m"
        result[lookup_col] = result["prediction_time"] - pd.Timedelta(minutes=minutes)
        lookback = _latest_at_or_before_expected(
            result,
            lookup,
            by=["location"],
            left_time_col=lookup_col,
            right_time_col="timestamp",
            value_cols=["lookup_temp"],
            tolerance=pd.Timedelta(minutes=45),
        )
        result[f"_lookup_temp_{minutes}m"] = lookback["lookup_temp"]
        result[f"expected_temp_change_{minutes}m"] = (
            result["expected_current_temp"] - result[f"_lookup_temp_{minutes}m"]
        )
        result = result.drop(columns=[lookup_col, f"_lookup_temp_{minutes}m"])

    result["expected_temp_acceleration_60m"] = (
        2.0 * result["expected_temp_change_60m"] - result["expected_temp_change_120m"]
    )
    result["expected_temp_change_60m_minus_3h_avg_rate"] = (
        result["expected_temp_change_60m"] - result["expected_temp_change_180m"] / 3.0
    )
    return result


def _add_time_expectations(expected: pd.DataFrame) -> pd.DataFrame:
    result = expected.copy()
    prediction_time = result["prediction_time"]
    target_date = result["target_date"]
    day_of_year = prediction_time.dt.dayofyear.astype(float)
    hour_fraction = (
        prediction_time.dt.hour.astype(float)
        + prediction_time.dt.minute.astype(float) / 60.0
        + prediction_time.dt.second.astype(float) / 3600.0
    )
    month = prediction_time.dt.month.astype(int)
    result["expected_day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    result["expected_day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    result["expected_hour_sin"] = np.sin(2.0 * np.pi * hour_fraction / 24.0)
    result["expected_hour_cos"] = np.cos(2.0 * np.pi * hour_fraction / 24.0)
    result["expected_month"] = month.astype(float)
    result["expected_season"] = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        [0, 1, 2, 3],
    ).astype(float)
    peak_time = target_date + pd.to_timedelta(15, unit="h")
    result["expected_forecast_horizon_hours"] = (
        peak_time - prediction_time
    ).dt.total_seconds() / 3600.0
    result["expected_minutes_until_typical_peak"] = (
        peak_time - prediction_time
    ).dt.total_seconds() / 60.0
    return result


def _write_report(checks: list[dict[str, Any]], feature_columns: list[str]) -> None:
    check_frame = pd.DataFrame(checks)
    CHECKS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    check_frame.to_csv(CHECKS_CSV_PATH, index=False)

    status = _overall_status(checks)
    counts = check_frame["status"].value_counts().to_dict()
    failed = check_frame[check_frame["status"] == "FAIL"]
    warned = check_frame[check_frame["status"] == "WARN"]

    lines = [
        "# Feature Integrity Report",
        "",
        f"Overall status: **{status}**",
        "",
        "## Summary",
        "",
        f"- Feature columns checked: {len(feature_columns)}",
        f"- PASS checks: {counts.get('PASS', 0)}",
        f"- WARN checks: {counts.get('WARN', 0)}",
        f"- FAIL checks: {counts.get('FAIL', 0)}",
        f"- Detailed checks CSV: `{CHECKS_CSV_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Important Limitations",
        "",
        "- This verifies internal provenance and formulas against the local processed source tables.",
        "- It does not prove NWS/NDFD grid forecasts equal Kalshi-visible forecasts or market-visible quotes.",
        "- NDFD forecast issue timestamps are checked for the daily-high baseline; the current archive does not provide an hourly forecast-temperature path.",
        "- `near_boundary_duration_so_far` follows the requested formula `abs(temp - round(temp)) <= 0.5`; this threshold counts every non-missing numeric hourly temperature.",
        "",
    ]

    if not failed.empty:
        lines.extend(["## Failed Checks", ""])
        for row in failed.to_dict("records"):
            lines.append(
                f"- {row['check']}: {row['affected_rows']} affected rows; {row['details']}"
            )
        lines.append("")

    if not warned.empty:
        lines.extend(["## Warnings", ""])
        for row in warned.to_dict("records"):
            lines.append(
                f"- {row['check']}: {row['affected_rows']} affected rows; {row['details']}"
            )
        lines.append("")

    lines.extend(["## Feature Provenance", ""])
    for feature in feature_columns:
        lines.append(f"- `{feature}`: {FEATURE_PROVENANCE.get(feature, 'not documented')}")
    lines.append("")

    lines.extend(["## Check Results", ""])
    lines.append("| Check | Status | Affected rows | Columns | Details |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for row in check_frame.to_dict("records"):
        details = str(row["details"]).replace("|", "\\|")
        lines.append(
            f"| {row['check']} | {row['status']} | {row['affected_rows']} | "
            f"{row['columns']} | {details} |"
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    checks: list[dict[str, Any]] = []

    rows = _read_csv(
        MODELING_ROWS_PATH,
        parse_dates=[
            "target_date",
            "prediction_time",
            "prediction_timestamp",
            "current_temp_source_time",
            "max_temp_so_far_source_time",
            "forecast_temp_source_valid_time",
            "forecast_max_so_far_source_valid_time",
        ],
    )
    hourly = _normalize_target_date(_read_csv(HOURLY_PATH, parse_dates=["timestamp"]))
    hourly_forecasts = _normalize_target_date(
        _read_csv(HOURLY_FORECASTS_PATH, parse_dates=["timestamp"]),
    )
    daily = _normalize_target_date(_read_csv(DAILY_PATH))
    forecasts = _normalize_target_date(_read_csv(FORECASTS_PATH))
    supervised = _normalize_target_date(
        _read_csv(SUPERVISED_PATH, parse_dates=["prediction_timestamp"]),
    )
    with FEATURE_COLUMNS_PATH.open(encoding="utf-8") as handle:
        feature_spec = json.load(handle)
    feature_columns = list(feature_spec["feature_columns"])
    hourly_forecasts_are_openmeteo = _is_openmeteo_forecast_frame(hourly_forecasts)
    if hourly_forecasts_are_openmeteo:
        hourly_forecasts = pd.DataFrame(columns=hourly_forecasts.columns)

    required_schema = {
        "modeling_rows_v1": {
            "df": rows,
            "columns": ["location", "target_date", "prediction_time", "actual_high", "forecast_high", "forecast_error"],
        },
        "hourly_clean": {
            "df": hourly,
            "columns": [
                "location",
                "target_date",
                "timestamp",
                "temperature_2m",
                "dew_point_2m",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
            ],
        },
        "hourly_forecasts_clean": {
            "df": hourly_forecasts,
            "columns": ["location", "target_date", "timestamp", "temperature_2m"],
        },
        "daily_clean": {"df": daily, "columns": ["location", "target_date", "actual_high"]},
        "forecasts_clean": {"df": forecasts, "columns": ["location", "target_date", "forecast_high"]},
        "supervised_forecast_error_rows": {
            "df": supervised,
            "columns": ["location", "target_date", "prediction_time", "prediction_timestamp"],
        },
    }
    for label, payload in required_schema.items():
        missing_columns = [column for column in payload["columns"] if column not in payload["df"].columns]
        _add_check(
            checks,
            f"Required schema: {label}",
            "FAIL" if missing_columns else "PASS",
            len(payload["df"]) if missing_columns else 0,
            missing_columns,
            "All required source columns are present." if not missing_columns else "Missing source columns.",
        )

    missing_features = [feature for feature in feature_columns if feature not in rows.columns]
    _add_check(
        checks,
        "Feature spec columns exist in modeling_rows_v1",
        "FAIL" if missing_features else "PASS",
        len(rows) if missing_features else 0,
        missing_features,
        "feature_columns.json is aligned to modeling_rows_v1.csv.",
    )

    if FINAL_FEATURE_LIST_PATH.exists():
        with FINAL_FEATURE_LIST_PATH.open(encoding="utf-8") as handle:
            final_payload = json.load(handle)
        final_features = list(final_payload.get("features", final_payload.get("feature_columns", [])))
        final_status = "PASS"
        final_details = "Final feature list columns are present, numeric, model-safe, and not entirely missing."
        try:
            validate_feature_columns_exist(rows, final_features)
        except (TypeError, ValueError) as exc:
            final_status = "FAIL"
            final_details = str(exc)
        _add_check(
            checks,
            "Final feature list is populated and model-safe",
            final_status,
            len(rows) if final_status == "FAIL" else 0,
            final_features,
            final_details,
        )

    forbidden_features = sorted({"forecast_error", "actual_high", "target"}.intersection(feature_columns))
    _add_check(
        checks,
        "Feature spec excludes target and final actuals",
        "FAIL" if forbidden_features else "PASS",
        len(rows) if forbidden_features else 0,
        forbidden_features,
        "Target/audit columns are not model features.",
    )

    unsafe_dtype_features = [
        feature
        for feature in feature_columns
        if feature in rows.columns
        and not (
            pd.api.types.is_numeric_dtype(rows[feature])
            or pd.api.types.is_bool_dtype(rows[feature])
        )
    ]
    _add_check(
        checks,
        "Feature spec columns are numeric",
        "FAIL" if unsafe_dtype_features else "PASS",
        len(rows) if unsafe_dtype_features else 0,
        unsafe_dtype_features,
        "All feature columns are numeric/model-safe.",
    )

    key_duplicate_count = int(rows.duplicated(["location", "target_date", "prediction_time"]).sum())
    per_day_counts = rows.groupby(["location", "target_date"]).size()
    wrong_count_days = int((per_day_counts != 24).sum())
    _add_check(
        checks,
        "Modeling row keys and hourly coverage",
        "FAIL" if key_duplicate_count else ("WARN" if wrong_count_days else "PASS"),
        key_duplicate_count + wrong_count_days,
        ["location", "target_date", "prediction_time"],
        (
            "Rows are unique; some location/date groups have fewer than 24 rows after "
            "dropping rows with missing critical weather fields."
        )
        if wrong_count_days and not key_duplicate_count
        else "Rows are unique and every location/date has 24 hourly prediction rows.",
    )

    expected_rows = len(supervised)
    _add_check(
        checks,
        "Modeling rows match supervised rows",
        "WARN" if len(rows) != expected_rows else "PASS",
        abs(len(rows) - expected_rows),
        ["modeling_rows_v1", "supervised_forecast_error_rows"],
        (
            f"modeling rows={len(rows)}, supervised rows={expected_rows}; "
            "feature builder drops rows missing critical weather/model fields."
        ),
    )

    target_mismatches = _mismatch_count(
        rows["forecast_error"],
        rows["actual_high"] - rows["forecast_high"],
    )
    _add_check(
        checks,
        "Target formula forecast_error = actual_high - forecast_high",
        "FAIL" if target_mismatches else "PASS",
        target_mismatches,
        ["forecast_error", "actual_high", "forecast_high"],
        "Target math matches stored columns.",
    )

    hourly_daily_highs = (
        hourly.groupby(["location", "target_date"], as_index=False)["temperature_2m"]
        .max()
        .rename(columns={"temperature_2m": "expected_actual_high"})
    )
    daily_compare = daily.merge(hourly_daily_highs, on=["location", "target_date"], how="left")
    daily_high_mismatches = _mismatch_count(
        daily_compare["actual_high"],
        daily_compare["expected_actual_high"],
        tolerance=DAILY_HIGH_TOLERANCE,
    )
    _add_check(
        checks,
        "daily_clean.actual_high matches max hourly_clean.temperature_2m",
        "WARN" if daily_high_mismatches else "PASS",
        daily_high_mismatches,
        ["actual_high", "temperature_2m"],
        (
            f"Tolerance: {DAILY_HIGH_TOLERANCE} F. ASOS hourly/special reports can "
            "disagree with the official daily TMAX climate product."
        ),
    )

    if hourly_forecasts.empty:
        _add_check(
            checks,
            "Hourly forecast source availability",
            "WARN",
            0,
            ["forecast_high", "temperature_2m"],
            (
                "hourly_forecasts_clean.csv is Open-Meteo proxy data and is ignored; "
                "forecast-relative columns use timestamp-safe NDFD daily-high fallback values."
            )
            if hourly_forecasts_are_openmeteo
            else "Skipped because hourly_forecasts_clean.csv is empty.",
        )
    else:
        hourly_forecast_highs = (
            hourly_forecasts.groupby(["location", "target_date"], as_index=False)["temperature_2m"]
            .max()
            .rename(columns={"temperature_2m": "expected_forecast_high"})
        )
        forecast_compare = forecasts.merge(hourly_forecast_highs, on=["location", "target_date"], how="left")
        forecast_high_mismatches = _mismatch_count(
            forecast_compare["forecast_high"],
            forecast_compare["expected_forecast_high"],
            tolerance=DAILY_HIGH_TOLERANCE,
        )
        _add_check(
            checks,
            "forecasts_clean.forecast_high matches max hourly_forecasts_clean.temperature_2m",
            "FAIL" if forecast_high_mismatches else "PASS",
            forecast_high_mismatches,
            ["forecast_high", "temperature_2m"],
            f"Tolerance: {DAILY_HIGH_TOLERANCE} F.",
        )

    expected = rows.loc[:, ["location", "target_date", "prediction_time"]].copy()
    observed_expected = _build_observed_expected(hourly)
    current_expected_cols = [
        "expected_current_temp",
        "expected_dew_point",
        "expected_cloud_cover_now",
        "expected_wind_speed",
        "expected_precipitation_now",
        "expected_wind_dir_sin",
        "expected_wind_dir_cos",
    ]
    current_expected = _latest_at_or_before_expected(
        expected,
        observed_expected,
        by=["location"],
        left_time_col="prediction_time",
        right_time_col="prediction_time",
        value_cols=current_expected_cols,
        tolerance=pd.Timedelta(hours=3),
    )
    for column in current_expected.columns:
        expected[column] = current_expected[column]

    cumulative_expected_cols = [
        "expected_max_temp_so_far",
        "expected_max_temp_so_far_source_time",
        "expected_current_temp_minus_max_so_far",
        "expected_minutes_since_max_temp_so_far",
        "expected_hour_of_max_temp_so_far",
        "expected_temp_range_so_far",
        "expected_area_under_temp_curve_so_far",
        "expected_near_boundary_duration_so_far",
        "expected_mean_temp_so_far",
        "expected_num_new_highs_last_3h",
    ]
    cumulative_expected = _latest_at_or_before_expected(
        expected,
        observed_expected,
        by=["location", "target_date"],
        left_time_col="prediction_time",
        right_time_col="prediction_time",
        value_cols=cumulative_expected_cols,
        tolerance=pd.Timedelta(hours=24),
    )
    for column in cumulative_expected.columns:
        expected[column] = cumulative_expected[column]
    expected["expected_current_temp_minus_max_so_far"] = (
        expected["expected_current_temp"] - expected["expected_max_temp_so_far"]
    )
    max_source_time = pd.to_datetime(expected["expected_max_temp_so_far_source_time"], errors="coerce")
    expected["expected_minutes_since_max_temp_so_far"] = (
        expected["prediction_time"] - max_source_time
    ).dt.total_seconds() / 60.0
    expected["expected_hour_of_max_temp_so_far"] = max_source_time.dt.hour.astype(float)

    expected = _add_temp_change_expectations(expected, hourly)

    forecast_expected = (
        _build_forecast_expected(hourly_forecasts)
        if not hourly_forecasts.empty
        else pd.DataFrame(
            columns=[
                "location",
                "target_date",
                "prediction_time",
                "expected_forecast_temp_current_hour",
                "expected_forecast_temp_source_valid_time",
                "expected_forecast_max_so_far",
                "expected_forecast_max_so_far_source_valid_time",
            ],
        )
    )
    aligned_forecast_expected = _latest_at_or_before_expected(
        expected,
        forecast_expected,
        by=["location", "target_date"],
        left_time_col="prediction_time",
        right_time_col="prediction_time",
        value_cols=[
            "expected_forecast_temp_current_hour",
            "expected_forecast_temp_source_valid_time",
            "expected_forecast_max_so_far",
            "expected_forecast_max_so_far_source_valid_time",
        ],
        tolerance=pd.Timedelta(hours=24),
    )
    for column in aligned_forecast_expected.columns:
        expected[column] = aligned_forecast_expected[column]
    if hourly_forecasts.empty:
        expected["expected_forecast_temp_current_hour"] = rows["forecast_high"].to_numpy()
        expected["expected_forecast_max_so_far"] = rows["forecast_high"].to_numpy()
        if "forecast_issue_time" in rows.columns:
            expected["expected_forecast_temp_source_issue_time"] = rows["forecast_issue_time"].to_numpy()
        if "ndfd_valid_time_utc" in rows.columns:
            expected["expected_forecast_temp_source_valid_time"] = rows["ndfd_valid_time_utc"].to_numpy()
            expected["expected_forecast_max_so_far_source_valid_time"] = rows["ndfd_valid_time_utc"].to_numpy()
    if hourly_forecasts.empty:
        expected["expected_mean_temp_error_so_far"] = (
            expected["expected_mean_temp_so_far"] - rows["forecast_high"]
        )
    else:
        error_source = hourly.loc[
            :,
            ["location", "target_date", "timestamp", "temperature_2m"],
        ].merge(
            hourly_forecasts.loc[
                :,
                ["location", "target_date", "timestamp", "temperature_2m"],
            ].rename(columns={"temperature_2m": "forecast_temperature_2m"}),
            on=["location", "target_date", "timestamp"],
            how="left",
            validate="one_to_one",
        )
        error_source["hourly_temp_error"] = (
            error_source["temperature_2m"] - error_source["forecast_temperature_2m"]
        )
        error_source = error_source.sort_values(["location", "target_date", "timestamp"])
        error_source["expected_mean_temp_error_so_far"] = (
            error_source.groupby(["location", "target_date"], dropna=False)["hourly_temp_error"]
            .expanding()
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        expected = expected.merge(
            error_source.loc[
                :,
                ["location", "target_date", "timestamp", "expected_mean_temp_error_so_far"],
            ].rename(columns={"timestamp": "prediction_time"}),
            on=["location", "target_date", "prediction_time"],
            how="left",
            validate="one_to_one",
        )
    expected = _add_time_expectations(expected)

    expected["expected_forecast_high"] = rows["forecast_high"]
    expected["expected_temp_minus_dew_point"] = expected["expected_current_temp"] - expected["expected_dew_point"]
    expected["expected_current_temp_minus_forecast_temp"] = (
        expected["expected_current_temp"] - expected["expected_forecast_temp_current_hour"]
    )
    expected["expected_max_so_far_minus_forecast_max_so_far"] = (
        expected["expected_max_temp_so_far"] - expected["expected_forecast_max_so_far"]
    )
    expected["expected_max_temp_error_so_far"] = (
        expected["expected_max_temp_so_far"] - expected["expected_forecast_max_so_far"]
    )
    expected["expected_max_so_far_minus_forecast_high"] = (
        expected["expected_max_temp_so_far"] - rows["forecast_high"]
    )
    hours_until_peak = expected["expected_minutes_until_typical_peak"] / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_gap_rate = (rows["forecast_high"] - expected["expected_current_temp"]) / hours_until_peak
    expected["expected_forecast_current_temp_gap_per_hour_to_peak"] = pd.Series(
        expected_gap_rate,
        index=expected.index,
        dtype=float,
    ).where(hours_until_peak.abs() > 1e-9)
    expected["expected_needed_warming_rate_minus_recent_rate"] = (
        expected["expected_forecast_current_temp_gap_per_hour_to_peak"]
        - expected["expected_temp_change_180m"] / 3.0
    )

    feature_to_expected = {
        "day_of_year_sin": "expected_day_of_year_sin",
        "day_of_year_cos": "expected_day_of_year_cos",
        "hour_sin": "expected_hour_sin",
        "hour_cos": "expected_hour_cos",
        "month": "expected_month",
        "season": "expected_season",
        "forecast_horizon_hours": "expected_forecast_horizon_hours",
        "current_temp": "expected_current_temp",
        "dew_point": "expected_dew_point",
        "cloud_cover_now": "expected_cloud_cover_now",
        "wind_speed": "expected_wind_speed",
        "precipitation_now": "expected_precipitation_now",
        "temp_minus_dew_point": "expected_temp_minus_dew_point",
        "wind_dir_sin": "expected_wind_dir_sin",
        "wind_dir_cos": "expected_wind_dir_cos",
        "max_temp_so_far": "expected_max_temp_so_far",
        "forecast_temp_current_hour": "expected_forecast_temp_current_hour",
        "current_temp_minus_forecast_temp": "expected_current_temp_minus_forecast_temp",
        "forecast_max_so_far": "expected_forecast_max_so_far",
        "max_so_far_minus_forecast_max_so_far": "expected_max_so_far_minus_forecast_max_so_far",
        "current_temp_minus_max_so_far": "expected_current_temp_minus_max_so_far",
        "minutes_since_max_temp_so_far": "expected_minutes_since_max_temp_so_far",
        "hour_of_max_temp_so_far": "expected_hour_of_max_temp_so_far",
        "max_so_far_minus_forecast_high": "expected_max_so_far_minus_forecast_high",
        "mean_temp_error_so_far": "expected_mean_temp_error_so_far",
        "max_temp_error_so_far": "expected_max_temp_error_so_far",
        "num_new_highs_last_3h": "expected_num_new_highs_last_3h",
        "temp_range_so_far": "expected_temp_range_so_far",
        "area_under_temp_curve_so_far": "expected_area_under_temp_curve_so_far",
        "near_boundary_duration_so_far": "expected_near_boundary_duration_so_far",
        "minutes_until_typical_peak": "expected_minutes_until_typical_peak",
        "forecast_current_temp_gap_per_hour_to_peak": (
            "expected_forecast_current_temp_gap_per_hour_to_peak"
        ),
        "needed_warming_rate_minus_recent_rate": (
            "expected_needed_warming_rate_minus_recent_rate"
        ),
    }
    for minutes in TEMP_CHANGE_WINDOWS_MINUTES:
        feature_to_expected[f"temp_change_{minutes}m"] = f"expected_temp_change_{minutes}m"
    feature_to_expected["temp_acceleration_60m"] = "expected_temp_acceleration_60m"
    feature_to_expected["temp_change_60m_minus_3h_avg_rate"] = (
        "expected_temp_change_60m_minus_3h_avg_rate"
    )

    for feature in feature_columns:
        expected_column = feature_to_expected.get(feature)
        if expected_column is None:
            if feature == "forecast_high":
                if "nws_forecast_high_f" in rows.columns:
                    _feature_check(checks, feature, rows["forecast_high"], rows["nws_forecast_high_f"])
                else:
                    _add_check(
                        checks,
                        "Feature formula: forecast_high",
                        "WARN",
                        0,
                        ["forecast_high"],
                        "No row-level forecast_high source column is available for the NDFD contract.",
                    )
            else:
                _add_check(
                    checks,
                    f"Feature formula: {feature}",
                    "WARN",
                    0,
                    [feature],
                    "No independent formula check is registered for this feature.",
                )
            continue
        _feature_check(checks, feature, rows[feature], expected[expected_column])

    source_time_checks = {
        column: rows[column]
        for column in [
            "current_temp_source_time",
            "max_temp_so_far_source_time",
            "forecast_temp_source_valid_time",
            "forecast_max_so_far_source_valid_time",
        ]
        if column in rows.columns
    }
    for column, source_time in source_time_checks.items():
        violations = pd.to_datetime(source_time, errors="coerce") > rows["prediction_time"]
        _add_check(
            checks,
            f"Timestamp safety: {column} <= prediction_time",
            "FAIL" if violations.any() else "PASS",
            int(violations.sum()),
            [column, "prediction_time"],
            "Source/valid timestamps are not after prediction_time.",
        )

    _add_check(
        checks,
        "near_boundary_duration_so_far interpretation",
        "WARN",
        0,
        ["near_boundary_duration_so_far"],
        "The requested abs(temp - round(temp)) <= 0.5 formula counts every non-missing numeric hourly temperature.",
    )

    _write_report(checks, feature_columns)
    status = _overall_status(checks)
    print(f"Feature integrity verification complete: {status}")
    print(f"Rows verified: {len(rows):,}")
    print(f"Feature columns verified: {len(feature_columns):,}")
    print(f"Report: {REPORT_PATH}")
    print(f"Checks CSV: {CHECKS_CSV_PATH}")

    if status == "FAIL":
        failed = pd.DataFrame(checks)
        print(failed[failed["status"] == "FAIL"].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
