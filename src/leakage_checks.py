from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


LEAKAGE_FEATURE_NAMES = {
    "forecast_error",
    "actual_high",
    "official_high",
    "final_high",
    "daily_high",
    "observed_daily_high",
    "actual_max_temp",
    "max_temp_today",
    "target",
}

SOURCE_TIMESTAMP_COLUMNS = [
    "current_temp_source_time",
    "max_temp_so_far_source_time",
    "forecast_source_issue_time",
    "forecast_temp_source_issue_time",
    "next_3h_forecast_source_issue_time",
]

VALID_TIMESTAMP_COLUMNS = [
    "forecast_temp_source_valid_time",
    "forecast_max_so_far_source_valid_time",
]

METADATA_FRAGMENTS = [
    "source_time",
    "issue_time",
    "valid_time",
    "timestamp",
    "created_at",
    "reference_time",
    "run_time",
    "as_of",
]


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(status, 2)


def _overall_status(items: list[dict[str, Any]]) -> str:
    if not items:
        return "PASS"
    return max((str(item["status"]) for item in items), key=_status_rank)


def _check_item(
    name: str,
    status: str,
    affected_rows: int = 0,
    affected_columns: list[str] | None = None,
    explanation: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "affected_rows": int(affected_rows),
        "affected_columns": affected_columns or [],
        "explanation": explanation,
    }


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _metadata_feature_columns(feature_columns: list[str]) -> list[str]:
    flagged: list[str] = []
    for column in feature_columns:
        lower = column.lower()
        if any(fragment in lower for fragment in METADATA_FRAGMENTS):
            flagged.append(column)
    return flagged


def run_leakage_checks(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    feature_name_map = {column.lower(): column for column in feature_columns}
    leaked = [
        feature_name_map[name]
        for name in LEAKAGE_FEATURE_NAMES
        if name in feature_name_map
    ]
    checks.append(
        _check_item(
            "Target leakage check",
            "FAIL" if leaked else "PASS",
            affected_rows=len(df) if leaked else 0,
            affected_columns=leaked,
            explanation=(
                "Feature columns include target/final actual high fields."
                if leaked
                else "No target or final actual high columns are included as features."
            ),
        )
    )

    timestamp_fail_columns: list[str] = []
    timestamp_warn_columns: list[str] = []
    timestamp_fail_rows = 0
    if "prediction_time" not in df.columns:
        checks.append(
            _check_item(
                "Future timestamp check",
                "FAIL",
                affected_rows=len(df),
                affected_columns=["prediction_time"],
                explanation="prediction_time is missing, so source timestamp safety cannot be verified.",
            )
        )
    else:
        prediction_time = _parse_datetime(df["prediction_time"])
        for column in SOURCE_TIMESTAMP_COLUMNS:
            if column not in df.columns:
                if column.startswith("forecast") or column.startswith("next_3h"):
                    timestamp_warn_columns.append(column)
                continue
            source_time = _parse_datetime(df[column])
            comparable = source_time.notna() & prediction_time.notna()
            violations = comparable & (source_time > prediction_time)
            if violations.any():
                timestamp_fail_columns.append(column)
                timestamp_fail_rows += int(violations.sum())

        for column in VALID_TIMESTAMP_COLUMNS:
            if column not in df.columns:
                continue
            valid_time = _parse_datetime(df[column])
            comparable = valid_time.notna() & prediction_time.notna()
            violations = comparable & (valid_time > prediction_time)
            if violations.any():
                timestamp_fail_columns.append(column)
                timestamp_fail_rows += int(violations.sum())

        if timestamp_fail_columns:
            status = "FAIL"
            explanation = "One or more source/valid timestamps occur after prediction_time."
        elif timestamp_warn_columns:
            status = "WARN"
            explanation = (
                "Available source timestamps are not after prediction_time, but forecast "
                "issue/run timestamp columns are missing, so forecast issuance safety is "
                "documented as a data limitation."
            )
        else:
            status = "PASS"
            explanation = "All available source timestamps are at or before prediction_time."

        checks.append(
            _check_item(
                "Future timestamp check",
                status,
                affected_rows=timestamp_fail_rows,
                affected_columns=timestamp_fail_columns + timestamp_warn_columns,
                explanation=explanation,
            )
        )

    if {"max_temp_so_far", "actual_high"}.issubset(df.columns):
        max_so_far = pd.to_numeric(df["max_temp_so_far"], errors="coerce")
        actual_high = pd.to_numeric(df["actual_high"], errors="coerce")
        comparable = max_so_far.notna() & actual_high.notna()
        violations = comparable & (max_so_far > actual_high + 0.5)
        uses_asos = (
            "observed_temperature_source" in df.columns
            and set(df["observed_temperature_source"].dropna().astype(str).unique()) == {"iem_nws_asos"}
        )
        status = "PASS"
        if violations.any():
            status = "WARN" if uses_asos else "FAIL"
        checks.append(
            _check_item(
                "Max-so-far sanity check",
                status,
                affected_rows=int(violations.sum()),
                affected_columns=["max_temp_so_far", "actual_high"] if violations.any() else [],
                explanation=(
                    "max_temp_so_far exceeds official daily actual_high for some rows. "
                    "Because observed_temperature_source is IEM/NWS ASOS, this is treated "
                    "as a source-disagreement warning: hourly/special ASOS reports can differ "
                    "from the official daily TMAX climate product. Future timestamp checks still "
                    "guard leakage."
                    if violations.any() and uses_asos
                    else (
                        "max_temp_so_far exceeds actual_high beyond tolerance. Possible causes: "
                        "unit mismatch, wrong actual_high column, timezone/date alignment problem, "
                        "or max_temp_so_far accidentally using future data."
                        if violations.any()
                        else "max_temp_so_far never exceeds actual_high beyond the 0.5 degree tolerance."
                    )
                ),
            )
        )
    else:
        missing = [
            column
            for column in ["max_temp_so_far", "actual_high"]
            if column not in df.columns
        ]
        checks.append(
            _check_item(
                "Max-so-far sanity check",
                "FAIL",
                affected_rows=len(df),
                affected_columns=missing,
                explanation="Required columns are missing for max-so-far sanity validation.",
            )
        )

    chrono_columns = [column for column in ["prediction_time", "target_date"] if column not in df.columns]
    if chrono_columns:
        checks.append(
            _check_item(
                "Chronological validity check",
                "FAIL",
                affected_rows=len(df),
                affected_columns=chrono_columns,
                explanation="Required chronological columns are missing.",
            )
        )
    else:
        prediction_time = _parse_datetime(df["prediction_time"])
        target_date = _parse_datetime(df["target_date"]).dt.normalize()
        parse_bad = prediction_time.isna() | target_date.isna()
        target_window_end = target_date + pd.Timedelta(days=1)
        invalid_window = (~parse_bad) & (prediction_time >= target_window_end)
        sort_columns = [
            column
            for column in ["location", "station", "target_date", "prediction_time"]
            if column in df.columns
        ]
        duplicate_keys = 0
        key_columns = [
            column for column in ["location", "target_date", "prediction_time"] if column in df.columns
        ]
        if len(key_columns) == 3:
            duplicate_keys = int(df.duplicated(subset=key_columns).sum())

        affected_rows = int(parse_bad.sum() + invalid_window.sum() + duplicate_keys)
        affected_columns = []
        if parse_bad.any():
            affected_columns.extend(["prediction_time", "target_date"])
        if invalid_window.any():
            affected_columns.extend(["prediction_time", "target_date"])
        if duplicate_keys:
            affected_columns.extend(key_columns)

        status = "FAIL" if parse_bad.any() or invalid_window.any() else "PASS"
        if status == "PASS" and duplicate_keys:
            status = "WARN"

        checks.append(
            _check_item(
                "Chronological validity check",
                status,
                affected_rows=affected_rows,
                affected_columns=sorted(set(affected_columns)),
                explanation=(
                    "prediction_time and target_date parse cleanly, prediction_time is inside "
                    "the target-date window, and rows can be sorted by "
                    f"{sort_columns}."
                    if status == "PASS"
                    else "Chronological parsing, target-date window, or duplicate-key validation found issues."
                ),
            )
        )

    missing_features = [column for column in feature_columns if column not in df.columns]
    unsafe_dtypes: list[str] = []
    for column in feature_columns:
        if column not in df.columns:
            continue
        dtype = df[column].dtype
        if not (
            pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            unsafe_dtypes.append(column)

    metadata_features = _metadata_feature_columns(feature_columns)
    reproducibility_fail_columns = missing_features + unsafe_dtypes + metadata_features
    checks.append(
        _check_item(
            "Feature reproducibility check",
            "FAIL" if reproducibility_fail_columns else "PASS",
            affected_rows=len(df) if reproducibility_fail_columns else 0,
            affected_columns=reproducibility_fail_columns,
            explanation=(
                "Feature columns are present, numeric/model-safe, and do not include metadata columns."
                if not reproducibility_fail_columns
                else "feature_columns.json includes missing, non-numeric, or metadata timestamp columns."
            ),
        )
    )

    return {
        "overall_status": _overall_status(checks),
        "checks": checks,
        "notes": list(df.attrs.get("feature_notes", [])),
    }


def write_leakage_report(checks: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Day 8 Leakage Check Report",
        "",
        f"Overall status: **{checks.get('overall_status', 'UNKNOWN')}**",
        "",
        "## Checks",
        "",
    ]

    for item in checks.get("checks", []):
        lines.extend(
            [
                f"### {item['name']}: {item['status']}",
                "",
                f"- Affected rows: {item['affected_rows']}",
                f"- Affected columns: {', '.join(item['affected_columns']) if item['affected_columns'] else 'none'}",
                f"- Explanation: {item['explanation']}",
                "",
            ]
        )

    notes = checks.get("notes", [])
    if notes:
        lines.extend(["## Feature Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
