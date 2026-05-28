from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import build_feature_matrix, load_inputs, write_feature_columns  # noqa: E402
from src.leakage_checks import run_leakage_checks, write_leakage_report  # noqa: E402


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "day8_features"

MODELING_ROWS_OUTPUT = PROCESSED_DIR / "modeling_rows_v1.csv"
PREVIEW_OUTPUT = OUTPUTS_DIR / "modeling_rows_v1_preview.csv"
MISSINGNESS_OUTPUT = OUTPUTS_DIR / "feature_missingness_report.csv"
FEATURE_COLUMNS_OUTPUT = OUTPUTS_DIR / "feature_columns.json"
LEAKAGE_REPORT_OUTPUT = OUTPUTS_DIR / "leakage_check_report.md"


def _range_text(series: pd.Series) -> str:
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min()} to {values.max()}"


def _status_counts(checks: dict[str, object]) -> str:
    items = checks.get("checks", [])
    if not isinstance(items, list):
        return "not available"
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for item in items:
        if isinstance(item, dict):
            status = str(item.get("status", "UNKNOWN"))
            if status in counts:
                counts[status] += 1
    return ", ".join(f"{status}: {count}" for status, count in counts.items())


def _run_basic_assertions(df: pd.DataFrame, feature_columns: list[str]) -> None:
    required_outputs = [
        MODELING_ROWS_OUTPUT,
        PREVIEW_OUTPUT,
        MISSINGNESS_OUTPUT,
        FEATURE_COLUMNS_OUTPUT,
        LEAKAGE_REPORT_OUTPUT,
    ]
    missing_outputs = [path for path in required_outputs if not path.exists()]
    if missing_outputs:
        raise AssertionError(f"Expected output files were not created: {missing_outputs}")

    forbidden_features = {"forecast_error", "actual_high", "target"}
    leaked = sorted(forbidden_features.intersection(feature_columns))
    if leaked:
        raise AssertionError(f"Forbidden target/audit columns in feature_columns: {leaked}")

    for source_col in ["current_temp_source_time", "max_temp_so_far_source_time"]:
        if source_col in df.columns:
            source_time = pd.to_datetime(df[source_col], errors="coerce")
            prediction_time = pd.to_datetime(df["prediction_time"], errors="coerce")
            violations = source_time.notna() & prediction_time.notna() & (source_time > prediction_time)
            if violations.any():
                raise AssertionError(
                    f"{source_col} has {int(violations.sum())} timestamps after prediction_time"
                )

    if {"location", "target_date", "prediction_time", "max_temp_so_far"}.issubset(df.columns):
        ordered = df.sort_values(["location", "target_date", "prediction_time"])
        diffs = ordered.groupby(["location", "target_date"])["max_temp_so_far"].diff()
        decreases = diffs.dropna() < -1e-9
        if decreases.any():
            raise AssertionError("max_temp_so_far decreases within at least one target_date")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    modeling_rows = build_feature_matrix(inputs, missingness_output_path=MISSINGNESS_OUTPUT)

    modeling_rows.to_csv(MODELING_ROWS_OUTPUT, index=False)
    preview = modeling_rows.sample(n=min(20, len(modeling_rows)), random_state=8)
    preview.to_csv(PREVIEW_OUTPUT, index=False)

    feature_spec = write_feature_columns(modeling_rows, FEATURE_COLUMNS_OUTPUT)
    feature_columns = list(feature_spec["feature_columns"])

    checks = run_leakage_checks(modeling_rows, feature_columns)
    write_leakage_report(checks, LEAKAGE_REPORT_OUTPUT)

    _run_basic_assertions(modeling_rows, feature_columns)

    date_range = _range_text(modeling_rows["target_date"]) if "target_date" in modeling_rows else "not available"
    prediction_range = (
        _range_text(modeling_rows["prediction_time"])
        if "prediction_time" in modeling_rows
        else "not available"
    )
    dropped_rows = int(modeling_rows.attrs.get("dropped_critical_rows", 0))

    print("Day 8 timestamp-safe feature build complete.")
    print(f"Rows: {len(modeling_rows):,}")
    print(f"Feature columns: {len(feature_columns):,}")
    print(f"Target date range: {date_range}")
    print(f"Prediction timestamp range: {prediction_range}")
    print(f"Missing critical rows dropped: {dropped_rows:,}")
    print(f"Leakage check status: {checks['overall_status']} ({_status_counts(checks)})")
    print(f"Modeling rows: {MODELING_ROWS_OUTPUT}")
    print(f"Feature columns: {FEATURE_COLUMNS_OUTPUT}")
    print(f"Leakage report: {LEAKAGE_REPORT_OUTPUT}")
    print(f"Random 20-row preview: {PREVIEW_OUTPUT}")

    notes = list(modeling_rows.attrs.get("feature_notes", []))
    if notes:
        print("Skipped/limited feature notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
