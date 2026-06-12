from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


OPENMETEO_METADATA_ROWS = 3
PREFERRED_DATETIME_COLUMNS = ("time", "date", "timestamp", "datetime")


def _looks_like_openmeteo_export(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first_line = handle.readline().strip().lower()
            second_line = handle.readline()
            third_line = handle.readline()
            fourth_line = handle.readline().strip().lower()
    except OSError:
        return False

    return (
        first_line.startswith("latitude,longitude")
        and bool(second_line.strip())
        and not third_line.strip()
        and fourth_line.startswith("time,")
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV, including Open-Meteo exports with metadata preambles."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")

    try:
        if _looks_like_openmeteo_export(csv_path):
            return pd.read_csv(csv_path, skiprows=OPENMETEO_METADATA_ROWS)
        return pd.read_csv(csv_path)
    except Exception as first_error:
        try:
            return pd.read_csv(csv_path, skiprows=OPENMETEO_METADATA_ROWS)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not read CSV file {csv_path}. "
                f"Initial error: {first_error}. Fallback error: {second_error}."
            ) from second_error


def read_openmeteo_metadata(path: str | Path) -> dict[str, str]:
    """Read the coordinate/timezone preamble from an Open-Meteo CSV if present."""
    csv_path = Path(path)
    if not csv_path.exists() or not _looks_like_openmeteo_export(csv_path):
        return {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
        values = next(csv.reader([handle.readline()]))

    return {key: value for key, value in zip(header, values, strict=False)}


def infer_datetime_column(df: pd.DataFrame) -> str | None:
    for preferred in PREFERRED_DATETIME_COLUMNS:
        exact_matches = [
            column
            for column in df.columns
            if str(column).strip().lower() == preferred
        ]
        if exact_matches:
            return str(exact_matches[0])

    best_column: str | None = None
    best_parse_rate = 0.0
    for column in df.columns:
        sample = df[column].dropna().head(200)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        parse_rate = float(parsed.notna().mean())
        if parse_rate > best_parse_rate:
            best_column = str(column)
            best_parse_rate = parse_rate

    if best_parse_rate >= 0.8:
        return best_column
    return None


def _date_range_for_column(df: pd.DataFrame, column: str | None) -> tuple[str | None, str | None]:
    if column is None or column not in df.columns or df.empty:
        return None, None

    parsed = pd.to_datetime(df[column], errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return None, None

    return parsed.min().isoformat(), parsed.max().isoformat()


def summarize_file(path: str | Path, df: pd.DataFrame) -> dict[str, Any]:
    datetime_column = infer_datetime_column(df)
    date_min, date_max = _date_range_for_column(df, datetime_column)
    missing_by_column = df.isna().sum().astype(int).to_dict()

    return {
        "file": str(Path(path)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_time_column": datetime_column,
        "date_min": date_min,
        "date_max": date_max,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "columns_list": json.dumps(list(df.columns), ensure_ascii=False),
        "missing_values_by_column": json.dumps(missing_by_column, ensure_ascii=False),
    }


def create_data_inventory(
    file_paths: list[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    rows = []
    for file_path in file_paths:
        df = load_csv(file_path)
        rows.append(summarize_file(file_path, df))

    inventory = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output, index=False)
    return inventory


def _as_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(str(row.get(column, "")).replace("\n", " ") for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "- not available"
    return "\n".join(f"- {key}: {_format_value(value)}" for key, value in mapping.items())


def write_verification_report(
    output_path: str | Path,
    inventory: pd.DataFrame,
    warnings: list[str],
    summary: dict,
) -> None:
    """Write the Day 6 markdown verification report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    inventory_rows = inventory[
        [
            "file",
            "rows",
            "columns",
            "date_time_column",
            "date_min",
            "date_max",
            "duplicate_rows",
            "missing_values_total",
        ]
    ].to_dict("records")

    missing_sections = summary.get("missing_sections", [])
    duplicate_sections = summary.get("duplicates", [])
    key_columns = summary.get("key_columns", {})
    unit_summary = summary.get("unit_summary", {})
    alignment = summary.get("alignment", {})
    forecast_error_summary = summary.get("forecast_error_summary", {})
    cleaned_outputs = summary.get("cleaned_outputs", [])
    known_risks = summary.get("known_risks", [])
    metadata = summary.get("metadata", {})

    warning_lines = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- none"
    risk_lines = "\n".join(f"- {risk}" for risk in known_risks) if known_risks else "- none"
    output_lines = "\n".join(f"- {path}" for path in cleaned_outputs) if cleaned_outputs else "- none"
    metadata_lines = _format_mapping(metadata)

    report = f"""# Day 6 Data Verification Report

## Files Audited
{_as_markdown_table(inventory_rows, ["file", "rows", "columns", "date_time_column", "date_min", "date_max", "duplicate_rows", "missing_values_total"])}

## Date Ranges
- Actual daily date range: {_format_value(alignment.get("actual_daily_date_range"))}
- Forecast daily date range: {_format_value(alignment.get("forecast_daily_date_range"))}
- Hourly actual timestamp range: {_format_value(alignment.get("actual_hourly_timestamp_range"))}
- Hourly forecast timestamp range: {_format_value(alignment.get("forecast_hourly_timestamp_range"))}

## Key Columns Identified
{_format_mapping(key_columns)}

## Location And Metadata
{metadata_lines}

## Unit Standardization
No unit conversions were performed. Units are documented from raw column names or metadata only when visible in the CSV export.

{_format_mapping(unit_summary)}

## Missing Values
{_as_markdown_table(missing_sections, ["dataset", "column", "missing_count", "missing_percent"])}

## Duplicate Timestamps/Dates
{_as_markdown_table(duplicate_sections, ["dataset", "duplicate_key", "duplicate_count"])}

## Actual High And Forecast High Alignment
- Daily actual rows: {_format_value(alignment.get("daily_actual_rows"))}
- Daily forecast rows: {_format_value(alignment.get("daily_forecast_rows"))}
- Overlapping date/location rows: {_format_value(alignment.get("overlap_rows"))}
- Merged preview rows: {_format_value(alignment.get("merged_preview_rows"))}

Forecast error is defined as `actual_high - forecast_high`.

### Forecast Error Summary
{_format_mapping(forecast_error_summary)}

## Forecast-Source Caveat
The canonical forecast baseline is the timestamp-safe historical NWS/NDFD MaxT forecast. Open-Meteo forecast history is retained only as legacy/auxiliary input and should not be described as the training forecast_high source when NDFD coverage is complete.

## Validation Warnings
{warning_lines}

## Known Risks Before Modeling
{risk_lines}

## Cleaned Outputs Created
{output_lines}
"""

    output.write_text(report, encoding="utf-8")
