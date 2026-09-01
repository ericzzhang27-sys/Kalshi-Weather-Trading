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
from src.artifact_io import atomic_write_csv  # noqa: E402
from src.forecast_data import (  # noqa: E402
    NDFD_FALLBACK_SOURCE,
    NDFD_FORECAST_SOURCE,
    build_prediction_time_forecasts,
    identify_forecast_high_column,
    load_ndfd_daily_high_forecasts,
    standardize_daily_forecasts,
    standardize_hourly_forecasts,
    validate_forecast_values,
    write_ndfd_forecast_reports,
)
from src.supervised_table import (  # noqa: E402
    PREDICTION_TIMES,
)
from src.weather_data import (  # noqa: E402
    OFFICIAL_DAILY_SOURCE,
    NWS_ASOS_SOURCE,
    identify_actual_high_column,
    load_nws_hourly_observations,
    load_official_daily_highs,
    standardize_daily_weather,
    standardize_hourly_weather,
    validate_weather_values,
)


OUTPUTS_DIR = REPO_ROOT / "outputs" / "data_audit"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
NDFD_DAILY_FEATURES_PATH = REPO_ROOT / "outputs" / "data" / "ndfd_knyc_daily_features.csv"
NDFD_DAILY_HIGH_ARCHIVE_CANDIDATES = [
    REPO_ROOT / "data" / "processed" / "ndfd_knyc_daily_high_forecasts.csv",
    REPO_ROOT / "outputs" / "data" / "ndfd_knyc_daily_high_forecasts.csv",
]
REQUIRE_FULL_NDFD_FORECAST_COVERAGE = False

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


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        result.append(path)
    return result


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


def _raise_incomplete_ndfd_coverage(forecasts_clean: pd.DataFrame) -> None:
    fallback = forecasts_clean[forecasts_clean["forecast_source"].ne(NDFD_FORECAST_SOURCE)].copy()
    if fallback.empty:
        return

    examples = fallback.loc[
        :,
        [column for column in ["date", "location", "prediction_time", "forecast_fallback_reason"] if column in fallback.columns],
    ].head(10)
    example_text = examples.to_string(index=False)
    raise ValueError(
        "NDFD forecast coverage is incomplete, so training CSVs were not rebuilt with "
        f"mixed forecast sources. {len(fallback):,} prediction rows are missing an "
        f"as-of-available NDFD forecast. First affected rows:\n{example_text}"
    )


def _drop_non_nws_forecast_audit_columns(forecasts_clean: pd.DataFrame) -> pd.DataFrame:
    return forecasts_clean.drop(
        columns=[
            "openmeteo_forecast_high_f",
            "forecast_fallback_reason",
        ],
        errors="ignore",
    )


def _source_usage_rows(
    *,
    daily_source: str,
    hourly_source: str,
    forecast_source: str,
    ndfd_note: str,
) -> list[dict[str, str]]:
    forecast_is_ndfd = "nws_ndfd" in forecast_source.lower()
    forecast_decision = "used"
    if forecast_is_ndfd and "fallback" in forecast_source.lower():
        forecast_decision = "used_with_fallback"
    elif not forecast_is_ndfd:
        forecast_decision = "fallback_used"
    return [
        {
            "field": "official_daily_high",
            "source_used": daily_source,
            "priority_rank": "1",
            "decision": "used",
            "reason": "NOAA/NWS daily TMAX for Central Park is the official Kalshi-style settlement label.",
        },
        {
            "field": "observed_temperature",
            "source_used": hourly_source,
            "priority_rank": "1",
            "decision": "used",
            "reason": "IEM/NWS ASOS observations provide timestamped station observations including special observations.",
        },
        {
            "field": "forecast_high",
            "source_used": forecast_source,
            "priority_rank": "1" if forecast_is_ndfd else "3",
            "decision": forecast_decision,
            "reason": ndfd_note,
        },
        {
            "field": "openmeteo_daily_historical_max",
            "source_used": "daily_raw_nyc_openmeteo.csv",
            "priority_rank": "rejected_for_label",
            "decision": "not_used_as_label",
            "reason": "Open-Meteo historical daily max is retained only for comparison, not as actual_high.",
        },
    ]


def _write_daily_label_coverage(daily_clean: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source": daily_clean.get("actual_source", pd.Series([""])).dropna().astype(str).iloc[0],
            "rows": int(len(daily_clean)),
            "date_min": _date_range(daily_clean, "date").split(" to ")[0],
            "date_max": _date_range(daily_clean, "date").split(" to ")[-1],
            "missing_official_daily_high_f": int(
                daily_clean.get("official_daily_high_f", pd.Series(dtype=float)).isna().sum()
            )
            if "official_daily_high_f" in daily_clean.columns
            else int(daily_clean["actual_high"].isna().sum()),
            "duplicate_date_location_rows": int(
                daily_clean.duplicated(subset=["date", "location"]).sum()
            ),
            "source_file": ";".join(
                sorted(daily_clean.get("source_file", pd.Series(dtype=str)).dropna().astype(str).unique())
            ),
            "source_station": ";".join(
                sorted(daily_clean.get("source_station", pd.Series(dtype=str)).dropna().astype(str).unique())
            ),
        }
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _write_hourly_obs_coverage(hourly_clean: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hourly_clean.empty:
        pd.DataFrame(
            [{"date": "", "rows": 0, "missing_nws_current_temp_f": 0, "max_hourly_tmpf": ""}]
        ).to_csv(output_path, index=False)
        return

    temp_col = "nws_current_temp_f" if "nws_current_temp_f" in hourly_clean.columns else "temperature_2m"
    grouped = (
        hourly_clean.assign(date=pd.to_datetime(hourly_clean["date"], errors="coerce").dt.date)
        .groupby("date", dropna=False)
        .agg(
            rows=("timestamp", "size"),
            missing_nws_current_temp_f=(temp_col, lambda value: int(pd.to_numeric(value, errors="coerce").isna().sum())),
            max_hourly_tmpf=(temp_col, lambda value: pd.to_numeric(value, errors="coerce").max()),
        )
        .reset_index()
    )
    grouped.insert(0, "source", hourly_clean.get("observation_source", pd.Series([""])).dropna().astype(str).iloc[0])
    grouped.insert(1, "station", ";".join(sorted(hourly_clean["station"].dropna().astype(str).unique())) if "station" in hourly_clean.columns else "")
    grouped.to_csv(output_path, index=False)


def _comparison_summary_rows(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("old_actual_high_vs_official", "old_minus_official"),
        ("openmeteo_daily_max_vs_official", "openmeteo_minus_official"),
        ("max_hourly_tmpf_vs_official", "max_hourly_minus_official"),
    ]
    for label, diff_col in comparisons:
        if diff_col not in comparison.columns:
            continue
        diff = pd.to_numeric(comparison[diff_col], errors="coerce").dropna()
        if diff.empty:
            continue
        abs_diff = diff.abs()
        largest = comparison.loc[abs_diff.sort_values(ascending=False).head(5).index]
        rows.append(
            {
                "comparison": label,
                "n_rows": int(len(diff)),
                "mean_abs_difference_f": float(abs_diff.mean()),
                "median_abs_difference_f": float(abs_diff.median()),
                "max_abs_difference_f": float(abs_diff.max()),
                "pct_abs_diff_ge_1f": float((abs_diff >= 1.0).mean() * 100.0),
                "pct_abs_diff_ge_2f": float((abs_diff >= 2.0).mean() * 100.0),
                "pct_abs_diff_ge_3f": float((abs_diff >= 3.0).mean() * 100.0),
                "largest_examples": "; ".join(
                    f"{pd.to_datetime(row['date']).date()}: {float(row[diff_col]):+.1f}F"
                    for _, row in largest.iterrows()
                ),
            }
        )
    return rows


def _write_high_source_comparison(
    *,
    official_daily: pd.DataFrame,
    openmeteo_daily: pd.DataFrame,
    hourly_clean: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    official = official_daily.loc[:, ["date", "location", "official_daily_high_f"]].copy()
    old = openmeteo_daily.loc[:, ["date", "location", "actual_high"]].rename(
        columns={"actual_high": "openmeteo_daily_max_f"}
    )
    comparison = official.merge(old, on=["date", "location"], how="left")
    comparison["old_actual_high_f"] = comparison["openmeteo_daily_max_f"]

    if not hourly_clean.empty:
        temp_col = "nws_current_temp_f" if "nws_current_temp_f" in hourly_clean.columns else "temperature_2m"
        hourly_daily = (
            hourly_clean.assign(date=pd.to_datetime(hourly_clean["date"], errors="coerce").dt.normalize())
            .groupby(["date", "location"], dropna=False)[temp_col]
            .max()
            .reset_index()
            .rename(columns={temp_col: "max_hourly_tmpf"})
        )
        comparison = comparison.merge(hourly_daily, on=["date", "location"], how="left")
    else:
        comparison["max_hourly_tmpf"] = pd.NA

    comparison["old_minus_official"] = (
        comparison["old_actual_high_f"] - comparison["official_daily_high_f"]
    )
    comparison["openmeteo_minus_official"] = (
        comparison["openmeteo_daily_max_f"] - comparison["official_daily_high_f"]
    )
    comparison["max_hourly_minus_official"] = (
        comparison["max_hourly_tmpf"] - comparison["official_daily_high_f"]
    )
    comparison.to_csv(output_path, index=False)
    pd.DataFrame(_comparison_summary_rows(comparison)).to_csv(summary_path, index=False)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        label: _resolve_data_path(label, candidates)
        for label, candidates in DATA_PATH_CANDIDATES.items()
    }

    raw_frames = {label: load_csv(path) for label, path in paths.items()}
    raw_inventory_paths = _unique_paths(
        [*RAW_DIR.glob("*.csv"), *paths.values(), *NDFD_DAILY_HIGH_ARCHIVE_CANDIDATES]
    )
    inventory = create_data_inventory(
        raw_inventory_paths,
        OUTPUTS_DIR / "data_inventory.csv",
    )

    openmeteo_daily_clean = standardize_daily_weather(raw_frames["daily_actual"], location="NYC")
    actual_high_column = identify_actual_high_column(openmeteo_daily_clean)
    forecast_high_column = identify_forecast_high_column(raw_frames["daily_forecast"])
    if forecast_high_column is None:
        raise ValueError("Could not identify the forecast high column in daily forecast data")

    warnings: list[str] = []
    try:
        daily_clean = load_official_daily_highs(RAW_DIR)
        actual_high_column = "TMAX"
        daily_source = OFFICIAL_DAILY_SOURCE
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Official NOAA/NWS daily TMAX is required for labels; refusing to rebuild "
            "targets from Open-Meteo daily historical max."
        ) from exc

    try:
        hourly_clean = load_nws_hourly_observations(RAW_DIR)
        hourly_source = NWS_ASOS_SOURCE
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "IEM/NWS ASOS KNYC hourly history is required for canonical training "
            "features; blocked instead of substituting Open-Meteo observations."
        ) from exc
    blocked_ranges = pd.DataFrame(hourly_clean.attrs.get("blocked_ranges", []))
    atomic_write_csv(
        blocked_ranges.reindex(columns=["station", "timestamp", "source_file", "reason"]),
        OUTPUTS_DIR / "blocked_source_ranges.csv",
        index=False,
    )

    openmeteo_forecasts_clean = standardize_daily_forecasts(raw_frames["daily_forecast"], location="NYC")
    forecasts_clean = openmeteo_forecasts_clean
    hourly_forecasts_clean = standardize_hourly_forecasts(
        raw_frames["hourly_forecast"],
        location="NYC",
    )

    forecast_source = "open_meteo_historical_forecast"
    ndfd_note = (
        "NDFD/NWS daily high forecast archive was not present, so Open-Meteo historical "
        "forecasts remain the forecast_high source."
    )
    ndfd_daily = load_ndfd_daily_high_forecasts(NDFD_DAILY_HIGH_ARCHIVE_CANDIDATES, location="NYC")
    if not ndfd_daily.empty:
        forecasts_clean = build_prediction_time_forecasts(
            openmeteo_daily=openmeteo_forecasts_clean,
            ndfd_daily=ndfd_daily,
            prediction_times=PREDICTION_TIMES,
        )
        if REQUIRE_FULL_NDFD_FORECAST_COVERAGE:
            _raise_incomplete_ndfd_coverage(forecasts_clean)
            forecasts_clean = _drop_non_nws_forecast_audit_columns(forecasts_clean)
        ndfd_prediction_rows = int(forecasts_clean["forecast_source"].eq(NDFD_FORECAST_SOURCE).sum())
        openmeteo_fallback_rows = int(forecasts_clean["forecast_source"].ne(NDFD_FORECAST_SOURCE).sum())
        forecast_source = (
            NDFD_FORECAST_SOURCE
            if openmeteo_fallback_rows == 0
            else NDFD_FALLBACK_SOURCE
        )
        ndfd_archive_range = _date_range(ndfd_daily, "date")
        ndfd_note = (
            "Historical NWS/NDFD MaxT forecast archive is used as forecast_high where an "
            f"issue is available as of the prediction time: {ndfd_prediction_rows:,} "
            f"prediction-time rows use NDFD over {ndfd_archive_range}; "
            f"{openmeteo_fallback_rows:,} rows use non-NDFD fallback."
        )
    elif NDFD_DAILY_FEATURES_PATH.exists():
        if REQUIRE_FULL_NDFD_FORECAST_COVERAGE:
            checked = "\n".join(f"  - {path}" for path in NDFD_DAILY_HIGH_ARCHIVE_CANDIDATES)
            raise FileNotFoundError(
                "A full NDFD daily-high forecast archive is required before rebuilding "
                "training CSVs. Build it with scripts/build_ndfd_daily_high_archive.py. "
                f"Checked:\n{checked}"
            )
        ndfd = pd.read_csv(NDFD_DAILY_FEATURES_PATH)
        non_missing_nws_highs = int(
            pd.to_numeric(ndfd.get("nws_forecast_high_f", pd.Series(dtype=float)), errors="coerce")
            .notna()
            .sum()
        )
        ndfd_range = _date_range(ndfd, "date")
        ndfd_note = (
            "A NDFD/NWS feature extract exists but was not used as forecast_high for the full "
            f"rebuild because it has only {non_missing_nws_highs} non-missing daily highs "
            f"over {ndfd_range}; Open-Meteo historical forecast remains the broader archive."
        )
    elif REQUIRE_FULL_NDFD_FORECAST_COVERAGE:
        checked = "\n".join(f"  - {path}" for path in NDFD_DAILY_HIGH_ARCHIVE_CANDIDATES)
        raise FileNotFoundError(
            "A full NDFD daily-high forecast archive is required before rebuilding "
            "training CSVs. Build it with scripts/build_ndfd_daily_high_archive.py. "
            f"Checked:\n{checked}"
        )

    daily_output = PROCESSED_DIR / "daily_clean.csv"
    hourly_output = PROCESSED_DIR / "hourly_clean.csv"
    forecasts_output = PROCESSED_DIR / "forecasts_clean.csv"
    hourly_forecasts_output = PROCESSED_DIR / "hourly_forecasts_clean.csv"
    preview_output = PROCESSED_DIR / "modeling_base_preview.csv"

    atomic_write_csv(daily_clean, daily_output, index=False)
    atomic_write_csv(hourly_clean, hourly_output, index=False)
    atomic_write_csv(forecasts_clean, forecasts_output, index=False)
    atomic_write_csv(hourly_forecasts_clean, hourly_forecasts_output, index=False)
    write_ndfd_forecast_reports(
        forecasts_clean,
        REPORTS_DIR / "ndfd_forecast_coverage.csv",
        REPORTS_DIR / "openmeteo_vs_ndfd_forecast_comparison.csv",
    )

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
        "prediction_time",
        "prediction_timestamp",
        "actual_high",
        "forecast_high",
        "forecast_error",
        "forecast_source",
        "forecast_issue_time",
        "openmeteo_forecast_high_f",
        "nws_forecast_high_f",
        "forecast_fallback_reason",
    ]
    merged.loc[:, [column for column in modeling_preview_columns if column in merged.columns]].to_csv(
        preview_output,
        index=False,
    )

    _append_alignment_warnings(warnings, merged, daily_clean, forecasts_clean)
    warnings.append(ndfd_note)
    if "location" not in raw_frames["daily_forecast"].columns:
        warnings.append("Location was filled as 'NYC' because row-level location/station information is missing")

    known_risks = [
        "Hourly/special ASOS observations can miss brief highs between reports, so official NOAA/NWS daily TMAX remains the label.",
        "High-missingness forecast precipitation-probability columns should not be used as complete features without a deliberate missing-data plan.",
    ]
    if "prediction_time" in forecasts_clean.columns:
        known_risks.insert(
            0,
            "NDFD forecast availability is enforced by forecast_issue_time <= prediction_timestamp; "
            "rows without an available NDFD issue block the canonical training rebuild.",
        )
    else:
        known_risks.insert(
            0,
            "Forecast rows do not include an as-of/model-run timestamp, so point-in-time forecast availability cannot be fully verified.",
        )
        known_risks.insert(
            1,
            "Open-Meteo forecast history is retained because a usable NDFD daily high archive was not found.",
        )

    _write_daily_label_coverage(daily_clean, REPORTS_DIR / "daily_label_coverage.csv")
    _write_hourly_obs_coverage(hourly_clean, REPORTS_DIR / "hourly_obs_coverage.csv")
    _write_high_source_comparison(
        official_daily=daily_clean,
        openmeteo_daily=openmeteo_daily_clean,
        hourly_clean=hourly_clean,
        output_path=REPORTS_DIR / "high_source_comparison.csv",
        summary_path=REPORTS_DIR / "high_source_comparison_summary.csv",
    )
    pd.DataFrame(
        _source_usage_rows(
            daily_source=daily_source,
            hourly_source=hourly_source,
            forecast_source=forecast_source,
            ndfd_note=ndfd_note,
        )
    ).to_csv(REPORTS_DIR / "source_usage_report.csv", index=False)

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

    forecast_duplicate_keys = (
        ["date", "location", "prediction_time"]
        if "prediction_time" in forecasts_clean.columns
        else ["date", "location"]
    )

    summary = {
        "key_columns": {
            "actual_daily_date_column": "DATE",
            "actual_high_column_used": actual_high_column,
            "forecast_daily_date_column": infer_datetime_column(raw_frames["daily_forecast"]),
            "forecast_high_column_used": forecast_high_column,
            "hourly_actual_time_column": "valid" if hourly_source == NWS_ASOS_SOURCE else infer_datetime_column(raw_frames["hourly_actual"]),
            "hourly_forecast_time_column": infer_datetime_column(raw_frames["hourly_forecast"]),
            "actual_source": daily_source,
            "hourly_observation_source": hourly_source,
            "forecast_source": forecast_source,
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
                "forecasts_clean": (forecasts_clean, forecast_duplicate_keys),
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
    print(f"Actual high source: {daily_source}")
    print(f"Hourly observation source: {hourly_source}")
    print(f"Forecast high source: {forecast_source}")
    print(f"Report: {OUTPUTS_DIR / 'data_verification_report.md'}")


if __name__ == "__main__":
    main()
