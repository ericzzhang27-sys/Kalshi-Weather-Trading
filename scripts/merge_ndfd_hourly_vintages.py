from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_TIMEZONE = "America/New_York"
VINTAGE_KEY = ["date", "location", "forecast_issue_time", "ndfd_valid_time_utc", "ndfd_wmo_header"]


def _prediction_timestamp_utc(values: pd.Series, timezone_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(
            timezone_name,
            ambiguous=True,
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


def _read_many(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frame["_input_file"] = str(path)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No non-empty NDFD vintage CSV inputs were found")
    return pd.concat(frames, ignore_index=True, sort=False)


def clean_vintages(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "location",
        "forecast_high",
        "forecast_issue_time",
        "ndfd_valid_time_utc",
        "ndfd_wmo_header",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"NDFD vintage inputs are missing required columns: {missing}")

    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
    clean["forecast_issue_time"] = pd.to_datetime(clean["forecast_issue_time"], errors="coerce", utc=True)
    clean["ndfd_valid_time_utc"] = pd.to_datetime(clean["ndfd_valid_time_utc"], errors="coerce", utc=True)
    clean["forecast_high"] = pd.to_numeric(clean["forecast_high"], errors="coerce")
    if "nws_forecast_high_f" in clean.columns:
        clean["nws_forecast_high_f"] = pd.to_numeric(clean["nws_forecast_high_f"], errors="coerce")
    else:
        clean["nws_forecast_high_f"] = clean["forecast_high"]

    clean = clean[
        clean["date"].notna()
        & clean["forecast_issue_time"].notna()
        & clean["ndfd_valid_time_utc"].notna()
        & clean["forecast_high"].between(-20, 120)
    ].copy()
    if clean.empty:
        raise ValueError("All NDFD vintage rows were invalid after cleaning")

    # NDFD MaxT valid_time is the GRIB maximum-period coordinate, not the
    # dissemination timestamp. Same-day updates can legitimately be issued
    # after that nominal period coordinate. Leakage is therefore enforced
    # against prediction_timestamp in build_hourly_asof(), not against
    # ndfd_valid_time_utc here.
    clean["ndfd_issue_after_valid_time"] = clean["forecast_issue_time"] > clean["ndfd_valid_time_utc"]

    clean["ndfd_wmo_header"] = clean["ndfd_wmo_header"].astype(str).str.upper()
    bad_header = clean["ndfd_wmo_header"].ne("YGUZ98")
    if bad_header.any():
        raise ValueError(f"Unexpected WMO headers: {sorted(clean.loc[bad_header, 'ndfd_wmo_header'].unique())}")

    clean = clean.drop_duplicates(subset=VINTAGE_KEY, keep="last")
    clean = clean.sort_values(["location", "date", "forecast_issue_time", "ndfd_valid_time_utc"]).reset_index(drop=True)
    return clean


def build_hourly_asof(
    vintages: pd.DataFrame,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    first_date = vintages["date"].min()
    last_date = vintages["date"].max()
    locations = sorted(vintages["location"].dropna().astype(str).unique())
    target_dates = pd.date_range(first_date, last_date, freq="D")
    hours = [f"{hour:02d}:00" for hour in range(24)]

    rows = pd.MultiIndex.from_product(
        [locations, target_dates, hours],
        names=["location", "date", "prediction_time"],
    ).to_frame(index=False)
    rows["prediction_timestamp"] = pd.to_datetime(
        rows["date"].dt.strftime("%Y-%m-%d") + " " + rows["prediction_time"],
        errors="raise",
    )
    rows["prediction_timestamp_utc"] = _prediction_timestamp_utc(
        rows["prediction_timestamp"],
        timezone_name,
    )

    right = vintages.copy().sort_values(["location", "date", "forecast_issue_time"])
    left = rows.sort_values(["location", "date", "prediction_timestamp_utc"])

    outputs: list[pd.DataFrame] = []
    for (location, target_date), left_group in left.groupby(["location", "date"], sort=False):
        right_group = right[(right["location"] == location) & (right["date"] == target_date)].copy()
        if right_group.empty:
            group = left_group.copy()
            group["forecast_high"] = pd.NA
            group["forecast_issue_time"] = pd.NaT
            group["forecast_source"] = pd.NA
            group["nws_forecast_high_f"] = pd.NA
            group["ndfd_valid_time_utc"] = pd.NaT
            group["ndfd_lead_hours"] = pd.NA
            group["ndfd_grid_distance_km"] = pd.NA
            group["ndfd_wmo_header"] = pd.NA
            group["ndfd_center"] = pd.NA
            group["ndfd_archive_root"] = pd.NA
            group["ndfd_archive_host"] = pd.NA
            group["ndfd_source_file"] = pd.NA
            group["ndfd_issue_after_valid_time"] = pd.NA
            outputs.append(group)
            continue

        right_group = right_group.sort_values("forecast_issue_time")
        keep_columns = [
            "forecast_issue_time",
            "forecast_high",
            "forecast_source",
            "nws_forecast_high_f",
            "ndfd_valid_time_utc",
            "ndfd_lead_hours",
            "ndfd_grid_distance_km",
            "ndfd_wmo_header",
            "ndfd_center",
            "ndfd_archive_root",
            "ndfd_archive_host",
            "ndfd_source_file",
            "ndfd_issue_after_valid_time",
        ]
        keep_columns = [column for column in keep_columns if column in right_group.columns]
        merged = pd.merge_asof(
            left_group.sort_values("prediction_timestamp_utc"),
            right_group.loc[:, keep_columns].sort_values("forecast_issue_time"),
            left_on="prediction_timestamp_utc",
            right_on="forecast_issue_time",
            direction="backward",
            allow_exact_matches=True,
        )
        outputs.append(merged)

    result = pd.concat(outputs, ignore_index=True, sort=False)
    issue = pd.to_datetime(result["forecast_issue_time"], errors="coerce", utc=True)
    prediction = pd.to_datetime(result["prediction_timestamp_utc"], errors="coerce", utc=True)
    result["forecast_age_minutes"] = (prediction - issue).dt.total_seconds() / 60.0
    leakage = issue.notna() & (issue > prediction)
    if leakage.any():
        raise ValueError(f"{int(leakage.sum())} hourly as-of rows use a future forecast")
    result["has_ndfd_asof"] = result["forecast_high"].notna()
    return result.sort_values(["location", "date", "prediction_time"]).reset_index(drop=True)


def overlap_validation(vintages: pd.DataFrame, reference_path: Path | None) -> dict[str, object]:
    if reference_path is None or not reference_path.exists():
        return {"reference_available": False}
    reference = pd.read_csv(reference_path)
    needed = {"date", "forecast_issue_time", "forecast_high"}
    if not needed.issubset(reference.columns):
        return {"reference_available": True, "reference_valid": False}

    left = vintages.copy()
    right = reference.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    left["forecast_issue_time"] = pd.to_datetime(left["forecast_issue_time"], errors="coerce", utc=True)
    right["forecast_issue_time"] = pd.to_datetime(right["forecast_issue_time"], errors="coerce", utc=True)
    left["forecast_high"] = pd.to_numeric(left["forecast_high"], errors="coerce")
    right["forecast_high"] = pd.to_numeric(right["forecast_high"], errors="coerce")

    merged = left.merge(
        right[["date", "forecast_issue_time", "forecast_high"]].rename(columns={"forecast_high": "reference_high"}),
        on=["date", "forecast_issue_time"],
        how="inner",
    )
    if merged.empty:
        return {"reference_available": True, "reference_valid": True, "overlap_rows": 0}
    diff = merged["forecast_high"] - merged["reference_high"]
    return {
        "reference_available": True,
        "reference_valid": True,
        "overlap_rows": int(len(merged)),
        "exact_match_rows": int(diff.abs().lt(1e-6).sum()),
        "exact_match_fraction": float(diff.abs().lt(1e-6).mean()),
        "mae_f": float(diff.abs().mean()),
        "max_abs_diff_f": float(diff.abs().max()),
    }


def coverage_report(
    vintages: pd.DataFrame,
    hourly: pd.DataFrame,
    overlap: dict[str, object],
) -> dict[str, object]:
    date_counts = vintages.groupby("date")["forecast_issue_time"].nunique()
    usable_hourly = int(hourly["has_ndfd_asof"].sum())
    total_hourly = int(len(hourly))
    issue_times = pd.to_datetime(vintages["forecast_issue_time"], utc=True)
    target_dates = pd.to_datetime(vintages["date"])
    issue_after_valid = int(vintages.get("ndfd_issue_after_valid_time", pd.Series(dtype=bool)).fillna(False).sum())
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vintage_rows": int(len(vintages)),
        "unique_target_days": int(target_dates.nunique()),
        "first_target_date": target_dates.min().date().isoformat(),
        "last_target_date": target_dates.max().date().isoformat(),
        "first_issue_time": issue_times.min().isoformat(),
        "last_issue_time": issue_times.max().isoformat(),
        "updates_per_target_day_mean": float(date_counts.mean()),
        "updates_per_target_day_median": float(date_counts.median()),
        "updates_per_target_day_p10": float(date_counts.quantile(0.10)),
        "updates_per_target_day_p90": float(date_counts.quantile(0.90)),
        "hourly_decision_rows": total_hourly,
        "hourly_rows_with_ndfd_asof": usable_hourly,
        "hourly_asof_coverage_fraction": usable_hourly / total_hourly if total_hourly else 0.0,
        "issue_after_nominal_valid_time_rows": issue_after_valid,
        "wmo_headers": sorted(vintages["ndfd_wmo_header"].dropna().astype(str).unique().tolist()),
        "archive_roots": sorted(vintages.get("ndfd_archive_root", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "overlap_validation": overlap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge NDFD point vintages and build timestamp-safe hourly replay rows.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--vintage-output", type=Path, required=True)
    parser.add_argument("--hourly-output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--minimum-hourly-coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vintages = clean_vintages(_read_many(args.inputs))
    hourly = build_hourly_asof(vintages, timezone_name=args.timezone)
    overlap = overlap_validation(vintages, args.reference)
    report = coverage_report(vintages, hourly, overlap)

    args.vintage_output.parent.mkdir(parents=True, exist_ok=True)
    args.hourly_output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
    vintages.to_csv(args.vintage_output, index=False)
    hourly.to_csv(args.hourly_output, index=False)
    args.coverage_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    coverage = float(report["hourly_asof_coverage_fraction"])
    if coverage < args.minimum_hourly_coverage:
        raise SystemExit(
            f"Hourly as-of coverage {coverage:.3%} is below required {args.minimum_hourly_coverage:.3%}"
        )


if __name__ == "__main__":
    main()
