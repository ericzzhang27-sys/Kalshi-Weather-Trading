from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ndfd_extract import (  # noqa: E402
    build_daily_features,
    extract_entries_to_dataframe,
    validate_point_forecasts,
    write_dataframe,
)
from src.data.ndfd_fetch import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    discover_range,
    download_entries,
    manifest_counts,
    write_manifest,
    write_missing_dates_report,
)


DEFAULT_VARIABLES = "maxt,temp,sky,wspd,wdir,pop12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NDFD point forecasts for Central Park / KNYC.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--lat", type=float, default=40.7812)
    parser.add_argument("--lon", type=float, default=-73.9665)
    parser.add_argument("--station", default="KNYC")
    parser.add_argument("--variables", default=DEFAULT_VARIABLES)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on discovered files to download/extract for debugging.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output", default="outputs/data/ndfd_knyc_point_forecasts.csv")
    parser.add_argument("--daily-output", default="outputs/data/ndfd_knyc_daily_features.csv")
    parser.add_argument("--manifest-output", default="outputs/data/ndfd_download_manifest.csv")
    parser.add_argument("--missing-dates-output", default="outputs/data/ndfd_missing_dates.csv")
    return parser.parse_args()


def _split_variables(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    variables = _split_variables(args.variables)

    entries = discover_range(
        args.start_date,
        args.end_date,
        variables=variables,
        limit_days=args.limit_days,
    )
    if args.max_files is not None:
        entries = [entry for entry in entries if entry.file_url][: args.max_files] + [
            entry for entry in entries if not entry.file_url
        ]
    entries = download_entries(
        entries,
        cache_dir=Path(args.cache_dir),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    manifest_path = write_manifest(entries, args.manifest_output)
    missing_dates_path = write_missing_dates_report(entries, args.missing_dates_output)

    point_path = Path(args.output)
    daily_path = Path(args.daily_output)
    if args.dry_run:
        point_df = extract_entries_to_dataframe([], lat=args.lat, lon=args.lon, station=args.station, requested_variables=variables)
    else:
        point_df = extract_entries_to_dataframe(entries, lat=args.lat, lon=args.lon, station=args.station, requested_variables=variables)
    daily_df = build_daily_features(point_df)

    write_dataframe(point_df, point_path)
    write_dataframe(daily_df, daily_path)
    warnings = validate_point_forecasts(point_df, args.lat, args.lon)
    counts = manifest_counts(entries)

    attempted_dates = len({entry.date for entry in entries})
    extracted = int((point_df.get("extraction_status") == "extracted").sum()) if not point_df.empty else 0
    print(f"number of dates attempted: {attempted_dates}")
    print(f"number of files discovered: {counts['files_discovered']}")
    print(f"number downloaded: {counts['downloaded']}")
    print(f"number cached: {counts['cached']}")
    print(f"number successfully extracted: {extracted}")
    print(f"number failed: {counts['failed'] + counts['missing_catalog']}")
    print(f"manifest path: {manifest_path}")
    print(f"missing dates report: {missing_dates_path}")
    print(f"point forecast output: {point_path}")
    print(f"daily feature output: {daily_path}")
    if warnings:
        print("validation warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print("first 10 rows of the final point forecast table:")
    print(point_df.head(10).to_string(index=False) if not point_df.empty else "(empty)")
    print("first 10 rows of the derived daily feature table:")
    print(daily_df.head(10).to_string(index=False) if not daily_df.empty else "(empty)")


if __name__ == "__main__":
    main()
