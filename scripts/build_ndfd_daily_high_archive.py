from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ndfd_extract import (  # noqa: E402
    build_daily_high_forecast_archive,
    extract_entries_to_dataframe,
    validate_point_forecasts,
    write_dataframe,
)
from src.data.ndfd_fetch import (  # noqa: E402
    CatalogEntry,
    DEFAULT_CACHE_DIR,
    discover_aws_wmo_range,
    download_entries,
    iter_dates,
    manifest_counts,
    parse_date,
    write_manifest,
    write_missing_dates_report,
)


DEFAULT_LAT = 40.7812
DEFAULT_LON = -73.9665
DEFAULT_WMO_PREFIXES = ["YGUZ98"]
DEFAULT_ISSUE_HOURS = [0, 6, 12, 18]


def _parse_csv_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def _filter_archive_dates(
    archive: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if archive.empty:
        return archive
    start = pd.Timestamp(parse_date(start_date))
    end = pd.Timestamp(parse_date(end_date))
    dates = pd.to_datetime(archive["date"], errors="coerce").dt.normalize()
    return archive[(dates >= start) & (dates <= end)].reset_index(drop=True)


def _latest_per_issue_hour(entries: list[CatalogEntry]) -> list[CatalogEntry]:
    selected: dict[tuple[str, str, str, int], CatalogEntry] = {}
    passthrough: list[CatalogEntry] = []
    for entry in entries:
        if not entry.file_url or not entry.issue_time:
            passthrough.append(entry)
            continue
        issue_time = pd.to_datetime(entry.issue_time, errors="coerce")
        if pd.isna(issue_time):
            passthrough.append(entry)
            continue
        wmo_prefix = entry.filename.split("_", 1)[0].upper()
        key = (entry.date, wmo_prefix, issue_time.strftime("%Y-%m-%d"), int(issue_time.hour))
        current = selected.get(key)
        if current is None:
            selected[key] = entry
            continue
        current_issue_time = pd.to_datetime(current.issue_time, errors="coerce")
        if pd.isna(current_issue_time) or issue_time > current_issue_time:
            selected[key] = entry
    return sorted(
        [*passthrough, *selected.values()],
        key=lambda entry: (entry.date, entry.issue_time or "", entry.filename),
    )


def _remove_cached_files(entries: list[CatalogEntry]) -> int:
    removed = 0
    for entry in entries:
        if not entry.local_path:
            continue
        path = Path(entry.local_path)
        try:
            if path.exists():
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a historical NWS/NDFD daily-high forecast archive for KNYC from "
            "NOAA's public noaa-ndfd-pds WMO files."
        )
    )
    parser.add_argument("--start-date", required=True, help="First target date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Last target date, YYYY-MM-DD")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Target latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Target longitude")
    parser.add_argument("--station", default="KNYC", help="Station/location label for extraction")
    parser.add_argument(
        "--wmo-prefixes",
        default=",".join(DEFAULT_WMO_PREFIXES),
        help="Comma-separated WMO file prefixes to keep. YGUZ98 covers the NYC CONUS MaxT grid.",
    )
    parser.add_argument(
        "--issue-hours",
        default=",".join(str(hour) for hour in DEFAULT_ISSUE_HOURS),
        help="Comma-separated UTC issue hours to keep, such as 0,6,12,18.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1,
        help="Extra issue dates before start-date to discover for early target-day prediction times.",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--point-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "data" / "ndfd_knyc_point_forecasts.csv",
    )
    parser.add_argument(
        "--archive-output",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "ndfd_knyc_daily_high_forecasts.csv",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "data" / "ndfd_download_manifest.csv",
    )
    parser.add_argument(
        "--missing-dates-output",
        type=Path,
        default=REPO_ROOT / "outputs" / "data" / "ndfd_missing_dates.csv",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument(
        "--keep-all-updates",
        action="store_true",
        help="Download every matching WMO update instead of the latest file per issue hour.",
    )
    parser.add_argument(
        "--stream-by-day",
        action="store_true",
        help="Discover, download, and extract one issue day at a time.",
    )
    parser.add_argument(
        "--purge-cache-after-extract",
        action="store_true",
        help="Delete downloaded GRIB files after their point rows are extracted.",
    )
    parser.add_argument(
        "--progress-every-days",
        type=int,
        default=25,
        help="Progress print cadence for --stream-by-day.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-download cached files")
    parser.add_argument("--dry-run", action="store_true", help="List matching files without downloading")
    return parser.parse_args()


def _prepare_discovered_entries(
    entries: list[CatalogEntry],
    *,
    keep_all_updates: bool,
) -> tuple[list[CatalogEntry], int]:
    entries = [entry for entry in entries if entry.file_url]
    before_count = len(entries)
    if not keep_all_updates:
        entries = _latest_per_issue_hour(entries)
    return entries, before_count


def _extract_downloaded_entries(
    entries: list[CatalogEntry],
    *,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()
    return extract_entries_to_dataframe(
        entries,
        lat=args.lat,
        lon=args.lon,
        station=args.station,
        requested_variables=["maxt"],
    )


def _finalize_outputs(
    *,
    args: argparse.Namespace,
    downloaded: list[CatalogEntry],
    point_df: pd.DataFrame,
    expected_target_days: list[object],
) -> None:
    write_manifest(downloaded, args.manifest_output)
    write_missing_dates_report(downloaded, args.missing_dates_output)

    counts = manifest_counts(downloaded)
    print(
        "NDFD discovery/download counts: "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
    if args.dry_run:
        print(f"Dry run complete. Manifest: {args.manifest_output}")
        return

    write_dataframe(point_df, args.point_output)

    archive = build_daily_high_forecast_archive(point_df)
    archive = _filter_archive_dates(archive, args.start_date, args.end_date)
    write_dataframe(archive, args.archive_output)

    warnings = validate_point_forecasts(point_df, target_lat=args.lat, target_lon=args.lon)
    coverage_days = pd.to_datetime(archive["date"], errors="coerce").dt.normalize().nunique() if not archive.empty else 0
    print(f"Extracted point rows: {len(point_df):,}")
    print(f"Daily NDFD high forecast rows: {len(archive):,}")
    print(f"Target-day coverage: {coverage_days:,} of {len(expected_target_days):,} days")
    print(f"Point forecasts: {args.point_output}")
    print(f"Daily high archive: {args.archive_output}")
    if warnings:
        print("Warnings:")
        for message in warnings:
            print(f"- {message}")


def _build_streamed(
    *,
    args: argparse.Namespace,
    discover_start: object,
    end: object,
    wmo_prefixes: list[str],
    issue_hours: list[int],
    expected_target_days: list[object],
) -> None:
    downloaded_all: list[CatalogEntry] = []
    point_frames: list[pd.DataFrame] = []
    selected_total = 0
    discovered_total = 0
    removed_total = 0
    issue_days = list(iter_dates(discover_start, end))

    for day_number, issue_day in enumerate(issue_days, start=1):
        day_entries = discover_aws_wmo_range(
            issue_day,
            issue_day,
            variable="maxt",
            wmo_prefixes=wmo_prefixes,
            issue_hours=issue_hours,
        )
        day_entries, before_count = _prepare_discovered_entries(
            day_entries,
            keep_all_updates=args.keep_all_updates,
        )
        discovered_total += before_count
        if args.max_files is not None:
            remaining = args.max_files - selected_total
            if remaining <= 0:
                break
            day_entries = day_entries[:remaining]
        selected_total += len(day_entries)

        downloaded = download_entries(
            day_entries,
            cache_dir=args.cache_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        downloaded_all.extend(downloaded)

        if not args.dry_run:
            point_frame = _extract_downloaded_entries(downloaded, args=args)
            if not point_frame.empty:
                point_frames.append(point_frame)
            if args.purge_cache_after_extract:
                removed_total += _remove_cached_files(downloaded)

        cadence = max(int(args.progress_every_days), 1)
        if day_number == 1 or day_number % cadence == 0 or day_number == len(issue_days):
            print(
                f"Processed issue days {day_number:,}/{len(issue_days):,}: "
                f"selected_files={selected_total:,}, "
                f"downloaded_or_cached={sum(1 for entry in downloaded_all if entry.download_status in {'downloaded', 'cached'}):,}, "
                f"point_rows={sum(len(frame) for frame in point_frames):,}, "
                f"purged_files={removed_total:,}",
                flush=True,
            )

    if not args.keep_all_updates:
        print(f"Kept latest WMO update per issue hour: {selected_total:,} of {discovered_total:,} files")

    point_df = pd.concat(point_frames, ignore_index=True, sort=False) if point_frames else pd.DataFrame()
    _finalize_outputs(
        args=args,
        downloaded=downloaded_all,
        point_df=point_df,
        expected_target_days=expected_target_days,
    )


def main() -> None:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    discover_start = start - timedelta(days=max(args.lookback_days, 0))
    wmo_prefixes = _parse_csv_strings(args.wmo_prefixes)
    issue_hours = _parse_csv_ints(args.issue_hours)

    expected_target_days = list(iter_dates(start, end))
    print(
        "Discovering NDFD MaxT WMO files "
        f"for target dates {start} to {end} "
        f"(issue dates {discover_start} to {end})"
    )
    if args.stream_by_day:
        _build_streamed(
            args=args,
            discover_start=discover_start,
            end=end,
            wmo_prefixes=wmo_prefixes,
            issue_hours=issue_hours,
            expected_target_days=expected_target_days,
        )
        return

    entries = discover_aws_wmo_range(
        discover_start,
        end,
        variable="maxt",
        wmo_prefixes=wmo_prefixes,
        issue_hours=issue_hours,
    )
    entries, before_count = _prepare_discovered_entries(
        entries,
        keep_all_updates=args.keep_all_updates,
    )
    if not args.keep_all_updates:
        print(f"Kept latest WMO update per issue hour: {len(entries):,} of {before_count:,} files")
    if args.max_files is not None:
        entries = entries[: args.max_files]

    downloaded = download_entries(
        entries,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    point_df = pd.DataFrame() if args.dry_run else _extract_downloaded_entries(downloaded, args=args)
    if args.purge_cache_after_extract and not args.dry_run:
        removed = _remove_cached_files(downloaded)
        print(f"Purged cached GRIB files after extraction: {removed:,}")
    _finalize_outputs(
        args=args,
        downloaded=downloaded,
        point_df=point_df,
        expected_target_days=expected_target_days,
    )


if __name__ == "__main__":
    main()
