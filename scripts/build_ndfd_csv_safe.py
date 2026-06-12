from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.ndfd_extract import (  # noqa: E402
    POINT_FORECAST_COLUMNS,
    build_daily_features,
    extract_file_to_rows,
    validate_point_forecasts,
    write_dataframe,
)
from src.data.ndfd_fetch import CatalogEntry, infer_center, infer_issue_time  # noqa: E402


DEFAULT_VARIABLES = "maxt,temp,sky,wspd,wdir,pop12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely convert cached NDFD GRIB files into point and daily CSV files."
    )
    parser.add_argument("--input-dir", default="data/raw/ndfd")
    parser.add_argument("--lat", type=float, default=40.7812)
    parser.add_argument("--lon", type=float, default=-73.9665)
    parser.add_argument("--station", default="KNYC")
    parser.add_argument("--variables", default=DEFAULT_VARIABLES)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--per-file-timeout", type=int, default=20)
    parser.add_argument("--output", default="outputs/data/ndfd_knyc_point_forecasts.csv")
    parser.add_argument("--daily-output", default="outputs/data/ndfd_knyc_daily_features.csv")
    parser.add_argument("--report-output", default="outputs/data/ndfd_safe_conversion_report.csv")
    parser.add_argument("--worker-file", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def split_variables(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def date_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if len(part) == 8 and part.isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:8]}"
    return ""


def entry_from_file(path: Path) -> CatalogEntry:
    filename = path.name
    return CatalogEntry(
        date=date_from_path(path),
        catalog_url="",
        file_url="",
        filename=filename,
        inferred_variable=None,
        issue_time=infer_issue_time(filename),
        center=infer_center(filename),
        local_path=str(path),
        download_status="cached",
        skip_reason=None,
    )


def discover_cached_files(input_dir: Path, max_files: int | None = None) -> list[Path]:
    files = sorted(path for path in input_dir.rglob("*") if path.is_file())
    return files[:max_files] if max_files is not None else files


def worker_main(args: argparse.Namespace) -> int:
    path = Path(args.worker_file)
    entry = entry_from_file(path)
    rows = extract_file_to_rows(
        entry,
        lat=args.lat,
        lon=args.lon,
        station=args.station,
        requested_variables=split_variables(args.variables),
    )
    frame = pd.DataFrame(rows, columns=POINT_FORECAST_COLUMNS)
    print(frame.to_json(orient="records", lines=True), end="")
    return 0


def run_worker(path: Path, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.monotonic()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-file",
        str(path),
        "--lat",
        str(args.lat),
        "--lon",
        str(args.lon),
        "--station",
        args.station,
        "--variables",
        args.variables,
    ]
    report: dict[str, object] = {
        "local_file": str(path),
        "status": "started",
        "row_count": 0,
        "elapsed_seconds": None,
        "error": None,
    }
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.per_file_timeout,
        )
    except subprocess.TimeoutExpired:
        report["status"] = "timed_out"
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["error"] = f"exceeded {args.per_file_timeout}s timeout"
        return [], report

    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if completed.returncode != 0:
        report["status"] = "failed"
        report["error"] = completed.stderr.strip()[-1000:]
        return [], report

    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)

    report["status"] = "extracted" if rows else "empty"
    report["row_count"] = len(rows)
    if completed.stderr.strip():
        report["error"] = completed.stderr.strip()[-1000:]
    return rows, report


def write_point_rows(rows: list[dict[str, object]], output_path: Path, append: bool) -> None:
    frame = pd.DataFrame(rows, columns=POINT_FORECAST_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, mode="a" if append else "w", header=not append, index=False)


def write_empty_outputs(point_path: Path, daily_path: Path, report_path: Path) -> None:
    write_dataframe(pd.DataFrame(columns=POINT_FORECAST_COLUMNS), point_path)
    write_dataframe(build_daily_features(pd.DataFrame(columns=POINT_FORECAST_COLUMNS)), daily_path)
    write_dataframe(pd.DataFrame(columns=["local_file", "status", "row_count", "elapsed_seconds", "error"]), report_path)


def main() -> int:
    args = parse_args()
    if args.worker_file:
        return worker_main(args)

    input_dir = Path(args.input_dir)
    point_path = Path(args.output)
    daily_path = Path(args.daily_output)
    report_path = Path(args.report_output)
    files = discover_cached_files(input_dir, args.max_files)
    if not files:
        write_empty_outputs(point_path, daily_path, report_path)
        print(f"no cached NDFD files found under {input_dir}")
        return 0

    all_rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    point_path.parent.mkdir(parents=True, exist_ok=True)
    if point_path.exists():
        point_path.unlink()

    for index, path in enumerate(files, start=1):
        rows, report = run_worker(path, args)
        reports.append(report)
        if rows:
            all_rows.extend(rows)
            write_point_rows(rows, point_path, append=point_path.exists())
        print(
            f"[{index}/{len(files)}] {report['status']}: {path.name} "
            f"({report['row_count']} rows, {report['elapsed_seconds']}s)"
        )

    if not point_path.exists():
        write_dataframe(pd.DataFrame(columns=POINT_FORECAST_COLUMNS), point_path)

    point_df = pd.DataFrame(all_rows, columns=POINT_FORECAST_COLUMNS)
    daily_df = build_daily_features(point_df)
    write_dataframe(daily_df, daily_path)
    write_dataframe(pd.DataFrame(reports), report_path)

    warnings = validate_point_forecasts(point_df, args.lat, args.lon)
    print(f"processed files: {len(files)}")
    print(f"point rows written: {len(point_df)}")
    print(f"daily rows written: {len(daily_df)}")
    print(f"point forecast output: {point_path}")
    print(f"daily feature output: {daily_path}")
    print(f"conversion report: {report_path}")
    if warnings:
        print("validation warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"finished at: {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
