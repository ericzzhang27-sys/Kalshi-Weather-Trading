from __future__ import annotations

import argparse
import io
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from xml.etree import ElementTree

import pandas as pd
import requests

NCEI_THREDDS_ROOT = "https://www.ncei.noaa.gov/thredds"
LEGACY_DATASET_ROOT = "model-ndfd-file_kwbn-old"
CURRENT_DATASET_ROOT = "model-ndfd-file"
CURRENT_ROOT_START = date(2020, 6, 1)
ONLINE_START = date(2018, 6, 1)
DEFAULT_WMO_HEADER = "YGUZ98"
DEFAULT_CENTER = "KWBN"
DEFAULT_LAT = 40.7812
DEFAULT_LON = -73.9665
DEFAULT_TIMEZONE = "America/New_York"
USER_AGENT = "Kalshi-Weather-Trading NDFD point-vintage backfill"

OUTPUT_COLUMNS = [
    "date",
    "location",
    "forecast_high",
    "forecast_source",
    "forecast_issue_time",
    "nws_forecast_high_f",
    "ndfd_valid_time_utc",
    "ndfd_lead_hours",
    "ndfd_grid_lat",
    "ndfd_grid_lon",
    "ndfd_grid_distance_km",
    "ndfd_wmo_header",
    "ndfd_center",
    "ndfd_archive_root",
    "ndfd_archive_host",
    "ndfd_source_file",
    "ndfd_catalog_url",
    "ndfd_ncss_url",
]


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def archive_root(day: date) -> str:
    return LEGACY_DATASET_ROOT if day < CURRENT_ROOT_START else CURRENT_DATASET_ROOT


def catalog_url(day: date) -> str:
    root = archive_root(day)
    return (
        f"{NCEI_THREDDS_ROOT}/catalog/{root}/"
        f"{day:%Y%m}/{day:%Y%m%d}/catalog.xml"
    )


def _get_text(
    url: str,
    *,
    params: dict[str, object] | None = None,
    timeout: int = 45,
    retries: int = 4,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 404:
                raise FileNotFoundError(f"404: {response.url}")
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.url}")
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


def infer_issue_time(filename: str) -> pd.Timestamp | None:
    match = re.search(r"_(\d{12})$", filename)
    if not match:
        return None
    return pd.Timestamp(datetime.strptime(match.group(1), "%Y%m%d%H%M"), tz="UTC")


def discover_day_files(
    day: date,
    wmo_header: str = DEFAULT_WMO_HEADER,
    center: str = DEFAULT_CENTER,
) -> tuple[list[dict[str, object]], str | None]:
    url = catalog_url(day)
    try:
        content = _get_text(url)
    except Exception as exc:
        return [], str(exc)

    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        return [], f"catalog_parse_error: {exc}"

    prefix = f"{wmo_header.upper()}_{center.upper()}_"
    discovered: list[dict[str, object]] = []
    seen: set[str] = set()
    for node in root.iter():
        url_path = node.attrib.get("urlPath")
        if not url_path:
            continue
        filename = Path(url_path).name
        if not filename.upper().startswith(prefix):
            continue
        if filename in seen:
            continue
        issue_time = infer_issue_time(filename)
        if issue_time is None:
            continue
        seen.add(filename)
        discovered.append(
            {
                "issue_date": day.isoformat(),
                "issue_time": issue_time,
                "filename": filename,
                "url_path": url_path.lstrip("/"),
                "catalog_url": url,
                "archive_root": archive_root(day),
            }
        )

    discovered.sort(key=lambda row: (row["issue_time"], row["filename"]))
    return discovered, None


def _normalise_header(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")


def _find_time_column(columns: Iterable[object]) -> str | None:
    columns = [str(column) for column in columns]
    for target in ["time", "date", "valid_time"]:
        for column in columns:
            base = _normalise_header(column)
            if base == target or base.startswith(target + "_"):
                return column
    for column in columns:
        base = _normalise_header(column)
        if "time" in base and "reftime" not in base:
            return column
    return None


def _find_coordinate_column(columns: Iterable[object], kind: str) -> str | None:
    for column in columns:
        base = _normalise_header(column)
        if kind == "lat" and (base == "latitude" or base.startswith("latitude_")):
            return str(column)
        if kind == "lon" and (base == "longitude" or base.startswith("longitude_")):
            return str(column)
    return None


def _find_maxt_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        base = _normalise_header(column)
        if "maximum" in base and "temperature" in base:
            return str(column)
    excluded = {
        _find_time_column(frame.columns),
        _find_coordinate_column(frame.columns, "lat"),
        _find_coordinate_column(frame.columns, "lon"),
    }
    candidates: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            candidates.append(str(column))
    return candidates[0] if len(candidates) == 1 else None


def _temperature_unit(column: str, values: pd.Series) -> str:
    text = column.lower()
    match = re.search(r"unit\s*=\s*[\"']?([^\]\"']+)", text)
    if match:
        unit = match.group(1).strip().lower()
        if unit in {"k", "kelvin"}:
            return "k"
        if unit in {"c", "degc", "celsius", "degree_celsius", "degrees_celsius"}:
            return "c"
        if unit in {"f", "degf", "fahrenheit", "degree_fahrenheit", "degrees_fahrenheit"}:
            return "f"
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if not numeric.empty and numeric.median() > 150:
        return "k"
    return "f"


def _to_fahrenheit(values: pd.Series, unit: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if unit == "k":
        return (numeric - 273.15) * 9.0 / 5.0 + 32.0
    if unit == "c":
        return numeric * 9.0 / 5.0 + 32.0
    return numeric


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def fetch_point_rows(
    item: dict[str, object],
    *,
    lat: float,
    lon: float,
    timezone_name: str,
    target_start: date,
    target_end: date,
    location: str,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    url_path = str(item["url_path"])
    ncss_url = f"{NCEI_THREDDS_ROOT}/ncss/grid/{url_path}"
    params: dict[str, object] = {
        "var": "all",
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "time": "all",
        "accept": "csv",
    }
    try:
        text = _get_text(ncss_url, params=params, timeout=60, retries=4)
        frame = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        return [], {"filename": item["filename"], "stage": "ncss_fetch", "error": str(exc)}

    if frame.empty:
        return [], {"filename": item["filename"], "stage": "ncss_parse", "error": "empty_csv"}

    time_column = _find_time_column(frame.columns)
    value_column = _find_maxt_column(frame)
    if time_column is None or value_column is None:
        return [], {
            "filename": item["filename"],
            "stage": "ncss_parse",
            "error": f"columns_not_identified: {list(frame.columns)}",
        }

    valid_times = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    unit = _temperature_unit(value_column, frame[value_column])
    highs_f = _to_fahrenheit(frame[value_column], unit)
    lat_column = _find_coordinate_column(frame.columns, "lat")
    lon_column = _find_coordinate_column(frame.columns, "lon")
    issue_time = pd.Timestamp(item["issue_time"])
    if issue_time.tzinfo is None:
        issue_time = issue_time.tz_localize("UTC")
    else:
        issue_time = issue_time.tz_convert("UTC")

    rows: list[dict[str, object]] = []
    for index in frame.index:
        valid_time = valid_times.loc[index]
        high_f = highs_f.loc[index]
        if pd.isna(valid_time) or pd.isna(high_f):
            continue
        if not (-20 <= float(high_f) <= 120):
            continue
        target_date = valid_time.tz_convert(timezone_name).date()
        if target_date < target_start or target_date > target_end:
            continue

        grid_lat = (
            pd.to_numeric(pd.Series([frame.loc[index, lat_column]]), errors="coerce").iloc[0]
            if lat_column
            else float("nan")
        )
        grid_lon = (
            pd.to_numeric(pd.Series([frame.loc[index, lon_column]]), errors="coerce").iloc[0]
            if lon_column
            else float("nan")
        )
        if pd.notna(grid_lon) and float(grid_lon) > 180:
            grid_lon = float(grid_lon) - 360.0
        grid_distance = (
            _haversine_km(float(grid_lat), float(grid_lon), lat, lon)
            if pd.notna(grid_lat) and pd.notna(grid_lon)
            else float("nan")
        )
        query_url = f"{ncss_url}?{urlencode(params)}"
        rows.append(
            {
                "date": target_date.isoformat(),
                "location": location,
                "forecast_high": round(float(high_f), 6),
                "forecast_source": "nws_ndfd_operational_maxt_archive",
                "forecast_issue_time": issue_time.isoformat(),
                "nws_forecast_high_f": round(float(high_f), 6),
                "ndfd_valid_time_utc": valid_time.isoformat(),
                "ndfd_lead_hours": round((valid_time - issue_time).total_seconds() / 3600.0, 6),
                "ndfd_grid_lat": None if pd.isna(grid_lat) else float(grid_lat),
                "ndfd_grid_lon": None if pd.isna(grid_lon) else float(grid_lon),
                "ndfd_grid_distance_km": None if pd.isna(grid_distance) else round(float(grid_distance), 6),
                "ndfd_wmo_header": DEFAULT_WMO_HEADER,
                "ndfd_center": DEFAULT_CENTER,
                "ndfd_archive_root": item["archive_root"],
                "ndfd_archive_host": "ncei_thredds",
                "ndfd_source_file": item["filename"],
                "ndfd_catalog_url": item["catalog_url"],
                "ndfd_ncss_url": query_url,
            }
        )
    return rows, None


def build_range(
    *,
    start: date,
    end: date,
    lat: float,
    lon: float,
    timezone_name: str,
    location: str,
    workers: int,
    progress_every_days: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if start < ONLINE_START:
        raise ValueError(f"Direct online NCEI archive begins at {ONLINE_START}; requested {start}")
    today = datetime.now(timezone.utc).date()
    if end > today:
        end = today
    if end < start:
        raise ValueError("end must not precede start")

    issue_start = start - timedelta(days=1)
    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    catalog_missing_days: list[str] = []
    files_discovered = 0
    days = list(iter_dates(issue_start, end))

    for day_number, issue_day in enumerate(days, start=1):
        files, catalog_error = discover_day_files(issue_day)
        if catalog_error:
            catalog_missing_days.append(issue_day.isoformat())
            failures.append(
                {
                    "filename": "",
                    "stage": "catalog",
                    "issue_date": issue_day.isoformat(),
                    "error": catalog_error,
                }
            )
            files = []
        files_discovered += len(files)

        if files:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = [
                    executor.submit(
                        fetch_point_rows,
                        item,
                        lat=lat,
                        lon=lon,
                        timezone_name=timezone_name,
                        target_start=start,
                        target_end=end,
                        location=location,
                    )
                    for item in files
                ]
                for future in as_completed(futures):
                    rows, failure = future.result()
                    all_rows.extend(rows)
                    if failure:
                        failures.append(failure)

        cadence = max(1, progress_every_days)
        if day_number == 1 or day_number % cadence == 0 or day_number == len(days):
            print(
                f"issue_days={day_number}/{len(days)} files={files_discovered} "
                f"rows={len(all_rows)} failures={len(failures)}",
                flush=True,
            )

    result = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    if not result.empty:
        result["forecast_issue_time"] = pd.to_datetime(result["forecast_issue_time"], utc=True)
        result["ndfd_valid_time_utc"] = pd.to_datetime(result["ndfd_valid_time_utc"], utc=True)
        result = result.drop_duplicates(
            subset=["date", "location", "forecast_issue_time", "ndfd_valid_time_utc", "ndfd_wmo_header"],
            keep="last",
        )
        result = result.sort_values(["date", "forecast_issue_time", "ndfd_valid_time_utc"]).reset_index(drop=True)
        result["forecast_issue_time"] = result["forecast_issue_time"].map(lambda value: value.isoformat())
        result["ndfd_valid_time_utc"] = result["ndfd_valid_time_utc"].map(lambda value: value.isoformat())

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "issue_start": issue_start.isoformat(),
        "files_discovered": files_discovered,
        "rows_extracted": int(len(result)),
        "unique_target_dates": int(pd.to_datetime(result["date"]).nunique()) if not result.empty else 0,
        "unique_issue_times": int(pd.to_datetime(result["forecast_issue_time"]).nunique()) if not result.empty else 0,
        "catalog_missing_day_count": len(catalog_missing_days),
        "catalog_missing_days": catalog_missing_days[:100],
        "failure_count": len(failures),
        "failures": failures[:100],
        "wmo_header": DEFAULT_WMO_HEADER,
        "center": DEFAULT_CENTER,
        "target_lat": lat,
        "target_lon": lon,
        "archive_roots": sorted(result["ndfd_archive_root"].dropna().unique().tolist()) if not result.empty else [],
    }
    return result, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill true point-in-time NDFD MaxT vintages at KNYC using "
            "NCEI THREDDS NCSS point extraction."
        )
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--location", default="NYC")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--progress-every-days", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, report = build_range(
        start=parse_date(args.start_date),
        end=parse_date(args.end_date),
        lat=args.lat,
        lon=args.lon,
        timezone_name=args.timezone,
        location=args.location,
        workers=args.workers,
        progress_every_days=args.progress_every_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if frame.empty:
        raise SystemExit("No NDFD point vintages were extracted")


if __name__ == "__main__":
    main()
