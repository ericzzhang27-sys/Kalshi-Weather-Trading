from __future__ import annotations

import argparse
import io
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py"

DEFAULT_START = "2018-01-01"
DEFAULT_END = date.today().isoformat()
DEFAULT_ASOS_OUTPUT = Path("data/raw/NYC_nws_hourly_2018_2026.csv")
DEFAULT_CLI_OUTPUT = Path("data/processed/knyc_cli_daily_2018_2026.csv")
DEFAULT_COVERAGE_OUTPUT = Path("outputs/data/knyc_backfill_coverage.json")

ASOS_COLUMNS = [
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "gust",
    "alti",
    "mslp",
    "p01i",
    "skyc1",
    "skyc2",
    "skyc3",
    "metar",
]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Kalshi-Weather-Trading research backfill/1.0"})
    return session


def _date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _monthly_windows(start: date, end: date):
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        window_start = max(start, cursor)
        window_end = min(end + timedelta(days=1), next_month)
        if window_start < window_end:
            yield window_start, window_end
        cursor = next_month


def fetch_asos(start: date, end: date, session: requests.Session, pause_seconds: float = 0.15) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    windows = list(_monthly_windows(start, end))
    for idx, (window_start, window_end) in enumerate(windows, start=1):
        params: list[tuple[str, str]] = [
            ("station", "NYC"),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"),
            ("elev", "no"),
            ("missing", "M"),
            ("trace", "T"),
            ("direct", "no"),
            ("report_type", "3"),
            ("report_type", "4"),
            ("sts", f"{window_start.isoformat()}T00:00:00Z"),
            ("ets", f"{window_end.isoformat()}T00:00:00Z"),
        ]
        params.extend(("data", col) for col in ASOS_COLUMNS)
        response = session.get(IEM_ASOS_URL, params=params, timeout=120)
        response.raise_for_status()
        text = response.text.strip()
        if text and not text.lower().startswith("error"):
            frame = pd.read_csv(io.StringIO(text), na_values=["M", "null", ""])
            if not frame.empty:
                frames.append(frame)
        print(f"ASOS {idx}/{len(windows)}: {window_start} -> {window_end}", flush=True)
        time.sleep(pause_seconds)

    if not frames:
        raise RuntimeError("IEM ASOS returned no KNYC/NYC observations")

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "valid" not in df.columns:
        raise RuntimeError(f"Unexpected IEM ASOS schema: {list(df.columns)}")
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce", utc=True)
    df = df[df["valid"].notna()].copy()
    df = df[(df["valid"].dt.date >= start) & (df["valid"].dt.date <= end)]
    df = df.drop_duplicates(subset=["station", "valid"], keep="last")
    return df.sort_values("valid").reset_index(drop=True)


def _cli_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "data", "records", "features"):
        value = payload.get(key)
        if isinstance(value, list):
            if key == "features":
                rows = []
                for feature in value:
                    if isinstance(feature, dict):
                        props = feature.get("properties")
                        if isinstance(props, dict):
                            rows.append(props)
                return rows
            return [row for row in value if isinstance(row, dict)]
    return []


def _first(row: dict, names: tuple[str, ...]):
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def fetch_cli(start: date, end: date, session: requests.Session) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(start.year, end.year + 1):
        response = session.get(IEM_CLI_URL, params={"station": "KNYC", "year": year}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        year_rows = _cli_records(payload)
        print(f"CLI {year}: {len(year_rows)} parsed records", flush=True)
        for raw in year_rows:
            valid = _first(raw, ("valid", "date", "day", "local_date"))
            high = _first(raw, ("high", "max_tmpf", "maximum", "max", "max_temp"))
            high_time = _first(raw, ("high_time", "max_tmpf_time", "max_time", "maximum_time"))
            low = _first(raw, ("low", "min_tmpf", "minimum", "min", "min_temp"))
            low_time = _first(raw, ("low_time", "min_tmpf_time", "min_time", "minimum_time"))
            precip = _first(raw, ("precip", "precipitation", "pday", "precip_today"))
            rows.append(
                {
                    "date": valid,
                    "station": "KNYC",
                    "cli_high_f": high,
                    "cli_high_time_local": high_time,
                    "cli_low_f": low,
                    "cli_low_time_local": low_time,
                    "cli_precip_in": precip,
                    "settlement_proxy_source": "iem_parsed_nws_cli",
                    "source_url": response.url,
                }
            )

    if not rows:
        raise RuntimeError("IEM CLI JSON returned no KNYC records")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["cli_high_f"] = pd.to_numeric(df["cli_high_f"], errors="coerce")
    df["cli_low_f"] = pd.to_numeric(df["cli_low_f"], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df[df["date"].between(start_ts, end_ts, inclusive="both")]
    df = df[df["date"].notna() & df["cli_high_f"].notna()]
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def coverage_report(asos: pd.DataFrame, cli: pd.DataFrame, start: date, end: date) -> dict:
    expected_days = (end - start).days + 1
    asos_dates = pd.to_datetime(asos["valid"], utc=True).dt.tz_convert("America/New_York").dt.normalize()
    cli_dates = pd.to_datetime(cli["date"]).dt.normalize()
    return {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "expected_days": expected_days,
        "asos_rows": int(len(asos)),
        "asos_first_utc": str(asos["valid"].min()),
        "asos_last_utc": str(asos["valid"].max()),
        "asos_unique_local_dates": int(asos_dates.nunique()),
        "cli_rows": int(len(cli)),
        "cli_first_date": str(cli_dates.min().date()) if len(cli) else None,
        "cli_last_date": str(cli_dates.max().date()) if len(cli) else None,
        "cli_day_coverage_fraction": round(float(cli_dates.nunique() / expected_days), 6),
        "notes": [
            "IEM ASOS is an observation archive; it is not the final settlement label.",
            "IEM parsed CLI is a strong historical proxy for the NWS CLI-based Kalshi settlement regime.",
            "Kalshi changed its temperature settlement provider to The Weather Company in August 2026; post-change rows should be cross-checked against TWC when an auditable archive is available.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill KNYC observations and parsed NWS CLI daily highs from IEM.")
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--asos-output", type=Path, default=DEFAULT_ASOS_OUTPUT)
    parser.add_argument("--cli-output", type=Path, default=DEFAULT_CLI_OUTPUT)
    parser.add_argument("--coverage-output", type=Path, default=DEFAULT_COVERAGE_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = _date(args.start_date)
    end = _date(args.end_date)
    if end < start:
        raise ValueError("end-date must be >= start-date")

    session = _session()
    asos = fetch_asos(start, end, session)
    cli = fetch_cli(start, end, session)

    args.asos_output.parent.mkdir(parents=True, exist_ok=True)
    args.cli_output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True)

    asos.to_csv(args.asos_output, index=False)
    cli.to_csv(args.cli_output, index=False)
    report = coverage_report(asos, cli, start, end)
    args.coverage_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote ASOS: {args.asos_output} ({len(asos):,} rows)")
    print(f"Wrote CLI: {args.cli_output} ({len(cli):,} rows)")
    print(f"Wrote coverage report: {args.coverage_output}")


if __name__ == "__main__":
    main()
