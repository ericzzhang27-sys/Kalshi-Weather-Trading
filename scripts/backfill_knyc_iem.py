from __future__ import annotations

import argparse
import io
import json
import time
from datetime import UTC, date, datetime, timedelta
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
ASOS_MIN_REQUEST_INTERVAL_SECONDS = 1.1

ASOS_COLUMNS = [
    "tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "mslp",
    "p01i", "skyc1", "skyc2", "skyc3", "metar",
]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Kalshi-Weather-Trading research backfill/1.0"})
    return session


def _date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _yearly_windows(start: date, end: date):
    for year in range(start.year, end.year + 1):
        window_start = max(start, date(year, 1, 1))
        window_end = min(end + timedelta(days=1), date(year + 1, 1, 1))
        if window_start < window_end:
            yield window_start, window_end


def _get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: list[tuple[str, str]] | dict[str, object],
    timeout: int,
    attempts: int = 5,
) -> requests.Response:
    delay = ASOS_MIN_REQUEST_INTERVAL_SECONDS
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in {429, 503} and attempt < attempts:
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def fetch_asos(start: date, end: date, session: requests.Session) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    windows = list(_yearly_windows(start, end))
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
        response = _get_with_retry(session, IEM_ASOS_URL, params=params, timeout=180)
        text = response.text.strip()
        if text and not text.lower().startswith("error"):
            frame = pd.read_csv(io.StringIO(text), na_values=["M", "null", ""])
            if not frame.empty:
                frames.append(frame)
        print(f"ASOS {idx}/{len(windows)}: {window_start} -> {window_end}", flush=True)
        time.sleep(ASOS_MIN_REQUEST_INTERVAL_SECONDS)

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
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [row for row in payload["results"] if isinstance(row, dict)]
    raise RuntimeError("Unexpected IEM CLI JSON schema: missing results list")


def _numeric_or_nan(value: object) -> float:
    if value in {None, "M", "T", ""}:
        return float("nan")
    return float(value)


def fetch_cli(start: date, end: date, session: requests.Session) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(start.year, end.year + 1):
        response = _get_with_retry(
            session,
            IEM_CLI_URL,
            params={"station": "KNYC", "year": year, "fmt": "json"},
            timeout=60,
        )
        year_rows = _cli_records(response.json())
        print(f"CLI {year}: {len(year_rows)} parsed records", flush=True)
        for raw in year_rows:
            high = _numeric_or_nan(raw.get("high"))
            low = _numeric_or_nan(raw.get("low"))
            rows.append(
                {
                    "date": raw.get("valid"),
                    "location": "NYC",
                    "actual_high": high,
                    "official_daily_high_f": high,
                    "actual_source": "iem_nws_cli_daily_high",
                    "source_file": raw.get("link") or response.url,
                    "source_station": raw.get("station", "KNYC"),
                    "source_station_name": raw.get("name", "NY CITY CENTRAL PARK"),
                    "cli_high_time_local": raw.get("high_time"),
                    "cli_low_f": low,
                    "cli_low_time_local": raw.get("low_time"),
                    "cli_precip_in": raw.get("precip"),
                    "cli_product_id": raw.get("product"),
                    "cli_wfo": raw.get("wfo"),
                }
            )

    if not rows:
        raise RuntimeError("IEM CLI JSON returned no KNYC records")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for column in ["actual_high", "official_daily_high_f", "cli_low_f"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["date"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")]
    df = df[df["date"].notna() & df["actual_high"].notna()]
    if not df["actual_high"].between(-20, 110).all():
        raise RuntimeError("CLI daily highs contain implausible Fahrenheit values")
    df = df.drop_duplicates(subset=["date", "location"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def coverage_report(asos: pd.DataFrame, cli: pd.DataFrame, start: date, end: date) -> dict:
    expected_days = (end - start).days + 1
    asos_dates = pd.to_datetime(asos["valid"], utc=True).dt.tz_convert("America/New_York").dt.normalize()
    cli_dates = pd.to_datetime(cli["date"]).dt.normalize()
    missing_cli_dates = pd.date_range(start, end, freq="D").difference(cli_dates)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
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
        "cli_missing_date_count": int(len(missing_cli_dates)),
        "cli_missing_dates": [d.date().isoformat() for d in missing_cli_dates[:100]],
        "notes": [
            "IEM ASOS is an observation archive; it is not the final settlement label.",
            "IEM CLI rows are parsed from NWS Daily Climate Report text products and are the preferred historical settlement proxy for the former NWS-CLI Kalshi regime.",
            "The CLI output is shaped to the project's existing daily-target audit contract: actual_high, official_daily_high_f, actual_source, station, and source metadata.",
            "Post-provider-change 2026 rows should be cross-checked against The Weather Company before treating CLI as exact settlement truth.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill KNYC ASOS observations and NWS CLI daily highs from IEM.")
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

    for path in [args.asos_output, args.cli_output, args.coverage_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    asos.to_csv(args.asos_output, index=False)
    cli.to_csv(args.cli_output, index=False)
    args.coverage_output.write_text(
        json.dumps(coverage_report(asos, cli, start, end), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote ASOS: {args.asos_output} ({len(asos):,} rows)")
    print(f"Wrote CLI: {args.cli_output} ({len(cli):,} rows)")
    print(f"Wrote coverage report: {args.coverage_output}")


if __name__ == "__main__":
    main()
