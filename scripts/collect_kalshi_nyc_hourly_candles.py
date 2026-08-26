from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = ["HIGHNY0", "HIGHNY", "KXHIGHNY"]
OUT = Path("outputs/kalshi_nyc_hourly")
PERIOD_MINUTES = 60
MAX_WORKERS = int(os.getenv("KALSHI_WORKERS", "8"))
TIMEOUT = 25
SESSION_LOCK = Lock()

MARKET_FIELDS = [
    "ticker", "event_ticker", "source_series_ticker", "tier", "market_type",
    "yes_sub_title", "no_sub_title", "strike_type", "floor_strike", "cap_strike",
    "created_time", "open_time", "close_time", "settlement_ts", "status", "result",
    "volume_fp", "open_interest_fp"
]

CANDLE_FIELDS = [
    "ticker", "event_ticker", "source_series_ticker", "tier", "end_period_ts", "end_period_time_utc",
    "yes_bid_open", "yes_bid_low", "yes_bid_high", "yes_bid_close",
    "yes_ask_open", "yes_ask_low", "yes_ask_high", "yes_ask_close",
    "price_open", "price_low", "price_high", "price_close", "price_mean", "price_previous",
    "volume", "open_interest", "strike_type", "floor_strike", "cap_strike", "result",
]


def request_json(path: str, params: dict[str, Any] | None = None, *, attempts: int = 6) -> dict[str, Any]:
    url = f"{BASE}{path}"
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": "Kalshi-Weather-Trading historical backtest collector"})
            if r.status_code in {429, 500, 502, 503, 504}:
                wait = min(30.0, (2 ** attempt) + random.random())
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(min(20.0, (2 ** attempt) + random.random()))
    raise RuntimeError(f"GET {url} params={params} failed after {attempts} attempts: {last}")


def paginate_markets(path: str, series_ticker: str, tier: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = ""
    page = 0
    while True:
        params: dict[str, Any] = {"limit": 1000, "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        data = request_json(path, params)
        page += 1
        rows = data.get("markets") or []
        for m in rows:
            m = dict(m)
            m["source_series_ticker"] = series_ticker
            m["tier"] = tier
            out.append(m)
        cursor = data.get("cursor") or ""
        print(f"markets tier={tier} series={series_ticker} page={page} rows={len(rows)} total={len(out)}", flush=True)
        if not cursor:
            break
    return out


def parse_ts(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        x = float(value)
        if x > 10_000_000_000:
            x /= 1000.0
        return int(x)
    text = str(value)
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def nested_value(obj: dict[str, Any] | None, name: str) -> Any:
    if not obj:
        return None
    for key in (f"{name}_dollars", name):
        if key in obj:
            return obj[key]
    return None


def scalar_value(c: dict[str, Any], fp_name: str, legacy_name: str) -> Any:
    if fp_name in c:
        return c.get(fp_name)
    return c.get(legacy_name)


def fetch_market_candles(m: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    ticker = m["ticker"]
    series = m["source_series_ticker"]
    tier = m["tier"]
    start = parse_ts(m.get("open_time")) or parse_ts(m.get("created_time"))
    end = parse_ts(m.get("close_time")) or parse_ts(m.get("settlement_ts")) or int(time.time())
    if start is None:
        return [], {"ticker": ticker, "error": "missing_start_time"}
    start = max(0, start - 3600)
    end = max(start + 3600, end + 3600)
    if tier == "historical":
        path = f"/historical/markets/{ticker}/candlesticks"
    else:
        path = f"/series/{series}/markets/{ticker}/candlesticks"
    try:
        data = request_json(path, {"start_ts": start, "end_ts": end, "period_interval": PERIOD_MINUTES})
    except Exception as exc:
        return [], {"ticker": ticker, "tier": tier, "error": str(exc)}

    rows: list[dict[str, Any]] = []
    for c in data.get("candlesticks") or []:
        ts = c.get("end_period_ts")
        ts_int = parse_ts(ts)
        bid = c.get("yes_bid") or {}
        ask = c.get("yes_ask") or {}
        price = c.get("price") or {}
        rows.append({
            "ticker": ticker,
            "event_ticker": m.get("event_ticker"),
            "source_series_ticker": series,
            "tier": tier,
            "end_period_ts": ts_int,
            "end_period_time_utc": datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat() if ts_int else None,
            "yes_bid_open": nested_value(bid, "open"),
            "yes_bid_low": nested_value(bid, "low"),
            "yes_bid_high": nested_value(bid, "high"),
            "yes_bid_close": nested_value(bid, "close"),
            "yes_ask_open": nested_value(ask, "open"),
            "yes_ask_low": nested_value(ask, "low"),
            "yes_ask_high": nested_value(ask, "high"),
            "yes_ask_close": nested_value(ask, "close"),
            "price_open": nested_value(price, "open"),
            "price_low": nested_value(price, "low"),
            "price_high": nested_value(price, "high"),
            "price_close": nested_value(price, "close"),
            "price_mean": nested_value(price, "mean"),
            "price_previous": nested_value(price, "previous"),
            "volume": scalar_value(c, "volume_fp", "volume"),
            "open_interest": scalar_value(c, "open_interest_fp", "open_interest"),
            "strike_type": m.get("strike_type"),
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike"),
            "result": m.get("result"),
        })
    return rows, None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def year_from_event(event_ticker: str | None) -> str:
    if not event_ticker:
        return "unknown"
    parts = event_ticker.split("-")
    if len(parts) < 2 or len(parts[1]) < 2:
        return "unknown"
    yy = parts[1][:2]
    if not yy.isdigit():
        return "unknown"
    year = 2000 + int(yy)
    return str(year)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    markets: list[dict[str, Any]] = []
    discovery_errors: list[dict[str, Any]] = []

    for series in SERIES:
        for path, tier in (("/historical/markets", "historical"), ("/markets", "live")):
            try:
                markets.extend(paginate_markets(path, series, tier))
            except Exception as exc:
                discovery_errors.append({"series": series, "tier": tier, "error": str(exc)})
                print(f"DISCOVERY_ERROR series={series} tier={tier}: {exc}", flush=True)

    # Deduplicate by ticker, preferring the historical tier if a market appears in both.
    by_ticker: dict[str, dict[str, Any]] = {}
    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        old = by_ticker.get(ticker)
        if old is None or (old.get("tier") == "live" and m.get("tier") == "historical"):
            by_ticker[ticker] = m
    markets = sorted(by_ticker.values(), key=lambda x: (x.get("event_ticker") or "", x.get("ticker") or ""))
    print(f"unique NYC high markets={len(markets)}", flush=True)
    if not markets:
        raise SystemExit(f"No NYC high markets discovered; errors={discovery_errors}")

    market_rows = [{field: m.get(field) for field in MARKET_FIELDS} for m in markets]
    write_csv(OUT / "kalshi_nyc_high_markets.csv", market_rows, MARKET_FIELDS)

    all_candles: list[dict[str, Any]] = []
    candle_errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_market_candles, m): m for m in markets}
        for idx, fut in enumerate(as_completed(futs), start=1):
            rows, err = fut.result()
            all_candles.extend(rows)
            if err:
                candle_errors.append(err)
            if idx == 1 or idx % 250 == 0 or idx == len(futs):
                print(f"candles markets={idx}/{len(futs)} rows={len(all_candles)} errors={len(candle_errors)}", flush=True)

    all_candles.sort(key=lambda r: (r.get("end_period_ts") or 0, r.get("ticker") or ""))
    write_csv(OUT / "kalshi_nyc_high_hourly_candles_all.csv", all_candles, CANDLE_FIELDS)

    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in all_candles:
        year = year_from_event(row.get("event_ticker"))
        by_year.setdefault(year, []).append(row)
    for year, rows in sorted(by_year.items()):
        write_csv(OUT / f"kalshi_nyc_high_hourly_candles_{year}.csv", rows, CANDLE_FIELDS)

    nonempty_markets = len({r["ticker"] for r in all_candles})
    timestamps = [r["end_period_ts"] for r in all_candles if r.get("end_period_ts")]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_attempted": SERIES,
        "period_interval_minutes": PERIOD_MINUTES,
        "market_count": len(markets),
        "markets_with_candles": nonempty_markets,
        "candle_row_count": len(all_candles),
        "earliest_candle_utc": datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat() if timestamps else None,
        "latest_candle_utc": datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat() if timestamps else None,
        "discovery_error_count": len(discovery_errors),
        "discovery_errors": discovery_errors,
        "candle_error_count": len(candle_errors),
        "candle_errors_sample": candle_errors[:100],
        "rows_by_event_year": {year: len(rows) for year, rows in sorted(by_year.items())},
    }
    (OUT / "coverage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not all_candles:
        raise SystemExit("No candlestick rows returned")


if __name__ == "__main__":
    main()
