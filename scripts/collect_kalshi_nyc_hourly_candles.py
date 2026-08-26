from __future__ import annotations

import csv
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = ["HIGHNY0", "HIGHNY", "KXHIGHNY"]
OUT = Path("outputs/kalshi_nyc_hourly")
PERIOD_MINUTES = 60
MAX_WORKERS = int(os.getenv("KALSHI_WORKERS", "16"))
TIMEOUT = 20
USER_AGENT = "Kalshi-Weather-Trading historical backtest collector"
_THREAD_LOCAL = threading.local()

MARKET_FIELDS = [
    "ticker", "event_ticker", "source_series_ticker", "tier", "market_type",
    "yes_sub_title", "no_sub_title", "strike_type", "floor_strike", "cap_strike",
    "created_time", "open_time", "close_time", "settlement_ts", "status", "result",
    "volume_fp", "open_interest_fp",
]
CANDLE_FIELDS = [
    "ticker", "event_ticker", "source_series_ticker", "tier", "end_period_ts", "end_period_time_utc",
    "yes_bid_open", "yes_bid_low", "yes_bid_high", "yes_bid_close",
    "yes_ask_open", "yes_ask_low", "yes_ask_high", "yes_ask_close",
    "price_open", "price_low", "price_high", "price_close", "price_mean", "price_previous",
    "volume", "open_interest", "strike_type", "floor_strike", "cap_strike", "result",
]


def session() -> requests.Session:
    s = getattr(_THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8))
        _THREAD_LOCAL.session = s
    return s


def request_json(path: str, params: dict[str, Any] | None = None, *, attempts: int = 5) -> dict[str, Any]:
    url = f"{BASE}{path}"
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = session().get(url, params=params, timeout=TIMEOUT)
            if r.status_code in {400, 401, 403, 404, 422}:
                raise RuntimeError(f"HTTP {r.status_code}: {r.url}: {r.text[:300]}")
            if r.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(20.0, (2 ** attempt) + random.random()))
                continue
            r.raise_for_status()
            return r.json()
        except RuntimeError:
            raise
        except Exception as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(min(12.0, (2 ** attempt) + random.random()))
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
        for market in rows:
            m = dict(market)
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
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def nested_value(obj: dict[str, Any] | None, name: str) -> Any:
    if not obj:
        return None
    for key in (f"{name}_dollars", name):
        if key in obj:
            return obj[key]
    return None


def scalar_value(candle: dict[str, Any], fp_name: str, legacy_name: str) -> Any:
    return candle.get(fp_name) if fp_name in candle else candle.get(legacy_name)


def candle_row(c: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    ts_int = parse_ts(c.get("end_period_ts"))
    bid, ask, price = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
    return {
        "ticker": market["ticker"], "event_ticker": market.get("event_ticker"),
        "source_series_ticker": market.get("source_series_ticker"), "tier": market.get("tier"),
        "end_period_ts": ts_int,
        "end_period_time_utc": datetime.fromtimestamp(ts_int, tz=timezone.utc).isoformat() if ts_int else None,
        "yes_bid_open": nested_value(bid, "open"), "yes_bid_low": nested_value(bid, "low"),
        "yes_bid_high": nested_value(bid, "high"), "yes_bid_close": nested_value(bid, "close"),
        "yes_ask_open": nested_value(ask, "open"), "yes_ask_low": nested_value(ask, "low"),
        "yes_ask_high": nested_value(ask, "high"), "yes_ask_close": nested_value(ask, "close"),
        "price_open": nested_value(price, "open"), "price_low": nested_value(price, "low"),
        "price_high": nested_value(price, "high"), "price_close": nested_value(price, "close"),
        "price_mean": nested_value(price, "mean"), "price_previous": nested_value(price, "previous"),
        "volume": scalar_value(c, "volume_fp", "volume"), "open_interest": scalar_value(c, "open_interest_fp", "open_interest"),
        "strike_type": market.get("strike_type"), "floor_strike": market.get("floor_strike"),
        "cap_strike": market.get("cap_strike"), "result": market.get("result"),
    }


def market_range(markets: list[dict[str, Any]]) -> tuple[int, int]:
    starts = [parse_ts(m.get("open_time")) or parse_ts(m.get("created_time")) for m in markets]
    ends = [parse_ts(m.get("close_time")) or parse_ts(m.get("settlement_ts")) for m in markets]
    starts, ends = [v for v in starts if v is not None], [v for v in ends if v is not None]
    start = min(starts) if starts else int(time.time()) - 172800
    end = max(ends) if ends else int(time.time())
    return max(0, start - 3600), max(start + 3600, end + 3600)


def fetch_single_market(market: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    start, end = market_range([market])
    ticker = market["ticker"]
    if market.get("tier") == "historical":
        path = f"/historical/markets/{quote(ticker, safe='')}/candlesticks"
    else:
        series = market["source_series_ticker"]
        path = f"/series/{quote(series, safe='')}/markets/{quote(ticker, safe='')}/candlesticks"
    try:
        data = request_json(path, {"start_ts": start, "end_ts": end, "period_interval": PERIOD_MINUTES})
        return [candle_row(c, market) for c in data.get("candlesticks") or []], None
    except Exception as exc:
        return [], {"ticker": ticker, "event_ticker": market.get("event_ticker"), "stage": "market_candles", "error": str(exc)}


def fetch_event(item: tuple[tuple[str, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    (series, event_ticker), markets = item
    start, end = market_range(markets)
    market_map = {m["ticker"]: m for m in markets}
    path = f"/series/{quote(series, safe='')}/events/{quote(event_ticker, safe='')}/candlesticks"
    try:
        data = request_json(path, {"start_ts": start, "end_ts": end, "period_interval": PERIOD_MINUTES})
        tickers, sets = data.get("market_tickers") or [], data.get("market_candlesticks") or []
        ticker_set = set(tickers)
        rows: list[dict[str, Any]] = []
        matched = 0
        for ticker, candles in zip(tickers, sets):
            market = market_map.get(ticker)
            if market is None:
                continue
            matched += 1
            rows.extend(candle_row(c, market) for c in candles or [])
        if matched:
            errors: list[dict[str, Any]] = []
            for market in markets:
                if market["ticker"] in ticker_set:
                    continue
                extra, err = fetch_single_market(market)
                rows.extend(extra)
                if err:
                    errors.append(err)
            return rows, errors, "event"
    except Exception as exc:
        event_error = str(exc)
    else:
        event_error = "event endpoint returned no matching market tickers"

    rows, errors = [], []
    for market in markets:
        extra, err = fetch_single_market(market)
        rows.extend(extra)
        if err:
            errors.append(err)
    if errors and len(errors) == len(markets):
        errors.insert(0, {"event_ticker": event_ticker, "stage": "event_candles", "error": event_error})
    return rows, errors, "market_fallback"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def event_year(event_ticker: str | None, ts: int | None = None) -> str:
    if event_ticker:
        parts = event_ticker.split("-")
        if len(parts) > 1 and len(parts[1]) >= 2 and parts[1][:2].isdigit():
            return str(2000 + int(parts[1][:2]))
    return str(datetime.fromtimestamp(ts, tz=timezone.utc).year) if ts else "unknown"


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

    by_ticker: dict[str, dict[str, Any]] = {}
    for market in markets:
        ticker = market.get("ticker")
        if not ticker:
            continue
        old = by_ticker.get(ticker)
        if old is None or (old.get("tier") == "live" and market.get("tier") == "historical"):
            by_ticker[ticker] = market
    markets = sorted(by_ticker.values(), key=lambda x: (x.get("event_ticker") or "", x.get("ticker") or ""))
    print(f"unique NYC high markets={len(markets)}", flush=True)
    if not markets:
        raise SystemExit(f"No NYC high markets discovered; errors={discovery_errors}")
    write_csv(OUT / "kalshi_nyc_high_markets.csv", [{field: m.get(field) for field in MARKET_FIELDS} for m in markets], MARKET_FIELDS)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    ungrouped: list[dict[str, Any]] = []
    for market in markets:
        event, series = market.get("event_ticker"), market.get("source_series_ticker")
        if event and series: grouped.setdefault((series, event), []).append(market)
        else: ungrouped.append(market)
    print(f"event groups={len(grouped)} ungrouped_markets={len(ungrouped)} workers={MAX_WORKERS}", flush=True)

    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    event_fast = fallback = 0
    with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as executor:
        futures = {executor.submit(fetch_event, item): item[0] for item in grouped.items()}
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try: rows, errs, mode = future.result()
            except Exception as exc: rows, errs, mode = [], [{"event_ticker": key[1], "stage": "event_worker", "error": str(exc)}], "market_fallback"
            all_rows.extend(rows); errors.extend(errs)
            event_fast += mode == "event"; fallback += mode != "event"
            if idx == 1 or idx % 100 == 0 or idx == total:
                print(f"events={idx}/{total} rows={len(all_rows)} errors={len(errors)} event_fast={event_fast} fallback={fallback}", flush=True)
    for market in ungrouped:
        rows, err = fetch_single_market(market); all_rows.extend(rows)
        if err: errors.append(err)

    deduped = {(r["ticker"], r.get("end_period_ts")): r for r in all_rows}
    all_rows = sorted(deduped.values(), key=lambda r: (r.get("end_period_ts") or 0, r.get("ticker") or ""))
    write_csv(OUT / "kalshi_nyc_high_hourly_candles_all.csv", all_rows, CANDLE_FIELDS)
    by_year: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows: by_year.setdefault(event_year(row.get("event_ticker"), row.get("end_period_ts")), []).append(row)
    for year, rows in sorted(by_year.items()): write_csv(OUT / f"kalshi_nyc_high_hourly_candles_{year}.csv", rows, CANDLE_FIELDS)

    timestamps = [int(r["end_period_ts"]) for r in all_rows if r.get("end_period_ts")]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "series_attempted": SERIES,
        "period_interval_minutes": PERIOD_MINUTES, "market_count": len(markets), "event_group_count": len(grouped),
        "event_fast_path_count": int(event_fast), "market_fallback_event_count": int(fallback),
        "markets_with_candles": len({r["ticker"] for r in all_rows}), "candle_row_count": len(all_rows),
        "earliest_candle_utc": datetime.fromtimestamp(min(timestamps), tz=timezone.utc).isoformat() if timestamps else None,
        "latest_candle_utc": datetime.fromtimestamp(max(timestamps), tz=timezone.utc).isoformat() if timestamps else None,
        "discovery_error_count": len(discovery_errors), "discovery_errors": discovery_errors,
        "candle_error_count": len(errors), "candle_errors_sample": errors[:100],
        "rows_by_event_year": {year: len(rows) for year, rows in sorted(by_year.items())},
    }
    (OUT / "coverage_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not all_rows: raise SystemExit("No candlestick rows returned")


if __name__ == "__main__":
    main()
