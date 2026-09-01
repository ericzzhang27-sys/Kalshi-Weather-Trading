#!/usr/bin/env python3
"""
Compile the deepest practical historical Kalshi NYC daily-high price dataset.

Sources
-------
1) TrevorJS/kalshi-trades (Hugging Face), public Kalshi trade/market archive:
   June 2021 -> January 2026.
2) Kalshi public REST API, historical + live tiers:
   used to extend the static archive from its last NYC trade through runtime.

Historical NYC ticker families handled:
- HIGHNY0-*   (early legacy series)
- HIGHNY-*    (legacy series)
- KXHIGHNY-*  (current series)

Outputs
-------
nyc_trades.parquet
nyc_markets.parquet
nyc_bars_15m.parquet
nyc_bars_1h.parquet
coverage_by_year.csv
coverage_report.json

Optional --csv also writes gzip-compressed CSV copies.

No Kalshi API key is required for the public market/trade endpoints used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import duckdb
import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
HF_TRADES_PATTERNS = [
    "hf://datasets/TrevorJS/kalshi-trades/trades-*.parquet",
    # Fallback in case the repository layout changes back to folders.
    "hf://datasets/TrevorJS/kalshi-trades/trades/*.parquet",
]
HF_MARKETS_PATTERNS = [
    "hf://datasets/TrevorJS/kalshi-trades/markets-*.parquet",
    "hf://datasets/TrevorJS/kalshi-trades/markets/*.parquet",
]
NYC_PREFIX_RE = r"^(HIGHNY0|HIGHNY|KXHIGHNY)-"
CURRENT_SERIES = "KXHIGHNY"
DEFAULT_OVERLAP_DAYS = 7
USER_AGENT = "kalshi-nyc-history-compiler/1.0"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def decimal_text(value: Any, scale: int = 6) -> str:
    """Normalize numeric-ish API values while preserving sub-cent prices."""
    if value is None or value == "":
        return ""
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return ""
    q = Decimal(1).scaleb(-scale)
    return format(d.quantize(q), "f")


class KalshiSession:
    def __init__(self, request_delay: float = 0.05, max_retries: int = 8):
        self.request_delay = max(0.0, request_delay)
        self.max_retries = max_retries
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})

    def get_json(self, path: str, params: dict[str, Any] | None = None,
                 allow_404: bool = False) -> dict[str, Any] | None:
        url = BASE_URL + path
        for attempt in range(self.max_retries):
            try:
                r = self.s.get(url, params=params, timeout=45)
            except requests.RequestException as exc:
                if attempt + 1 == self.max_retries:
                    raise RuntimeError(f"Request failed after retries: {url}") from exc
                time.sleep(min(30, 0.75 * (2 ** attempt)))
                continue

            if allow_404 and r.status_code == 404:
                return None
            if r.status_code == 429 or 500 <= r.status_code < 600:
                retry_after = r.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else min(30, 0.75 * (2 ** attempt))
                except ValueError:
                    delay = min(30, 0.75 * (2 ** attempt))
                time.sleep(delay)
                continue

            try:
                r.raise_for_status()
            except requests.HTTPError as exc:
                body = r.text[:500]
                raise RuntimeError(f"HTTP {r.status_code} for {r.url}: {body}") from exc

            if self.request_delay:
                time.sleep(self.request_delay)
            return r.json()

        raise RuntimeError(f"Request failed after retries: {url}")

    def paginate(self, path: str, collection_key: str,
                 params: dict[str, Any] | None = None,
                 allow_404: bool = False) -> Iterable[dict[str, Any]]:
        p = dict(params or {})
        p.setdefault("limit", 1000)
        cursor: str | None = None
        while True:
            if cursor:
                p["cursor"] = cursor
            elif "cursor" in p:
                p.pop("cursor", None)

            payload = self.get_json(path, p, allow_404=allow_404)
            if payload is None:
                return
            for row in payload.get(collection_key, []) or []:
                yield row
            cursor = payload.get("cursor")
            if not cursor:
                break


def init_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # Modern DuckDB autoloads httpfs, but explicitly load/install for portability.
    try:
        con.execute("LOAD httpfs")
    except Exception:
        try:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
        except Exception as exc:
            raise RuntimeError(
                "DuckDB could not load httpfs, which is required for hf:// access. "
                "Check that this machine has internet access, then rerun."
            ) from exc
    return con


def select_working_hf_pattern(con: duckdb.DuckDBPyConnection,
                              patterns: list[str]) -> str:
    last_exc: Exception | None = None
    for pattern in patterns:
        try:
            con.execute(
                f"SELECT 1 FROM read_parquet('{pattern}') LIMIT 1"
            ).fetchone()
            return pattern
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        "Could not read the Hugging Face Parquet archive with any known layout."
    ) from last_exc


def extract_static_archive(con: duckdb.DuckDBPyConnection, out: Path) -> tuple[Path, Path]:
    print("[1/6] Locating Hugging Face archive...")
    trades_pattern = select_working_hf_pattern(con, HF_TRADES_PATTERNS)
    markets_pattern = select_working_hf_pattern(con, HF_MARKETS_PATTERNS)
    print(f"      trades:  {trades_pattern}")
    print(f"      markets: {markets_pattern}")

    archive_trades = out / "_archive_nyc_trades.parquet"
    archive_markets = out / "_archive_nyc_markets.parquet"

    print("[2/6] Extracting NYC trade rows from 2021-2026 archive...")
    con.execute(f"""
        COPY (
            SELECT
                CAST(trade_id AS VARCHAR) AS trade_id,
                CAST(ticker AS VARCHAR) AS ticker,
                CAST(count AS DECIMAL(20,2)) AS count_fp,
                CAST(yes_price AS DECIMAL(20,6)) / 100 AS yes_price_dollars,
                CAST(no_price AS DECIMAL(20,6)) / 100 AS no_price_dollars,
                CAST(taker_side AS VARCHAR) AS taker_outcome_side,
                CAST(created_time AS TIMESTAMPTZ) AS created_time,
                'hf_archive'::VARCHAR AS source
            FROM read_parquet('{trades_pattern}')
            WHERE regexp_matches(CAST(ticker AS VARCHAR), '{NYC_PREFIX_RE}')
        )
        TO '{archive_trades.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print("[3/6] Extracting NYC market metadata from archive...")
    con.execute(f"""
        COPY (
            SELECT
                CAST(ticker AS VARCHAR) AS ticker,
                CAST(event_ticker AS VARCHAR) AS event_ticker,
                CAST(title AS VARCHAR) AS title,
                CAST(yes_sub_title AS VARCHAR) AS yes_sub_title,
                CAST(no_sub_title AS VARCHAR) AS no_sub_title,
                CAST(status AS VARCHAR) AS status,
                CAST(result AS VARCHAR) AS result,
                CAST(created_time AS TIMESTAMPTZ) AS created_time,
                CAST(open_time AS TIMESTAMPTZ) AS open_time,
                CAST(close_time AS TIMESTAMPTZ) AS close_time,
                CAST(yes_bid AS DECIMAL(20,6)) / 100 AS yes_bid_dollars,
                CAST(yes_ask AS DECIMAL(20,6)) / 100 AS yes_ask_dollars,
                CAST(no_bid AS DECIMAL(20,6)) / 100 AS no_bid_dollars,
                CAST(no_ask AS DECIMAL(20,6)) / 100 AS no_ask_dollars,
                CAST(last_price AS DECIMAL(20,6)) / 100 AS last_price_dollars,
                CAST(volume AS DECIMAL(24,2)) AS volume_fp,
                CAST(open_interest AS DECIMAL(24,2)) AS open_interest_fp,
                NULL::VARCHAR AS strike_type,
                NULL::DOUBLE AS floor_strike,
                NULL::DOUBLE AS cap_strike,
                'hf_archive'::VARCHAR AS source
            FROM read_parquet('{markets_pattern}')
            WHERE regexp_matches(CAST(ticker AS VARCHAR), '{NYC_PREFIX_RE}')
        )
        TO '{archive_markets.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    return archive_trades, archive_markets


def get_archive_max_trade(con: duckdb.DuckDBPyConnection, archive_trades: Path) -> datetime:
    row = con.execute(
        f"SELECT max(created_time) FROM read_parquet('{archive_trades.as_posix()}')"
    ).fetchone()
    if not row or row[0] is None:
        raise RuntimeError("Archive extraction produced no NYC trades.")
    value = row[0]
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def enumerate_kxhighny_markets(k: KalshiSession) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    # Historical tier first, then live tier so fresher live metadata wins on overlap.
    for source, path in [
        ("kalshi_historical", "/historical/markets"),
        ("kalshi_live", "/markets"),
    ]:
        params = {
            "series_ticker": CURRENT_SERIES,
            "limit": 1000,
            "mve_filter": "exclude",
        }
        for m in k.paginate(path, "markets", params=params, allow_404=True):
            ticker = m.get("ticker")
            if not ticker:
                continue
            row = dict(m)
            row["_source"] = source
            merged[ticker] = row

    return list(merged.values())


def market_latest_time(m: dict[str, Any]) -> datetime | None:
    vals = [
        parse_rfc3339(m.get("close_time")),
        parse_rfc3339(m.get("latest_expiration_time")),
        parse_rfc3339(m.get("settlement_ts")),
        parse_rfc3339(m.get("open_time")),
        parse_rfc3339(m.get("created_time")),
    ]
    vals = [x for x in vals if x is not None]
    return max(vals) if vals else None


def fetch_incremental_api(k: KalshiSession, out: Path,
                          archive_max: datetime,
                          overlap_days: int) -> tuple[Path | None, Path | None, int]:
    print("[4/6] Extending with current Kalshi historical/live API...")
    markets = enumerate_kxhighny_markets(k)
    if not markets:
        print("      WARNING: Kalshi returned no KXHIGHNY markets; keeping static archive only.")
        return None, None, 0

    threshold = archive_max - timedelta(days=overlap_days)
    relevant = []
    for m in markets:
        t = market_latest_time(m)
        if t is None or t >= threshold:
            relevant.append(m)

    print(
        f"      static NYC archive ends {archive_max.isoformat()}; "
        f"checking {len(relevant)} KXHIGHNY markets on/after {threshold.date()}."
    )

    api_markets_csv = out / "_api_markets.csv"
    market_fields = [
        "ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title",
        "status", "result", "created_time", "open_time", "close_time",
        "yes_bid_dollars", "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars",
        "last_price_dollars", "volume_fp", "open_interest_fp",
        "strike_type", "floor_strike", "cap_strike", "source",
    ]
    with api_markets_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=market_fields)
        w.writeheader()
        for m in relevant:
            w.writerow({
                "ticker": m.get("ticker", ""),
                "event_ticker": m.get("event_ticker", ""),
                "title": m.get("title", ""),
                "yes_sub_title": m.get("yes_sub_title", ""),
                "no_sub_title": m.get("no_sub_title", ""),
                "status": m.get("status", ""),
                "result": m.get("result", ""),
                "created_time": m.get("created_time", ""),
                "open_time": m.get("open_time", ""),
                "close_time": m.get("close_time", ""),
                "yes_bid_dollars": m.get("yes_bid_dollars", ""),
                "yes_ask_dollars": m.get("yes_ask_dollars", ""),
                "no_bid_dollars": m.get("no_bid_dollars", ""),
                "no_ask_dollars": m.get("no_ask_dollars", ""),
                "last_price_dollars": m.get("last_price_dollars", ""),
                "volume_fp": m.get("volume_fp", m.get("volume", "")),
                "open_interest_fp": m.get("open_interest_fp", m.get("open_interest", "")),
                "strike_type": m.get("strike_type", ""),
                "floor_strike": m.get("floor_strike", ""),
                "cap_strike": m.get("cap_strike", ""),
                "source": m.get("_source", "kalshi_api"),
            })

    api_trades_csv = out / "_api_trades.csv"
    trade_fields = [
        "trade_id", "ticker", "count_fp", "yes_price_dollars",
        "no_price_dollars", "taker_outcome_side", "created_time", "source",
    ]

    min_ts = int(threshold.timestamp())
    seen_trade_ids: set[str] = set()
    written = 0

    with api_trades_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=trade_fields)
        w.writeheader()

        total = len(relevant)
        for idx, m in enumerate(sorted(relevant, key=lambda x: x.get("ticker", "")), 1):
            ticker = m.get("ticker")
            if not ticker:
                continue

            # A market lives in one tier according to the current cutoff, but querying
            # both safely handles overlap/cutoff movement. trade_id dedupes duplicates.
            for source, path in [
                ("kalshi_historical", "/historical/trades"),
                ("kalshi_live", "/markets/trades"),
            ]:
                params = {"ticker": ticker, "limit": 1000, "min_ts": min_ts}
                try:
                    rows = k.paginate(path, "trades", params=params, allow_404=True)
                    for tr in rows:
                        trade_id = str(tr.get("trade_id") or "")
                        if not trade_id or trade_id in seen_trade_ids:
                            continue
                        seen_trade_ids.add(trade_id)

                        yes_price = tr.get("yes_price_dollars")
                        no_price = tr.get("no_price_dollars")
                        count = tr.get("count_fp")

                        # Compatibility with pre-fixed-point fields if Kalshi serves them.
                        if yes_price in (None, "") and tr.get("yes_price") not in (None, ""):
                            yes_price = Decimal(str(tr["yes_price"])) / Decimal("100")
                        if no_price in (None, "") and tr.get("no_price") not in (None, ""):
                            no_price = Decimal(str(tr["no_price"])) / Decimal("100")
                        if count in (None, ""):
                            count = tr.get("count", "")

                        taker = (
                            tr.get("taker_outcome_side")
                            or tr.get("taker_side")
                            or ""
                        )
                        w.writerow({
                            "trade_id": trade_id,
                            "ticker": tr.get("ticker", ticker),
                            "count_fp": decimal_text(count, 2),
                            "yes_price_dollars": decimal_text(yes_price, 6),
                            "no_price_dollars": decimal_text(no_price, 6),
                            "taker_outcome_side": taker,
                            "created_time": tr.get("created_time", ""),
                            "source": source,
                        })
                        written += 1
                except RuntimeError as exc:
                    # A moved cutoff can make one tier reject a ticker. The other tier
                    # is still attempted, but surface unexpected failures.
                    print(f"      warning {ticker} via {source}: {exc}", file=sys.stderr)

            if idx % 100 == 0 or idx == total:
                print(f"      markets checked: {idx}/{total}; API trades captured: {written}")

    return api_trades_csv, api_markets_csv, written


def merge_outputs(con: duckdb.DuckDBPyConnection, out: Path,
                  archive_trades: Path, archive_markets: Path,
                  api_trades_csv: Path | None, api_markets_csv: Path | None) -> tuple[Path, Path]:
    print("[5/6] Normalizing, deduplicating, and merging...")

    final_trades = out / "nyc_trades.parquet"
    final_markets = out / "nyc_markets.parquet"

    trade_parts = [f"SELECT * FROM read_parquet('{archive_trades.as_posix()}')"]
    if api_trades_csv and api_trades_csv.exists() and api_trades_csv.stat().st_size > 100:
        trade_parts.append(f"""
            SELECT
                trade_id::VARCHAR AS trade_id,
                ticker::VARCHAR AS ticker,
                try_cast(count_fp AS DECIMAL(20,2)) AS count_fp,
                try_cast(yes_price_dollars AS DECIMAL(20,6)) AS yes_price_dollars,
                try_cast(no_price_dollars AS DECIMAL(20,6)) AS no_price_dollars,
                taker_outcome_side::VARCHAR AS taker_outcome_side,
                try_cast(created_time AS TIMESTAMPTZ) AS created_time,
                source::VARCHAR AS source
            FROM read_csv_auto('{api_trades_csv.as_posix()}', header=true, all_varchar=true)
        """)

    trade_union = "\nUNION ALL\n".join(trade_parts)
    con.execute(f"""
        COPY (
            WITH combined AS (
                {trade_union}
            ),
            ranked AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY trade_id
                           ORDER BY CASE WHEN source = 'kalshi_live' THEN 3
                                         WHEN source = 'kalshi_historical' THEN 2
                                         ELSE 1 END DESC
                       ) AS rn
                FROM combined
                WHERE trade_id IS NOT NULL
                  AND ticker IS NOT NULL
                  AND created_time IS NOT NULL
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1
            ORDER BY created_time, ticker, trade_id
        )
        TO '{final_trades.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    market_parts = [f"SELECT * FROM read_parquet('{archive_markets.as_posix()}')"]
    if api_markets_csv and api_markets_csv.exists() and api_markets_csv.stat().st_size > 100:
        market_parts.append(f"""
            SELECT
                ticker::VARCHAR AS ticker,
                event_ticker::VARCHAR AS event_ticker,
                title::VARCHAR AS title,
                yes_sub_title::VARCHAR AS yes_sub_title,
                no_sub_title::VARCHAR AS no_sub_title,
                status::VARCHAR AS status,
                result::VARCHAR AS result,
                try_cast(created_time AS TIMESTAMPTZ) AS created_time,
                try_cast(open_time AS TIMESTAMPTZ) AS open_time,
                try_cast(close_time AS TIMESTAMPTZ) AS close_time,
                try_cast(yes_bid_dollars AS DECIMAL(20,6)) AS yes_bid_dollars,
                try_cast(yes_ask_dollars AS DECIMAL(20,6)) AS yes_ask_dollars,
                try_cast(no_bid_dollars AS DECIMAL(20,6)) AS no_bid_dollars,
                try_cast(no_ask_dollars AS DECIMAL(20,6)) AS no_ask_dollars,
                try_cast(last_price_dollars AS DECIMAL(20,6)) AS last_price_dollars,
                try_cast(volume_fp AS DECIMAL(24,2)) AS volume_fp,
                try_cast(open_interest_fp AS DECIMAL(24,2)) AS open_interest_fp,
                nullif(strike_type, '')::VARCHAR AS strike_type,
                try_cast(floor_strike AS DOUBLE) AS floor_strike,
                try_cast(cap_strike AS DOUBLE) AS cap_strike,
                source::VARCHAR AS source
            FROM read_csv_auto('{api_markets_csv.as_posix()}', header=true, all_varchar=true)
        """)

    market_union = "\nUNION ALL\n".join(market_parts)
    con.execute(f"""
        COPY (
            WITH combined AS (
                {market_union}
            ),
            ranked AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY ticker
                           ORDER BY CASE WHEN source = 'kalshi_live' THEN 3
                                         WHEN source = 'kalshi_historical' THEN 2
                                         ELSE 1 END DESC,
                                    coalesce(close_time, open_time, created_time) DESC NULLS LAST
                       ) AS rn
                FROM combined
                WHERE ticker IS NOT NULL
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1
            ORDER BY coalesce(open_time, created_time), ticker
        )
        TO '{final_markets.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    return final_trades, final_markets


def build_bars_and_coverage(con: duckdb.DuckDBPyConnection, out: Path,
                            final_trades: Path, final_markets: Path,
                            retrieved_at: datetime,
                            write_csv: bool) -> None:
    print("[6/6] Building 15-minute/hourly bars and coverage report...")

    def build_bars(minutes: int, filename: str) -> None:
        dest = out / filename
        con.execute(f"""
            COPY (
                SELECT
                    ticker,
                    time_bucket(INTERVAL '{minutes} minutes', created_time) AS window_start,
                    arg_min(yes_price_dollars, created_time) AS yes_open,
                    max(yes_price_dollars) AS yes_high,
                    min(yes_price_dollars) AS yes_low,
                    arg_max(yes_price_dollars, created_time) AS yes_close,
                    sum(yes_price_dollars * count_fp) / nullif(sum(count_fp), 0) AS yes_vwap,
                    arg_min(no_price_dollars, created_time) AS no_open,
                    max(no_price_dollars) AS no_high,
                    min(no_price_dollars) AS no_low,
                    arg_max(no_price_dollars, created_time) AS no_close,
                    sum(no_price_dollars * count_fp) / nullif(sum(count_fp), 0) AS no_vwap,
                    sum(count_fp) AS contracts,
                    count(*) AS trade_count
                FROM read_parquet('{final_trades.as_posix()}')
                GROUP BY ticker, window_start
                ORDER BY window_start, ticker
            )
            TO '{dest.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    build_bars(15, "nyc_bars_15m.parquet")
    build_bars(60, "nyc_bars_1h.parquet")

    coverage_csv = out / "coverage_by_year.csv"
    con.execute(f"""
        COPY (
            SELECT
                year(created_time) AS year,
                CASE
                    WHEN starts_with(ticker, 'HIGHNY0-') THEN 'HIGHNY0'
                    WHEN starts_with(ticker, 'HIGHNY-') THEN 'HIGHNY'
                    WHEN starts_with(ticker, 'KXHIGHNY-') THEN 'KXHIGHNY'
                    ELSE 'other'
                END AS ticker_family,
                min(created_time) AS first_trade,
                max(created_time) AS last_trade,
                count(*) AS trade_rows,
                count(DISTINCT ticker) AS traded_markets,
                sum(count_fp) AS contracts
            FROM read_parquet('{final_trades.as_posix()}')
            GROUP BY year, ticker_family
            ORDER BY year, ticker_family
        )
        TO '{coverage_csv.as_posix()}'
        (FORMAT CSV, HEADER)
    """)

    stats = con.execute(f"""
        SELECT
            count(*) AS trade_rows,
            count(DISTINCT trade_id) AS unique_trade_ids,
            count(DISTINCT ticker) AS traded_markets,
            min(created_time) AS first_trade_time,
            max(created_time) AS last_trade_time,
            sum(count_fp) AS contracts
        FROM read_parquet('{final_trades.as_posix()}')
    """).fetchone()

    market_stats = con.execute(f"""
        SELECT
            count(*) AS market_rows,
            min(coalesce(open_time, created_time)) AS first_market_time,
            max(coalesce(close_time, open_time, created_time)) AS last_market_time
        FROM read_parquet('{final_markets.as_posix()}')
    """).fetchone()

    first_trade = con.execute(f"""
        SELECT ticker, created_time, yes_price_dollars, no_price_dollars, count_fp
        FROM read_parquet('{final_trades.as_posix()}')
        ORDER BY created_time
        LIMIT 1
    """).fetchone()

    report = {
        "retrieved_at_utc": retrieved_at.isoformat(),
        "ticker_families": ["HIGHNY0", "HIGHNY", "KXHIGHNY"],
        "trade_rows": int(stats[0] or 0),
        "unique_trade_ids": int(stats[1] or 0),
        "traded_markets": int(stats[2] or 0),
        "first_trade_time": stats[3].isoformat() if stats[3] else None,
        "last_trade_time": stats[4].isoformat() if stats[4] else None,
        "contracts": str(stats[5]) if stats[5] is not None else None,
        "market_rows": int(market_stats[0] or 0),
        "first_market_time": market_stats[1].isoformat() if market_stats[1] else None,
        "last_market_time": market_stats[2].isoformat() if market_stats[2] else None,
        "first_trade": {
            "ticker": first_trade[0],
            "created_time": first_trade[1].isoformat(),
            "yes_price_dollars": str(first_trade[2]),
            "no_price_dollars": str(first_trade[3]),
            "count_fp": str(first_trade[4]),
        } if first_trade else None,
        "sources": [
            {
                "name": "TrevorJS/kalshi-trades",
                "type": "static public archive",
                "url": "https://huggingface.co/datasets/TrevorJS/kalshi-trades",
                "published_global_span": "June 2021 through January 2026",
            },
            {
                "name": "Kalshi public REST API",
                "type": "historical + live incremental extension",
                "url": "https://external-api.kalshi.com/trade-api/v2",
            },
        ],
    }
    (out / "coverage_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    if write_csv:
        for parquet_name in [
            "nyc_trades.parquet",
            "nyc_markets.parquet",
            "nyc_bars_15m.parquet",
            "nyc_bars_1h.parquet",
        ]:
            src = out / parquet_name
            dst = out / (src.stem + ".csv.gz")
            con.execute(f"""
                COPY (SELECT * FROM read_parquet('{src.as_posix()}'))
                TO '{dst.as_posix()}'
                (FORMAT CSV, HEADER, COMPRESSION GZIP)
            """)


def cleanup_intermediate(out: Path) -> None:
    for name in [
        "_archive_nyc_trades.parquet",
        "_archive_nyc_markets.parquet",
        "_api_trades.csv",
        "_api_markets.csv",
    ]:
        p = out / name
        if p.exists():
            p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compile all available Kalshi NYC daily-high historical price/trade data."
    )
    ap.add_argument(
        "--output", default="kalshi_nyc_history",
        help="Output directory (default: kalshi_nyc_history)"
    )
    ap.add_argument(
        "--archive-only", action="store_true",
        help="Use the static 2021-Jan-2026 archive only; skip Kalshi API extension."
    )
    ap.add_argument(
        "--csv", action="store_true",
        help="Also write gzip-compressed CSV copies of the final Parquet datasets."
    )
    ap.add_argument(
        "--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS,
        help="Overlap around static archive endpoint when appending API data (default: 7)."
    )
    ap.add_argument(
        "--request-delay", type=float, default=0.05,
        help="Delay after successful Kalshi API requests in seconds (default: 0.05)."
    )
    ap.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep intermediate archive/API extraction files."
    )
    args = ap.parse_args()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    retrieved_at = utcnow()

    con = init_duckdb()
    archive_trades, archive_markets = extract_static_archive(con, out)
    archive_max = get_archive_max_trade(con, archive_trades)

    api_trades_csv = None
    api_markets_csv = None
    if not args.archive_only:
        k = KalshiSession(request_delay=args.request_delay)
        api_trades_csv, api_markets_csv, _ = fetch_incremental_api(
            k, out, archive_max, max(0, args.overlap_days)
        )

    final_trades, final_markets = merge_outputs(
        con, out, archive_trades, archive_markets,
        api_trades_csv, api_markets_csv
    )
    build_bars_and_coverage(
        con, out, final_trades, final_markets, retrieved_at, args.csv
    )

    if not args.keep_intermediate:
        cleanup_intermediate(out)

    print()
    print("Done.")
    print(f"Output directory: {out}")
    print(f"Trades:            {out / 'nyc_trades.parquet'}")
    print(f"Markets:           {out / 'nyc_markets.parquet'}")
    print(f"15-minute bars:    {out / 'nyc_bars_15m.parquet'}")
    print(f"Hourly bars:       {out / 'nyc_bars_1h.parquet'}")
    print(f"Coverage:          {out / 'coverage_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
