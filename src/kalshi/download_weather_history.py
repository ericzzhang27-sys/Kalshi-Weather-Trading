from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import time
from src.kalshi.historical_client import CITY_SERIES_CONFIG, KalshiHistoricalClient, create_historical_client_from_env
from src.trading.kalshi_client import KalshiAPIError

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "kalshi" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "kalshi" / "processed"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "kalshi" / "metadata"

# ------------------------------------------------------------
# Config dataclass
# ------------------------------------------------------------
@dataclass
class DownloadConfig:
    cities: list[str] = field(default_factory=lambda: ["NYC"])
    series_tickers: dict[str, list[str]] = field(default_factory=lambda: {"NYC": ["KXHIGHNY", "HIGHNY"]})
    raw_dir: Path = DEFAULT_RAW_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    metadata_dir: Path = DEFAULT_METADATA_DIR
    period_interval: int = 1  # 1-min candles
    force_refresh: bool = False
    max_pages_per_series: int = 100
    markets_per_event_limit: int = 1000
    trades_limit: int = 1000
    max_trades_pages: int = 5
    include_trades: bool = True
    target_start: str | None = None
    target_end: str | None = None
    max_markets: int | None = None
    publish_as_canonical: bool = False
    market_tickers_from: Path | None = None

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.metadata_dir = Path(self.metadata_dir)
        self.market_tickers_from = Path(self.market_tickers_from) if self.market_tickers_from else None

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _ensure_dirs(cfg: DownloadConfig) -> None:
    for p in [cfg.raw_dir, cfg.processed_dir, cfg.metadata_dir,
              cfg.raw_dir / "markets", cfg.raw_dir / "candles",
              cfg.raw_dir / "candles" / f"{cfg.period_interval}m", cfg.raw_dir / "trades"]:
        p.mkdir(parents=True, exist_ok=True)

def _hash_market(market: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(market, sort_keys=True).encode()).hexdigest()[:8]

def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _unix_ts(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp())

def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None

def _infer_bucket_label(market: dict[str, Any]) -> str:
    for k in ["yes_sub_title", "subtitle", "title"]:
        v = market.get(k)
        if v and str(v).strip():
            return str(v).strip()
    # fallback from strikes
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")
    if floor is not None and cap is not None:
        return f"{floor} to {cap}"
    if cap is not None:
        return f"<= {cap}"
    if floor is not None:
        return f">= {floor}"
    return market.get("ticker", "unknown")

def _market_to_row(market: dict[str, Any], city: str, series_ticker: str) -> dict[str, Any]:
    return {
        "series_ticker": series_ticker,
        "event_ticker": market.get("event_ticker", ""),
        "market_ticker": market.get("ticker", ""),
        "city": city,
        "floor_strike": market.get("floor_strike"),
        "cap_strike": market.get("cap_strike"),
        "yes_sub_title": market.get("yes_sub_title", ""),
        "subtitle": market.get("subtitle", ""),
        "title": market.get("title", ""),
        "bucket_label": _infer_bucket_label(market),
        "status": market.get("status", ""),
        "result": market.get("result", ""),
        "settlement_value": market.get("settlement_value", market.get("result", "")),
        "expiration_value": market.get("expiration_value", ""),
        "open_time": market.get("open_time", ""),
        "close_time": market.get("close_time", ""),
        "settlement_time": market.get("settlement_time", market.get("expiration_time", "")),
        "expected_expiration_time": market.get("expected_expiration_time", ""),
        "latest_expiration_time": market.get("latest_expiration_time", ""),
        "volume": market.get("volume_fp", market.get("volume", "")),
        "volume_24h": market.get("volume_24h_fp", ""),
        "open_interest": market.get("open_interest_fp", market.get("open_interest", "")),
        "yes_bid": market.get("yes_bid_dollars", market.get("yes_bid", "")),
        "yes_ask": market.get("yes_ask_dollars", market.get("yes_ask", "")),
        "last_price": market.get("last_price_dollars", market.get("last_price", "")),
        "notional_value": market.get("notional_value_dollars", ""),
        "rules_primary": market.get("rules_primary", ""),
        "raw_market_json": json.dumps(market, sort_keys=True),
    }

def _infer_target_date(market: dict[str, Any]) -> str:
    # Try event_ticker like KXHIGHNY-26AUG23 or KXHIGHNY-25JAN15
    et = str(market.get("event_ticker", ""))
    # The event ticker encodes the weather target date.  It must take
    # precedence over settlement/latest-expiration metadata, which can be up
    # to a week later for legacy contracts.
    import re
    m = re.search(r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})", et.upper())
    if m:
        try:
            year = 2000 + int(m.group(1))
            mon_str = m.group(2)
            day = int(m.group(3))
            mon = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}[mon_str]
            return f"{year:04d}-{mon:02d}-{day:02d}"
        except Exception:
            pass
    # Fallback for nonstandard event tickers: use close/expiration metadata.
    for field in ["expiration_time", "close_time", "expected_expiration_time", "latest_expiration_time"]:
        dt = _parse_iso(market.get(field))
        if dt:
            # Kalshi weather markets close early next day UTC, expiration is target date
            # Convert to America/New_York date
            try:
                import zoneinfo
                ny = zoneinfo.ZoneInfo("America/New_York")
                local = dt.astimezone(ny)
                # If close_time is 05:00 UTC next day, local is 01:00 next day but target_date is previous day?
                # Use expiration_value or title to disambiguate? Simpler: use local date minus 1 if hour <6?
                # Check target_date via event ticker first
                return local.date().isoformat()
            except Exception:
                return dt.date().isoformat()
    return et or "unknown"

# ------------------------------------------------------------
# Core download functions
# ------------------------------------------------------------
def download_historical_markets(
    client: KalshiHistoricalClient,
    cfg: DownloadConfig,
) -> pd.DataFrame:
    """Discover archived events and individual bucket contracts. Save raw + processed."""
    _ensure_dirs(cfg)
    all_rows: list[dict[str, Any]] = []
    raw_payloads: list[dict[str, Any]] = []

    for city in cfg.cities:
        series_list = cfg.series_tickers.get(city, CITY_SERIES_CONFIG.get(city, [f"KXHIGH{city}"]))
        for series in series_list:
            cache_path = cfg.raw_dir / "markets" / f"historical_markets_{series}.json"
            if cache_path.exists() and not cfg.force_refresh:
                logger.info("Using cached markets for %s from %s", series, cache_path)
                try:
                    cached = json.loads(cache_path.read_text())
                    for m in cached:
                        row = _market_to_row(m, city, series)
                        row["target_date"] = _infer_target_date(m)
                        all_rows.append(row)
                        raw_payloads.append(m)
                    continue
                except Exception as exc:
                    logger.warning("Failed to load cache %s: %s, refetching", cache_path, exc)

            logger.info("Fetching historical markets for series %s (city %s)", series, city)
            series_markets: list[dict[str, Any]] = []
            try:
                for m in client.iter_historical_markets(series_ticker=series, limit=1000, max_pages=cfg.max_pages_per_series, auth=False):
                    series_markets.append(m)
                    row = _market_to_row(m, city, series)
                    row["target_date"] = _infer_target_date(m)
                    all_rows.append(row)
                    raw_payloads.append(m)
            except KalshiAPIError as exc:
                logger.warning("Historical markets fetch failed for %s: %s (trying live markets)", series, exc)
                try:
                    for m in client.iter_markets(status=None, series_ticker=series, limit=1000, max_pages=cfg.max_pages_per_series, auth=False):
                        series_markets.append(m)
                        row = _market_to_row(m, city, series)
                        row["target_date"] = _infer_target_date(m)
                        all_rows.append(row)
                        raw_payloads.append(m)
                except Exception as e2:
                    logger.error("Live fallback also failed for %s: %s", series, e2)

            # Save raw
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(series_markets, indent=2), encoding="utf-8")
            # Also save individual raw payloads aggregate
            logger.info("Series %s: fetched %d markets", series, len(series_markets))

    # Deduplicate by market_ticker
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["market_ticker"], keep="last")
        # Validation: dates/tickers/buckets consistent
        dup_markets = df[df.duplicated(subset=["market_ticker"], keep=False)]
        if not dup_markets.empty:
            logger.warning("Duplicate market tickers after dedup: %d", len(dup_markets))
        # Save processed
        out_path = cfg.processed_dir / "historical_markets_processed.csv"
        df.to_csv(out_path, index=False)
        # Also save raw aggregate
        raw_agg = cfg.raw_dir / "historical_markets_raw.json"
        raw_agg.write_text(json.dumps(raw_payloads, indent=2), encoding="utf-8")
        # Metadata
        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cities": cfg.cities,
            "series_tickers": cfg.series_tickers,
            "market_count": len(df),
            "event_count": int(df["event_ticker"].nunique()) if "event_ticker" in df.columns else 0,
            "date_range": [str(df["target_date"].min()), str(df["target_date"].max())] if "target_date" in df.columns and not df.empty else [],
        }
        (cfg.metadata_dir / "historical_markets_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Saved %d historical markets to %s", len(df), out_path)
    else:
        logger.warning("No historical markets discovered (API may require auth or no history). Creating empty frame.")
        df = pd.DataFrame(columns=["series_ticker","event_ticker","market_ticker","city","floor_strike","cap_strike","bucket_label","status","target_date"])

    return df


def download_historical_candles(
    client: KalshiHistoricalClient,
    markets_df: pd.DataFrame,
    cfg: DownloadConfig,
) -> pd.DataFrame:
    """Download 1-minute candles for every bucket. Resumable, cached, rate-limited."""
    _ensure_dirs(cfg)
    if markets_df.empty:
        logger.warning("No markets to download candles for")
        return pd.DataFrame()

    candle_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_interval": cfg.period_interval,
        "market_count": len(markets_df),
        "candle_count": 0,
        "earliest_candle": None,
        "latest_candle": None,
        "markets_with_candles": 0,
        "markets_without_candles": 0,
    }

    # Cutoff for routing
    cutoff_ts = None
    try:
        cutoff_payload = client.get_historical_cutoff()
        cutoff_ts_str = cutoff_payload.get("market_settled_ts")
        if cutoff_ts_str:
            cutoff_ts = _parse_iso(cutoff_ts_str)
            logger.info("Historical cutoff market_settled_ts: %s", cutoff_ts_str)
    except Exception as exc:
        logger.info("Could not fetch cutoff: %s (will try both endpoints)", exc)

    for idx, row in markets_df.iterrows():
        ticker = str(row["market_ticker"])
        series = str(row["series_ticker"])
        event = str(row["event_ticker"])
        city = str(row.get("city", "NYC"))
        floor_strike = row.get("floor_strike")
        cap_strike = row.get("cap_strike")
        target_date = str(row.get("target_date", ""))

        # Determine time range: from open_time to close_time/settlement_time, or target_date +- 2 days
        open_dt = _parse_iso(str(row.get("open_time") or ""))
        close_dt = _parse_iso(str(row.get("close_time") or "")) or _parse_iso(str(row.get("expected_expiration_time") or "")) or _parse_iso(str(row.get("latest_expiration_time") or ""))
        settlement_dt = _parse_iso(str(row.get("settlement_time") or ""))

        # Fallback: infer from target_date: candles for target day 00:00 to next day 06:00 UTC?
        if not open_dt or not close_dt:
            try:
                td = datetime.fromisoformat(target_date)
                # Assume market opens 7 days before at 00:00 UTC and closes next day 05:00 UTC
                open_dt = datetime(td.year, td.month, td.day, 0, 0, tzinfo=timezone.utc) - timedelta(days=7)
                close_dt = datetime(td.year, td.month, td.day, 5, 0, tzinfo=timezone.utc) + timedelta(days=1)
            except Exception:
                open_dt = datetime.now(timezone.utc) - timedelta(days=30)
                close_dt = datetime.now(timezone.utc)

        start_ts = _unix_ts(open_dt)
        end_ts = _unix_ts(close_dt)

        # Check cache
        # Interval-specific caches prevent a previous 60-minute payload from
        # silently satisfying a later one-minute request.
        cache_file = cfg.raw_dir / "candles" / f"{cfg.period_interval}m" / f"{ticker}.json"
        raw_candles_payload: dict[str, Any] | None = None
        if cache_file.exists() and not cfg.force_refresh:
            try:
                raw_candles_payload = json.loads(cache_file.read_text())
                logger.debug("Using cached candles for %s", ticker)
            except Exception:
                raw_candles_payload = None

        if raw_candles_payload is None:
            # Try historical then live
            try:
                # Prefer historical for settled markets
                if cutoff_ts and settlement_dt and settlement_dt < cutoff_ts:
                    raw_candles_payload = client.get_historical_candlesticks(ticker, start_ts, end_ts, cfg.period_interval, auth=False)
                else:
                    # Try historical first, fallback to live
                    try:
                        raw_candles_payload = client.get_historical_candlesticks(ticker, start_ts, end_ts, cfg.period_interval, auth=False)
                    except KalshiAPIError as e:
                        if e.status_code in (404, 400):
                            raw_candles_payload = client.get_live_candlesticks(series, ticker, start_ts, end_ts, cfg.period_interval, auth=False)
                        else:
                            raise
            except KalshiAPIError as exc:
                logger.warning("Candles fetch failed for %s (%s): %s", ticker, series, exc)
                raw_candles_payload = {"ticker": ticker, "candlesticks": []}
            except Exception as exc:
                logger.warning("Candles fetch error for %s: %s", ticker, exc)
                raw_candles_payload = {"ticker": ticker, "candlesticks": []}

            # Save raw
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(raw_candles_payload, indent=2), encoding="utf-8")

        # Normalize candles
        candles = raw_candles_payload.get("candlesticks", []) if isinstance(raw_candles_payload, dict) else []
        if not candles:
            metadata["markets_without_candles"] += 1
            continue
        metadata["markets_with_candles"] += 1

        for c in candles:
            # candle schema historical vs live differs slightly
            end_ts_val = c.get("end_period_ts")
            if end_ts_val is None:
                continue
            candle_dt = datetime.fromtimestamp(int(end_ts_val), tz=timezone.utc)
            # yes_bid / yes_ask / price can be nested dicts with different keys
            yes_bid = c.get("yes_bid", {})
            yes_ask = c.get("yes_ask", {})
            price = c.get("price", {}) or c.get("price_distribution", {})  # live vs historical naming
            # Historical keys: open/high/low/close (without _dollars); live: open_dollars etc.
            def _extract_bidask(d: dict[str, Any]) -> dict[str, Any]:
                if not isinstance(d, dict):
                    return {"open": None, "high": None, "low": None, "close": None}
                # Try historical
                if "open" in d or "close" in d:
                    return {"open": d.get("open"), "high": d.get("high"), "low": d.get("low"), "close": d.get("close")}
                # live
                return {"open": d.get("open_dollars"), "high": d.get("high_dollars"), "low": d.get("low_dollars"), "close": d.get("close_dollars")}

            def _extract_price(d: dict[str, Any]) -> dict[str, Any]:
                if not isinstance(d, dict):
                    return {"open": None, "high": None, "low": None, "close": None, "mean": None, "previous": None}
                if "open" in d or "mean" in d:
                    return {"open": d.get("open"), "high": d.get("high"), "low": d.get("low"), "close": d.get("close"), "mean": d.get("mean"), "previous": d.get("previous")}
                return {"open": d.get("open_dollars"), "high": d.get("high_dollars"), "low": d.get("low_dollars"), "close": d.get("close_dollars"), "mean": d.get("mean_dollars"), "previous": d.get("previous_dollars")}

            yb = _extract_bidask(yes_bid)
            ya = _extract_bidask(yes_ask)
            pr = _extract_price(price)

            candle_rows.append({
                "market_ticker": ticker,
                "event_ticker": event,
                "series_ticker": series,
                "city": city,
                "target_date": target_date,
                "floor_strike": floor_strike,
                "cap_strike": cap_strike,
                "end_period_ts": int(end_ts_val),
                "timestamp": candle_dt.isoformat(),
                "yes_bid_open": yb["open"],
                "yes_bid_high": yb["high"],
                "yes_bid_low": yb["low"],
                "yes_bid_close": yb["close"],
                "yes_ask_open": ya["open"],
                "yes_ask_high": ya["high"],
                "yes_ask_low": ya["low"],
                "yes_ask_close": ya["close"],
                "price_open": pr["open"],
                "price_high": pr["high"],
                "price_low": pr["low"],
                "price_close": pr["close"],
                "price_mean": pr["mean"],
                "price_previous": pr["previous"],
                "volume": c.get("volume") or c.get("volume_fp"),
                "open_interest": c.get("open_interest") or c.get("open_interest_fp"),
                "raw_candle_json": json.dumps(c, sort_keys=True),
            })

        # Rate limit respect: small sleep every 20 markets
        if (idx + 1) % 20 == 0:
            time.sleep(0.5)

        if (idx + 1) % 100 == 0:
            logger.info("Candles progress %d/%d markets", idx+1, len(markets_df))

    df_candles = pd.DataFrame(candle_rows)
    if not df_candles.empty:
        # Duplicate detection
        dup = df_candles.duplicated(subset=["market_ticker", "end_period_ts"], keep=False)
        if dup.any():
            logger.warning("Duplicate candles detected: %d rows", dup.sum())
            df_candles = df_candles.drop_duplicates(subset=["market_ticker", "end_period_ts"], keep="last")

        # Validation: bid > ask?
        # Clean numeric
        for col in ["yes_bid_close", "yes_ask_close"]:
            if col in df_candles.columns:
                df_candles[col] = pd.to_numeric(df_candles[col], errors="coerce")
        bad = df_candles[(df_candles["yes_bid_close"].notna()) & (df_candles["yes_ask_close"].notna()) & (df_candles["yes_bid_close"] > df_candles["yes_ask_close"])]
        if not bad.empty:
            logger.warning("Impossible bid > ask in %d candles", len(bad))

        # Always retain interval-specific processed data. Publishing it as the
        # generic canonical file is explicit and happens only after validation.
        stem = f"historical_candles_{cfg.period_interval}m_processed"
        if len(df_candles) < 500000:
            df_candles.to_parquet(cfg.processed_dir / f"{stem}.parquet", index=False)
        df_candles.to_csv(cfg.processed_dir / f"{stem}.csv", index=False)
        if cfg.publish_as_canonical:
            df_candles.to_csv(cfg.processed_dir / "historical_candles_processed.csv", index=False)
        metadata["candle_count"] = len(df_candles)
        metadata["earliest_candle"] = str(df_candles["timestamp"].min())
        metadata["latest_candle"] = str(df_candles["timestamp"].max())
        metadata["published_as_canonical"] = bool(cfg.publish_as_canonical)
        metadata_path = cfg.metadata_dir / f"candles_{cfg.period_interval}m_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved %d candles across %d markets", len(df_candles), metadata["markets_with_candles"])
        # Report earliest date with usable candles
        # Usable = has yes_bid_close and yes_ask_close not null
        usable = df_candles[df_candles["yes_bid_close"].notna() & df_candles["yes_ask_close"].notna()]
        if not usable.empty:
            earliest_usable = usable["timestamp"].min()
            logger.info("Earliest usable 1-min candle: %s", earliest_usable)
            metadata["earliest_usable_candle"] = str(earliest_usable)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        logger.warning("No candles downloaded")
        # Create empty files for downstream
        df_candles = pd.DataFrame(columns=["market_ticker","event_ticker","series_ticker","city","target_date","timestamp","yes_bid_close","yes_ask_close","price_close"])
        df_candles.to_csv(cfg.processed_dir / f"historical_candles_{cfg.period_interval}m_processed.csv", index=False)

    return df_candles


def download_historical_trades(
    client: KalshiHistoricalClient,
    markets_df: pd.DataFrame,
    cfg: DownloadConfig,
) -> pd.DataFrame:
    if not cfg.include_trades or markets_df.empty:
        return pd.DataFrame()
    _ensure_dirs(cfg)
    all_trades: list[dict[str, Any]] = []
    for idx, row in markets_df.iterrows():
        ticker = str(row["market_ticker"])
        cache_file = cfg.raw_dir / "trades" / f"{ticker}.json"
        if cache_file.exists() and not cfg.force_refresh:
            try:
                trades = json.loads(cache_file.read_text())
                for t in trades:
                    t["_market_ticker"] = ticker
                    t["_event_ticker"] = str(row["event_ticker"])
                    t["_series_ticker"] = str(row["series_ticker"])
                    t["_city"] = str(row.get("city","NYC"))
                    all_trades.append(t)
                continue
            except Exception:
                pass
        # Fetch trades
        fetched: list[dict[str, Any]] = []
        try:
            # Try historical trades with ticker filter
            for t in client.iter_historical_trades(ticker=ticker, limit=cfg.trades_limit, max_pages=cfg.max_trades_pages, auth=False):
                t["_market_ticker"] = ticker
                fetched.append(t)
            # If empty, try live
            if not fetched:
                for t in client.iter_live_trades(ticker=ticker, limit=cfg.trades_limit, max_pages=2, auth=False):
                    t["_market_ticker"] = ticker
                    fetched.append(t)
        except Exception as exc:
            logger.warning("Trades fetch failed for %s: %s", ticker, exc)
        # Save raw
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(fetched, indent=2), encoding="utf-8")
        for t in fetched:
            t["_event_ticker"] = str(row["event_ticker"])
            t["_series_ticker"] = str(row["series_ticker"])
            t["_city"] = str(row.get("city","NYC"))
        all_trades.extend(fetched)
        if (idx+1) % 50 == 0:
            logger.info("Trades progress %d/%d", idx+1, len(markets_df))
            time.sleep(0.3)

    if all_trades:
        df = pd.DataFrame([
            {
                "market_ticker": t.get("_market_ticker") or t.get("ticker",""),
                "event_ticker": t.get("_event_ticker",""),
                "series_ticker": t.get("_series_ticker",""),
                "city": t.get("_city","NYC"),
                "trade_id": t.get("trade_id",""),
                "timestamp": t.get("created_time",""),
                "yes_price": t.get("yes_price_dollars", t.get("yes_price","")),
                "no_price": t.get("no_price_dollars", t.get("no_price","")),
                "quantity": t.get("count_fp", t.get("count","")),
                "taker_side": t.get("taker_outcome_side") or t.get("taker_side",""),
                "taker_book_side": t.get("taker_book_side",""),
                "is_block_trade": t.get("is_block_trade",""),
                "raw_trade_json": json.dumps(t, sort_keys=True),
            }
            for t in all_trades
        ])
        df = df.drop_duplicates(subset=["trade_id"], keep="last") if "trade_id" in df.columns else df
        df.to_csv(cfg.processed_dir / "historical_trades_processed.csv", index=False)
        logger.info("Saved %d trades", len(df))
        return df
    else:
        pd.DataFrame(columns=["market_ticker","trade_id","timestamp","yes_price"]).to_csv(cfg.processed_dir / "historical_trades_processed.csv", index=False)
        return pd.DataFrame()


def run_full_download(cfg: DownloadConfig | None = None, client: KalshiHistoricalClient | None = None) -> dict[str, Any]:
    """Orchestrate full historical download."""
    cfg = cfg or DownloadConfig()
    client = client or create_historical_client_from_env()
    logger.info("Starting full historical download for cities %s", cfg.cities)
    markets_df = download_historical_markets(client, cfg)
    selected_markets = markets_df.copy()
    if cfg.market_tickers_from:
        source = pd.read_csv(cfg.market_tickers_from, usecols=["market_ticker"])
        requested_tickers = set(source["market_ticker"].dropna().astype(str))
        selected_markets = selected_markets[selected_markets["market_ticker"].astype(str).isin(requested_tickers)]
    if cfg.target_start:
        selected_markets = selected_markets[pd.to_datetime(selected_markets["target_date"], errors="coerce") >= pd.Timestamp(cfg.target_start)]
    if cfg.target_end:
        selected_markets = selected_markets[pd.to_datetime(selected_markets["target_date"], errors="coerce") <= pd.Timestamp(cfg.target_end)]
    if cfg.max_markets is not None:
        selected_markets = selected_markets.sort_values(["target_date", "market_ticker"], kind="stable").head(int(cfg.max_markets))
    selected_markets = selected_markets.reset_index(drop=True)
    candles_df = download_historical_candles(client, selected_markets, cfg)
    trades_df = download_historical_trades(client, selected_markets, cfg)

    earliest = None
    if not candles_df.empty and "yes_bid_close" in candles_df.columns:
        usable = candles_df[candles_df["yes_bid_close"].notna() & candles_df["yes_ask_close"].notna()]
        if not usable.empty:
            earliest = str(usable["timestamp"].min())

    summary = {
        "markets": len(markets_df),
        "selected_markets": len(selected_markets),
        "events": int(markets_df["event_ticker"].nunique()) if not markets_df.empty else 0,
        "candles": len(candles_df),
        "trades": len(trades_df),
        "earliest_candle": str(candles_df["timestamp"].min()) if not candles_df.empty else None,
        "earliest_usable_1min_candle": earliest,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_interval": cfg.period_interval,
        "published_as_canonical": cfg.publish_as_canonical,
    }
    (cfg.metadata_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Download historical Kalshi weather markets")
    parser.add_argument("--city", default="NYC", help="City code (NYC, CHI, ...)")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache")
    parser.add_argument("--no-trades", action="store_true", help="Skip trades download")
    parser.add_argument("--period-interval", type=int, choices=[1, 60], default=1)
    parser.add_argument("--target-start", default=None)
    parser.add_argument("--target-end", default=None)
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--publish-as-canonical", action="store_true")
    parser.add_argument("--market-tickers-from", type=Path, default=None, help="CSV containing a market_ticker column")
    args = parser.parse_args()
    cfg = DownloadConfig(
        cities=[args.city], force_refresh=args.force_refresh,
        include_trades=not args.no_trades, period_interval=args.period_interval,
        target_start=args.target_start, target_end=args.target_end,
        max_markets=args.max_markets, publish_as_canonical=args.publish_as_canonical,
        market_tickers_from=args.market_tickers_from,
    )
    run_full_download(cfg)
