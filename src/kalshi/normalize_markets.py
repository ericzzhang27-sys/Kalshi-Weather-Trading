from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.trading.contract_mapping import ContractMappingError, parse_contract_bucket


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data/kalshi/processed"


def parse_kalshi_bucket(market: dict[str, Any]):
    return parse_contract_bucket(market)


def normalize_historical_markets(
    markets_df: pd.DataFrame | None = None,
    candles_df: pd.DataFrame | None = None,
    trades_df: pd.DataFrame | None = None,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """Pure canonical normalization; never infers settlement or writes files."""
    base = Path(processed_dir)
    markets = (
        pd.read_csv(base / "historical_markets_processed.csv")
        if markets_df is None
        else markets_df.copy()
    )
    candles = (
        pd.read_csv(base / "historical_candles_processed.csv")
        if candles_df is None
        else candles_df.copy()
    )
    if markets.empty or candles.empty:
        raise ValueError("Historical market and candle tables must both be nonempty")
    ticker_column = "market_ticker" if "market_ticker" in markets else "ticker"
    if ticker_column not in markets or "market_ticker" not in candles:
        raise ValueError("Historical markets/candles require market ticker columns")
    if "raw_market_json" not in markets or markets["raw_market_json"].isna().any():
        raise ValueError("Historical markets lack per-row raw API provenance")
    if "raw_candle_json" not in candles or candles["raw_candle_json"].isna().any():
        raise ValueError("Historical candles lack per-row raw API provenance")

    records: list[dict[str, Any]] = []
    for _, row in markets.iterrows():
        payload = _combined_payload(row.to_dict())
        try:
            bucket = parse_contract_bucket(payload)
        except ContractMappingError as exc:
            raise ValueError(f"Ambiguous bucket for {row[ticker_column]}: {exc}") from exc
        result = str(payload.get("result", row.get("result", ""))).lower()
        if result not in {"yes", "no"}:
            raise ValueError(f"Unresolved Kalshi result for {row[ticker_column]}")
        settlement_timestamp = _first_present(
            payload, "settlement_time", "settlement_ts", "settlement_timestamp"
        )
        if settlement_timestamp is None:
            raise ValueError(f"Missing settlement timestamp for {row[ticker_column]}")
        records.append({
            "market_ticker": str(row[ticker_column]),
            "event_ticker": payload.get("event_ticker", row.get("event_ticker")),
            "series_ticker": payload.get("series_ticker", row.get("series_ticker")),
            "city": str(row.get("city", "NYC")).upper(),
            "target_date": row.get("target_date"),
            "bucket_lower": bucket.lower_temp,
            "bucket_upper": bucket.upper_temp,
            "bucket_label": bucket.label,
            "result": result,
            "settlement_timestamp": settlement_timestamp,
            "raw_market_json": row["raw_market_json"],
        })
    market_frame = pd.DataFrame(records)
    if market_frame["market_ticker"].duplicated().any():
        raise ValueError("Historical markets contain duplicate tickers")
    if market_frame[["event_ticker", "target_date", "settlement_timestamp"]].isna().any().any():
        raise ValueError("Settled market metadata is incomplete")
    market_frame["settlement_timestamp"] = pd.to_datetime(
        market_frame["settlement_timestamp"], errors="raise", utc=True
    )

    candles["timestamp"] = _candle_timestamp(candles)
    rename = {
        "yes_bid_open_dollars": "yes_bid_open", "yes_ask_open_dollars": "yes_ask_open",
        "yes_bid_close_dollars": "yes_bid_close", "yes_ask_close_dollars": "yes_ask_close",
    }
    candles = candles.rename(columns={old: new for old, new in rename.items() if old in candles})
    quote_columns = ["yes_bid_open", "yes_ask_open", "yes_bid_close", "yes_ask_close"]
    for column in quote_columns:
        if column not in candles:
            candles[column] = np.nan
        candles[column] = pd.to_numeric(candles[column], errors="coerce")
        invalid = candles[column].notna() & ~candles[column].between(0.0, 1.0)
        if invalid.any():
            raise ValueError(f"Historical candles contain invalid {column} values")
    if candles.duplicated(["market_ticker", "timestamp"]).any():
        raise ValueError("Historical candles contain duplicate ticker/timestamp rows")

    metadata_columns = set(market_frame.columns) - {"market_ticker"}
    candles = candles.drop(
        columns=[name for name in metadata_columns if name in candles], errors="ignore"
    )
    canonical = candles.merge(market_frame, on="market_ticker", how="inner", validate="many_to_one")
    if canonical.empty:
        raise ValueError("No historical candles matched settled markets")
    keep = [
        "timestamp", "settlement_timestamp", "target_date", "city", "series_ticker",
        "event_ticker", "market_ticker", "bucket_lower", "bucket_upper", "bucket_label",
        *quote_columns, "yes_bid_size_open", "yes_ask_size_open", "volume", "open_interest",
        "result", "raw_market_json", "raw_candle_json",
    ]
    return canonical[[name for name in keep if name in canonical]].sort_values(
        ["timestamp", "market_ticker"], kind="stable"
    ).reset_index(drop=True)


def normalize_historical_files(
    markets_path: str | Path,
    candles_path: str | Path,
    output_path: str | Path,
    *,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Normalize a large candle CSV to an immutable, compact Parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    market_source = Path(markets_path)
    candle_source = Path(candles_path)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"canonical output is immutable: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    markets = pd.read_csv(market_source)
    market_hash = _sha256_file(market_source)
    candle_hash = _sha256_file(candle_source)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    market_tickers: set[str] = set()
    try:
        for chunk in pd.read_csv(candle_source, chunksize=int(chunksize)):
            tickers = set(chunk["market_ticker"].dropna().astype(str))
            matching_markets = markets[markets["market_ticker"].astype(str).isin(tickers)]
            canonical = normalize_historical_markets(matching_markets, chunk)
            canonical["source_markets_sha256"] = market_hash
            canonical["source_candles_sha256"] = candle_hash
            canonical = canonical.drop(columns=["raw_market_json", "raw_candle_json"], errors="ignore")
            table = pa.Table.from_pandas(canonical, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(canonical)
            market_tickers.update(canonical["market_ticker"].astype(str))
    except Exception:
        if writer is not None:
            writer.close()
        if destination.exists():
            destination.unlink()
        raise
    if writer is None:
        raise ValueError("no canonical rows were produced")
    writer.close()
    return {
        "output_path": str(destination.resolve()),
        "rows": int(total_rows),
        "markets": int(len(market_tickers)),
        "source_markets_sha256": market_hash,
        "source_candles_sha256": candle_hash,
        "output_sha256": _sha256_file(destination),
    }


def _combined_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw = row.get("raw_market_json")
    if isinstance(raw, str) and raw.strip():
        decoded = json.loads(raw)
        if isinstance(decoded, dict):
            payload.update(decoded.get("market", decoded))
    payload.update({key: value for key, value in row.items() if not _missing(value)})
    return payload


def _candle_timestamp(candles: pd.DataFrame) -> pd.Series:
    if "timestamp" in candles:
        return pd.to_datetime(candles["timestamp"], errors="raise", utc=True)
    if "end_period_ts" in candles:
        return pd.to_datetime(candles["end_period_ts"], unit="s", errors="raise", utc=True)
    raise ValueError("Historical candles require timestamp or end_period_ts")


def _first_present(payload: dict[str, Any], *names: str) -> Any:
    return next((payload[name] for name in names if name in payload and not _missing(payload[name])), None)


def _missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize Kalshi historical candles")
    parser.add_argument("--markets", type=Path, default=DEFAULT_PROCESSED_DIR / "historical_markets_processed.csv")
    parser.add_argument("--candles", type=Path, default=DEFAULT_PROCESSED_DIR / "historical_candles_processed.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    print(json.dumps(normalize_historical_files(args.markets, args.candles, args.output, chunksize=args.chunksize), indent=2))
