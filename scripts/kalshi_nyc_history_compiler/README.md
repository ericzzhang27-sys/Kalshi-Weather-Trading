# Kalshi NYC Daily-High Historical Price Dataset Compiler

This package compiles the deepest practical public history for Kalshi's **New York City daily high temperature** market.

## Why this is not just `KXHIGHNY`

The NYC temperature contract changed ticker families over time:

- `HIGHNY0-*` — earliest legacy NYC high-temperature contracts
- `HIGHNY-*` — later legacy contracts
- `KXHIGHNY-*` — current series

The earliest verified NYC high-temperature market I found is `HIGHNY0-21JUL17-T90`, created on **2021-07-16 13:43:19 UTC**. Filtering only `KXHIGHNY` would therefore discard years of history.

## What the compiler produces

- `nyc_trades.parquet` — every captured executed trade, canonical dataset
- `nyc_markets.parquet` — market metadata/resolution
- `nyc_bars_15m.parquet` — 15-minute OHLC/VWAP/volume
- `nyc_bars_1h.parquet` — hourly OHLC/VWAP/volume
- `coverage_by_year.csv` — coverage by year and ticker family
- `coverage_report.json` — exact first/last trade and row counts from your run

Optional `--csv` writes gzip-compressed CSV copies as well.

## Data sources

1. TrevorJS Kalshi archive on Hugging Face

   https://huggingface.co/datasets/TrevorJS/kalshi-trades

   Published coverage: **June 2021 through January 2026** across Kalshi. It contains 154,505,005 trade rows and 17,464,713 market rows and is built from Kalshi public API data.

2. Kalshi public Trade API

   https://external-api.kalshi.com/trade-api/v2

   The script uses both the live and historical tiers to append data after the static archive's last NYC trade. Public market/trade endpoints do not require an API key.

Official documentation:
- https://docs.kalshi.com/getting_started/historical_data
- https://docs.kalshi.com/api-reference/market/get-trades
- https://docs.kalshi.com/api-reference/historical/get-historical-trades
- https://docs.kalshi.com/api-reference/historical/get-historical-markets

## Install

Python 3.11+ recommended.

```powershell
py -m pip install -r requirements.txt
```

## Build the full dataset

From this folder:

```powershell
py compile_nyc_kalshi.py --output .\kalshi_nyc_history --csv
```

If you only want the static 2021–Jan-2026 archive extraction:

```powershell
py compile_nyc_kalshi.py --output .\kalshi_nyc_history --archive-only --csv
```

Parquet is the recommended format for backtesting. The CSV copies are convenient, but larger/slower.

## Canonical trade schema

| column | meaning |
|---|---|
| `trade_id` | unique public trade ID |
| `ticker` | Kalshi market ticker |
| `count_fp` | contracts traded; normalized fixed-point size |
| `yes_price_dollars` | YES execution price in dollars |
| `no_price_dollars` | NO execution price in dollars |
| `taker_outcome_side` | outcome side that took liquidity |
| `created_time` | execution timestamp in UTC |
| `source` | `hf_archive`, `kalshi_historical`, or `kalshi_live` |

The static archive stores legacy prices in integer cents and counts as integers. The current Kalshi API uses fixed-point strings (`*_dollars`, `*_fp`). The compiler normalizes both before merging.

## 15-minute / hourly bar schema

Each row is per **market ticker + time bucket** and contains:

- YES OHLC
- YES VWAP
- NO OHLC
- NO VWAP
- contracts traded
- trade count

This is usually the best input for your backtest dashboard, while `nyc_trades.parquet` remains the lossless source.

## Important caveats

- This is **executed-trade history**, not historical full order-book depth. Historical bid/ask snapshots cannot be reconstructed perfectly from trades alone.
- Old Kalshi aliases are included because the archive shows the NYC high-temperature product under those prefixes. Kalshi's general guidance is not to infer arbitrary series relationships only from ticker strings.
- The static third-party archive is public and derived from Kalshi's API, but it is still a third-party archive. Keep `trade_id` and source fields so you can audit/deduplicate.
- Kalshi can move its historical cutoff over time. The script deliberately queries both live and historical tiers and deduplicates by `trade_id`.
- The actual earliest **trade** may be later than the earliest market creation time. `coverage_report.json` records the exact earliest trade found by the extraction.

## Direct DuckDB extraction

If you already use DuckDB, see `extract_archive_only.sql` for the minimal 2021–Jan-2026 extraction.
