-- Archive-only extraction from TrevorJS/kalshi-trades.
-- Run with DuckDB on an internet-connected machine.

INSTALL httpfs;
LOAD httpfs;

COPY (
    SELECT
        trade_id,
        ticker,
        CAST(count AS DECIMAL(20,2)) AS count_fp,
        CAST(yes_price AS DECIMAL(20,6)) / 100 AS yes_price_dollars,
        CAST(no_price AS DECIMAL(20,6)) / 100 AS no_price_dollars,
        taker_side AS taker_outcome_side,
        created_time,
        'hf_archive' AS source
    FROM read_parquet('hf://datasets/TrevorJS/kalshi-trades/trades-*.parquet')
    WHERE regexp_matches(ticker, '^(HIGHNY0|HIGHNY|KXHIGHNY)-')
)
TO 'nyc_trades_archive.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);

COPY (
    SELECT *
    FROM read_parquet('hf://datasets/TrevorJS/kalshi-trades/markets-*.parquet')
    WHERE regexp_matches(ticker, '^(HIGHNY0|HIGHNY|KXHIGHNY)-')
)
TO 'nyc_markets_archive.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD);
