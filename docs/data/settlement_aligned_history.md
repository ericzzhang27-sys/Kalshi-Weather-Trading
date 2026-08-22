# Settlement-aligned historical weather data

This branch extends the NYC weather dataset toward a settlement-aligned 2018-present history.

## Sources

| Source | Intended coverage | Role |
|---|---:|---|
| IEM parsed NWS CLI (`KNYC`) | 2018-present | Historical daily-high settlement proxy and audit trail |
| IEM ASOS/METAR (`NYC` / KNYC) | 2018-present | Timestamped intraday observation path |
| NOAA/NWS NDFD MaxT | existing archive/fetcher | Timestamp-safe official forecast-high anchor |
| NCEI Central Park (`USW00094728`) | existing 2022+ file | Independent overlap validation for daily TMAX |

The backfill command is:

```bash
python scripts/backfill_knyc_iem.py --start-date 2018-01-01 --end-date 2026-08-22
python scripts/validate_knyc_settlement_sources.py
```

Generated outputs:

- `data/raw/NYC_nws_hourly_2018_2026.csv`
- `data/processed/knyc_cli_daily_2018_2026.csv`
- `outputs/data/knyc_backfill_coverage.json`
- `outputs/data/knyc_cli_vs_ncei_validation.json`
- `outputs/data/knyc_cli_vs_ncei_mismatches.csv`

## Important interpretation

The IEM CLI dataset is parsed from NWS Daily Climate Report text products. It is preferred over reconstructing a daily maximum from hourly observations because a settlement contract is resolved from a reported climate value, not necessarily the maximum visible in an hourly series.

The processed CLI table deliberately includes the project's canonical target/audit fields: `actual_high`, `official_daily_high_f`, `actual_source`, `source_station`, and source metadata.

The ASOS archive is used only for information that was observable by a prediction timestamp. It must not be used to reconstruct future portions of a target day's path.

For the post-provider-change 2026 regime, NWS CLI remains a useful cross-check but should not be assumed to equal the final Kalshi settlement value without comparing it to the applicable The Weather Company report.

## Next forecast-history additions

The existing NDFD point-forecast builder should be backfilled separately because multi-year GRIB retrieval is much heavier than the ASOS/CLI downloads. HRRR can provide additional model guidance for the full 2018-present window, while useful NBM public archival coverage is shorter. Raw GRIB archives should not be committed; only extracted Central Park point/feature tables belong in Git.
