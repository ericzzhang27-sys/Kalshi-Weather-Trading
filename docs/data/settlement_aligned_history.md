# Settlement-aligned historical weather data

This branch extends the NYC weather dataset toward a settlement-aligned 2018-present history.

The full June-2018-through-present NDFD matrix build is triggered by normal (non-`[skip ci]`) branch updates and commits validated compact outputs back to this branch only after coverage checks pass.

## Sources

| Source | Intended coverage | Role |
|---|---:|---|
| IEM parsed NWS CLI (`KNYC`) | 2018-present | Historical daily-high settlement proxy and audit trail |
| IEM ASOS/METAR (`NYC` / KNYC) | 2018-present | Timestamped intraday observation path |
| NOAA/NWS NDFD MaxT | June 2018-present online archive | Timestamp-safe official forecast-high vintages |
| NCEI Central Park (`USW00094728`) | existing 2022+ file | Independent overlap validation for daily TMAX |

The IEM downloads are executed in year-sized requests. This deliberately respects the ASOS service throttle and avoids unnecessarily hammering the public archive.

The observation/target backfill command is:

```bash
python scripts/backfill_knyc_iem.py --start-date 2018-01-01 --end-date 2026-08-22
python scripts/validate_knyc_settlement_sources.py
```

Generated observation/target outputs:

- `data/raw/NYC_nws_hourly_2018_2026.csv`
- `data/processed/knyc_cli_daily_2018_2026.csv`
- `outputs/data/knyc_backfill_coverage.json`
- `outputs/data/knyc_cli_vs_ncei_validation.json`
- `outputs/data/knyc_cli_vs_ncei_mismatches.csv`

These generated outputs are also packaged by the PR validation workflow for reproducible handoff without modifying the underlying CSV contents.

## NDFD point-in-time forecast history

The forecast backfill uses actual operational NDFD `Daytime Maximum Temperature` vintages rather than reconstructed forecasts. The production contract is fixed to WMO header `YGUZ98`, center `KWBN`, and the Central Park point.

Online NCEI THREDDS coverage is split across two archive roots:

- `model-ndfd-file_kwbn-old`: legacy operational archive from June 2018 through May 2020.
- `model-ndfd-file`: June 2020 onward.

`scripts/backfill_ndfd_hourly_vintages.py` discovers every real `YGUZ98` update and uses the NCEI NetCDF Subset Service to request only the Central Park point instead of downloading the full CONUS GRIB. It preserves original file issue time, MaxT valid time, grid coordinates, archive root, WMO header, and source file.

The archive is intentionally not interpolated to manufacture forecasts. `scripts/merge_ndfd_hourly_vintages.py` builds one decision row per local clock hour by selecting only the latest real NDFD vintage with `forecast_issue_time <= prediction_timestamp`. The resulting hourly replay is therefore a record of what a live process could actually have known at that hour.

The final forecast/replay outputs are:

- `data/processed/ndfd_knyc_daily_high_forecasts.csv`: compact canonical operational MaxT vintage table, June 2018-present.
- `data/processed/ndfd_knyc_daily_high_forecasts_2018_2026.csv`: explicit versioned copy of the same compact vintage history.
- `data/processed/ndfd_knyc_hourly_asof_forecasts_2018_2026.csv`: hourly decision snapshots carrying forward only the latest already-issued forecast.
- `data/processed/production_replay_base_2018_2026.csv`: hourly NDFD as-of forecast joined to only already-observed KNYC ASOS data plus the final daily target/audit fields.
- `outputs/data/ndfd_hourly_vintage_coverage.json`: coverage, update-frequency, and overlap-validation audit.
- `outputs/data/production_replay_base_coverage.json`: combined replay coverage and target-regime audit.

Forecast vintage rows must preserve source provenance and must never have an issue timestamp after the corresponding prediction timestamp. Older data is not treated as interchangeable solely because it is NOAA/NWS; WMO header, product, location mapping, units, and time semantics are validated explicitly.

## Important interpretation

The IEM CLI dataset is parsed from NWS Daily Climate Report text products. It is preferred over reconstructing a daily maximum from hourly observations because a settlement contract is resolved from a reported climate value, not necessarily the maximum visible in an hourly series.

The processed CLI table deliberately includes the project's canonical target/audit fields: `actual_high`, `official_daily_high_f`, `actual_source`, `source_station`, and source metadata.

The ASOS archive is used only for information that was observable by a prediction timestamp. It must not be used to reconstruct future portions of a target day's path.

For the post-provider-change 2026 regime, NWS CLI remains a useful cross-check but should not be assumed to equal the final Kalshi settlement value without comparing it to the applicable The Weather Company report. `production_replay_base_2018_2026.csv` therefore marks post-2026-08-13 CLI targets as TWC-regime proxies and excludes them from exact-regime training eligibility.

## Next forecast-history additions

After NDFD is validated and merged, add production-parity sources such as LAMP/MOS, NBM, HRRR, and nearby ASOS stations. Raw large model archives should not be committed; only extracted timestamped point/feature tables belong in Git.