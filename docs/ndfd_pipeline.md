# NDFD Forecast Pipeline

NDFD is historical National Weather Service gridded forecast data. It is not CSV
data; NOAA stores the files as GRIB, so the pipeline downloads GRIB files and
extracts the nearest grid point to Central Park / KNYC.

The target point is latitude `40.7812`, longitude `-73.9665`. Extracted rows are written as a tidy point forecast table, and a second daily feature table is built for merge-friendly modeling features.

Use this pipeline as the historical NWS forecast source. NOAA daily TMAX remains the official observed label, and IEM ASOS remains the hourly observed feature source. Open-Meteo can remain auxiliary, but it should not be the NWS forecast anchor.

## Training Archive

The training forecast-high archive is built from NOAA's public
`noaa-ndfd-pds` S3 bucket, using WMO MaxT files. For NYC/Central Park, the
`YGUZ98` MaxT files cover the nearest CONUS grid point used by this project.
By default the builder keeps the latest WMO update per selected issue hour
(`00Z`, `06Z`, `12Z`, and `18Z`) to keep the archive smaller while preserving
point-in-time forecast availability. Use `--keep-all-updates` only if you want
every intermediate WMO update.

Smoke test:

```powershell
python scripts/build_ndfd_daily_high_archive.py `
  --start-date 2023-06-06 `
  --end-date 2023-06-07 `
  --max-files 4 `
  --archive-output outputs/data/ndfd_knyc_daily_high_forecasts_smoke.csv
```

Full configured run:

```powershell
python scripts/build_ndfd_daily_high_archive.py `
  --start-date 2022-01-01 `
  --end-date 2026-05-20 `
  --stream-by-day `
  --purge-cache-after-extract
```

Default outputs:

- `data/processed/ndfd_knyc_daily_high_forecasts.csv`
- `outputs/data/ndfd_knyc_point_forecasts.csv`
- `outputs/data/ndfd_download_manifest.csv`
- `outputs/data/ndfd_missing_dates.csv`

`scripts/run_day6_data_verification.py` consumes
`data/processed/ndfd_knyc_daily_high_forecasts.csv`. It expands the forecast
table to hourly prediction rows and uses the latest NDFD issue available as of
each `prediction_timestamp`. The Day 6 rebuild now fails if any prediction row
lacks an as-of-available NDFD forecast, so the training CSVs remain NWS-forecast only.

## Legacy Point Probe

The older point-forecast probe remains useful for inspecting additional NDFD
variables from NCEI THREDDS catalogs.

Smoke test:

```powershell
python scripts/build_ndfd_point_forecasts.py `
  --start-date 2023-06-01 `
  --end-date 2023-06-03 `
  --lat 40.7812 `
  --lon -73.9665 `
  --variables maxt,temp,sky,wspd,wdir,pop12 `
  --output outputs/data/ndfd_knyc_point_forecasts.csv
```

Full configured run, after the smoke test passes:

```powershell
python scripts/build_ndfd_point_forecasts.py `
  --start-date 2022-01-01 `
  --end-date 2026-05-23 `
  --lat 40.7812 `
  --lon -73.9665 `
  --variables maxt,temp,sky,wspd,wdir,pop12 `
  --output outputs/data/ndfd_knyc_point_forecasts.csv
```

GRIB extraction requires:

```powershell
pip install xarray cfgrib eccodes
```

Catalog discovery starts with the NCEI THREDDS pattern:

```text
https://www.ncei.noaa.gov/thredds/catalog/model-ndfd-file/access/YYYYMM/YYYYMMDD/catalog.xml
```

The code also tries sibling non-`access` and `historical` paths because the archive layout changes across periods. Missing catalogs, skipped centers, failed downloads, and local cache paths are written to `outputs/data/ndfd_download_manifest.csv`.
