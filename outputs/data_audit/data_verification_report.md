# Day 6 Data Verification Report

## Files Audited
| file | rows | columns | date_time_column | date_min | date_max | duplicate_rows | missing_values_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C:\Weather Trading\Kalshi-Weather-Trading\data\raw\daily_raw_nyc_openmeteo.csv | 1601 | 19 | time | 2022-01-01T00:00:00 | 2026-05-20T00:00:00 | 0 | 0 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\raw\hourly_raw_nyc_openmeteo.csv | 38424 | 19 | time | 2022-01-01T00:00:00 | 2026-05-20T23:00:00 | 0 | 0 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\raw\nws_daily).csv | 1615 | 22 | DATE | 2022-01-01T00:00:00 | 2026-06-03T00:00:00 | 0 | 13420 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\raw\NYC_nws_hourly.csv | 48462 | 32 | valid | 2022-01-01T00:51:00 | 2026-05-23T23:51:00 | 0 | 0 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\forecasts\daily_forecasts_nyc_openmeteo.csv | 1601 | 20 | time | 2022-01-01T00:00:00 | 2026-05-20T00:00:00 | 0 | 1010 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\forecasts\hourly_forecasts_nyc_openmeteo.csv | 38424 | 19 | time | 2022-01-01T00:00:00 | 2026-05-20T23:00:00 | 0 | 24261 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\processed\ndfd_knyc_daily_high_forecasts.csv | 17564 | 10 | date | 2022-01-01T00:00:00 | 2026-05-20T00:00:00 | 0 | 0 |

## Date Ranges
- Actual daily date range: 2022-01-01 to 2026-06-03
- Forecast daily date range: 2022-01-01 to 2026-05-20
- Hourly actual timestamp range: 2022-01-01 00:51:00 to 2026-05-23 23:51:00
- Hourly forecast timestamp range: 2022-01-01 00:00:00 to 2026-05-20 23:00:00

## Key Columns Identified
- actual_daily_date_column: DATE
- actual_high_column_used: TMAX
- forecast_daily_date_column: time
- forecast_high_column_used: temperature_2m_max (°F)
- hourly_actual_time_column: valid
- hourly_forecast_time_column: time
- actual_source: noaa_nws_daily_tmax
- hourly_observation_source: iem_nws_asos
- forecast_source: nws_ndfd_historical_forecast

## Location And Metadata
- daily_actual: latitude=40.808434, longitude=-74.0199, elevation=33.0, utc_offset_seconds=-14400, timezone=America/New_York, timezone_abbreviation=GMT-4
- hourly_actual: latitude=40.808434, longitude=-74.0199, elevation=33.0, utc_offset_seconds=-14400, timezone=America/New_York, timezone_abbreviation=GMT-4
- daily_forecast: latitude=40.78858, longitude=-73.9661, elevation=33.0, utc_offset_seconds=-14400, timezone=America/New_York, timezone_abbreviation=GMT-4
- hourly_forecast: latitude=40.78858, longitude=-73.9661, elevation=33.0, utc_offset_seconds=-14400, timezone=America/New_York, timezone_abbreviation=GMT-4
- location_column: No row-level location/station column was present in the CSV data; the cleaning step filled location='NYC'.

## Unit Standardization
No unit conversions were performed. Units are documented from raw column names or metadata only when visible in the CSV export.

- temperature: Units visible in column names: °F. Columns: dew_point_2m (°F), dew_point_2m_mean (°F), temperature_2m (°F), temperature_2m_max (°F), temperature_2m_min (°F)
- wind: Units visible in column names: mp/h. Columns: wind_gusts_10m (mp/h), wind_gusts_10m_max (mp/h), wind_speed_10m (mp/h), wind_speed_10m_max (mp/h), wind_speed_10m_mean (mp/h)
- precipitation: Units visible in column names: %, h, inch. Columns: precipitation (inch), precipitation_hours (h), precipitation_probability (%), precipitation_probability_max (%), precipitation_sum (inch), rain (inch)
- conversion_policy: No unit conversion was applied during Day 6 cleaning.

## Missing Values
| dataset | column | missing_count | missing_percent |
| --- | --- | --- | --- |
| hourly_clean | nws_current_temp_f | 51 | 0.11 |
| hourly_clean | nws_dew_point_f | 88 | 0.18 |
| hourly_clean | nws_relative_humidity | 88 | 0.18 |
| hourly_clean | nws_wind_dir | 18457 | 38.09 |
| hourly_clean | nws_wind_speed_kt | 3666 | 7.56 |
| hourly_clean | nws_wind_gust_kt | 40338 | 83.24 |
| hourly_clean | nws_altimeter | 99 | 0.20 |
| hourly_clean | nws_mslp | 10399 | 21.46 |
| hourly_clean | nws_precip_1h | 1 | 0.00 |
| hourly_clean | nws_skyc1 | 223 | 0.46 |
| hourly_clean | nws_skyc2 | 37258 | 76.88 |
| hourly_clean | nws_skyc3 | 44093 | 90.98 |
| hourly_clean | nws_cloud_cover_pct | 433 | 0.89 |
| hourly_clean | temperature_2m | 51 | 0.11 |
| hourly_clean | dew_point_2m | 88 | 0.18 |
| hourly_clean | relative_humidity_2m | 88 | 0.18 |
| hourly_clean | wind_direction_10m | 18457 | 38.09 |
| hourly_clean | wind_speed_10m | 3666 | 7.56 |
| hourly_clean | wind_gusts_10m | 40338 | 83.24 |
| hourly_clean | precipitation | 1 | 0.00 |
| hourly_clean | cloud_cover | 433 | 0.89 |
| forecasts_clean | precipitation_probability_max | 24240 | 63.09 |
| hourly_forecasts_clean | precipitation_probability | 24261 | 63.14 |

## Duplicate Timestamps/Dates
| dataset | duplicate_key | duplicate_count |
| --- | --- | --- |
| daily_clean | date, location | 0 |
| hourly_clean | timestamp, location | 4 |
| forecasts_clean | date, location, prediction_time | 0 |
| hourly_forecasts_clean | timestamp, location | 0 |

## Actual High And Forecast High Alignment
- Daily actual rows: 1615
- Daily forecast rows: 38424
- Overlapping date/location rows: 38424
- Merged preview rows: 38424

Forecast error is defined as `actual_high - forecast_high`.

### Forecast Error Summary
- count: 38424
- mean: 0.405
- std: 2.637
- min: -8.990
- 25%: -1.050
- 50%: 0.050
- 75%: 1.990
- max: 15.010

## Forecast-Source Caveat
The canonical forecast baseline is the timestamp-safe historical NWS/NDFD MaxT forecast. Open-Meteo forecast history is retained only as legacy/auxiliary input and should not be described as the training forecast_high source when NDFD coverage is complete.

## Validation Warnings
- hourly weather: 4 duplicate rows by timestamp, location
- hourly weather: 'nws_wind_dir' is 38.09% missing (18457 rows); do not silently treat this as complete
- hourly weather: 'nws_wind_gust_kt' is 83.24% missing (40338 rows); do not silently treat this as complete
- hourly weather: 'nws_skyc2' is 76.88% missing (37258 rows); do not silently treat this as complete
- hourly weather: 'nws_skyc3' is 90.98% missing (44093 rows); do not silently treat this as complete
- hourly weather: 'wind_direction_10m' is 38.09% missing (18457 rows); do not silently treat this as complete
- hourly weather: 'wind_gusts_10m' is 83.24% missing (40338 rows); do not silently treat this as complete
- daily forecast: 'precipitation_probability_max' is 63.09% missing (24240 rows); do not silently treat this as complete
- hourly forecast: 'precipitation_probability' is 63.14% missing (24261 rows); do not silently treat this as complete
- hourly forecast: legacy Open-Meteo hourly forecast file has no model-run timestamp and is ignored by NWS/NDFD training feature rebuilds
- Historical NWS/NDFD MaxT forecast archive is used as forecast_high where an issue is available as of the prediction time: 38,424 prediction-time rows use NDFD over 2022-01-01 to 2026-05-20; 0 rows use non-NDFD fallback.
- Location was filled as 'NYC' because row-level location/station information is missing

## Known Risks Before Modeling
- NDFD forecast availability is enforced by forecast_issue_time <= prediction_timestamp; rows without an available NDFD issue block the canonical training rebuild.
- Hourly/special ASOS observations can miss brief highs between reports, so official NOAA/NWS daily TMAX remains the label.
- High-missingness forecast precipitation-probability columns should not be used as complete features without a deliberate missing-data plan.

## Cleaned Outputs Created
- C:\Weather Trading\Kalshi-Weather-Trading\outputs\data_audit\data_inventory.csv
- C:\Weather Trading\Kalshi-Weather-Trading\outputs\data_audit\data_verification_report.md
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\hourly_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\daily_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\forecasts_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\hourly_forecasts_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\modeling_base_preview.csv
