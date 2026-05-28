# Day 6 Data Verification Report

## Files Audited
| file | rows | columns | date_time_column | date_min | date_max | duplicate_rows | missing_values_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C:\Weather Trading\Kalshi-Weather-Trading\data\raw\daily_raw_nyc_openmeteo.csv | 1601 | 19 | time | 2022-01-01T00:00:00 | 2026-05-20T00:00:00 | 0 | 0 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\raw\hourly_raw_nyc_openmeteo.csv | 38424 | 19 | time | 2022-01-01T00:00:00 | 2026-05-20T23:00:00 | 0 | 0 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\forecasts\daily_forecasts_nyc_openmeteo.csv | 1601 | 20 | time | 2022-01-01T00:00:00 | 2026-05-20T00:00:00 | 0 | 1010 |
| C:\Weather Trading\Kalshi-Weather-Trading\data\forecasts\hourly_forecasts_nyc_openmeteo.csv | 38424 | 19 | time | 2022-01-01T00:00:00 | 2026-05-20T23:00:00 | 0 | 24261 |

## Date Ranges
- Actual daily date range: 2022-01-01 to 2026-05-20
- Forecast daily date range: 2022-01-01 to 2026-05-20
- Hourly actual timestamp range: 2022-01-01 00:00:00 to 2026-05-20 23:00:00
- Hourly forecast timestamp range: 2022-01-01 00:00:00 to 2026-05-20 23:00:00

## Key Columns Identified
- actual_daily_date_column: time
- actual_high_column_used: temperature_2m_max (°F)
- forecast_daily_date_column: time
- forecast_high_column_used: temperature_2m_max (°F)
- hourly_actual_time_column: time
- hourly_forecast_time_column: time
- forecast_source: open_meteo_historical_forecast

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
| forecasts_clean | precipitation_probability_max | 1010 | 63.09 |
| hourly_forecasts_clean | precipitation_probability | 24261 | 63.14 |

## Duplicate Timestamps/Dates
| dataset | duplicate_key | duplicate_count |
| --- | --- | --- |
| daily_clean | date, location | 0 |
| hourly_clean | timestamp, location | 0 |
| forecasts_clean | date, location | 0 |
| hourly_forecasts_clean | timestamp, location | 0 |

## Actual High And Forecast High Alignment
- Daily actual rows: 1601
- Daily forecast rows: 1601
- Overlapping date/location rows: 1601
- Merged preview rows: 1601

Forecast error is defined as `actual_high - forecast_high`.

### Forecast Error Summary
- count: 1601
- mean: -0.863
- std: 2.261
- min: -15.400
- 25%: -2.300
- 50%: -0.900
- 75%: 0.500
- max: 10.500

## Forecast-Source Caveat
The forecast dataset is treated as a historical forecast proxy from Open-Meteo, not confirmed official NWS forecast data. Therefore, the current target should be described as actual high minus Open-Meteo historical forecast high unless true archived NWS forecast data is later substituted.

## Validation Warnings
- daily forecast: 'precipitation_probability_max' is 63.09% missing (1010 rows); do not silently treat this as complete
- daily forecast: forecast creation/model-run timestamp is missing; true as-of-time forecast availability is not fully verifiable
- hourly forecast: 'precipitation_probability' is 63.14% missing (24261 rows); do not silently treat this as complete
- hourly forecast: forecast creation/model-run timestamp is missing; true as-of-time forecast availability is not fully verifiable
- Forecast data is an Open-Meteo historical forecast proxy, not confirmed official NWS archived forecast data
- Location was filled as 'NYC' because row-level location/station information is missing

## Known Risks Before Modeling
- Forecast rows do not include an as-of/model-run timestamp, so point-in-time forecast availability cannot be fully verified.
- Open-Meteo forecast history is not proven to match official NWS forecasts or Kalshi trader-visible forecasts.
- Actual and forecast files include coordinate metadata, but no official station identifier.
- High-missingness forecast precipitation-probability columns should not be used as complete features without a deliberate missing-data plan.

## Cleaned Outputs Created
- C:\Weather Trading\Kalshi-Weather-Trading\outputs\data_audit\data_inventory.csv
- C:\Weather Trading\Kalshi-Weather-Trading\outputs\data_audit\data_verification_report.md
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\hourly_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\daily_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\forecasts_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\hourly_forecasts_clean.csv
- C:\Weather Trading\Kalshi-Weather-Trading\data\processed\modeling_base_preview.csv
