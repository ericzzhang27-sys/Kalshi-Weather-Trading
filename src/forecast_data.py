from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .weather_data import (
        CSV_METADATA_ROWS,
        HIGH_MISSING_THRESHOLD_PERCENT,
        _candidate_matches,
        _coerce_numeric_columns,
        _drop_duplicate_keys,
        _existing_columns,
        _find_column,
        _normalise_for_match,
        _print_profile,
        _validate_time_column,
        standardize_openmeteo_columns,
    )
except ImportError:
    from weather_data import (
        CSV_METADATA_ROWS,
        HIGH_MISSING_THRESHOLD_PERCENT,
        _candidate_matches,
        _coerce_numeric_columns,
        _drop_duplicate_keys,
        _existing_columns,
        _find_column,
        _normalise_for_match,
        _print_profile,
        _validate_time_column,
        standardize_openmeteo_columns,
    )


FORECAST_SOURCE = "open_meteo_historical_forecast"
NDFD_FORECAST_SOURCE = "nws_ndfd_historical_forecast"
NDFD_FALLBACK_SOURCE = "nws_ndfd_historical_forecast_with_open_meteo_fallback"


def _read_openmeteo_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), skiprows=CSV_METADATA_ROWS)


def identify_forecast_high_column(df: pd.DataFrame) -> str | None:
    return _find_column(
        df,
        [
            "forecast_high",
            "temperature_2m_max",
            "daily_forecast_high",
            "forecast_daily_high",
            "high_temperature",
        ],
    )


def _standardize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_openmeteo_columns(
        df,
        {
            "weather_code": ["weather_code", "weather_code (wmo code)"],
            "temperature_2m": ["temperature_2m"],
            "temperature_2m_min": ["temperature_2m_min", "forecast_low"],
            "relative_humidity_2m": ["relative_humidity_2m"],
            "dew_point_2m": ["dew_point_2m"],
            "precipitation": ["precipitation"],
            "precipitation_sum": ["precipitation_sum"],
            "precipitation_hours": ["precipitation_hours"],
            "precipitation_probability": ["precipitation_probability"],
            "precipitation_probability_max": ["precipitation_probability_max"],
            "rain": ["rain"],
            "rain_sum": ["rain_sum"],
            "snowfall": ["snowfall"],
            "snowfall_sum": ["snowfall_sum"],
            "surface_pressure": ["surface_pressure"],
            "surface_pressure_mean": ["surface_pressure_mean"],
            "cloud_cover": ["cloud_cover"],
            "cloud_cover_mean": ["cloud_cover_mean"],
            "cloud_cover_low": ["cloud_cover_low"],
            "cloud_cover_mid": ["cloud_cover_mid"],
            "cloud_cover_high": ["cloud_cover_high"],
            "wind_speed_10m": ["wind_speed_10m"],
            "wind_speed_10m_max": ["wind_speed_10m_max"],
            "wind_speed_10m_mean": ["wind_speed_10m_mean"],
            "wind_direction_10m": ["wind_direction_10m"],
            "wind_direction_10m_dominant": ["wind_direction_10m_dominant"],
            "wind_gusts_10m": ["wind_gusts_10m"],
            "wind_gusts_10m_max": ["wind_gusts_10m_max"],
            "daylight_duration": ["daylight_duration"],
            "sunshine_duration": ["sunshine_duration"],
            "shortwave_radiation": ["shortwave_radiation"],
            "shortwave_radiation_sum": ["shortwave_radiation_sum"],
            "direct_radiation": ["direct_radiation"],
            "diffuse_radiation": ["diffuse_radiation"],
            "is_day": ["is_day"],
            "sunrise": ["sunrise"],
            "sunset": ["sunset"],
        },
    )


def load_hourly_forecasts(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["date"] = df["time"].dt.date
    return df


def load_daily_forecasts(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise").dt.date
    df["date"] = df["time"]
    df = standardize_openmeteo_columns(
        df,
        {
            "forecast_high": ["temperature_2m_max"],
            "forecast_low": ["temperature_2m_min"],
        },
    )
    return df


def standardize_daily_forecasts(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    date_column = _find_column(clean, ["date", "time", "timestamp", "datetime"])
    if date_column is None:
        raise ValueError("Could not identify a daily forecast target date column")
    if date_column != "date":
        clean = clean.rename(columns={date_column: "date"})

    clean["date"] = pd.to_datetime(clean["date"], errors="raise").dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    forecast_high_column = identify_forecast_high_column(clean)
    if forecast_high_column is None:
        raise ValueError(
            "Could not identify forecast daily high column. "
            f"Available columns: {list(clean.columns)}"
        )
    if forecast_high_column != "forecast_high":
        clean = clean.rename(columns={forecast_high_column: "forecast_high"})

    clean["forecast_source"] = FORECAST_SOURCE
    clean = _standardize_forecast_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "forecast_high",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_hours",
            "precipitation_probability_max",
            "rain_sum",
            "snowfall_sum",
            "wind_gusts_10m_max",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "weather_code",
            "cloud_cover_mean",
            "dew_point_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
        ],
    )
    clean = _drop_duplicate_keys(clean, ["date", "location"])
    clean = clean.sort_values(["location", "date"]).reset_index(drop=True)

    columns = _existing_columns(
        clean,
        [
            "date",
            "location",
            "forecast_high",
            "forecast_source",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_hours",
            "precipitation_probability_max",
            "rain_sum",
            "snowfall_sum",
            "wind_gusts_10m_max",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "weather_code",
            "cloud_cover_mean",
            "dew_point_2m_mean",
            "relative_humidity_2m_mean",
            "surface_pressure_mean",
            "wind_speed_10m_mean",
            "daylight_duration",
            "sunshine_duration",
            "sunrise",
            "sunset",
        ],
    )
    return clean.loc[:, columns]


def _normalise_prediction_time(value: object) -> str:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"Prediction time must be HH:MM, got {value!r}")
    return f"{int(parts[0]):02d}:{int(parts[1]):02d}"


def _prediction_timestamp_utc(
    timestamps: pd.Series,
    timezone_name: str,
) -> pd.Series:
    parsed = pd.to_datetime(timestamps, errors="raise")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(
            timezone_name,
            ambiguous=True,
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


def load_ndfd_daily_high_forecasts(
    paths: str | Path | Iterable[str | Path],
    location: str = "NYC",
) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(path) for path in paths]

    frames: list[pd.DataFrame] = []
    for path in path_list:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame)

    columns = [
        "date",
        "location",
        "forecast_high",
        "forecast_source",
        "forecast_issue_time",
        "nws_forecast_high_f",
        "ndfd_valid_time_utc",
        "ndfd_lead_hours",
        "ndfd_grid_distance_km",
        "ndfd_source_files",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)

    clean = pd.concat(frames, ignore_index=True, sort=False)
    clean["date"] = pd.to_datetime(clean["date"], errors="raise").dt.normalize()
    if "location" not in clean.columns:
        clean["location"] = location
    clean["location"] = clean["location"].fillna(location).astype(str)

    if "forecast_high" not in clean.columns:
        if "nws_forecast_high_f" not in clean.columns:
            raise ValueError("NDFD archive must include forecast_high or nws_forecast_high_f")
        clean["forecast_high"] = clean["nws_forecast_high_f"]
    clean["forecast_high"] = pd.to_numeric(clean["forecast_high"], errors="coerce")
    clean["nws_forecast_high_f"] = pd.to_numeric(
        clean.get("nws_forecast_high_f", clean["forecast_high"]),
        errors="coerce",
    )
    if "forecast_issue_time" not in clean.columns:
        raise ValueError("NDFD archive must include forecast_issue_time")
    clean["forecast_issue_time"] = pd.to_datetime(
        clean["forecast_issue_time"],
        errors="coerce",
        utc=True,
    )
    clean["forecast_source"] = NDFD_FORECAST_SOURCE
    clean = clean[clean["forecast_high"].notna() & clean["forecast_issue_time"].notna()]
    if clean.empty:
        return pd.DataFrame(columns=columns)

    for column in ["ndfd_lead_hours", "ndfd_grid_distance_km"]:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = _drop_duplicate_keys(clean, ["date", "location", "forecast_issue_time"])
    clean = clean.sort_values(["location", "date", "forecast_issue_time"]).reset_index(drop=True)
    return clean.loc[:, _existing_columns(clean, columns)]


def expand_daily_forecasts_to_prediction_times(
    daily_forecasts: pd.DataFrame,
    prediction_times: Iterable[str],
) -> pd.DataFrame:
    daily = daily_forecasts.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    daily["location"] = daily["location"].astype(str)

    time_df = pd.DataFrame(
        {"prediction_time": [_normalise_prediction_time(value) for value in prediction_times]}
    )
    expanded = daily.assign(_forecast_cross_join_key=1).merge(
        time_df.assign(_forecast_cross_join_key=1),
        on="_forecast_cross_join_key",
        how="inner",
    )
    expanded = expanded.drop(columns="_forecast_cross_join_key")
    expanded["prediction_timestamp"] = pd.to_datetime(
        expanded["date"].dt.strftime("%Y-%m-%d") + " " + expanded["prediction_time"],
        errors="raise",
    )
    return expanded.sort_values(["location", "date", "prediction_timestamp"]).reset_index(drop=True)


def build_prediction_time_forecasts(
    openmeteo_daily: pd.DataFrame,
    ndfd_daily: pd.DataFrame,
    prediction_times: Iterable[str],
    timezone_name: str = "America/New_York",
) -> pd.DataFrame:
    forecasts = expand_daily_forecasts_to_prediction_times(openmeteo_daily, prediction_times)
    forecasts["openmeteo_forecast_high_f"] = pd.to_numeric(
        forecasts["forecast_high"],
        errors="coerce",
    )
    forecasts["nws_forecast_high_f"] = pd.NA
    forecasts["forecast_issue_time"] = pd.Series([pd.NA] * len(forecasts), dtype="object")
    forecasts["ndfd_valid_time_utc"] = pd.NA
    forecasts["ndfd_lead_hours"] = pd.NA
    forecasts["ndfd_grid_distance_km"] = pd.NA
    forecasts["forecast_fallback_reason"] = "no_nws_ndfd_for_target_date"

    ndfd = ndfd_daily.copy()
    if ndfd.empty:
        forecasts["forecast_source"] = FORECAST_SOURCE
        return forecasts

    ndfd["date"] = pd.to_datetime(ndfd["date"], errors="raise").dt.normalize()
    ndfd["location"] = ndfd["location"].astype(str)
    ndfd["forecast_issue_time"] = pd.to_datetime(
        ndfd["forecast_issue_time"],
        errors="coerce",
        utc=True,
    )
    ndfd["forecast_high"] = pd.to_numeric(ndfd["forecast_high"], errors="coerce")
    ndfd = ndfd[ndfd["forecast_issue_time"].notna() & ndfd["forecast_high"].notna()]
    if ndfd.empty:
        forecasts["forecast_source"] = FORECAST_SOURCE
        return forecasts

    forecasts["_prediction_timestamp_utc"] = _prediction_timestamp_utc(
        forecasts["prediction_timestamp"],
        timezone_name,
    )
    ndfd_groups = {
        key: group.sort_values("forecast_issue_time").reset_index(drop=True)
        for key, group in ndfd.groupby(["date", "location"], dropna=False)
    }

    for index, row in forecasts.iterrows():
        key = (row["date"], row["location"])
        candidates = ndfd_groups.get(key)
        if candidates is None or candidates.empty:
            continue
        usable = candidates[candidates["forecast_issue_time"] <= row["_prediction_timestamp_utc"]]
        if usable.empty:
            forecasts.at[index, "forecast_fallback_reason"] = "no_nws_ndfd_issue_as_of_prediction_time"
            continue

        chosen = usable.iloc[-1]
        forecasts.at[index, "forecast_high"] = float(chosen["forecast_high"])
        forecasts.at[index, "forecast_source"] = NDFD_FORECAST_SOURCE
        forecasts.at[index, "forecast_issue_time"] = chosen["forecast_issue_time"]
        forecasts.at[index, "nws_forecast_high_f"] = float(chosen["forecast_high"])
        forecasts.at[index, "forecast_fallback_reason"] = pd.NA
        for column in ["ndfd_valid_time_utc", "ndfd_lead_hours", "ndfd_grid_distance_km"]:
            if column in chosen.index:
                forecasts.at[index, column] = chosen[column]

    forecasts = forecasts.drop(columns="_prediction_timestamp_utc")
    forecasts["forecast_source"] = forecasts["forecast_source"].fillna(FORECAST_SOURCE)
    return forecasts.sort_values(["location", "date", "prediction_timestamp"]).reset_index(drop=True)


def write_ndfd_forecast_reports(
    forecasts: pd.DataFrame,
    coverage_path: str | Path,
    comparison_path: str | Path,
) -> None:
    coverage_output = Path(coverage_path)
    comparison_output = Path(comparison_path)
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    if forecasts.empty or "prediction_time" not in forecasts.columns:
        pd.DataFrame().to_csv(coverage_output, index=False)
        pd.DataFrame().to_csv(comparison_output, index=False)
        return

    report = forecasts.copy()
    report["used_nws_ndfd"] = report["forecast_source"].eq(NDFD_FORECAST_SOURCE)
    coverage = (
        report.groupby(["date", "location"], dropna=False)
        .agg(
            prediction_rows=("prediction_time", "size"),
            nws_ndfd_rows=("used_nws_ndfd", "sum"),
            openmeteo_fallback_rows=("used_nws_ndfd", lambda value: int((~value).sum())),
            earliest_forecast_issue_time=("forecast_issue_time", "min"),
            latest_forecast_issue_time=("forecast_issue_time", "max"),
        )
        .reset_index()
    )
    coverage["nws_ndfd_row_share"] = coverage["nws_ndfd_rows"] / coverage["prediction_rows"]
    coverage.to_csv(coverage_output, index=False)

    comparison_columns = _existing_columns(
        report,
        [
            "date",
            "location",
            "prediction_time",
            "prediction_timestamp",
            "forecast_source",
            "forecast_high",
            "nws_forecast_high_f",
            "openmeteo_forecast_high_f",
            "forecast_issue_time",
            "forecast_fallback_reason",
        ],
    )
    comparison = report.loc[:, comparison_columns].copy()
    if {"nws_forecast_high_f", "openmeteo_forecast_high_f"}.issubset(comparison.columns):
        comparison["nws_minus_openmeteo_f"] = (
            pd.to_numeric(comparison["nws_forecast_high_f"], errors="coerce")
            - pd.to_numeric(comparison["openmeteo_forecast_high_f"], errors="coerce")
        )
    comparison.to_csv(comparison_output, index=False)


def standardize_hourly_forecasts(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    timestamp_column = _find_column(clean, ["timestamp", "time", "datetime", "date"])
    if timestamp_column is None:
        raise ValueError("Could not identify an hourly forecast timestamp column")
    if timestamp_column != "timestamp":
        clean = clean.rename(columns={timestamp_column: "timestamp"})

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="raise")
    clean["date"] = clean["timestamp"].dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    clean["forecast_source"] = FORECAST_SOURCE
    clean = _standardize_forecast_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "precipitation_probability",
            "rain",
            "snowfall",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "is_day",
        ],
    )
    clean = _drop_duplicate_keys(clean, ["timestamp", "location"])
    clean = clean.sort_values(["location", "timestamp"]).reset_index(drop=True)

    columns = _existing_columns(
        clean,
        [
            "timestamp",
            "date",
            "location",
            "forecast_source",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "precipitation_probability",
            "rain",
            "snowfall",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "is_day",
        ],
    )
    return clean.loc[:, columns]


def _temperature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        normalised = _normalise_for_match(column)
        if (
            "temperature" in normalised
            or "dewpoint" in normalised
            or column in {"forecast_high", "forecast_low"}
        ):
            columns.append(str(column))
    return columns


def _append_missing_key_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
    columns: list[str],
) -> None:
    for column in columns:
        if column not in df.columns:
            warnings.append(f"{dataset_name}: missing required column '{column}'")
            continue

        missing_count = int(df[column].isna().sum())
        if missing_count:
            warnings.append(f"{dataset_name}: {missing_count} missing values in '{column}'")


def _append_duplicate_key_warning(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
    keys: list[str],
) -> None:
    if not all(key in df.columns for key in keys):
        return

    duplicate_count = int(df.duplicated(subset=keys).sum())
    if duplicate_count:
        warnings.append(
            f"{dataset_name}: {duplicate_count} duplicate rows by {', '.join(keys)}"
        )


def _append_range_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    for column in _temperature_columns(df):
        values = pd.to_numeric(df[column], errors="coerce")
        bad_count = int(((values < -100) | (values > 140)).sum())
        if bad_count:
            warnings.append(
                f"{dataset_name}: {bad_count} impossible Fahrenheit temperature values in '{column}'"
            )

    for column in df.columns:
        normalised = _normalise_for_match(column)
        values = pd.to_numeric(df[column], errors="coerce")
        if (
            ("precipitation" in normalised or "rain" in normalised or "snowfall" in normalised)
            and int((values < 0).sum())
        ):
            warnings.append(f"{dataset_name}: negative precipitation/rain/snowfall in '{column}'")
        if (
            ("windspeed" in normalised or "windgusts" in normalised)
            and int((values < 0).sum())
        ):
            warnings.append(f"{dataset_name}: negative wind speed/gust values in '{column}'")


def _append_high_missing_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    if df.empty:
        warnings.append(f"{dataset_name}: dataframe is empty")
        return

    missing_percentages = df.isna().mean().mul(100)
    for column, missing_percent in missing_percentages.items():
        if missing_percent >= HIGH_MISSING_THRESHOLD_PERCENT and missing_percent > 0:
            missing_count = int(df[column].isna().sum())
            warnings.append(
                f"{dataset_name}: '{column}' is {missing_percent:.2f}% missing "
                f"({missing_count} rows); do not silently treat this as complete"
            )


def _append_forecast_source_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    if "forecast_source" not in df.columns:
        warnings.append(f"{dataset_name}: missing forecast_source")
    else:
        missing_source = int(df["forecast_source"].isna().sum())
        if missing_source:
            warnings.append(f"{dataset_name}: {missing_source} rows missing forecast_source")

        unique_sources = sorted(str(value) for value in df["forecast_source"].dropna().unique())
        allowed_prediction_time_mix = (
            "prediction_time" in df.columns
            and set(unique_sources).issubset({FORECAST_SOURCE, NDFD_FORECAST_SOURCE})
        )
        if len(unique_sources) != 1 and not allowed_prediction_time_mix:
            warnings.append(
                f"{dataset_name}: ambiguous forecast_source values {unique_sources}"
            )

    as_of_candidates = [
        "forecast_created_at",
        "forecast_issue_time",
        "forecast_reference_time",
        "model_run_time",
        "run_timestamp",
        "as_of",
        "issued_at",
    ]
    has_as_of_column = any(
        any(_candidate_matches(str(column), candidate) for candidate in as_of_candidates)
        for column in df.columns
    )
    if not has_as_of_column:
        source_values = (
            df["forecast_source"].dropna().astype(str).str.lower().unique()
            if "forecast_source" in df.columns
            else []
        )
        is_legacy_openmeteo_hourly = (
            dataset_name.lower().startswith("hourly")
            and len(source_values) > 0
            and all("open_meteo" in source for source in source_values)
        )
        if is_legacy_openmeteo_hourly:
            warnings.append(
                f"{dataset_name}: legacy Open-Meteo hourly forecast file has no model-run "
                "timestamp and is ignored by NWS/NDFD training feature rebuilds"
            )
        else:
            warnings.append(
                f"{dataset_name}: forecast creation/model-run timestamp is missing; "
                "true as-of-time forecast availability is not fully verifiable"
            )


def validate_forecast_values(df: pd.DataFrame, granularity: str) -> list[str]:
    warnings: list[str] = []
    dataset_name = f"{granularity} forecast"
    granularity_normalised = granularity.strip().lower()

    if granularity_normalised.startswith("daily"):
        daily_keys = ["date", "location", "prediction_time"] if "prediction_time" in df.columns else ["date", "location"]
        required_columns = [*daily_keys, "forecast_high"]
        _append_missing_key_warnings(
            warnings,
            df,
            dataset_name,
            required_columns,
        )
        _append_duplicate_key_warning(warnings, df, dataset_name, daily_keys)
    elif granularity_normalised.startswith("hour"):
        _append_missing_key_warnings(warnings, df, dataset_name, ["timestamp", "location"])
        _append_duplicate_key_warning(warnings, df, dataset_name, ["timestamp", "location"])
    else:
        warnings.append(f"Unknown forecast granularity '{granularity}'")

    _append_range_warnings(warnings, df, dataset_name)
    _append_high_missing_warnings(warnings, df, dataset_name)
    _append_forecast_source_warnings(warnings, df, dataset_name)
    return warnings


def validate_hourly_forecasts(df: pd.DataFrame) -> None:
    _validate_time_column(df)

    if df["time"].duplicated().any():
        duplicate_count = int(df["time"].duplicated().sum())
        raise ValueError(f"Hourly forecast data has {duplicate_count} duplicate timestamps")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("Hourly forecast timestamps must be monotonic increasing")

    _print_profile(df, "Hourly forecast weather")


def validate_daily_forecasts(df: pd.DataFrame) -> None:
    if "date" not in df.columns:
        raise ValueError("Daily forecast data must include a 'date' column")

    if df["date"].duplicated().any():
        duplicate_count = int(df["date"].duplicated().sum())
        raise ValueError(f"Daily forecast data has {duplicate_count} duplicate dates")

    if "forecast_high" not in df.columns:
        raise ValueError("Daily forecast data must include 'forecast_high'")

    _print_profile(df, "Daily forecast weather")
