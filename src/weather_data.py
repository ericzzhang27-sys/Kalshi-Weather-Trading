from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


CSV_METADATA_ROWS = 3
TOP_MISSING_COLUMNS = 10
HIGH_MISSING_THRESHOLD_PERCENT = 25.0
NYC_LOCATION = "NYC"
OFFICIAL_DAILY_SOURCE = "noaa_nws_daily_tmax"
NWS_ASOS_SOURCE = "iem_nws_asos"
CENTRAL_PARK_STATION = "USW00094728"
CENTRAL_PARK_NAME_FRAGMENT = "central park"
NYC_HIGH_MIN_F = -20.0
NYC_HIGH_MAX_F = 110.0
ASOS_TEMP_MIN_F = -40.0
ASOS_TEMP_MAX_F = 130.0
ASOS_MISSING_VALUES = ["", "M", "NA", "NaN", "nan"]
ASOS_TRACE_VALUES = {"T", "TRACE", "trace"}

OFFICIAL_DAILY_COLUMNS = [
    "date",
    "location",
    "actual_high",
    "official_daily_high_f",
    "actual_source",
    "source_file",
    "source_station",
    "source_station_name",
]

NWS_HOURLY_COLUMNS = [
    "timestamp",
    "date",
    "location",
    "station",
    "nws_current_temp_f",
    "nws_dew_point_f",
    "nws_relative_humidity",
    "nws_wind_dir",
    "nws_wind_speed_kt",
    "nws_wind_gust_kt",
    "nws_altimeter",
    "nws_mslp",
    "nws_precip_1h",
    "nws_skyc1",
    "nws_skyc2",
    "nws_skyc3",
    "nws_metar",
    "nws_cloud_cover_pct",
    "source_file",
    "observation_source",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "wind_direction_10m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
]

SKY_COVER_TO_PERCENT = {
    "CLR": 0.0,
    "SKC": 0.0,
    "NCD": 0.0,
    "NSC": 0.0,
    "FEW": 20.0,
    "SCT": 40.0,
    "BKN": 75.0,
    "OVC": 100.0,
    "VV": 100.0,
}


def _normalise_for_match(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.replace("Adeg", "").replace("A", "")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _base_openmeteo_name(value: str) -> str:
    return str(value).split("(", 1)[0].strip()


def _candidate_matches(column: str, candidate: str) -> bool:
    column_norm = _normalise_for_match(column)
    candidate_norm = _normalise_for_match(candidate)

    if column_norm == candidate_norm:
        return True

    candidate_base_norm = _normalise_for_match(_base_openmeteo_name(candidate))
    column_base_norm = _normalise_for_match(_base_openmeteo_name(column))
    return bool(candidate_base_norm) and column_base_norm == candidate_base_norm


def _iter_rename_targets(rename_map: dict[str, str | Iterable[str]]) -> Iterable[tuple[str, list[str]]]:
    for source_or_target, target_or_sources in rename_map.items():
        if isinstance(target_or_sources, str):
            yield target_or_sources, [source_or_target]
        else:
            yield source_or_target, list(target_or_sources)


def standardize_openmeteo_columns(
    df: pd.DataFrame,
    rename_map: dict[str, str | Iterable[str]],
) -> pd.DataFrame:
    """Rename Open-Meteo columns while tolerating unit suffix differences."""
    rename_lookup: dict[str, str] = {}

    for target, candidates in _iter_rename_targets(rename_map):
        if target in df.columns:
            continue

        for candidate in candidates:
            exact_matches = [column for column in df.columns if column == candidate]
            if exact_matches:
                rename_lookup[exact_matches[0]] = target
                break

            fuzzy_matches = [
                column
                for column in df.columns
                if column not in rename_lookup and _candidate_matches(column, candidate)
            ]
            if len(fuzzy_matches) == 1:
                rename_lookup[fuzzy_matches[0]] = target
                break

    return df.rename(columns=rename_lookup)


def _read_openmeteo_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path), skiprows=CSV_METADATA_ROWS)


def _missing_percentages(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().mul(100).sort_values(ascending=False)


def _print_missing_summary(df: pd.DataFrame) -> None:
    missing = _missing_percentages(df).head(TOP_MISSING_COLUMNS)
    missing = missing[missing > 0]

    print("Top missing-value percentages:")
    if missing.empty:
        print("  none")
        return

    for column, percent in missing.items():
        print(f"  {column}: {percent:.2f}%")


def _print_profile(df: pd.DataFrame, label: str, date_column: str = "date") -> None:
    print(f"\n{label}")
    print(f"Shape: {df.shape}")

    if date_column in df.columns and not df.empty:
        print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
    elif "time" in df.columns and not df.empty:
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")

    print(f"Columns: {list(df.columns)}")
    _print_missing_summary(df)


def _validate_time_column(df: pd.DataFrame) -> None:
    if "time" not in df.columns:
        raise ValueError("Expected Open-Meteo CSV to include a 'time' column")


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        exact_matches = [
            column for column in df.columns if str(column).strip().lower() == candidate.lower()
        ]
        if exact_matches:
            return str(exact_matches[0])

        fuzzy_matches = [
            str(column) for column in df.columns if _candidate_matches(str(column), candidate)
        ]
        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0]

    return None


def _iter_csv_paths(path_or_dir: str | Path) -> list[Path]:
    path = Path(path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.glob("*.csv") if candidate.is_file())


def _read_standard_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=ASOS_MISSING_VALUES, keep_default_na=True)


def _select_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return _find_column(df, candidates)


def _source_file_value(path: Path) -> str:
    return str(path).replace("\\", "/")


def _daily_source_score(df: pd.DataFrame) -> int:
    score = 0
    station_col = _select_column(df, ["STATION", "station", "station_id"])
    name_col = _select_column(df, ["NAME", "name", "station_name"])
    if station_col is not None:
        stations = set(df[station_col].dropna().astype(str).str.upper())
        if CENTRAL_PARK_STATION in stations:
            score += 1000
    if name_col is not None:
        names = " ".join(df[name_col].dropna().astype(str).str.lower().unique())
        if CENTRAL_PARK_NAME_FRAGMENT in names:
            score += 500
        if "ny city" in names or "new york" in names:
            score += 50
    score += min(len(df), 100)
    return score


def _candidate_official_daily_files(path_or_dir: str | Path) -> list[tuple[int, Path, pd.DataFrame]]:
    candidates: list[tuple[int, Path, pd.DataFrame]] = []
    for path in _iter_csv_paths(path_or_dir):
        try:
            df = _read_standard_csv(path)
        except Exception:
            continue
        date_col = _select_column(df, ["DATE", "date"])
        tmax_col = _select_column(df, ["TMAX", "tmax"])
        if date_col is None or tmax_col is None:
            continue
        candidates.append((_daily_source_score(df), path, df))
    return sorted(candidates, key=lambda item: (item[0], item[1].name), reverse=True)


def _validate_official_tmax_units(values: pd.Series, *, path: Path) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    non_missing = numeric.dropna()
    if non_missing.empty:
        raise ValueError(f"Official daily TMAX has no numeric values: {path}")

    plausible_f = non_missing.between(NYC_HIGH_MIN_F, NYC_HIGH_MAX_F).mean()
    if plausible_f >= 0.99:
        return numeric.astype(float)

    tenths_c_to_f = non_missing.div(10.0).mul(9.0 / 5.0).add(32.0)
    if tenths_c_to_f.between(NYC_HIGH_MIN_F, NYC_HIGH_MAX_F).mean() >= 0.99:
        raise ValueError(
            "Official daily TMAX appears to be encoded in tenths Celsius, not Fahrenheit. "
            f"Refusing to silently convert without explicit metadata: {path}"
        )

    raise ValueError(
        "Official daily TMAX values are outside plausible NYC Fahrenheit bounds "
        f"({NYC_HIGH_MIN_F:g}F to {NYC_HIGH_MAX_F:g}F): {path}"
    )


def _resolve_duplicate_daily_dates(df: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    if not df.duplicated(subset=["date"]).any():
        return df

    pieces = []
    for date_value, group in df.groupby("date", sort=True, dropna=False):
        unique_highs = group["official_daily_high_f"].dropna().unique()
        unique_station = group.get("source_station", pd.Series(dtype=object)).dropna().unique()
        exact_duplicate_count = int(group.drop_duplicates().shape[0])
        if len(unique_highs) <= 1 and len(unique_station) <= 1:
            pieces.append(group.iloc[[0]])
            continue
        if exact_duplicate_count == 1:
            pieces.append(group.iloc[[0]])
            continue
        raise ValueError(
            "Official daily high file has unresolved duplicate DATE rows for "
            f"{pd.Timestamp(date_value).date()}: {path}"
        )
    return pd.concat(pieces, ignore_index=True)


def load_official_daily_highs(path_or_dir: str | Path) -> pd.DataFrame:
    """
    Load official NOAA/NWS daily TMAX rows, preferring Central Park / USW00094728.

    The expected NOAA daily export has DATE and TMAX columns plus optional station
    identifiers. TMAX is accepted only when the raw values are already plausible
    NYC Fahrenheit highs; likely tenths-Celsius encodings raise instead of being
    silently converted.
    """
    candidates = _candidate_official_daily_files(path_or_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No official NOAA/NWS daily TMAX CSV found under {Path(path_or_dir)}"
        )

    _, path, raw = candidates[0]
    date_col = _select_column(raw, ["DATE", "date"])
    tmax_col = _select_column(raw, ["TMAX", "tmax"])
    station_col = _select_column(raw, ["STATION", "station", "station_id"])
    name_col = _select_column(raw, ["NAME", "name", "station_name"])
    if date_col is None or tmax_col is None:
        raise ValueError(f"Official daily high file is missing DATE or TMAX: {path}")

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(raw[date_col], errors="raise").dt.normalize()
    result["location"] = NYC_LOCATION
    result["official_daily_high_f"] = _validate_official_tmax_units(raw[tmax_col], path=path)
    result["actual_high"] = result["official_daily_high_f"]
    result["actual_source"] = OFFICIAL_DAILY_SOURCE
    result["source_file"] = _source_file_value(path)
    if station_col is not None:
        result["source_station"] = raw[station_col].astype(str)
    else:
        result["source_station"] = ""
    if name_col is not None:
        result["source_station_name"] = raw[name_col].astype(str)
    else:
        result["source_station_name"] = ""

    if result["date"].isna().any():
        raise ValueError(f"Official daily high file contains unparseable dates: {path}")
    missing_highs = int(result["official_daily_high_f"].isna().sum())
    if missing_highs:
        raise ValueError(f"Official daily high file has {missing_highs} missing TMAX values: {path}")

    bad_highs = result[
        ~result["official_daily_high_f"].between(NYC_HIGH_MIN_F, NYC_HIGH_MAX_F)
    ]
    if not bad_highs.empty:
        raise ValueError(
            f"Official daily high file has {len(bad_highs)} implausible NYC highs: {path}"
        )

    result = result.drop_duplicates()
    result = _resolve_duplicate_daily_dates(result, path=path)
    result = result.sort_values("date").reset_index(drop=True)
    return result.loc[:, OFFICIAL_DAILY_COLUMNS]


def _candidate_nws_hourly_files(path_or_dir: str | Path) -> list[tuple[int, Path, pd.DataFrame]]:
    candidates: list[tuple[int, Path, pd.DataFrame]] = []
    for path in _iter_csv_paths(path_or_dir):
        try:
            df = _read_standard_csv(path)
        except Exception:
            continue
        station_col = _select_column(df, ["station", "STATION"])
        valid_col = _select_column(df, ["valid", "timestamp", "time"])
        temp_col = _select_column(df, ["tmpf", "temp_f", "nws_current_temp_f"])
        if station_col is None or valid_col is None or temp_col is None:
            continue
        stations = set(df[station_col].dropna().astype(str).str.upper().unique())
        score = 1000 if stations.intersection({"NYC", "KNYC"}) else 0
        score += min(len(df), 100)
        candidates.append((score, path, df))
    return sorted(candidates, key=lambda item: (item[0], item[1].name), reverse=True)


def _numeric_asos_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype("object").where(~series.astype(str).str.strip().isin(ASOS_TRACE_VALUES), "0")
    return pd.to_numeric(cleaned, errors="coerce")


def _sky_cover_percent(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().map(SKY_COVER_TO_PERCENT)


def _validate_asos_ranges(df: pd.DataFrame, *, path: Path) -> None:
    for column in ["nws_current_temp_f", "nws_dew_point_f"]:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        bad_count = int(((values < ASOS_TEMP_MIN_F) | (values > ASOS_TEMP_MAX_F)).sum())
        if bad_count:
            raise ValueError(f"{path} has {bad_count} implausible ASOS temperature values in {column}")

    range_checks = {
        "nws_relative_humidity": (0.0, 100.0),
        "nws_wind_dir": (0.0, 360.0),
        "nws_wind_speed_kt": (0.0, 150.0),
        "nws_wind_gust_kt": (0.0, 200.0),
        "nws_precip_1h": (0.0, 20.0),
    }
    for column, (lower, upper) in range_checks.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        bad_count = int(((values < lower) | (values > upper)).sum())
        if bad_count:
            raise ValueError(f"{path} has {bad_count} implausible ASOS values in {column}")


def _standardize_nws_hourly_frame(raw: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    valid_col = _select_column(raw, ["valid", "timestamp", "time"])
    station_col = _select_column(raw, ["station", "STATION"])
    if valid_col is None or station_col is None:
        raise ValueError(f"NWS hourly file is missing station or valid timestamp: {path}")

    result = pd.DataFrame()
    timestamp = pd.to_datetime(raw[valid_col], errors="raise")
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    result["timestamp"] = timestamp
    result["date"] = result["timestamp"].dt.normalize()
    result["location"] = NYC_LOCATION
    result["station"] = raw[station_col].astype(str)

    column_map = {
        "tmpf": "nws_current_temp_f",
        "dwpf": "nws_dew_point_f",
        "relh": "nws_relative_humidity",
        "drct": "nws_wind_dir",
        "sknt": "nws_wind_speed_kt",
        "gust": "nws_wind_gust_kt",
        "alti": "nws_altimeter",
        "mslp": "nws_mslp",
        "p01i": "nws_precip_1h",
    }
    for source, target in column_map.items():
        source_col = _select_column(raw, [source])
        if source_col is not None:
            result[target] = _numeric_asos_column(raw[source_col])

    for source, target in [
        ("skyc1", "nws_skyc1"),
        ("skyc2", "nws_skyc2"),
        ("skyc3", "nws_skyc3"),
        ("metar", "nws_metar"),
    ]:
        source_col = _select_column(raw, [source])
        if source_col is not None:
            result[target] = raw[source_col]

    if "nws_skyc1" in result.columns:
        result["nws_cloud_cover_pct"] = _sky_cover_percent(result["nws_skyc1"])

    result["source_file"] = _source_file_value(path)
    result["observation_source"] = NWS_ASOS_SOURCE

    alias_pairs = {
        "nws_current_temp_f": "temperature_2m",
        "nws_dew_point_f": "dew_point_2m",
        "nws_relative_humidity": "relative_humidity_2m",
        "nws_wind_dir": "wind_direction_10m",
        "nws_wind_speed_kt": "wind_speed_10m",
        "nws_wind_gust_kt": "wind_gusts_10m",
        "nws_precip_1h": "precipitation",
        "nws_cloud_cover_pct": "cloud_cover",
    }
    for source, alias in alias_pairs.items():
        if source in result.columns:
            result[alias] = result[source]

    _validate_asos_ranges(result, path=path)
    return result


def load_nws_hourly_observations(path_or_dir: str | Path) -> pd.DataFrame:
    """
    Load IEM/NWS ASOS hourly and special observations for NYC/KNYC.

    IEM CSV exports used by this project have naive `valid` timestamps already in
    America/New_York local clock time, as confirmed by the METAR Z timestamps in
    the raw file. Timezone-aware inputs are converted to America/New_York and then
    stored as naive local timestamps to match the rest of the pipeline.
    """
    candidates = _candidate_nws_hourly_files(path_or_dir)
    if not candidates:
        raise FileNotFoundError(f"No IEM/NWS ASOS hourly CSV found under {Path(path_or_dir)}")

    frames = []
    best_score = candidates[0][0]
    for score, path, raw in candidates:
        if score < best_score:
            continue
        frames.append(_standardize_nws_hourly_frame(raw, path=path))

    result = pd.concat(frames, ignore_index=True)
    station_upper = result["station"].astype(str).str.upper()
    preferred = result[station_upper.isin(["NYC", "KNYC"])].copy()
    if not preferred.empty:
        result = preferred

    result = result.drop_duplicates()
    result = result.sort_values(["station", "timestamp"]).reset_index(drop=True)
    for column in NWS_HOURLY_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return result.loc[:, NWS_HOURLY_COLUMNS]


def identify_actual_high_column(df: pd.DataFrame) -> str | None:
    return _find_column(
        df,
        [
            "official_daily_high_f",
            "actual_high",
            "actual_daily_high",
            "temperature_2m_max",
            "daily_high",
            "high_temperature",
        ],
    )


def _standardize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    return standardize_openmeteo_columns(
        df,
        {
            "weather_code": ["weather_code", "weather_code (wmo code)"],
            "temperature_2m": ["temperature_2m"],
            "temperature_2m_min": ["temperature_2m_min", "actual_daily_low"],
            "temperature_2m_mean": ["temperature_2m_mean"],
            "relative_humidity_2m": ["relative_humidity_2m"],
            "dew_point_2m": ["dew_point_2m"],
            "apparent_temperature": ["apparent_temperature"],
            "precipitation": ["precipitation"],
            "precipitation_sum": ["precipitation_sum"],
            "precipitation_hours": ["precipitation_hours"],
            "rain": ["rain"],
            "rain_sum": ["rain_sum"],
            "snowfall": ["snowfall"],
            "snowfall_sum": ["snowfall_sum"],
            "pressure_msl": ["pressure_msl"],
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
            "et0_fao_evapotranspiration": ["et0_fao_evapotranspiration"],
            "is_day": ["is_day"],
            "sunrise": ["sunrise"],
            "sunset": ["sunset"],
        },
    )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _drop_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.drop_duplicates()
    if all(key in df.columns for key in keys):
        df = df.drop_duplicates(subset=keys, keep="first")
    return df


def _existing_columns(df: pd.DataFrame, ordered_columns: Iterable[str]) -> list[str]:
    columns: list[str] = []
    for column in ordered_columns:
        if column in df.columns and column not in columns:
            columns.append(column)
    return columns


def load_hourly_weather(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["date"] = df["time"].dt.date
    return df


def load_daily_weather(path: str | Path) -> pd.DataFrame:
    df = _read_openmeteo_csv(path)
    _validate_time_column(df)

    df["time"] = pd.to_datetime(df["time"], errors="raise").dt.date
    df["date"] = df["time"]
    df = standardize_openmeteo_columns(
        df,
        {
            "actual_daily_high": ["temperature_2m_max", "actual_high"],
            "actual_daily_low": ["temperature_2m_min", "actual_low"],
        },
    )
    return df


def standardize_daily_weather(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    date_column = _find_column(clean, ["date", "time", "timestamp", "datetime"])
    if date_column is None:
        raise ValueError("Could not identify a daily weather date/time column")
    if date_column != "date":
        clean = clean.rename(columns={date_column: "date"})

    clean["date"] = pd.to_datetime(clean["date"], errors="raise").dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    actual_high_column = identify_actual_high_column(clean)
    if actual_high_column is None:
        raise ValueError(
            "Could not identify actual daily high column. "
            f"Available columns: {list(clean.columns)}"
        )
    if actual_high_column == "official_daily_high_f":
        clean["actual_high"] = pd.to_numeric(clean[actual_high_column], errors="coerce")
        if "actual_source" not in clean.columns:
            clean["actual_source"] = OFFICIAL_DAILY_SOURCE
    elif actual_high_column != "actual_high":
        clean = clean.rename(columns={actual_high_column: "actual_high"})

    clean = _standardize_weather_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "actual_high",
            "official_daily_high_f",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "weather_code",
            "daylight_duration",
            "sunshine_duration",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ],
    )
    clean = _drop_duplicate_keys(clean, ["date", "location"])
    clean = clean.sort_values(["location", "date"]).reset_index(drop=True)

    columns = _existing_columns(
        clean,
        [
            "date",
            "location",
            "actual_high",
            "official_daily_high_f",
            "actual_source",
            "source_file",
            "source_station",
            "source_station_name",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "weather_code",
            "temperature_2m_min",
            "temperature_2m_mean",
            "daylight_duration",
            "sunshine_duration",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
            "precipitation_hours",
            "wind_direction_10m_dominant",
            "cloud_cover_mean",
            "dew_point_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
            "surface_pressure_mean",
            "sunrise",
            "sunset",
        ],
    )
    return clean.loc[:, columns]


def standardize_hourly_weather(df: pd.DataFrame, location: str = "NYC") -> pd.DataFrame:
    clean = df.copy()

    timestamp_column = _find_column(clean, ["timestamp", "time", "datetime", "date"])
    if timestamp_column is None:
        raise ValueError("Could not identify an hourly weather timestamp column")
    if timestamp_column != "timestamp":
        clean = clean.rename(columns={timestamp_column: "timestamp"})

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="raise")
    clean["date"] = clean["timestamp"].dt.normalize()

    if "location" not in clean.columns:
        clean["location"] = location
    else:
        clean["location"] = clean["location"].fillna(location).astype(str)

    clean = _standardize_weather_columns(clean)
    clean = _coerce_numeric_columns(
        clean,
        [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "weather_code",
            "pressure_msl",
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
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "weather_code",
            "pressure_msl",
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


def _append_missing_key_warnings(
    warnings: list[str],
    df: pd.DataFrame,
    dataset_name: str,
    columns: Iterable[str],
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


def _temperature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        normalised = _normalise_for_match(column)
        if (
            "temperature" in normalised
            or "dewpoint" in normalised
            or column in {"actual_high", "actual_daily_high"}
        ):
            columns.append(str(column))
    return columns


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


def validate_weather_values(df: pd.DataFrame, granularity: str) -> list[str]:
    warnings: list[str] = []
    dataset_name = f"{granularity} weather"
    granularity_normalised = granularity.strip().lower()

    if granularity_normalised.startswith("daily"):
        _append_missing_key_warnings(warnings, df, dataset_name, ["date", "location", "actual_high"])
        _append_duplicate_key_warning(warnings, df, dataset_name, ["date", "location"])
    elif granularity_normalised.startswith("hour"):
        _append_missing_key_warnings(warnings, df, dataset_name, ["timestamp", "location"])
        _append_duplicate_key_warning(warnings, df, dataset_name, ["timestamp", "location"])
    else:
        warnings.append(f"Unknown weather granularity '{granularity}'")

    _append_range_warnings(warnings, df, dataset_name)
    _append_high_missing_warnings(warnings, df, dataset_name)
    return warnings


def validate_hourly_weather(df: pd.DataFrame) -> None:
    _validate_time_column(df)

    if df["time"].duplicated().any():
        duplicate_count = int(df["time"].duplicated().sum())
        raise ValueError(f"Hourly weather data has {duplicate_count} duplicate timestamps")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("Hourly weather timestamps must be monotonic increasing")

    _print_profile(df, "Hourly actual weather")


def validate_daily_weather(df: pd.DataFrame) -> None:
    if "date" not in df.columns:
        raise ValueError("Daily weather data must include a 'date' column")

    if df["date"].duplicated().any():
        duplicate_count = int(df["date"].duplicated().sum())
        raise ValueError(f"Daily weather data has {duplicate_count} duplicate dates")

    if "actual_daily_high" not in df.columns:
        raise ValueError("Daily weather data must include 'actual_daily_high'")

    if "actual_daily_low" in df.columns:
        comparable = df[["actual_daily_high", "actual_daily_low"]].dropna()
        bad_rows = comparable["actual_daily_high"] < comparable["actual_daily_low"]
        if bad_rows.any():
            raise ValueError("Daily weather data has actual_daily_high below actual_daily_low")

    _print_profile(df, "Daily actual weather")
