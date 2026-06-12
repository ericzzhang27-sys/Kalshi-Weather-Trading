from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.ndfd_fetch import CatalogEntry, infer_variable_from_text


GRIB_INSTALL_MESSAGE = (
    "NDFD GRIB extraction requires xarray, cfgrib, and eccodes. "
    "Install them with: pip install xarray cfgrib eccodes"
)

VARIABLE_CANONICAL = {
    "t": "temp",
    "2t": "temp",
    "temp": "temp",
    "2d": "dewpoint",
    "dewpoint": "dewpoint",
    "dew point": "dewpoint",
    "dewpoint temperature": "dewpoint",
    "dew point temperature": "dewpoint",
    "maxt": "maxt",
    "mx2t": "maxt",
    "tmax": "maxt",
    "tcc": "sky",
    "sky": "sky",
    "wspd": "wspd",
    "si10": "wspd",
    "10si": "wspd",
    "wdir": "wdir",
    "10wdir": "wdir",
    "pop": "pop12",
    "pop12": "pop12",
}

GRIB_CODE_CANONICAL = {
    (0, 0, 0): "temp",
    (0, 0, 4): "maxt",
    (0, 1, 8): "pop12",
    (0, 1, 192): "pop12",
    (0, 2, 0): "wdir",
    (0, 2, 1): "wspd",
    (0, 6, 1): "sky",
}

POINT_FORECAST_COLUMNS = [
    "source",
    "station",
    "target_lat",
    "target_lon",
    "grid_lat",
    "grid_lon",
    "grid_distance_km",
    "forecast_issue_time",
    "valid_time",
    "variable",
    "value_raw",
    "units_raw",
    "value_f",
    "value_standardized",
    "nws_forecast_high_f",
    "nws_forecast_temp_f",
    "file_url",
    "local_file",
    "ingest_date",
    "extraction_status",
]

NDFD_FORECAST_SOURCE = "nws_ndfd_historical_forecast"
DEFAULT_TARGET_TIMEZONE = "America/New_York"


@dataclass
class PointSelection:
    grid_lat: float
    grid_lon: float
    distance_km: float


def normalize_target_lon(target_lon: float, dataset_lons: object) -> float:
    values = np.asarray(dataset_lons, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() >= 0 and target_lon < 0:
        return target_lon % 360
    return target_lon


def temperature_to_fahrenheit(value: float | int | None, units: str | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    unit = (units or "").strip().lower()
    numeric = float(value)
    if unit in {"k", "kelvin", "degk"}:
        return (numeric - 273.15) * 9.0 / 5.0 + 32.0
    if unit in {"c", "degc", "degree celsius", "degrees celsius", "celsius", "°c"}:
        return numeric * 9.0 / 5.0 + 32.0
    if unit in {"f", "degf", "degree fahrenheit", "degrees fahrenheit", "fahrenheit", "°f"}:
        return numeric
    return None


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius_km = 6371.0
    lat1_rad = np.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = np.radians(lat1 - lat2)
    delta_lon = np.radians(lon1 - lon2)
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * math.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def _coord_name(dataset: object, candidates: Iterable[str]) -> str:
    coords = getattr(dataset, "coords", {})
    for candidate in candidates:
        if candidate in coords:
            return candidate
    raise ValueError(f"Could not find coordinate in {list(candidates)}")


def select_nearest_point(dataset: object, lat: float, lon: float) -> tuple[object, PointSelection]:
    lat_name = _coord_name(dataset, ["latitude", "lat", "y"])
    lon_name = _coord_name(dataset, ["longitude", "lon", "x"])
    lat_coord = dataset[lat_name]
    lon_coord = dataset[lon_name]
    lats = np.asarray(dataset[lat_name].values, dtype=float)
    lons = np.asarray(dataset[lon_name].values, dtype=float)
    target_lon = normalize_target_lon(lon, lons)

    if lats.ndim == 1 and lons.ndim == 1 and lats.shape == lons.shape and lat_coord.dims == lon_coord.dims:
        distances = _haversine_km(lats, lons, lat, target_lon)
        flat_index = int(np.nanargmin(distances))
        dim_name = lat_coord.dims[0]
        selected = dataset.isel({dim_name: flat_index})
        grid_lat = float(lats[flat_index])
        grid_lon = float(lons[flat_index])
        return selected, PointSelection(grid_lat=grid_lat, grid_lon=grid_lon, distance_km=float(distances[flat_index]))

    if lats.ndim == 1 and lons.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        select_by_index = False
    else:
        lat_grid, lon_grid = lats, lons
        select_by_index = lat_coord.dims == lon_coord.dims

    distances = _haversine_km(lat_grid, lon_grid, lat, target_lon)
    flat_index = int(np.nanargmin(distances))
    grid_index = np.unravel_index(flat_index, distances.shape)
    grid_lat = float(lat_grid[grid_index])
    grid_lon = float(lon_grid[grid_index])
    if select_by_index:
        selected = dataset.isel(dict(zip(lat_coord.dims, grid_index)))
    else:
        selected = dataset.sel({lat_name: grid_lat, lon_name: grid_lon}, method="nearest")
    return selected, PointSelection(grid_lat=grid_lat, grid_lon=grid_lon, distance_km=float(distances[grid_index]))


def _require_xarray():
    try:
        import xarray as xr
        import cfgrib  # noqa: F401
    except ImportError as exc:
        raise ImportError(GRIB_INSTALL_MESSAGE) from exc
    return xr


def _candidate_filter_kwargs() -> list[dict[str, object]]:
    return [
        {},
        {"filter_by_keys": {"typeOfLevel": "surface"}},
        {"filter_by_keys": {"typeOfLevel": "heightAboveGround"}},
        {"filter_by_keys": {"stepType": "instant"}},
        {"filter_by_keys": {"stepType": "max"}},
        {"filter_by_keys": {"stepType": "avg"}},
    ]


def _dataset_fingerprint(dataset: object) -> tuple[object, ...]:
    data_vars = getattr(dataset, "data_vars", {})
    parts: list[object] = [tuple(sorted(getattr(dataset, "sizes", {}).items()))]
    for name, data_array in data_vars.items():
        attrs = getattr(data_array, "attrs", {})
        parts.append(
            (
                name,
                attrs.get("GRIB_discipline"),
                attrs.get("GRIB_parameterCategory"),
                attrs.get("GRIB_parameterNumber"),
                attrs.get("GRIB_typeOfLevel"),
                attrs.get("GRIB_stepType"),
            )
        )
    return tuple(parts)


def open_grib_datasets(path: str | Path) -> list[object]:
    xr = _require_xarray()
    datasets: list[object] = []
    errors: list[str] = []
    seen: set[tuple[object, ...]] = set()
    for backend_kwargs in _candidate_filter_kwargs():
        try:
            dataset = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={
                    **backend_kwargs,
                    "indexpath": "",
                    "read_keys": ["discipline", "parameterCategory", "parameterNumber"],
                },
            )
            fingerprint = _dataset_fingerprint(dataset)
            if dataset.data_vars and fingerprint not in seen:
                datasets.append(dataset)
                seen.add(fingerprint)
        except Exception as exc:
            errors.append(str(exc))
    if not datasets:
        raise ValueError(f"Could not open GRIB file {path}. Last errors: {' | '.join(errors[-3:])}")
    return datasets


def _grib_code_variable(attrs: dict[str, object]) -> str | None:
    try:
        code = (
            int(attrs["GRIB_discipline"]),
            int(attrs["GRIB_parameterCategory"]),
            int(attrs["GRIB_parameterNumber"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return GRIB_CODE_CANONICAL.get(code)


def _canonical_variable(data_array: object, fallback_text: str = "") -> str | None:
    attrs = getattr(data_array, "attrs", {})
    candidates = [
        attrs.get("GRIB_shortName"),
        attrs.get("shortName"),
        attrs.get("GRIB_name"),
        attrs.get("long_name"),
        getattr(data_array, "name", None),
        fallback_text,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        key = str(candidate).strip().lower()
        if key in VARIABLE_CANONICAL:
            return VARIABLE_CANONICAL[key]
        inferred = infer_variable_from_text(key)
        if inferred:
            return inferred
    code_variable = _grib_code_variable(attrs)
    if code_variable:
        return code_variable
    return None


def _timestamp_scalar(value: object) -> str | None:
    try:
        if hasattr(value, "values"):
            value = value.values
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            return None
        return timestamp.isoformat()
    except Exception:
        return None


def _valid_time_for_array(data_array: object, dataset: object) -> str | None:
    for holder in (data_array, dataset):
        coords = getattr(holder, "coords", {})
        for name in ("valid_time", "time"):
            if name in coords:
                return _timestamp_scalar(coords[name])
    return None


def _coord_timestamp_at_index(coord: object, data_dims: tuple[str, ...], index: tuple[int, ...]) -> str | None:
    values = np.asarray(getattr(coord, "values", coord))
    if values.size == 0:
        return None
    if values.shape == ():
        return _timestamp_scalar(values.item())

    coord_dims = tuple(getattr(coord, "dims", ()))
    if not coord_dims:
        return _timestamp_scalar(values.reshape(-1)[0])

    indexer: list[int] = []
    for dim in coord_dims:
        if dim not in data_dims:
            return None
        indexer.append(index[data_dims.index(dim)])
    try:
        return _timestamp_scalar(values[tuple(indexer)])
    except Exception:
        return None


def _valid_time_for_index(data_array: object, dataset: object, index: tuple[int, ...]) -> str | None:
    data_dims = tuple(getattr(data_array, "dims", ()))
    for holder in (data_array, dataset):
        coords = getattr(holder, "coords", {})
        for name in ("valid_time", "time"):
            if name in coords:
                timestamp = _coord_timestamp_at_index(coords[name], data_dims, index)
                if timestamp:
                    return timestamp
    return _valid_time_for_array(data_array, dataset)


def _extract_scalar(data_array: object) -> float | None:
    try:
        values = np.asarray(data_array.values)
        if values.size == 0:
            return None
        return float(values.reshape(-1)[0])
    except Exception:
        return None


def _iter_scalar_values(data_array: object, dataset: object) -> Iterable[tuple[float | None, str | None]]:
    values = np.asarray(data_array.values)
    if values.size == 0:
        return
    if values.shape == ():
        yield _extract_scalar(data_array), _valid_time_for_array(data_array, dataset)
        return
    for index in np.ndindex(values.shape):
        value = values[index]
        yield (float(value) if pd.notna(value) else None), _valid_time_for_index(data_array, dataset, index)


def _fallback_units(variable: str | None, units: str | None) -> str | None:
    normalised = (units or "").strip().lower()
    if normalised and normalised != "unknown":
        return units
    if variable in {"maxt", "temp", "dewpoint"}:
        return "K"
    if variable == "sky":
        return "%"
    if variable == "wspd":
        return "m/s"
    if variable == "wdir":
        return "degree"
    if variable == "pop12":
        return "%"
    return units


def _standardized_value(variable: str | None, value_raw: float | None, units_raw: str | None) -> tuple[float | None, float | None]:
    value_f = temperature_to_fahrenheit(value_raw, units_raw) if variable in {"maxt", "temp"} else None
    if variable in {"maxt", "temp"}:
        return value_f, value_f
    return None, value_raw


def extract_file_to_rows(
    entry: CatalogEntry,
    lat: float,
    lon: float,
    station: str = "KNYC",
    requested_variables: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    if not entry.local_path:
        return []
    requested = {value.strip().lower() for value in requested_variables or [] if value.strip()}
    rows: list[dict[str, object]] = []
    ingest_date = datetime.now(timezone.utc).isoformat()

    for dataset in open_grib_datasets(entry.local_path):
        try:
            point_dataset, selection = select_nearest_point(dataset, lat, lon)
        except Exception as exc:
            rows.append(_failed_row(entry, lat, lon, station, f"point_selection_failed: {exc}", ingest_date))
            continue

        for name, data_array in point_dataset.data_vars.items():
            variable = _canonical_variable(data_array, f"{entry.filename} {name}")
            if requested and variable not in requested:
                continue
            units_raw = _fallback_units(variable, getattr(data_array, "attrs", {}).get("units"))
            for value_raw, valid_time in _iter_scalar_values(data_array, dataset):
                value_f, value_standardized = _standardized_value(variable, value_raw, units_raw)
                row = {
                    "source": "NDFD",
                    "station": station,
                    "target_lat": lat,
                    "target_lon": lon,
                    "grid_lat": selection.grid_lat,
                    "grid_lon": selection.grid_lon,
                    "grid_distance_km": selection.distance_km,
                    "forecast_issue_time": entry.issue_time,
                    "valid_time": valid_time,
                    "variable": variable,
                    "value_raw": value_raw,
                    "units_raw": units_raw,
                    "value_f": value_f,
                    "value_standardized": value_standardized,
                    "nws_forecast_high_f": value_f if variable == "maxt" else None,
                    "nws_forecast_temp_f": value_f if variable == "temp" else None,
                    "file_url": entry.file_url,
                    "local_file": entry.local_path,
                    "ingest_date": ingest_date,
                    "extraction_status": "extracted" if variable else "skipped_unknown_variable",
                }
                rows.append(row)
    return rows


def _failed_row(
    entry: CatalogEntry,
    lat: float,
    lon: float,
    station: str,
    status: str,
    ingest_date: str,
) -> dict[str, object]:
    return {
        "source": "NDFD",
        "station": station,
        "target_lat": lat,
        "target_lon": lon,
        "grid_lat": None,
        "grid_lon": None,
        "grid_distance_km": None,
        "forecast_issue_time": entry.issue_time,
        "valid_time": None,
        "variable": None,
        "value_raw": None,
        "units_raw": None,
        "value_f": None,
        "value_standardized": None,
        "nws_forecast_high_f": None,
        "nws_forecast_temp_f": None,
        "file_url": entry.file_url,
        "local_file": entry.local_path,
        "ingest_date": ingest_date,
        "extraction_status": status,
    }


def extract_entries_to_dataframe(
    entries: Iterable[CatalogEntry],
    lat: float,
    lon: float,
    station: str = "KNYC",
    requested_variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in entries:
        if entry.download_status not in {"downloaded", "cached"}:
            continue
        try:
            rows.extend(
                extract_file_to_rows(
                    entry,
                    lat=lat,
                    lon=lon,
                    station=station,
                    requested_variables=requested_variables,
                )
            )
        except Exception as exc:
            rows.append(_failed_row(entry, lat, lon, station, f"open_failed: {exc}", datetime.now(timezone.utc).isoformat()))
    return pd.DataFrame(rows, columns=POINT_FORECAST_COLUMNS)


def validate_point_forecasts(df: pd.DataFrame, target_lat: float, target_lon: float) -> list[str]:
    warnings: list[str] = []
    if df.empty:
        return ["no extracted NDFD rows"]

    valid_time = pd.to_datetime(df["valid_time"], errors="coerce") if "valid_time" in df else pd.Series(dtype="datetime64[ns]")
    if valid_time.isna().all():
        warnings.append("valid_time did not parse for any extracted rows")

    if "value_f" in df:
        temps = pd.to_numeric(df.loc[df["variable"].isin(["maxt", "temp"]), "value_f"], errors="coerce")
        bad_temps = temps.notna() & ((temps < -20) | (temps > 110))
        if bad_temps.any():
            warnings.append(f"{int(bad_temps.sum())} NYC temperature forecasts outside -20F to 110F")

    if {"grid_lat", "grid_lon"}.issubset(df.columns):
        grid = df[["grid_lat", "grid_lon"]].dropna()
        if not grid.empty:
            target_grid_lon = normalize_target_lon(target_lon, grid["grid_lon"].to_numpy())
            distances = _haversine_km(grid["grid_lat"].to_numpy(), grid["grid_lon"].to_numpy(), target_lat, target_grid_lon)
            if float(np.nanmin(distances)) > 50:
                warnings.append("nearest selected grid point is more than 50 km from Central Park")

    if {"forecast_issue_time", "valid_time"}.issubset(df.columns):
        issue = pd.to_datetime(df["forecast_issue_time"], errors="coerce")
        leakage = issue.notna() & valid_time.notna() & (issue > valid_time)
        if leakage.any():
            warnings.append(f"{int(leakage.sum())} rows have forecast_issue_time after valid_time")

    return warnings


def maxt_valid_time_to_target_date(
    valid_time: object,
    timezone_name: str = DEFAULT_TARGET_TIMEZONE,
) -> pd.Timestamp:
    valid = pd.to_datetime(pd.Series([valid_time]), errors="coerce", utc=True).iloc[0]
    if pd.isna(valid):
        return pd.NaT
    return valid.tz_convert(timezone_name).normalize().tz_localize(None)


def _station_to_location(value: object) -> str:
    station = str(value).strip().upper()
    if station == "KNYC":
        return "NYC"
    return station or "NYC"


def build_daily_high_forecast_archive(
    point_df: pd.DataFrame,
    timezone_name: str = DEFAULT_TARGET_TIMEZONE,
) -> pd.DataFrame:
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
    if point_df.empty:
        return pd.DataFrame(columns=columns)

    df = point_df.copy()
    df["variable"] = df.get("variable", pd.Series(dtype=str)).astype(str).str.lower()
    df = df[df["variable"] == "maxt"]
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"] == "extracted"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    high_column = "nws_forecast_high_f" if "nws_forecast_high_f" in df.columns else "value_f"
    df["nws_forecast_high_f"] = pd.to_numeric(df[high_column], errors="coerce")
    df["forecast_issue_time"] = pd.to_datetime(df["forecast_issue_time"], errors="coerce", utc=True)
    df["valid_time_utc"] = pd.to_datetime(df["valid_time"], errors="coerce", utc=True)
    df = df[df["nws_forecast_high_f"].notna() & df["forecast_issue_time"].notna() & df["valid_time_utc"].notna()]
    df = df[df["nws_forecast_high_f"].between(-20, 120)]
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["date"] = (
        df["valid_time_utc"]
        .dt.tz_convert(timezone_name)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    station_series = df["station"] if "station" in df.columns else pd.Series(["KNYC"] * len(df), index=df.index)
    df["location"] = station_series.map(_station_to_location)
    df["ndfd_lead_hours"] = (
        df["valid_time_utc"] - df["forecast_issue_time"]
    ).dt.total_seconds() / 3600.0
    df["ndfd_grid_distance_km"] = pd.to_numeric(
        df.get("grid_distance_km", pd.Series(dtype=float)),
        errors="coerce",
    )

    records: list[dict[str, object]] = []
    group_keys = ["date", "location", "forecast_issue_time"]
    for (target_date, location, issue_time), group in df.groupby(group_keys, dropna=False):
        group = group.sort_values(["ndfd_grid_distance_km", "valid_time_utc"], na_position="last")
        chosen = group.iloc[0]
        source_files = sorted({str(value) for value in group.get("local_file", pd.Series(dtype=str)).dropna().unique()})
        records.append(
            {
                "date": target_date,
                "location": location,
                "forecast_high": float(chosen["nws_forecast_high_f"]),
                "forecast_source": NDFD_FORECAST_SOURCE,
                "forecast_issue_time": issue_time.isoformat() if pd.notna(issue_time) else None,
                "nws_forecast_high_f": float(chosen["nws_forecast_high_f"]),
                "ndfd_valid_time_utc": chosen["valid_time_utc"].isoformat(),
                "ndfd_lead_hours": float(chosen["ndfd_lead_hours"]),
                "ndfd_grid_distance_km": float(chosen["ndfd_grid_distance_km"])
                if pd.notna(chosen["ndfd_grid_distance_km"])
                else None,
                "ndfd_source_files": ";".join(source_files),
            }
        )

    result = pd.DataFrame(records, columns=columns)
    return result.sort_values(["location", "date", "forecast_issue_time"]).reset_index(drop=True)


def build_daily_features(point_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "forecast_issue_time",
        "nws_forecast_high_f",
        "nws_forecast_temp_near_09_f",
        "nws_forecast_temp_near_12_f",
        "nws_forecast_temp_near_15_f",
        "nws_forecast_temp_near_18_f",
        "nws_sky_mean_daytime",
        "nws_wind_mean_daytime",
        "source_files",
    ]
    if point_df.empty:
        return pd.DataFrame(columns=columns)

    df = point_df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df["forecast_issue_time"] = pd.to_datetime(df["forecast_issue_time"], errors="coerce")
    df = df[df["valid_time"].notna()]
    df["date"] = df["valid_time"].dt.date

    records: list[dict[str, object]] = []
    for (day, issue_time), group in df.groupby(["date", "forecast_issue_time"], dropna=False):
        record: dict[str, object] = {
            "date": day,
            "forecast_issue_time": issue_time.isoformat() if pd.notna(issue_time) else None,
            "nws_forecast_high_f": _first_numeric(group.loc[group["variable"] == "maxt", "nws_forecast_high_f"]),
            "nws_forecast_temp_near_09_f": _temp_near_hour(group, 9),
            "nws_forecast_temp_near_12_f": _temp_near_hour(group, 12),
            "nws_forecast_temp_near_15_f": _temp_near_hour(group, 15),
            "nws_forecast_temp_near_18_f": _temp_near_hour(group, 18),
            "nws_sky_mean_daytime": _daytime_mean(group, "sky"),
            "nws_wind_mean_daytime": _daytime_mean(group, "wspd"),
            "source_files": ";".join(sorted({str(value) for value in group["local_file"].dropna().unique()})),
        }
        records.append(record)
    return pd.DataFrame(records, columns=columns).sort_values(["date", "forecast_issue_time"], na_position="last").reset_index(drop=True)


def _first_numeric(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.iloc[0]) if not numeric.empty else None


def _temp_near_hour(group: pd.DataFrame, hour: int) -> float | None:
    temps = group[group["variable"] == "temp"].copy()
    temps["value"] = pd.to_numeric(temps["nws_forecast_temp_f"], errors="coerce")
    temps = temps[temps["value"].notna()]
    if temps.empty:
        return None
    target = temps["valid_time"].dt.normalize() + pd.to_timedelta(hour, unit="h")
    idx = (temps["valid_time"] - target).abs().idxmin()
    return float(temps.loc[idx, "value"])


def _daytime_mean(group: pd.DataFrame, variable: str) -> float | None:
    subset = group[(group["variable"] == variable) & (group["valid_time"].dt.hour.between(9, 18))].copy()
    values = pd.to_numeric(subset["value_standardized"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path
