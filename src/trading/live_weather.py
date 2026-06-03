from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.trading.config import TradingConfig


LIVE_FORECAST_SOURCE = "open_meteo_live_forecast"
LIVE_OBSERVATION_SOURCE = "open_meteo_live_current"

OBSERVED_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
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
]

FORECAST_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "precipitation_probability",
    "rain",
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
]

DAILY_FORECAST_VARIABLES = [
    "temperature_2m_max",
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
]

ISSUE_TIME_KEYS = [
    "issue_time",
    "issued_at",
    "forecast_issue_time",
    "forecast_created_at",
    "forecast_reference_time",
    "model_run_time",
    "run_time",
    "run_timestamp",
    "reference_time",
    "as_of",
]


class WeatherProviderClient(Protocol):
    def fetch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class LiveWeatherSnapshot:
    location: str
    target_date: date
    prediction_time: datetime
    fetched_at: datetime
    hourly_observations: pd.DataFrame
    hourly_forecasts: pd.DataFrame
    daily_forecast: pd.DataFrame
    diagnostics: pd.DataFrame

    @property
    def no_trade(self) -> bool:
        if self.diagnostics.empty or "status" not in self.diagnostics.columns:
            return False
        return bool((self.diagnostics["status"] == "NO_TRADE").any())


class OpenMeteoClient:
    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1/forecast",
        timeout_seconds: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()

    def fetch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        response = self.session.get(
            self.base_url,
            params=dict(params),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Open-Meteo response must be a JSON object")
        if payload.get("error") is True:
            reason = str(payload.get("reason", "unknown Open-Meteo error"))
            raise ValueError(f"Open-Meteo error: {reason}")
        return payload


def fetch_live_weather(
    location: str,
    target_date: date,
    prediction_time: datetime,
    config: TradingConfig,
    client: WeatherProviderClient | None = None,
    fetched_at: datetime | None = None,
) -> LiveWeatherSnapshot:
    settings = config.weather
    provider_client = client or OpenMeteoClient(
        base_url=settings.forecast_base_url,
        timeout_seconds=config.kalshi.request_timeout_seconds,
    )
    prediction_local = _local_naive_datetime(prediction_time, settings.timezone)
    fetched_at_local = _local_naive_datetime(fetched_at or datetime.now(ZoneInfo(settings.timezone)), settings.timezone)

    observation_payload = provider_client.fetch(
        _weather_params(
            latitude=settings.observation_grid.latitude,
            longitude=settings.observation_grid.longitude,
            timezone=settings.timezone,
            temperature_unit=settings.temperature_unit,
            wind_speed_unit=settings.wind_speed_unit,
            precipitation_unit=settings.precipitation_unit,
            hourly=OBSERVED_HOURLY_VARIABLES,
            current=OBSERVED_HOURLY_VARIABLES,
            daily=[],
            past_hours=settings.observed_past_hours,
            forecast_hours=1,
            forecast_days=1,
        )
    )
    forecast_payload = provider_client.fetch(
        _weather_params(
            latitude=settings.forecast_grid.latitude,
            longitude=settings.forecast_grid.longitude,
            timezone=settings.timezone,
            temperature_unit=settings.temperature_unit,
            wind_speed_unit=settings.wind_speed_unit,
            precipitation_unit=settings.precipitation_unit,
            hourly=FORECAST_HOURLY_VARIABLES,
            current=[],
            daily=DAILY_FORECAST_VARIABLES,
            past_hours=settings.forecast_past_hours,
            forecast_hours=24,
            forecast_days=settings.forecast_days,
        )
    )

    hourly_observations = _hourly_frame_from_payload(
        observation_payload,
        location=location,
        source_role="hourly_observations",
        source_name=LIVE_OBSERVATION_SOURCE,
        include_current=True,
    )
    hourly_forecasts = _hourly_frame_from_payload(
        forecast_payload,
        location=location,
        source_role="hourly_forecasts",
        source_name=LIVE_FORECAST_SOURCE,
        include_current=False,
    )
    daily_forecast = _daily_forecast_frame_from_payload(
        forecast_payload,
        location=location,
        source_name=LIVE_FORECAST_SOURCE,
    )

    diagnostics = _build_weather_diagnostics(
        observation_payload=observation_payload,
        forecast_payload=forecast_payload,
        hourly_observations=hourly_observations,
        hourly_forecasts=hourly_forecasts,
        daily_forecast=daily_forecast,
        target_date=target_date,
        prediction_time=prediction_local,
        fetched_at=fetched_at_local,
        config=config,
    )

    return LiveWeatherSnapshot(
        location=location,
        target_date=target_date,
        prediction_time=prediction_local,
        fetched_at=fetched_at_local,
        hourly_observations=hourly_observations,
        hourly_forecasts=hourly_forecasts,
        daily_forecast=daily_forecast,
        diagnostics=diagnostics,
    )


def save_live_weather_snapshot(snapshot: LiveWeatherSnapshot, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        snapshot.hourly_observations,
        snapshot.hourly_forecasts,
        snapshot.daily_forecast,
        snapshot.diagnostics.assign(source_role="weather_diagnostics"),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    combined.insert(0, "snapshot_location", snapshot.location)
    combined.insert(1, "snapshot_target_date", snapshot.target_date.isoformat())
    combined.insert(2, "snapshot_prediction_time", snapshot.prediction_time.isoformat())
    combined.insert(3, "snapshot_fetched_at", snapshot.fetched_at.isoformat())
    combined.to_csv(path, index=False)


def save_live_weather_diagnostics(snapshot: LiveWeatherSnapshot, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.diagnostics.to_csv(path, index=False)


def _weather_params(
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    temperature_unit: str,
    wind_speed_unit: str,
    precipitation_unit: str,
    hourly: list[str],
    current: list[str],
    daily: list[str],
    past_hours: int,
    forecast_hours: int,
    forecast_days: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
        "precipitation_unit": precipitation_unit,
        "past_hours": int(past_hours),
        "forecast_hours": int(forecast_hours),
        "forecast_days": int(forecast_days),
    }
    if hourly:
        params["hourly"] = ",".join(hourly)
    if current:
        params["current"] = ",".join(current)
    if daily:
        params["daily"] = ",".join(daily)
    return params


def _hourly_frame_from_payload(
    payload: Mapping[str, Any],
    *,
    location: str,
    source_role: str,
    source_name: str,
    include_current: bool,
) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    rows = _time_series_records(hourly)

    if include_current and isinstance(payload.get("current"), dict):
        current = dict(payload["current"])
        current_time = current.get("time")
        if current_time:
            row = {"timestamp": current_time}
            for key, value in current.items():
                if key not in {"time", "interval"}:
                    row[key] = value
            rows.append(row)

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return _empty_weather_frame(source_role=source_role)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["target_date"] = frame["date"]
    frame["location"] = location
    frame["source_role"] = source_role
    frame["forecast_source"] = source_name
    issue_time = _payload_issue_time(payload)
    if issue_time is not None:
        frame["forecast_issue_time"] = issue_time
    frame = _coerce_numeric_weather_columns(frame)
    frame = frame.drop_duplicates(subset=["timestamp", "location"], keep="last")
    return frame.sort_values(["location", "timestamp"]).reset_index(drop=True)


def _daily_forecast_frame_from_payload(
    payload: Mapping[str, Any],
    *,
    location: str,
    source_name: str,
) -> pd.DataFrame:
    daily = payload.get("daily", {})
    rows = _time_series_records(daily, time_column_name="date")
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return _empty_weather_frame(source_role="daily_forecast")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["location"] = location
    frame["source_role"] = "daily_forecast"
    frame["forecast_source"] = source_name
    if "temperature_2m_max" in frame.columns:
        frame = frame.rename(columns={"temperature_2m_max": "forecast_high"})
    issue_time = _payload_issue_time(payload)
    if issue_time is not None:
        frame["forecast_issue_time"] = issue_time
    frame = _coerce_numeric_weather_columns(frame)
    frame = frame.drop_duplicates(subset=["date", "location"], keep="last")
    return frame.sort_values(["location", "date"]).reset_index(drop=True)


def _time_series_records(
    payload_section: Any,
    *,
    time_column_name: str = "timestamp",
) -> list[dict[str, Any]]:
    if not isinstance(payload_section, Mapping):
        return []
    times = payload_section.get("time")
    if not isinstance(times, list):
        return []
    records: list[dict[str, Any]] = []
    value_columns = [
        key
        for key, value in payload_section.items()
        if key != "time" and isinstance(value, list) and len(value) == len(times)
    ]
    for index, time_value in enumerate(times):
        record = {time_column_name: time_value}
        for column in value_columns:
            record[column] = payload_section[column][index]
        records.append(record)
    return records


def _build_weather_diagnostics(
    *,
    observation_payload: Mapping[str, Any],
    forecast_payload: Mapping[str, Any],
    hourly_observations: pd.DataFrame,
    hourly_forecasts: pd.DataFrame,
    daily_forecast: pd.DataFrame,
    target_date: date,
    prediction_time: datetime,
    fetched_at: datetime,
    config: TradingConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    records.extend(_unit_diagnostics(observation_payload, "hourly_observations", config))
    records.extend(_unit_diagnostics(forecast_payload, "hourly_forecasts", config))

    records.append(
        _freshness_record(
            name="latest_observation_age",
            source_role="hourly_observations",
            frame=hourly_observations,
            time_col="timestamp",
            prediction_time=prediction_time,
            max_age_minutes=config.weather.max_observation_age_minutes,
        )
    )
    records.append(
        _freshness_record(
            name="latest_hourly_forecast_valid_time",
            source_role="hourly_forecasts",
            frame=hourly_forecasts,
            time_col="timestamp",
            prediction_time=prediction_time,
            max_age_minutes=config.weather.max_forecast_age_minutes,
        )
    )

    target_ts = pd.Timestamp(target_date)
    if daily_forecast.empty or "forecast_high" not in daily_forecast.columns:
        records.append(
            _diagnostic_record(
                "daily_forecast_high_present",
                "daily_forecast",
                "NO_TRADE",
                "missing_forecast_high",
                fetched_at=fetched_at,
            )
        )
    else:
        target_rows = daily_forecast[daily_forecast["date"] == target_ts]
        status = "OK" if not target_rows.empty and target_rows["forecast_high"].notna().any() else "NO_TRADE"
        records.append(
            _diagnostic_record(
                "daily_forecast_high_present",
                "daily_forecast",
                status,
                "" if status == "OK" else "missing_target_date_forecast_high",
                fetched_at=fetched_at,
                detail=f"target_date={target_date.isoformat()}",
            )
        )

    issue_time = _payload_issue_time(forecast_payload)
    if issue_time is None:
        records.append(
            _diagnostic_record(
                "forecast_issue_time_present",
                "hourly_forecasts",
                "NO_TRADE" if config.weather.require_forecast_issue_time else "WARN",
                "missing_forecast_issue_time",
                fetched_at=fetched_at,
                detail="Open-Meteo forecast payload did not expose a model issue/run timestamp.",
            )
        )
    else:
        records.append(
            _diagnostic_record(
                "forecast_issue_time_present",
                "hourly_forecasts",
                "OK",
                "",
                fetched_at=fetched_at,
                source_time=issue_time,
            )
        )

    records.append(
        _diagnostic_record(
            "live_trading_eligibility",
            "all",
            "SHADOW_ONLY",
            "day2_read_only_and_forecast_provenance_warning",
            fetched_at=fetched_at,
            detail="Day 2 creates scoreable rows only; no order books, edge, or order intents.",
        )
    )
    return pd.DataFrame.from_records(records)


def _freshness_record(
    *,
    name: str,
    source_role: str,
    frame: pd.DataFrame,
    time_col: str,
    prediction_time: datetime,
    max_age_minutes: int,
) -> dict[str, Any]:
    if frame.empty or time_col not in frame.columns:
        return _diagnostic_record(
            name,
            source_role,
            "NO_TRADE",
            "missing_source_times",
            source_time=None,
            age_minutes=None,
        )
    times = pd.to_datetime(frame[time_col], errors="coerce")
    usable = times[times <= pd.Timestamp(prediction_time)]
    if usable.empty:
        return _diagnostic_record(
            name,
            source_role,
            "NO_TRADE",
            "no_source_time_at_or_before_prediction_time",
            source_time=None,
            age_minutes=None,
        )
    latest = usable.max().to_pydatetime()
    age_minutes = (prediction_time - latest).total_seconds() / 60.0
    status = "OK" if age_minutes <= max_age_minutes else "NO_TRADE"
    return _diagnostic_record(
        name,
        source_role,
        status,
        "" if status == "OK" else "stale_weather_data",
        source_time=latest,
        age_minutes=age_minutes,
    )


def _unit_diagnostics(
    payload: Mapping[str, Any],
    source_role: str,
    config: TradingConfig,
) -> list[dict[str, Any]]:
    units = {}
    for key in ["current_units", "hourly_units", "daily_units"]:
        section = payload.get(key)
        if isinstance(section, Mapping):
            units.update(section)
    checks = [
        ("temperature_2m", "F", config.weather.temperature_unit),
        ("wind_speed_10m", "mp/h", config.weather.wind_speed_unit),
        ("precipitation", "inch", config.weather.precipitation_unit),
    ]
    records = []
    for variable, expected_fragment, configured_unit in checks:
        actual_unit = str(units.get(variable, ""))
        if not actual_unit:
            status = "WARN"
            reason = "unit_missing"
        elif expected_fragment.lower() in actual_unit.lower() or configured_unit.lower() in actual_unit.lower():
            status = "OK"
            reason = ""
        else:
            status = "NO_TRADE"
            reason = "unit_mismatch"
        records.append(
            _diagnostic_record(
                f"{variable}_unit",
                source_role,
                status,
                reason,
                detail=f"actual_unit={actual_unit or 'missing'}",
            )
        )
    return records


def _diagnostic_record(
    diagnostic_name: str,
    source_role: str,
    status: str,
    no_trade_reason: str,
    *,
    fetched_at: datetime | None = None,
    source_time: datetime | pd.Timestamp | None = None,
    age_minutes: float | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "diagnostic_name": diagnostic_name,
        "source_role": source_role,
        "status": status,
        "no_trade_reason": no_trade_reason,
        "source_time": "" if source_time is None else pd.Timestamp(source_time).isoformat(),
        "fetched_at": "" if fetched_at is None else pd.Timestamp(fetched_at).isoformat(),
        "age_minutes": age_minutes,
        "detail": detail,
    }


def _payload_issue_time(payload: Mapping[str, Any]) -> pd.Timestamp | None:
    for key in ISSUE_TIME_KEYS:
        value = payload.get(key)
        if value:
            return pd.Timestamp(value).tz_localize(None)
    return None


def _coerce_numeric_weather_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column in {
            "timestamp",
            "date",
            "target_date",
            "location",
            "source_role",
            "forecast_source",
            "forecast_issue_time",
            "sunrise",
            "sunset",
        }:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _empty_weather_frame(*, source_role: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["source_role"]).assign(source_role=source_role).iloc[0:0]


def _local_naive_datetime(value: datetime, timezone_name: str) -> datetime:
    timezone = ZoneInfo(timezone_name)
    if value.tzinfo is None:
        return value.replace(tzinfo=None)
    return value.astimezone(timezone).replace(tzinfo=None)
