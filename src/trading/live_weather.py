from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
import re
from typing import Any, Mapping, Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.trading.config import TradingConfig


LIVE_FORECAST_SOURCE = "open_meteo_live_forecast"
NWS_FORECAST_SOURCE = "nws_gridpoint_forecast"
LIVE_OBSERVATION_SOURCE = "open_meteo_live_current"
LIVE_OBSERVATION_FALLBACK_SOURCE = "open_meteo_observation_fallback"

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


class NwsForecastProviderClient(Protocol):
    def fetch_point_metadata(self, latitude: float, longitude: float) -> dict[str, Any]:
        ...

    def fetch_forecast_url(self, url: str) -> dict[str, Any]:
        ...


class NwsObservationClient(Protocol):
    def fetch_observations(
        self,
        station_id: str,
        *,
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> dict[str, Any]:
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


class NwsStationClient:
    def __init__(
        self,
        base_url: str = "https://api.weather.gov",
        user_agent: str = "KalshiWeatherTrading/0.1 (local research; contact user)",
        timeout_seconds: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()

    def fetch_observations(
        self,
        station_id: str,
        *,
        start: datetime,
        end: datetime,
        limit: int = 500,
    ) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/stations/{station_id}/observations",
            params={
                "start": pd.Timestamp(start).isoformat(),
                "end": pd.Timestamp(end).isoformat(),
                "limit": int(limit),
            },
            headers={
                "Accept": "application/geo+json",
                "User-Agent": self.user_agent,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("NWS observations response must be a JSON object")
        return payload


class NwsForecastClient:
    def __init__(
        self,
        base_url: str = "https://api.weather.gov",
        user_agent: str = "KalshiWeatherTrading/0.1 (local research; contact user)",
        timeout_seconds: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()

    def fetch_point_metadata(self, latitude: float, longitude: float) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/points/{float(latitude):.5f},{float(longitude):.5f}",
            headers={
                "Accept": "application/geo+json",
                "User-Agent": self.user_agent,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("NWS points response must be a JSON object")
        return payload

    def fetch_forecast_url(self, url: str) -> dict[str, Any]:
        response = self.session.get(
            url,
            headers={
                "Accept": "application/geo+json",
                "User-Agent": self.user_agent,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("NWS forecast response must be a JSON object")
        return payload


def fetch_live_weather(
    location: str,
    target_date: date,
    prediction_time: datetime,
    config: TradingConfig,
    client: WeatherProviderClient | None = None,
    observation_client: NwsObservationClient | None = None,
    fetched_at: datetime | None = None,
) -> LiveWeatherSnapshot:
    settings = config.weather
    open_meteo_client = (
        client
        if settings.provider == "open_meteo" and client is not None
        else OpenMeteoClient(
            base_url=settings.forecast_base_url,
            timeout_seconds=config.kalshi.request_timeout_seconds,
        )
    )
    nws_forecast_client = (
        client
        if settings.provider == "nws" and client is not None
        else NwsForecastClient(
            base_url=settings.nws_station.base_url,
            user_agent=settings.nws_station.user_agent,
            timeout_seconds=config.kalshi.request_timeout_seconds,
        )
    )
    provider_client = open_meteo_client
    if settings.provider == "open_meteo":
        provider_client = open_meteo_client
    elif settings.provider == "nws":
        provider_client = nws_forecast_client
    else:
        raise ValueError(f"Unsupported live weather provider: {settings.provider}")
    open_meteo_fallback_client = (
        open_meteo_client
        if settings.provider == "open_meteo"
        else OpenMeteoClient(
            base_url=settings.forecast_base_url,
            timeout_seconds=config.kalshi.request_timeout_seconds,
        )
    )
    prediction_local = _local_naive_datetime(prediction_time, settings.timezone)
    fetched_at_local = _local_naive_datetime(fetched_at or datetime.now(ZoneInfo(settings.timezone)), settings.timezone)

    if settings.observations_provider == "nws_station":
        station_client = observation_client or NwsStationClient(
            base_url=settings.nws_station.base_url,
            user_agent=settings.nws_station.user_agent,
            timeout_seconds=config.kalshi.request_timeout_seconds,
        )
        observation_start, observation_end = _nws_observation_window(
            target_date=target_date,
            prediction_time=prediction_local,
            timezone_name=settings.timezone,
        )
        observation_payload = station_client.fetch_observations(
            settings.nws_station.station_id,
            start=observation_start,
            end=observation_end,
            limit=max(48, settings.observed_past_hours + 24),
        )
        observation_payload = dict(observation_payload)
        observation_payload["hourly_units"] = {
            "time": "iso8601",
            "temperature_2m": "F",
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        }
    else:
        observation_payload = open_meteo_client.fetch(
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
    if settings.provider == "nws":
        forecast_payload = _fetch_nws_forecast_payload(
            provider_client,
            latitude=settings.forecast_grid.latitude,
            longitude=settings.forecast_grid.longitude,
            timezone_name=settings.timezone,
        )
    else:
        forecast_payload = open_meteo_client.fetch(
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

    observation_fallback_reason = ""
    if settings.observations_provider == "nws_station":
        hourly_observations = _hourly_frame_from_nws_payload(
            observation_payload,
            location=location,
            timezone_name=settings.timezone,
            station_id=settings.nws_station.station_id,
            station_name=settings.nws_station.station_name,
        )
        observation_fallback_reason = _nws_observation_fallback_reason(
            hourly_observations,
            prediction_time=prediction_local,
            max_age_minutes=settings.max_observation_age_minutes,
        )
        if observation_fallback_reason:
            fallback_payload = open_meteo_fallback_client.fetch(
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
            fallback_observations = _hourly_frame_from_payload(
                fallback_payload,
                location=location,
                source_role="hourly_observations",
                source_name=LIVE_OBSERVATION_FALLBACK_SOURCE,
                include_current=True,
            )
            if not fallback_observations.empty:
                fallback_observations["provider_station_id"] = settings.nws_station.station_id
                fallback_observations["provider_station_name"] = (
                    f"Open-Meteo proxy fallback for {settings.nws_station.station_name}"
                )
                fallback_observations["provider_units"] = (
                    "temperature=F;wind_speed=mph;precipitation=inch"
                )
                fallback_observations["nws_fallback_reason"] = observation_fallback_reason
                hourly_observations = fallback_observations
                observation_payload = fallback_payload
    else:
        hourly_observations = _hourly_frame_from_payload(
            observation_payload,
            location=location,
            source_role="hourly_observations",
            source_name=LIVE_OBSERVATION_SOURCE,
            include_current=True,
        )
    if settings.provider == "nws":
        hourly_forecasts = _hourly_frame_from_nws_forecast_payload(
            forecast_payload,
            location=location,
            timezone_name=settings.timezone,
        )
        daily_forecast = _daily_forecast_frame_from_nws_payload(
            forecast_payload,
            hourly_forecasts=hourly_forecasts,
            location=location,
            timezone_name=settings.timezone,
        )
    else:
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
        observation_fallback_reason=observation_fallback_reason,
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
        _diagnostics_for_combined_frame(snapshot.diagnostics),
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


def diagnostics_for_combined_frame(diagnostics: pd.DataFrame) -> pd.DataFrame:
    return _diagnostics_for_combined_frame(diagnostics)


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


def _fetch_nws_forecast_payload(
    client: NwsForecastProviderClient,
    *,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> dict[str, Any]:
    point_payload = client.fetch_point_metadata(latitude, longitude)
    properties = point_payload.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("NWS points response is missing properties")
    hourly_url = properties.get("forecastHourly")
    daily_url = properties.get("forecast")
    if not hourly_url:
        raise ValueError("NWS points response is missing forecastHourly URL")
    hourly_payload = client.fetch_forecast_url(str(hourly_url))
    daily_payload = client.fetch_forecast_url(str(daily_url)) if daily_url else {}
    issue_time = _nws_forecast_issue_time(hourly_payload, timezone_name) or _nws_forecast_issue_time(
        daily_payload,
        timezone_name,
    )
    return {
        "provider": NWS_FORECAST_SOURCE,
        "point": point_payload,
        "nws_hourly": hourly_payload,
        "nws_daily": daily_payload,
        "forecast_issue_time": "" if issue_time is None else issue_time.isoformat(),
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "F",
            "wind_speed_10m": "mph",
            "precipitation": "inch",
        },
        "daily_units": {
            "time": "iso8601",
            "temperature_2m": "F",
            "wind_speed_10m": "mph",
            "precipitation": "inch",
        },
    }


def _hourly_frame_from_nws_forecast_payload(
    payload: Mapping[str, Any],
    *,
    location: str,
    timezone_name: str,
) -> pd.DataFrame:
    hourly_payload = payload.get("nws_hourly", payload)
    periods = _nws_periods(hourly_payload)
    if not periods:
        return _empty_weather_frame(source_role="hourly_forecasts")
    issue_time = _payload_issue_time(payload) or _nws_forecast_issue_time(
        hourly_payload,
        timezone_name,
    )

    rows: list[dict[str, Any]] = []
    for period in periods:
        if not isinstance(period, Mapping):
            continue
        timestamp = _nws_local_time(period.get("startTime"), timezone_name)
        if timestamp is None:
            continue
        short_forecast = str(period.get("shortForecast", "") or "")
        detailed_forecast = str(period.get("detailedForecast", "") or "")
        row = {
            "timestamp": timestamp,
            "temperature_2m": _nws_temperature_value(
                period.get("temperature"),
                period.get("temperatureUnit"),
            ),
            "relative_humidity_2m": _quantitative_value(period.get("relativeHumidity")),
            "dew_point_2m": _unit_quantity(
                period.get("dewpoint"),
                target_unit="fahrenheit",
            ),
            "precipitation": 0.0,
            "rain": 0.0,
            "precipitation_probability": _quantitative_value(
                period.get("probabilityOfPrecipitation")
            ),
            "weather_code": _weather_code_from_text(
                f"{short_forecast} {detailed_forecast}"
            ),
            "cloud_cover": _cloud_cover_from_forecast_text(short_forecast),
            "surface_pressure": None,
            "wind_speed_10m": _wind_speed_from_text(period.get("windSpeed")),
            "wind_direction_10m": _wind_direction_degrees(period.get("windDirection")),
            "wind_gusts_10m": _wind_gust_from_text(detailed_forecast),
            "is_day": 1.0 if bool(period.get("isDaytime")) else 0.0,
            "nws_forecast_name": period.get("name", ""),
            "nws_short_forecast": short_forecast,
            "nws_detailed_forecast": detailed_forecast,
        }
        rows.append(row)

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return _empty_weather_frame(source_role="hourly_forecasts")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["target_date"] = frame["date"]
    frame["location"] = location
    frame["source_role"] = "hourly_forecasts"
    frame["forecast_source"] = NWS_FORECAST_SOURCE
    if issue_time is not None:
        frame["forecast_issue_time"] = issue_time
    frame["provider_units"] = (
        "temperature=F;wind_speed=mph;precipitation_probability=percent"
    )
    frame = _coerce_numeric_weather_columns(frame)
    frame = frame.drop_duplicates(subset=["timestamp", "location"], keep="last")
    return frame.sort_values(["location", "timestamp"]).reset_index(drop=True)


def _daily_forecast_frame_from_nws_payload(
    payload: Mapping[str, Any],
    *,
    hourly_forecasts: pd.DataFrame,
    location: str,
    timezone_name: str,
) -> pd.DataFrame:
    daily_periods = _daily_period_frame_from_nws_payload(
        payload,
        location=location,
        timezone_name=timezone_name,
    )
    hourly_daily = _daily_frame_from_hourly_nws_forecasts(hourly_forecasts, location=location)
    if hourly_daily.empty:
        result = daily_periods
    elif daily_periods.empty:
        result = hourly_daily
    else:
        result = hourly_daily.merge(
            daily_periods,
            on=["date", "location", "source_role", "forecast_source"],
            how="outer",
            suffixes=("", "_period"),
        )
        for column in ["forecast_high", "temperature_2m_min", "weather_code"]:
            period_column = f"{column}_period"
            if period_column in result.columns:
                if column in result.columns:
                    result[column] = result[column].combine_first(result[period_column])
                else:
                    result[column] = result[period_column]
                result = result.drop(columns=[period_column])
        if "forecast_issue_time_period" in result.columns:
            result["forecast_issue_time"] = result["forecast_issue_time"].combine_first(
                result["forecast_issue_time_period"]
            )
            result = result.drop(columns=["forecast_issue_time_period"])
    if result.empty:
        return _empty_weather_frame(source_role="daily_forecast")
    return result.sort_values(["location", "date"]).reset_index(drop=True)


def _daily_frame_from_hourly_nws_forecasts(
    hourly_forecasts: pd.DataFrame,
    *,
    location: str,
) -> pd.DataFrame:
    if hourly_forecasts.empty or not {"date", "temperature_2m"}.issubset(hourly_forecasts.columns):
        return pd.DataFrame()
    frame = hourly_forecasts.copy()
    frame["temperature_2m"] = pd.to_numeric(frame["temperature_2m"], errors="coerce")
    if "precipitation_probability" in frame.columns:
        frame["precipitation_probability"] = pd.to_numeric(
            frame["precipitation_probability"],
            errors="coerce",
        )
    if "wind_speed_10m" in frame.columns:
        frame["wind_speed_10m"] = pd.to_numeric(frame["wind_speed_10m"], errors="coerce")
    records = []
    for day, group in frame.dropna(subset=["date"]).groupby("date", sort=False):
        records.append(
            {
                "date": pd.Timestamp(day).normalize(),
                "location": location,
                "source_role": "daily_forecast",
                "forecast_source": NWS_FORECAST_SOURCE,
                "forecast_issue_time": _first_valid(group, "forecast_issue_time"),
                "forecast_high": _max_numeric(group, "temperature_2m"),
                "temperature_2m_min": _min_numeric(group, "temperature_2m"),
                "precipitation_probability_max": _max_numeric(
                    group,
                    "precipitation_probability",
                ),
                "wind_speed_10m_max": _max_numeric(group, "wind_speed_10m"),
                "weather_code": _first_valid(group, "weather_code"),
            }
        )
    return pd.DataFrame.from_records(records)


def _daily_period_frame_from_nws_payload(
    payload: Mapping[str, Any],
    *,
    location: str,
    timezone_name: str,
) -> pd.DataFrame:
    daily_payload = payload.get("nws_daily", {})
    periods = _nws_periods(daily_payload)
    if not periods:
        return pd.DataFrame()
    issue_time = _payload_issue_time(payload) or _nws_forecast_issue_time(
        daily_payload,
        timezone_name,
    )
    records_by_date: dict[pd.Timestamp, dict[str, Any]] = {}
    for period in periods:
        if not isinstance(period, Mapping):
            continue
        start = _nws_local_time(period.get("startTime"), timezone_name)
        if start is None:
            continue
        day = pd.Timestamp(start).normalize()
        record = records_by_date.setdefault(
            day,
            {
                "date": day,
                "location": location,
                "source_role": "daily_forecast",
                "forecast_source": NWS_FORECAST_SOURCE,
                "forecast_issue_time": issue_time,
            },
        )
        temp = _nws_temperature_value(period.get("temperature"), period.get("temperatureUnit"))
        if bool(period.get("isDaytime")):
            record["forecast_high"] = _max_optional(record.get("forecast_high"), temp)
        else:
            record["temperature_2m_min"] = _min_optional(record.get("temperature_2m_min"), temp)
        record["precipitation_probability_max"] = _max_optional(
            record.get("precipitation_probability_max"),
            _quantitative_value(period.get("probabilityOfPrecipitation")),
        )
        record["wind_speed_10m_max"] = _max_optional(
            record.get("wind_speed_10m_max"),
            _wind_speed_from_text(period.get("windSpeed")),
        )
        record["weather_code"] = _weather_code_from_text(
            f"{period.get('shortForecast', '')} {period.get('detailedForecast', '')}"
        )
    return pd.DataFrame.from_records(list(records_by_date.values()))


def _hourly_frame_from_nws_payload(
    payload: Mapping[str, Any],
    *,
    location: str,
    timezone_name: str,
    station_id: str,
    station_name: str,
) -> pd.DataFrame:
    features = payload.get("features", [])
    if not isinstance(features, list):
        return _empty_weather_frame(source_role="hourly_observations")

    rows: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        properties = feature.get("properties", {})
        if not isinstance(properties, Mapping):
            continue
        timestamp = properties.get("timestamp")
        if not timestamp:
            continue
        row = {
            "timestamp": timestamp,
            "temperature_2m": _unit_value(properties, "temperature", "fahrenheit"),
            "relative_humidity_2m": _unit_value(properties, "relativeHumidity", "percent"),
            "dew_point_2m": _unit_value(properties, "dewpoint", "fahrenheit"),
            "precipitation": _unit_value(properties, "precipitationLastHour", "inch"),
            "rain": _unit_value(properties, "precipitationLastHour", "inch"),
            "weather_code": _weather_code_from_nws(properties),
            "surface_pressure": _unit_value(properties, "barometricPressure", "hpa"),
            "cloud_cover": _cloud_cover_from_nws(properties),
            "wind_speed_10m": _unit_value(properties, "windSpeed", "mph"),
            "wind_direction_10m": _unit_value(properties, "windDirection", "degree"),
            "wind_gusts_10m": _unit_value(properties, "windGust", "mph"),
            "nws_station_id": station_id,
            "nws_station_name": station_name,
            "nws_observation_raw": properties.get("rawMessage", ""),
            "nws_text_description": properties.get("textDescription", ""),
        }
        six_hour_max, daily_max = _nws_remark_max_temperatures(row["nws_observation_raw"])
        row["nws_6h_max_temp"] = six_hour_max
        row["nws_24h_max_temp"] = daily_max
        rows.append(row)

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return _empty_weather_frame(source_role="hourly_observations")

    timezone = ZoneInfo(timezone_name)
    frame["timestamp"] = (
        pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)
    )
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["target_date"] = frame["date"]
    frame["location"] = location
    frame["source_role"] = "hourly_observations"
    frame["forecast_source"] = "nws_station_observations"
    frame["provider_station_id"] = station_id
    frame["provider_station_name"] = station_name
    frame["provider_units"] = "temperature=F;wind_speed=mph;precipitation=inch;pressure=hPa"
    frame = _coerce_numeric_weather_columns(frame)
    frame = frame.drop_duplicates(subset=["timestamp", "location"], keep="last")
    frame = frame.sort_values(["location", "timestamp"]).reset_index(drop=True)
    return _attach_observed_high_so_far(frame)


def _nws_observation_fallback_reason(
    frame: pd.DataFrame,
    *,
    prediction_time: datetime,
    max_age_minutes: int,
) -> str:
    if frame.empty or "timestamp" not in frame.columns:
        return "missing_nws_station_observations"
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    usable = timestamps[timestamps <= pd.Timestamp(prediction_time)]
    if usable.empty:
        return "no_nws_station_observation_at_or_before_prediction_time"
    latest = usable.max().to_pydatetime()
    age_minutes = (prediction_time - latest).total_seconds() / 60.0
    if age_minutes > max_age_minutes:
        return "stale_nws_station_observations"
    return ""


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


def _nws_observation_window(
    *,
    target_date: date,
    prediction_time: datetime,
    timezone_name: str,
) -> tuple[datetime, datetime]:
    timezone = ZoneInfo(timezone_name)
    start_local = datetime.combine(target_date, time.min).replace(tzinfo=timezone)
    if prediction_time.tzinfo is None:
        end_local = prediction_time.replace(tzinfo=timezone)
    else:
        end_local = prediction_time.astimezone(timezone)
    if end_local < start_local:
        end_local = start_local
    return start_local.astimezone(ZoneInfo("UTC")), end_local.astimezone(ZoneInfo("UTC"))


def _unit_value(properties: Mapping[str, Any], field: str, target_unit: str) -> float | None:
    value = properties.get(field)
    if not isinstance(value, Mapping):
        return None
    raw = value.get("value")
    if raw is None:
        return None
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None
    unit_code = str(value.get("unitCode", ""))
    if target_unit == "fahrenheit":
        if "degC" in unit_code:
            return numeric * 9.0 / 5.0 + 32.0
        return numeric
    if target_unit == "mph":
        if "km_h-1" in unit_code:
            return numeric * 0.621371
        if "m_s-1" in unit_code:
            return numeric * 2.236936
        return numeric
    if target_unit == "inch":
        if unit_code.endswith(":mm") or "wmoUnit:mm" in unit_code:
            return numeric / 25.4
        return numeric
    if target_unit == "hpa":
        if unit_code.endswith(":Pa") or "wmoUnit:Pa" in unit_code:
            return numeric / 100.0
        return numeric
    if target_unit == "degree":
        return numeric
    if target_unit == "percent":
        return numeric
    return numeric


def _cloud_cover_from_nws(properties: Mapping[str, Any]) -> float | None:
    layers = properties.get("cloudLayers")
    if not isinstance(layers, list) or not layers:
        return None
    amounts = {
        "CLR": 0.0,
        "SKC": 0.0,
        "FEW": 20.0,
        "SCT": 40.0,
        "BKN": 75.0,
        "OVC": 100.0,
        "VV": 100.0,
    }
    values = []
    for layer in layers:
        if isinstance(layer, Mapping):
            amount = str(layer.get("amount", "")).upper()
            if amount in amounts:
                values.append(amounts[amount])
    if not values:
        return None
    return max(values)


def _weather_code_from_nws(properties: Mapping[str, Any]) -> float | None:
    text = str(properties.get("textDescription", "") or "").lower()
    if not text:
        return None
    if "thunder" in text:
        return 95.0
    if "snow" in text:
        return 71.0
    if "rain" in text or "shower" in text or "drizzle" in text:
        return 61.0
    if "fog" in text or "mist" in text or "haze" in text:
        return 45.0
    if "overcast" in text:
        return 3.0
    if "cloud" in text:
        return 2.0
    if "clear" in text or "fair" in text:
        return 0.0
    return None


def _nws_remark_max_temperatures(raw_message: Any) -> tuple[float | None, float | None]:
    text = str(raw_message or "")
    six_hour_max: float | None = None
    daily_max: float | None = None
    for token in text.split():
        if len(token) == 5 and token.startswith("1") and token[1] in {"0", "1"} and token[2:].isdigit():
            six_hour_max = _signed_tenths_celsius_to_fahrenheit(token[1], token[2:])
        elif len(token) == 9 and token.startswith("4") and token[1] in {"0", "1"} and token[5] in {"0", "1"}:
            if token[2:5].isdigit() and token[6:9].isdigit():
                daily_max = _signed_tenths_celsius_to_fahrenheit(token[1], token[2:5])
    return six_hour_max, daily_max


def _nws_periods(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    properties = payload.get("properties", {})
    if not isinstance(properties, Mapping):
        return []
    periods = properties.get("periods", [])
    return periods if isinstance(periods, list) else []


def _nws_forecast_issue_time(
    payload: Mapping[str, Any],
    timezone_name: str,
) -> pd.Timestamp | None:
    properties = payload.get("properties", {})
    if not isinstance(properties, Mapping):
        return None
    value = properties.get("updateTime") or properties.get("generatedAt")
    if not value:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    timezone = ZoneInfo(timezone_name)
    if timestamp.tzinfo is None:
        return pd.Timestamp(timestamp)
    return pd.Timestamp(timestamp.tz_convert(timezone).tz_localize(None))


def _nws_local_time(value: Any, timezone_name: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    timezone = ZoneInfo(timezone_name)
    if timestamp.tzinfo is None:
        return pd.Timestamp(timestamp)
    return pd.Timestamp(timestamp.tz_convert(timezone).tz_localize(None))


def _nws_temperature_value(value: Any, unit: Any) -> float | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    unit_text = str(unit or "").upper()
    if unit_text == "C":
        return numeric * 9.0 / 5.0 + 32.0
    return numeric


def _quantitative_value(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    return _optional_float(value.get("value"))


def _unit_quantity(value: Any, *, target_unit: str) -> float | None:
    if not isinstance(value, Mapping):
        return None
    numeric = _optional_float(value.get("value"))
    if numeric is None:
        return None
    unit_code = str(value.get("unitCode", ""))
    if target_unit == "fahrenheit" and "degC" in unit_code:
        return numeric * 9.0 / 5.0 + 32.0
    if target_unit == "mph":
        if "km_h-1" in unit_code:
            return numeric * 0.621371
        if "m_s-1" in unit_code:
            return numeric * 2.236936
    return numeric


def _wind_speed_from_text(value: Any) -> float | None:
    text = str(value or "")
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    speed = max(numbers)
    lower = text.lower()
    if "km/h" in lower or "km h" in lower:
        speed *= 0.621371
    elif "m/s" in lower:
        speed *= 2.236936
    return speed


def _wind_gust_from_text(value: Any) -> float | None:
    text = str(value or "")
    match = re.search(r"gusts? (?:as high as |up to )?(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _wind_direction_degrees(value: Any) -> float | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text.endswith("DEG"):
        text = text[:-3].strip()
    numeric = _optional_float(text)
    if numeric is not None:
        return numeric
    lookup = {
        "N": 0.0,
        "NNE": 22.5,
        "NE": 45.0,
        "ENE": 67.5,
        "E": 90.0,
        "ESE": 112.5,
        "SE": 135.0,
        "SSE": 157.5,
        "S": 180.0,
        "SSW": 202.5,
        "SW": 225.0,
        "WSW": 247.5,
        "W": 270.0,
        "WNW": 292.5,
        "NW": 315.0,
        "NNW": 337.5,
        "VRB": None,
        "VARIABLE": None,
    }
    return lookup.get(text)


def _weather_code_from_text(value: Any) -> float | None:
    text = str(value or "").lower()
    if not text:
        return None
    if "thunder" in text:
        return 95.0
    if "snow" in text:
        return 71.0
    if "sleet" in text or "freezing rain" in text:
        return 66.0
    if "rain" in text or "shower" in text or "drizzle" in text:
        return 61.0
    if "fog" in text or "mist" in text or "haze" in text:
        return 45.0
    if "overcast" in text:
        return 3.0
    if "cloud" in text:
        return 2.0
    if "sun" in text or "clear" in text or "fair" in text:
        return 0.0
    return None


def _cloud_cover_from_forecast_text(value: Any) -> float | None:
    text = str(value or "").lower()
    if not text:
        return None
    if "clear" in text or "sunny" in text:
        return 0.0
    if "mostly sunny" in text or "mostly clear" in text:
        return 25.0
    if "partly" in text:
        return 50.0
    if "mostly cloudy" in text:
        return 75.0
    if "cloudy" in text or "overcast" in text:
        return 100.0
    return None


def _first_valid(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    return None if values.empty else values.iloc[0]


def _max_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.max())


def _min_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.min())


def _max_optional(left: Any, right: Any) -> float | None:
    values = [value for value in [_optional_float(left), _optional_float(right)] if value is not None]
    return None if not values else max(values)


def _min_optional(left: Any, right: Any) -> float | None:
    values = [value for value in [_optional_float(left), _optional_float(right)] if value is not None]
    return None if not values else min(values)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _signed_tenths_celsius_to_fahrenheit(sign: str, tenths: str) -> float:
    celsius = float(int(tenths)) / 10.0
    if sign == "1":
        celsius = -celsius
    return celsius * 9.0 / 5.0 + 32.0


def _attach_observed_high_so_far(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "temperature_2m" not in frame.columns:
        return frame
    result = frame.copy()
    high_columns = ["temperature_2m"]
    if "nws_6h_max_temp" in result.columns:
        timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
        result["_safe_nws_6h_max_temp"] = result["nws_6h_max_temp"].where(timestamps.dt.hour >= 6)
        high_columns.append("_safe_nws_6h_max_temp")
    result["_candidate_high"] = result[high_columns].max(axis=1, skipna=True)
    result["observed_high_so_far"] = result["_candidate_high"].cummax()

    source_times = []
    best_value = float("-inf")
    best_time = pd.NaT
    for _, row in result.iterrows():
        candidate = row["_candidate_high"]
        if pd.notna(candidate) and float(candidate) >= best_value:
            best_value = float(candidate)
            best_time = row.get("timestamp", pd.NaT)
        source_times.append(best_time)
    result["observed_high_so_far_source_time"] = source_times
    drop_columns = ["_candidate_high"]
    if "_safe_nws_6h_max_temp" in result.columns:
        drop_columns.append("_safe_nws_6h_max_temp")
    return result.drop(columns=drop_columns)


def _diagnostics_for_combined_frame(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics.assign(source_role="weather_diagnostics")
    frame = diagnostics.copy()
    if "source_role" in frame.columns:
        frame = frame.rename(columns={"source_role": "diagnostic_source_role"})
    frame["source_role"] = "weather_diagnostics"
    return frame


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
    observation_fallback_reason: str = "",
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
    if observation_fallback_reason:
        records.append(
            _diagnostic_record(
                "nws_station_observation_fallback",
                "hourly_observations",
                "NO_TRADE",
                "nws_station_observation_fallback",
                fetched_at=fetched_at,
                detail=(
                    f"{observation_fallback_reason}; using Open-Meteo proxy "
                    "observations so the model can still score for display."
                ),
            )
        )
    verified_high_record = _verified_observed_high_record(
        hourly_observations=hourly_observations,
        target_date=target_date,
        prediction_time=prediction_time,
        config=config,
    )
    if verified_high_record is not None:
        records.append(verified_high_record)

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


def _verified_observed_high_record(
    *,
    hourly_observations: pd.DataFrame,
    target_date: date,
    prediction_time: datetime,
    config: TradingConfig,
) -> dict[str, Any] | None:
    if config.weather.observations_provider != "nws_station":
        return None
    name = "verified_observed_high_window"
    if hourly_observations.empty or "timestamp" not in hourly_observations.columns:
        return _diagnostic_record(
            name,
            "hourly_observations",
            "NO_TRADE",
            "missing_observed_high_times",
        )
    if "nws_6h_max_temp" not in hourly_observations.columns:
        return _diagnostic_record(
            name,
            "hourly_observations",
            "NO_TRADE",
            "missing_nws_max_temperature_report",
        )

    frame = hourly_observations.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["nws_6h_max_temp"] = pd.to_numeric(frame["nws_6h_max_temp"], errors="coerce")
    target_ts = pd.Timestamp(target_date).normalize()
    usable = frame[
        (frame["timestamp"].notna())
        & (frame["timestamp"] <= pd.Timestamp(prediction_time))
        & (frame["timestamp"].dt.normalize() == target_ts)
        & (frame["nws_6h_max_temp"].notna())
    ]
    if usable.empty:
        return _diagnostic_record(
            name,
            "hourly_observations",
            "NO_TRADE",
            "missing_nws_max_temperature_report",
            detail="No target-date NWS 6-hour max-temperature remark is available yet.",
        )

    latest_summary = usable["timestamp"].max().to_pydatetime()
    age_minutes = (prediction_time - latest_summary).total_seconds() / 60.0
    max_age = config.weather.max_unverified_observed_high_minutes
    status = "OK" if age_minutes <= max_age else "NO_TRADE"
    reason = "" if status == "OK" else "unverified_observed_high_window"
    detail = (
        "NWS max-temperature remarks verify the station high only through the "
        f"latest summary report; max_unverified_minutes={max_age}."
    )
    return _diagnostic_record(
        name,
        "hourly_observations",
        status,
        reason,
        source_time=latest_summary,
        age_minutes=age_minutes,
        detail=detail,
    )


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
            "provider_station_id",
            "provider_station_name",
            "provider_units",
            "nws_fallback_reason",
            "nws_station_id",
            "nws_station_name",
            "nws_observation_raw",
            "nws_text_description",
            "nws_forecast_name",
            "nws_short_forecast",
            "nws_detailed_forecast",
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
