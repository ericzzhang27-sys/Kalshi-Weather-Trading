from __future__ import annotations

from datetime import date, datetime

from src.trading.config import load_trading_config, parse_trading_config
from src.trading.live_weather import fetch_live_weather


class FakeOpenMeteoClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def fetch(self, params):
        self.calls.append(dict(params))
        if not self.payloads:
            raise AssertionError("No fake payloads left")
        return self.payloads.pop(0)


class FakeNwsObservationClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def fetch_observations(self, station_id, *, start, end, limit=500):
        self.calls.append(
            {
                "station_id": station_id,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        return self.payload


class FakeNwsForecastClient:
    def __init__(self, point_payload=None, hourly_payload=None, daily_payload=None):
        self.point_payload = point_payload or _nws_point_payload()
        self.hourly_payload = hourly_payload or _nws_hourly_forecast_payload()
        self.daily_payload = daily_payload or _nws_daily_forecast_payload()
        self.point_calls = []
        self.forecast_calls = []

    def fetch_point_metadata(self, latitude, longitude):
        self.point_calls.append({"latitude": latitude, "longitude": longitude})
        return self.point_payload

    def fetch_forecast_url(self, url):
        self.forecast_calls.append(url)
        if str(url).endswith("/hourly"):
            return self.hourly_payload
        return self.daily_payload


def _open_meteo_observation_config(**weather_overrides):
    weather = {"observations_provider": "open_meteo"}
    weather.update(weather_overrides)
    return parse_trading_config({"weather": weather})


def _hourly_payload(times=None, unit="°F"):
    times = times or [
        "2026-06-02T09:00",
        "2026-06-02T10:00",
        "2026-06-02T11:00",
        "2026-06-02T12:00",
    ]
    values = [74.0 + index for index in range(len(times))]
    return {
        "current_units": {
            "time": "iso8601",
            "temperature_2m": unit,
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": unit,
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "current": {
            "time": times[-1],
            "temperature_2m": values[-1],
            "relative_humidity_2m": 50,
            "dew_point_2m": 55.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "weather_code": 1,
            "surface_pressure": 1010.0,
            "cloud_cover": 25,
            "wind_speed_10m": 5.0,
            "wind_direction_10m": 180,
        },
        "hourly": {
            "time": times,
            "temperature_2m": values,
            "relative_humidity_2m": [50] * len(times),
            "dew_point_2m": [55.0] * len(times),
            "precipitation": [0.0] * len(times),
            "rain": [0.0] * len(times),
            "weather_code": [1] * len(times),
            "surface_pressure": [1010.0] * len(times),
            "cloud_cover": [25] * len(times),
            "wind_speed_10m": [5.0] * len(times),
            "wind_direction_10m": [180] * len(times),
        },
    }


def _forecast_payload(include_high=True, unit="°F", issue_time=None):
    daily = {
        "time": ["2026-06-02"],
        "temperature_2m_min": [69.0],
        "precipitation_sum": [0.0],
        "precipitation_hours": [0.0],
        "wind_speed_10m_max": [9.0],
        "wind_direction_10m_dominant": [180],
        "weather_code": [1],
        "cloud_cover_mean": [25],
        "dew_point_2m_mean": [55.0],
        "relative_humidity_2m_mean": [50],
        "surface_pressure_mean": [1010.0],
        "wind_speed_10m_mean": [5.0],
    }
    if include_high:
        daily["temperature_2m_max"] = [78.0]
    payload = {
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": unit,
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "daily_units": {
            "time": "iso8601",
            "temperature_2m": unit,
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "hourly": {
            "time": [
                "2026-06-02T09:00",
                "2026-06-02T10:00",
                "2026-06-02T11:00",
                "2026-06-02T12:00",
            ],
            "temperature_2m": [73.0, 74.0, 75.0, 76.0],
            "relative_humidity_2m": [55, 54, 53, 52],
            "dew_point_2m": [54.0, 54.5, 55.0, 55.5],
            "precipitation": [0.0, 0.0, 0.0, 0.0],
            "precipitation_probability": [0, 0, 0, 0],
            "rain": [0.0, 0.0, 0.0, 0.0],
            "weather_code": [1, 1, 1, 1],
            "surface_pressure": [1010.0, 1010.0, 1010.0, 1010.0],
            "cloud_cover": [20, 20, 25, 25],
            "wind_speed_10m": [4.0, 4.5, 5.0, 5.5],
            "wind_direction_10m": [180, 180, 180, 180],
        },
        "daily": daily,
    }
    if issue_time is not None:
        payload["forecast_issue_time"] = issue_time
    return payload


def _nws_point_payload():
    return {
        "properties": {
            "forecastHourly": "https://api.weather.gov/gridpoints/OKX/34,46/forecast/hourly",
            "forecast": "https://api.weather.gov/gridpoints/OKX/34,46/forecast",
        }
    }


def _nws_hourly_forecast_payload():
    return {
        "properties": {
            "units": "us",
            "updateTime": "2026-06-02T10:45:00+00:00",
            "generatedAt": "2026-06-02T11:00:00+00:00",
            "periods": [
                {
                    "number": 1,
                    "name": "",
                    "startTime": "2026-06-02T09:00:00-04:00",
                    "endTime": "2026-06-02T10:00:00-04:00",
                    "isDaytime": True,
                    "temperature": 76,
                    "temperatureUnit": "F",
                    "probabilityOfPrecipitation": {
                        "unitCode": "wmoUnit:percent",
                        "value": 10,
                    },
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 13},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 48},
                    "windSpeed": "5 mph",
                    "windDirection": "SW",
                    "shortForecast": "Mostly Sunny",
                    "detailedForecast": "",
                },
                {
                    "number": 2,
                    "name": "",
                    "startTime": "2026-06-02T10:00:00-04:00",
                    "endTime": "2026-06-02T11:00:00-04:00",
                    "isDaytime": True,
                    "temperature": 78,
                    "temperatureUnit": "F",
                    "probabilityOfPrecipitation": {
                        "unitCode": "wmoUnit:percent",
                        "value": 12,
                    },
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 14},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 45},
                    "windSpeed": "6 to 10 mph",
                    "windDirection": "SW",
                    "shortForecast": "Mostly Sunny",
                    "detailedForecast": "Southwest wind 6 to 10 mph.",
                },
                {
                    "number": 3,
                    "name": "",
                    "startTime": "2026-06-02T11:00:00-04:00",
                    "endTime": "2026-06-02T12:00:00-04:00",
                    "isDaytime": True,
                    "temperature": 79,
                    "temperatureUnit": "F",
                    "probabilityOfPrecipitation": {
                        "unitCode": "wmoUnit:percent",
                        "value": 15,
                    },
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 15},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 43},
                    "windSpeed": "8 mph",
                    "windDirection": "SW",
                    "shortForecast": "Partly Sunny",
                    "detailedForecast": "Southwest wind 8 mph, with gusts as high as 18 mph.",
                },
            ],
        }
    }


def _nws_daily_forecast_payload():
    return {
        "properties": {
            "units": "us",
            "updateTime": "2026-06-02T10:45:00+00:00",
            "generatedAt": "2026-06-02T11:00:00+00:00",
            "periods": [
                {
                    "number": 1,
                    "name": "Today",
                    "startTime": "2026-06-02T06:00:00-04:00",
                    "endTime": "2026-06-02T18:00:00-04:00",
                    "isDaytime": True,
                    "temperature": 80,
                    "temperatureUnit": "F",
                    "probabilityOfPrecipitation": {
                        "unitCode": "wmoUnit:percent",
                        "value": 15,
                    },
                    "windSpeed": "6 to 12 mph",
                    "windDirection": "SW",
                    "shortForecast": "Partly Sunny",
                    "detailedForecast": "Partly sunny, with a high near 80.",
                },
                {
                    "number": 2,
                    "name": "Tonight",
                    "startTime": "2026-06-02T18:00:00-04:00",
                    "endTime": "2026-06-03T06:00:00-04:00",
                    "isDaytime": False,
                    "temperature": 69,
                    "temperatureUnit": "F",
                    "probabilityOfPrecipitation": {
                        "unitCode": "wmoUnit:percent",
                        "value": 5,
                    },
                    "windSpeed": "5 mph",
                    "windDirection": "SW",
                    "shortForecast": "Mostly Clear",
                    "detailedForecast": "Mostly clear, with a low around 69.",
                },
            ],
        }
    }


def test_fetch_live_weather_builds_frames_and_warns_on_missing_issue_time() -> None:
    config = _open_meteo_observation_config()
    client = FakeOpenMeteoClient([_hourly_payload(), _forecast_payload()])

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=client,
        fetched_at=datetime(2026, 6, 2, 12, 5),
    )

    assert len(client.calls) == 2
    assert snapshot.no_trade is False
    assert "forecast_high" in snapshot.daily_forecast.columns
    issue_diag = snapshot.diagnostics[
        snapshot.diagnostics["diagnostic_name"] == "forecast_issue_time_present"
    ].iloc[0]
    assert issue_diag["status"] == "WARN"


def test_stale_observation_produces_no_trade_diagnostic() -> None:
    config = _open_meteo_observation_config()
    client = FakeOpenMeteoClient(
        [
            _hourly_payload(times=["2026-06-02T06:00", "2026-06-02T07:00"]),
            _forecast_payload(),
        ]
    )

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=client,
    )

    assert snapshot.no_trade is True
    assert "stale_weather_data" in set(snapshot.diagnostics["no_trade_reason"])


def test_missing_daily_forecast_high_produces_no_trade_diagnostic() -> None:
    config = _open_meteo_observation_config()
    client = FakeOpenMeteoClient([_hourly_payload(), _forecast_payload(include_high=False)])

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=client,
    )

    assert snapshot.no_trade is True
    assert "missing_forecast_high" in set(snapshot.diagnostics["no_trade_reason"])


def test_unit_mismatch_produces_no_trade_diagnostic() -> None:
    config = _open_meteo_observation_config()
    client = FakeOpenMeteoClient([_hourly_payload(unit="°C"), _forecast_payload()])

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=client,
    )

    assert snapshot.no_trade is True
    assert "unit_mismatch" in set(snapshot.diagnostics["no_trade_reason"])


def test_required_issue_time_turns_missing_issue_warning_into_no_trade() -> None:
    config = _open_meteo_observation_config(require_forecast_issue_time=True)
    client = FakeOpenMeteoClient([_hourly_payload(), _forecast_payload()])

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=client,
    )

    assert snapshot.no_trade is True
    issue_diag = snapshot.diagnostics[
        snapshot.diagnostics["diagnostic_name"] == "forecast_issue_time_present"
    ].iloc[0]
    assert issue_diag["status"] == "NO_TRADE"


def test_nws_forecasts_are_converted_to_feature_compatible_frames() -> None:
    config = parse_trading_config(
        {
            "weather": {
                "provider": "nws",
                "observations_provider": "nws_station",
            }
        }
    )
    forecast_client = FakeNwsForecastClient()

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=forecast_client,
        observation_client=FakeNwsObservationClient(_nws_observation_payload()),
        fetched_at=datetime(2026, 6, 2, 12, 5),
    )

    assert forecast_client.point_calls
    assert len(forecast_client.forecast_calls) == 2
    assert snapshot.hourly_forecasts["forecast_source"].iloc[0] == "nws_gridpoint_forecast"
    assert snapshot.hourly_forecasts["forecast_issue_time"].notna().all()
    assert snapshot.hourly_forecasts["temperature_2m"].max() == 79.0
    assert snapshot.hourly_forecasts["dew_point_2m"].iloc[0] == 55.4
    assert snapshot.daily_forecast["forecast_high"].iloc[0] == 79.0
    assert "forecast_high" in snapshot.daily_forecast.columns
    issue_diag = snapshot.diagnostics[
        snapshot.diagnostics["diagnostic_name"] == "forecast_issue_time_present"
    ].iloc[0]
    assert issue_diag["status"] == "OK"


def test_nws_station_observations_are_converted_to_feature_compatible_units() -> None:
    config = parse_trading_config({"weather": {"observations_provider": "nws_station"}})
    forecast_client = FakeOpenMeteoClient([_forecast_payload()])
    observation_client = FakeNwsObservationClient(_nws_observation_payload())

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=forecast_client,
        observation_client=observation_client,
        fetched_at=datetime(2026, 6, 2, 12, 5),
    )

    assert len(forecast_client.calls) == 1
    assert observation_client.calls[0]["station_id"] == "KNYC"
    assert snapshot.hourly_observations["forecast_source"].iloc[0] == "nws_station_observations"
    assert snapshot.hourly_observations["provider_station_id"].iloc[0] == "KNYC"
    latest = snapshot.hourly_observations.sort_values("timestamp").iloc[-1]
    assert latest["temperature_2m"] == 77.0
    assert latest["dew_point_2m"] == 50.0
    assert round(float(latest["wind_speed_10m"]), 4) == 6.2137
    assert latest["surface_pressure"] == 1012.0
    assert latest["precipitation"] == 0.1
    assert latest["nws_6h_max_temp"] == 82.94
    assert latest["observed_high_so_far"] == 82.94
    assert snapshot.no_trade is False


def test_nws_station_observed_high_window_blocks_stale_max_summary() -> None:
    config = parse_trading_config(
        {
            "weather": {
                "observations_provider": "nws_station",
                "max_unverified_observed_high_minutes": 20,
            }
        }
    )
    forecast_client = FakeOpenMeteoClient([_forecast_payload()])
    payload = {
        "features": [
            {
                "properties": {
                    "timestamp": "2026-06-02T11:51:00+00:00",
                    "temperature": {"unitCode": "wmoUnit:degC", "value": 22.0},
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 10.0},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 50.0},
                    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 0.0},
                    "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101000},
                    "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 0.0},
                    "cloudLayers": [{"amount": "CLR"}],
                    "textDescription": "Clear",
                    "rawMessage": "KNYC 021151Z AUTO RMK AO2 T02200100 10283",
                }
            },
            {
                "properties": {
                    "timestamp": "2026-06-02T16:00:00+00:00",
                    "temperature": {"unitCode": "wmoUnit:degC", "value": 25.0},
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 10.0},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 38.0},
                    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 0.0},
                    "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101000},
                    "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 0.0},
                    "cloudLayers": [{"amount": "CLR"}],
                    "textDescription": "Clear",
                    "rawMessage": "KNYC 021600Z AUTO RMK AO2 T02500100",
                }
            },
        ]
    }

    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=forecast_client,
        observation_client=FakeNwsObservationClient(payload),
        fetched_at=datetime(2026, 6, 2, 12, 0),
    )

    diagnostic = snapshot.diagnostics[
        snapshot.diagnostics["diagnostic_name"] == "verified_observed_high_window"
    ].iloc[0]
    assert diagnostic["status"] == "NO_TRADE"
    assert diagnostic["no_trade_reason"] == "unverified_observed_high_window"
    assert snapshot.no_trade is True


def test_nws_daily_max_remark_does_not_inflate_intraday_high_so_far() -> None:
    config = parse_trading_config({"weather": {"observations_provider": "nws_station"}})
    forecast_client = FakeOpenMeteoClient([_forecast_payload()])
    payload = {
        "features": [
            {
                "properties": {
                    "timestamp": "2026-06-02T04:51:00+00:00",
                    "temperature": {"unitCode": "wmoUnit:degC", "value": 20.0},
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 10.0},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 50.0},
                    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 0.0},
                    "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101000},
                    "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 0.0},
                    "cloudLayers": [{"amount": "CLR"}],
                    "textDescription": "Clear",
                    "rawMessage": "KNYC 020451Z AUTO RMK AO2 T02000100 403000100",
                }
            },
            {
                "properties": {
                    "timestamp": "2026-06-02T12:51:00+00:00",
                    "temperature": {"unitCode": "wmoUnit:degC", "value": 21.0},
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 10.0},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 45.0},
                    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 0.0},
                    "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101000},
                    "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 0.0},
                    "cloudLayers": [{"amount": "CLR"}],
                    "textDescription": "Clear",
                    "rawMessage": "KNYC 021251Z AUTO RMK AO2 T02100100",
                }
            },
        ]
    }
    snapshot = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 9, 0),
        config=config,
        client=forecast_client,
        observation_client=FakeNwsObservationClient(payload),
        fetched_at=datetime(2026, 6, 2, 9, 0),
    )

    ordered = snapshot.hourly_observations.sort_values("timestamp")
    assert ordered.iloc[0]["nws_24h_max_temp"] == 86.0
    assert ordered.iloc[-1]["observed_high_so_far"] == 69.8


def _nws_observation_payload():
    return {
        "features": [
            {
                "properties": {
                    "timestamp": "2026-06-02T15:00:00+00:00",
                    "temperature": {"unitCode": "wmoUnit:degC", "value": 23.0},
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 10.0},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 43.0},
                    "windDirection": {"unitCode": "wmoUnit:degree_(angle)", "value": 180},
                    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 8.0},
                    "windGust": {"unitCode": "wmoUnit:km_h-1", "value": 16.0},
                    "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101000},
                    "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 0.0},
                    "cloudLayers": [{"amount": "CLR", "base": {"value": None}}],
                    "textDescription": "Clear",
                    "rawMessage": "KNYC 021500Z AUTO",
                }
            },
            {
                "properties": {
                    "timestamp": "2026-06-02T16:00:00+00:00",
                    "temperature": {"unitCode": "wmoUnit:degC", "value": 25.0},
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 10.0},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 38.0},
                    "windDirection": {"unitCode": "wmoUnit:degree_(angle)", "value": 200},
                    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 10.0},
                    "windGust": {"unitCode": "wmoUnit:km_h-1", "value": None},
                    "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101200},
                    "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 2.54},
                    "cloudLayers": [{"amount": "SCT", "base": {"value": 1200}}],
                    "textDescription": "Partly Cloudy",
                    "rawMessage": "KNYC 021600Z AUTO 20006KT 10SM SCT040 25/10 A2988 RMK AO2 T02500100 10283",
                }
            },
        ]
    }
