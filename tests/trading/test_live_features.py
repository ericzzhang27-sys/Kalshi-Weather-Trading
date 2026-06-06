from __future__ import annotations

from datetime import date, datetime
import json
import subprocess
import sys

import pandas as pd

from src.predict_distribution import DEFAULT_FEATURE_LIST_PATH, load_probability_engine
from src.trading.config import parse_trading_config
from src.trading.contract_mapping import map_event_contracts
from src.trading.live_features import build_live_feature_rows
from src.trading.live_weather import fetch_live_weather


class FakeOpenMeteoClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def fetch(self, params):
        if not self.payloads:
            raise AssertionError("No fake payloads left")
        return self.payloads.pop(0)


class FakeNwsObservationClient:
    def __init__(self, payload):
        self.payload = payload

    def fetch_observations(self, station_id, *, start, end, limit=500):
        return self.payload


class FakeNwsForecastClient:
    def __init__(self):
        self.point_payload = _nws_point_payload()
        self.hourly_payload = _nws_hourly_forecast_payload()
        self.daily_payload = _nws_daily_forecast_payload()

    def fetch_point_metadata(self, latitude, longitude):
        return self.point_payload

    def fetch_forecast_url(self, url):
        if str(url).endswith("/hourly"):
            return self.hourly_payload
        return self.daily_payload


def _observation_payload():
    times = [
        "2026-06-02T06:00",
        "2026-06-02T07:00",
        "2026-06-02T08:00",
        "2026-06-02T09:00",
        "2026-06-02T10:00",
        "2026-06-02T11:00",
        "2026-06-02T12:00",
    ]
    temps = [70.0, 71.0, 72.5, 74.0, 75.0, 76.0, 77.0]
    return {
        "current_units": {
            "time": "iso8601",
            "temperature_2m": "°F",
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°F",
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "current": {
            "time": times[-1],
            "temperature_2m": temps[-1],
            "relative_humidity_2m": 52,
            "dew_point_2m": 56.0,
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
            "temperature_2m": temps,
            "relative_humidity_2m": [52] * len(times),
            "dew_point_2m": [56.0] * len(times),
            "precipitation": [0.0] * len(times),
            "rain": [0.0] * len(times),
            "weather_code": [1] * len(times),
            "surface_pressure": [1010.0] * len(times),
            "cloud_cover": [25] * len(times),
            "wind_speed_10m": [5.0] * len(times),
            "wind_direction_10m": [180] * len(times),
        },
    }


def _forecast_payload():
    times = [
        "2026-06-02T06:00",
        "2026-06-02T07:00",
        "2026-06-02T08:00",
        "2026-06-02T09:00",
        "2026-06-02T10:00",
        "2026-06-02T11:00",
        "2026-06-02T12:00",
    ]
    return {
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°F",
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "daily_units": {
            "time": "iso8601",
            "temperature_2m": "°F",
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "hourly": {
            "time": times,
            "temperature_2m": [71.0, 72.0, 73.0, 74.0, 75.0, 75.5, 76.0],
            "relative_humidity_2m": [55] * len(times),
            "dew_point_2m": [55.0] * len(times),
            "precipitation": [0.0] * len(times),
            "precipitation_probability": [0] * len(times),
            "rain": [0.0] * len(times),
            "weather_code": [1] * len(times),
            "surface_pressure": [1010.0] * len(times),
            "cloud_cover": [25] * len(times),
            "wind_speed_10m": [5.0] * len(times),
            "wind_direction_10m": [180] * len(times),
        },
        "daily": {
            "time": ["2026-06-02"],
            "temperature_2m_max": [78.0],
            "temperature_2m_min": [69.0],
            "precipitation_sum": [0.0],
            "precipitation_hours": [0.0],
            "wind_speed_10m_max": [9.0],
            "wind_direction_10m_dominant": [180],
            "weather_code": [1],
            "cloud_cover_mean": [25],
            "dew_point_2m_mean": [55.0],
            "relative_humidity_2m_mean": [55],
            "surface_pressure_mean": [1010.0],
            "wind_speed_10m_mean": [5.0],
        },
    }


def _mapping():
    markets = pd.DataFrame(
        [
            {
                "event_ticker": "KXHIGHNY-26JUN02",
                "ticker": "KXHIGHNY-26JUN02-T73",
                "status": "active",
                "eligible": True,
                "title": "Will the high temp in NYC be <73?",
                "subtitle": "72 or below",
                "strike_type": "less",
                "cap_strike": 73,
            },
            {
                "event_ticker": "KXHIGHNY-26JUN02",
                "ticker": "KXHIGHNY-26JUN02-B73.5",
                "status": "active",
                "eligible": True,
                "title": "Will the high temp in NYC be 73-74?",
                "subtitle": "73 to 74",
                "strike_type": "between",
                "floor_strike": 73,
                "cap_strike": 74,
            },
            {
                "event_ticker": "KXHIGHNY-26JUN02",
                "ticker": "KXHIGHNY-26JUN02-B75.5",
                "status": "active",
                "eligible": True,
                "title": "Will the high temp in NYC be 75-76?",
                "subtitle": "75 to 76",
                "strike_type": "between",
                "floor_strike": 75,
                "cap_strike": 76,
            },
            {
                "event_ticker": "KXHIGHNY-26JUN02",
                "ticker": "KXHIGHNY-26JUN02-B77.5",
                "status": "active",
                "eligible": True,
                "title": "Will the high temp in NYC be 77-78?",
                "subtitle": "77 to 78",
                "strike_type": "between",
                "floor_strike": 77,
                "cap_strike": 78,
            },
            {
                "event_ticker": "KXHIGHNY-26JUN02",
                "ticker": "KXHIGHNY-26JUN02-B79.5",
                "status": "active",
                "eligible": True,
                "title": "Will the high temp in NYC be 79-80?",
                "subtitle": "79 to 80",
                "strike_type": "between",
                "floor_strike": 79,
                "cap_strike": 80,
            },
            {
                "event_ticker": "KXHIGHNY-26JUN02",
                "ticker": "KXHIGHNY-26JUN02-T80",
                "status": "active",
                "eligible": True,
                "title": "Will the high temp in NYC be >80?",
                "subtitle": "81 or above",
                "strike_type": "greater",
                "floor_strike": 80,
            },
        ]
    )
    return map_event_contracts(markets, "KXHIGHNY-26JUN02")


def _feature_rows():
    config = parse_trading_config({"weather": {"observations_provider": "open_meteo"}})
    client = FakeOpenMeteoClient([_observation_payload(), _forecast_payload()])
    weather = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=client,
    )
    return build_live_feature_rows(
        weather=weather,
        mapping=_mapping(),
        feature_list_path=DEFAULT_FEATURE_LIST_PATH,
    )


def test_live_feature_rows_have_final_feature_columns_and_freshness() -> None:
    rows = _feature_rows()
    feature_columns = json.loads(DEFAULT_FEATURE_LIST_PATH.read_text(encoding="utf-8"))["features"]

    assert len(rows) == 1
    assert feature_columns == list(rows.columns[-len(feature_columns):])
    assert rows["live_feature_status"].iloc[0] == "SCOREABLE_SHADOW"
    freshness = rows.attrs["freshness"]
    assert len(freshness) == len(feature_columns)
    assert set(freshness["feature"]) == set(feature_columns)


def test_probability_engine_accepts_live_feature_row() -> None:
    rows = _feature_rows()
    engine = load_probability_engine(feature_list_path=DEFAULT_FEATURE_LIST_PATH)

    params = engine.predict_distribution_params(rows)

    assert len(params) == 1
    assert params["sigma"].iloc[0] > 0


def test_nws_forecasts_build_live_feature_rows() -> None:
    config = parse_trading_config(
        {"weather": {"provider": "nws", "observations_provider": "nws_station"}}
    )
    weather = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=FakeNwsForecastClient(),
        observation_client=FakeNwsObservationClient(_nws_observation_payload()),
    )

    rows = build_live_feature_rows(
        weather=weather,
        mapping=_mapping(),
        feature_list_path=DEFAULT_FEATURE_LIST_PATH,
    )

    assert len(rows) == 1
    assert rows["target_date"].iloc[0] == pd.Timestamp("2026-06-02")
    assert rows["live_feature_status"].iloc[0] == "SCOREABLE_SHADOW"
    assert weather.daily_forecast["target_date"].iloc[0] == pd.Timestamp("2026-06-02")


def test_stale_nws_observations_fall_back_to_proxy_features() -> None:
    config = parse_trading_config({"weather": {"observations_provider": "nws_station"}})
    client = FakeOpenMeteoClient([_forecast_payload(), _observation_payload()])
    weather = fetch_live_weather(
        location="NYC",
        target_date=date(2026, 6, 2),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        config=config,
        client=client,
        observation_client=FakeNwsObservationClient(_stale_nws_payload()),
    )

    rows = build_live_feature_rows(
        weather=weather,
        mapping=_mapping(),
        feature_list_path=DEFAULT_FEATURE_LIST_PATH,
    )

    assert len(rows) == 1
    assert weather.hourly_observations["forecast_source"].iloc[-1] == "open_meteo_observation_fallback"
    assert rows["live_feature_status"].iloc[0] == "NO_TRADE"
    assert "nws_station_observation_fallback" in rows["no_trade_reason"].iloc[0]


def test_day2_script_dry_run_writes_expected_outputs(tmp_path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    weather_path = tmp_path / "weather.csv"
    features_path = tmp_path / "features.csv"
    freshness_path = tmp_path / "freshness.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_live_feature_rows.py",
            "--dry-run-fixture",
            "--skip-scoreability-check",
            "--mapping-output-path",
            str(mapping_path),
            "--weather-output-path",
            str(weather_path),
            "--feature-output-path",
            str(features_path),
            "--freshness-output-path",
            str(freshness_path),
        ],
        cwd=".",
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Saved Day 2 live feature rows" in completed.stdout
    assert mapping_path.exists()
    assert weather_path.exists()
    assert features_path.exists()
    assert freshness_path.exists()


def _stale_nws_payload():
    return {
        "features": [
            {
                "properties": {
                    "timestamp": "2026-06-02T10:00:00+00:00",
                    "temperature": {"unitCode": "wmoUnit:degC", "value": 22.0},
                    "dewpoint": {"unitCode": "wmoUnit:degC", "value": 12.0},
                    "relativeHumidity": {"unitCode": "wmoUnit:percent", "value": 55.0},
                    "windDirection": {"unitCode": "wmoUnit:degree_(angle)", "value": 180},
                    "windSpeed": {"unitCode": "wmoUnit:km_h-1", "value": 8.0},
                    "barometricPressure": {"unitCode": "wmoUnit:Pa", "value": 101000},
                    "precipitationLastHour": {"unitCode": "wmoUnit:mm", "value": 0.0},
                    "cloudLayers": [{"amount": "CLR"}],
                    "textDescription": "Clear",
                    "rawMessage": "KNYC 021000Z AUTO 18004KT 10SM CLR 22/12 A2983 RMK AO2 T02200120",
                }
            }
        ]
    }


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
