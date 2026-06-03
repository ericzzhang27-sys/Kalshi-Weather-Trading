from __future__ import annotations

from datetime import date, datetime
import json
import subprocess
import sys

import pandas as pd

from src.predict_distribution import DEFAULT_FEATURE_LIST_PATH, load_probability_engine
from src.trading.config import load_trading_config
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
    config = load_trading_config()
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
