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


def test_fetch_live_weather_builds_frames_and_warns_on_missing_issue_time() -> None:
    config = load_trading_config()
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
    config = load_trading_config()
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
    config = load_trading_config()
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
    config = load_trading_config()
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
    config = parse_trading_config({"weather": {"require_forecast_issue_time": True}})
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
