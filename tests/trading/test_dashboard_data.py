from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.trading.dashboard_data as dashboard_data
from src.trading.config import load_trading_config, parse_trading_config
from src.trading.dashboard_data import (
    build_bucket_board,
    load_dashboard_state,
    load_dashboard_state_from_artifacts,
    _select_event_ticker,
)


class FakeKalshiClient:
    def __init__(self) -> None:
        self.market_calls = []
        self.orderbook_calls = []

    def iter_markets(self, **kwargs):
        self.market_calls.append(dict(kwargs))
        yield from _markets()

    def get(self, path, params=None, auth=False):
        self.orderbook_calls.append((path, dict(params or {}), auth))
        return {"yes_dollars": [[0.40, 10]], "no_dollars": [[0.55, 7]]}


class FakeOpenMeteoClient:
    def __init__(self) -> None:
        self.payloads = [_observation_payload(), _forecast_payload()]
        self.calls = []

    def fetch(self, params):
        self.calls.append(dict(params))
        if not self.payloads:
            raise AssertionError("No fake payloads left")
        return self.payloads.pop(0)


def test_build_bucket_board_combines_mapping_probabilities_and_prices() -> None:
    mapping = pd.DataFrame(
        [
            {
                "event_ticker": "KXHIGHNY-26JUN02",
                "ticker": "KXHIGHNY-26JUN02-T73",
                "mapping_status": "MAPPED",
                "bucket_name": "72 or below",
                "bucket_lower_temp": None,
                "bucket_upper_temp": 72.5,
            }
        ]
    )
    probabilities = pd.DataFrame(
        [
            {
                "ticker": "KXHIGHNY-26JUN02-T73",
                "probability": 0.25,
                "mu": -1.0,
                "sigma": 2.5,
                "model_name": "demo_model",
            }
        ]
    )
    orderbook_summary = pd.DataFrame(
        [
            {
                "ticker": "KXHIGHNY-26JUN02-T73",
                "best_yes_bid": 0.40,
                "best_yes_ask": 0.45,
                "yes_spread": 0.05,
                "orderbook_status": "OK",
            }
        ]
    )

    board = build_bucket_board(mapping, probabilities, orderbook_summary)

    assert len(board) == 1
    assert board.iloc[0]["probability"] == 0.25
    assert board.iloc[0]["best_yes_bid"] == 0.40
    assert board.iloc[0]["orderbook_status"] == "OK"


def test_select_event_ticker_prefers_nearest_eligible_close() -> None:
    markets = pd.DataFrame(
        [
            {
                "event_ticker": "KXHIGHNY-26JUN04",
                "eligible": True,
                "close_time": "2026-06-05T04:59:00Z",
            },
            {
                "event_ticker": "KXHIGHNY-26JUN03",
                "eligible": True,
                "close_time": "2026-06-04T04:59:00Z",
            },
        ]
    )

    assert _select_event_ticker(markets, None) == "KXHIGHNY-26JUN03"


def test_load_dashboard_state_from_saved_artifacts_scores_when_needed() -> None:
    config = load_trading_config()
    feature_rows_path = ROOT / "outputs/live_trading/live_feature_rows.csv"
    mapping_path = ROOT / "outputs/live_trading/contract_bucket_mapping.csv"
    if not feature_rows_path.exists() or not mapping_path.exists():
        pytest.skip("Day 2 live feature artifacts are not available")

    state = load_dashboard_state_from_artifacts(config)

    assert state.status["data_source"] == "saved_artifacts"
    assert state.status["trading_enabled"] is False
    assert not state.mapping.empty
    assert not state.live_feature_rows.empty
    assert not state.bucket_board.empty


def test_load_dashboard_state_orchestrates_read_only_refresh_with_fake_clients() -> None:
    config = parse_trading_config({"weather": {"observations_provider": "open_meteo"}})
    kalshi_client = FakeKalshiClient()
    weather_client = FakeOpenMeteoClient()

    state = load_dashboard_state(
        config,
        event_ticker="KXHIGHNY-26JUN02",
        target_date=date(2026, 6, 2),
        depth=5,
        kalshi_client=kalshi_client,
        weather_client=weather_client,
        prediction_time=datetime(2026, 6, 2, 12, 0),
        write_outputs=False,
    )

    assert state.status["read_only"] is True
    assert state.status["trading_enabled"] is False
    assert len(kalshi_client.orderbook_calls) == 6
    assert len(weather_client.calls) == 2
    assert len(state.bucket_board) == 6
    assert state.bucket_board["probability"].sum() == pytest.approx(1.0, abs=1e-6)
    assert state.bucket_board["best_yes_bid"].notna().all()
    assert not state.live_feature_rows.empty
    assert not state.feature_freshness.empty


def test_load_dashboard_state_keeps_live_orderbooks_when_features_are_not_scoreable() -> None:
    class BrokenWeatherClient:
        def fetch(self, params):
            raise RuntimeError("weather unavailable")

    config = parse_trading_config({"weather": {"observations_provider": "open_meteo"}})
    state = load_dashboard_state(
        config,
        event_ticker="KXHIGHNY-26JUN02",
        target_date=date(2026, 6, 2),
        depth=5,
        kalshi_client=FakeKalshiClient(),
        weather_client=BrokenWeatherClient(),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        write_outputs=False,
    )

    assert state.status["data_source"] == "live_partial"
    assert state.status["probability_rows"] == 0
    assert any("live_feature_scoring_unavailable" in item for item in state.status["warnings"])
    assert state.bucket_probabilities.empty
    assert state.live_feature_rows.empty
    assert not state.orderbook.empty
    assert len(state.bucket_board) == 6
    assert state.bucket_board["best_yes_bid"].notna().all()


def test_load_dashboard_state_records_probability_scoring_error_after_features(monkeypatch) -> None:
    def broken_score(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(dashboard_data, "score_live_probabilities", broken_score)
    config = parse_trading_config({"weather": {"observations_provider": "open_meteo"}})

    state = load_dashboard_state(
        config,
        event_ticker="KXHIGHNY-26JUN02",
        target_date=date(2026, 6, 2),
        depth=5,
        kalshi_client=FakeKalshiClient(),
        weather_client=FakeOpenMeteoClient(),
        prediction_time=datetime(2026, 6, 2, 12, 0),
        write_outputs=False,
    )

    assert not state.live_feature_rows.empty
    assert state.bucket_probabilities.empty
    assert state.edge_table.empty
    assert state.status["feature_rows"] == len(state.live_feature_rows)
    assert state.status["probability_rows"] == 0
    assert state.status["edge_rows"] == 0
    assert state.status["probability_scoring_status"] == "ERROR"
    assert state.status["probability_scoring_error"] == "RuntimeError: model unavailable"
    assert any(
        "probability_scoring_unavailable:RuntimeError: model unavailable" == item
        for item in state.status["warnings"]
    )


def _markets() -> list[dict[str, object]]:
    close_time = "2030-06-02T23:00:00Z"
    base = {
        "event_ticker": "KXHIGHNY-26JUN02",
        "status": "open",
        "close_time": close_time,
        "rules_primary": "The highest temperature in New York City determines the result.",
    }
    rows = [
        {
            "ticker": "KXHIGHNY-26JUN02-T73",
            "title": "Will the high temperature in NYC be 72 or below on Jun 2, 2026?",
            "subtitle": "72 or below",
            "strike_type": "less",
            "cap_strike": 73,
        },
        {
            "ticker": "KXHIGHNY-26JUN02-B73.5",
            "title": "Will the high temperature in NYC be 73 to 74 on Jun 2, 2026?",
            "subtitle": "73 to 74",
            "strike_type": "between",
            "floor_strike": 73,
            "cap_strike": 74,
        },
        {
            "ticker": "KXHIGHNY-26JUN02-B75.5",
            "title": "Will the high temperature in NYC be 75 to 76 on Jun 2, 2026?",
            "subtitle": "75 to 76",
            "strike_type": "between",
            "floor_strike": 75,
            "cap_strike": 76,
        },
        {
            "ticker": "KXHIGHNY-26JUN02-B77.5",
            "title": "Will the high temperature in NYC be 77 to 78 on Jun 2, 2026?",
            "subtitle": "77 to 78",
            "strike_type": "between",
            "floor_strike": 77,
            "cap_strike": 78,
        },
        {
            "ticker": "KXHIGHNY-26JUN02-B79.5",
            "title": "Will the high temperature in NYC be 79 to 80 on Jun 2, 2026?",
            "subtitle": "79 to 80",
            "strike_type": "between",
            "floor_strike": 79,
            "cap_strike": 80,
        },
        {
            "ticker": "KXHIGHNY-26JUN02-T80",
            "title": "Will the high temperature in NYC be 81 or above on Jun 2, 2026?",
            "subtitle": "81 or above",
            "strike_type": "greater",
            "floor_strike": 80,
        },
    ]
    return [dict(base, **row) for row in rows]


def _observation_payload() -> dict[str, object]:
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
            "temperature_2m": "F",
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "F",
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


def _forecast_payload() -> dict[str, object]:
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
            "temperature_2m": "F",
            "wind_speed_10m": "mp/h",
            "precipitation": "inch",
        },
        "daily_units": {
            "time": "iso8601",
            "temperature_2m": "F",
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
