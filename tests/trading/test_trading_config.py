from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.trading.config import TradingConfigError, load_trading_config, parse_trading_config, validate_trading_config


def test_default_trading_config_is_shadow_and_not_live_auto() -> None:
    config = load_trading_config()

    assert config.mode == "shadow"
    assert config.trading_enabled is False
    assert config.live_auto_enabled is False
    assert config.kalshi.base_url.endswith("/trade-api/v2")
    assert config.markets.default_location == "NYC"
    assert config.weather.provider == "nws"
    assert config.settlement.typical_peak_hour == 15
    assert config.outputs.settlement_state_path.name == "settlement_state.csv"


def test_live_auto_requires_explicit_enablement() -> None:
    with pytest.raises(TradingConfigError, match="live_auto_enabled"):
        parse_trading_config(
            {
                "mode": "live_auto",
                "trading_enabled": True,
                "live_auto_enabled": False,
            }
        )


def test_live_modes_require_trading_enabled() -> None:
    with pytest.raises(TradingConfigError, match="trading_enabled"):
        parse_trading_config(
            {
                "mode": "live_manual_approve",
                "trading_enabled": False,
                "live_auto_enabled": False,
            }
        )


def test_missing_config_path_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trading_config(tmp_path / "missing.yaml")


def test_invalid_settlement_hours_fail_validation() -> None:
    with pytest.raises(TradingConfigError, match="peak_window_end_hour"):
        parse_trading_config(
            {
                "settlement": {
                    "typical_peak_hour": 18,
                    "peak_window_end_hour": 15,
                }
            }
        )


def test_weather_provider_aliases_are_normalized() -> None:
    config = parse_trading_config(
        {
            "weather": {
                "provider": "NWS Station",
                "observations_provider": "weather.gov",
            }
        }
    )

    assert config.weather.provider == "nws"
    assert config.weather.observations_provider == "nws_station"


def test_open_meteo_provider_alias_is_normalized() -> None:
    config = parse_trading_config(
        {
            "weather": {
                "provider": "open-meteo",
                "observations_provider": "openmeteo",
            }
        }
    )

    assert config.weather.provider == "open_meteo"
    assert config.weather.observations_provider == "open_meteo"


def test_unknown_live_weather_provider_falls_back_to_nws() -> None:
    config = parse_trading_config(
        {
            "weather": {
                "provider": "nws-current-live-forecast-v2",
                "observations_provider": "station-observations",
            }
        }
    )

    assert config.weather.provider == "nws"
    assert config.weather.observations_provider == "nws_station"


def test_weather_provider_validation_is_fail_open_for_dashboard_boot() -> None:
    config = parse_trading_config({})
    config = replace(
        config,
        weather=replace(
            config.weather,
            provider="legacy_streamlit_value",
            observations_provider="legacy_observation_value",
        ),
    )

    validate_trading_config(config)
