from __future__ import annotations

from pathlib import Path

import pytest

from src.trading.config import TradingConfigError, load_trading_config, parse_trading_config


def test_default_trading_config_is_shadow_and_not_live_auto() -> None:
    config = load_trading_config()

    assert config.mode == "shadow"
    assert config.trading_enabled is False
    assert config.live_auto_enabled is False
    assert config.kalshi.base_url.endswith("/trade-api/v2")
    assert config.markets.default_location == "NYC"


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
