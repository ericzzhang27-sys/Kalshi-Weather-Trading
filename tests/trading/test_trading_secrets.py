from __future__ import annotations

from pathlib import Path

import pytest

from src.trading.secrets import TradingSecretsError, load_trading_secrets


def test_missing_required_kalshi_secrets_fail_closed() -> None:
    with pytest.raises(TradingSecretsError, match="KALSHI_API_KEY_ID"):
        load_trading_secrets(env={}, require_private_key=True)


def test_public_market_data_can_load_without_secrets() -> None:
    secrets = load_trading_secrets(env={}, require_private_key=False)

    assert secrets.kalshi_env == "demo"
    assert secrets.has_kalshi_credentials is False


def test_missing_private_key_file_fails(tmp_path: Path) -> None:
    missing_key_path = tmp_path / "missing.key"

    with pytest.raises(TradingSecretsError, match="does not exist"):
        load_trading_secrets(
            env={
                "KALSHI_API_KEY_ID": "fake-key-id",
                "KALSHI_PRIVATE_KEY_PATH": str(missing_key_path),
            },
            require_private_key=True,
        )


def test_unused_missing_private_key_does_not_block_public_market_data(tmp_path: Path) -> None:
    secrets = load_trading_secrets(
        env={"KALSHI_PRIVATE_KEY_PATH": str(tmp_path / "missing.key")},
        require_private_key=False,
    )

    assert secrets.private_key_path == tmp_path / "missing.key"
