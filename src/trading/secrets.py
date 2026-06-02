from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


class TradingSecretsError(ValueError):
    """Raised when required local trading secrets are missing or malformed."""


@dataclass(frozen=True)
class TradingSecrets:
    kalshi_env: str
    api_key_id: str | None
    private_key_path: Path | None

    @property
    def has_kalshi_credentials(self) -> bool:
        return bool(self.api_key_id and self.private_key_path)


def load_trading_secrets(
    env: Mapping[str, str] | None = None,
    require_private_key: bool = True,
) -> TradingSecrets:
    source = os.environ if env is None else env
    kalshi_env = _optional_env(source, "KALSHI_ENV") or "demo"
    api_key_id = _optional_env(source, "KALSHI_API_KEY_ID")
    private_key_path_value = _optional_env(source, "KALSHI_PRIVATE_KEY_PATH")
    private_key_path = Path(private_key_path_value) if private_key_path_value else None

    if require_private_key:
        missing = [
            name
            for name, value in {
                "KALSHI_API_KEY_ID": api_key_id,
                "KALSHI_PRIVATE_KEY_PATH": private_key_path_value,
            }.items()
            if not value
        ]
        if missing:
            raise TradingSecretsError(
                "Missing required Kalshi credential environment variables: "
                + ", ".join(missing)
            )
    if require_private_key and private_key_path is not None and not private_key_path.exists():
        raise TradingSecretsError(f"Kalshi private key file does not exist: {private_key_path}")

    return TradingSecrets(
        kalshi_env=kalshi_env.strip().lower(),
        api_key_id=api_key_id,
        private_key_path=private_key_path,
    )


def load_private_key(path: str | Path) -> RSAPrivateKey:
    key_path = Path(path)
    if not key_path.exists():
        raise TradingSecretsError(f"Private key file does not exist: {key_path}")
    with key_path.open("rb") as key_file:
        key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend(),
        )
    if not isinstance(key, RSAPrivateKey):
        raise TradingSecretsError("Kalshi private key must be an RSA private key")
    return key


def _optional_env(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None
