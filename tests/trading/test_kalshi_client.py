from __future__ import annotations

import base64

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.trading.kalshi_client import (
    KalshiClient,
    create_kalshi_signature,
    signing_path_from_url,
)


def test_signing_path_strips_query_and_keeps_api_prefix() -> None:
    path = signing_path_from_url(
        "https://external-api.demo.kalshi.co/trade-api/v2",
        "/markets?limit=5",
    )

    assert path == "/trade-api/v2/markets"


def test_create_kalshi_signature_uses_timestamp_method_and_path() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    timestamp = "1703123456789"
    method = "GET"
    path = "/trade-api/v2/portfolio/balance"

    signature = create_kalshi_signature(private_key, timestamp, method, path)

    private_key.public_key().verify(
        base64.b64decode(signature),
        f"{timestamp}{method}{path}".encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_auth_headers_include_required_kalshi_fields() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client = KalshiClient(
        base_url="https://external-api.demo.kalshi.co/trade-api/v2",
        api_key_id="fake-key-id",
        private_key=private_key,
        timestamp_ms_fn=lambda: "1703123456789",
    )

    headers = client.auth_headers("GET", "/markets?limit=5")

    assert headers["KALSHI-ACCESS-KEY"] == "fake-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1703123456789"
    assert "KALSHI-ACCESS-SIGNATURE" in headers
