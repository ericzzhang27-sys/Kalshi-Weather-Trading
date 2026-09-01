from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence

import websockets

from src.research.interfaces import FixedPointLevel, MarketSnapshot
from src.trading.kalshi_client import create_kalshi_signature


WS_PATH = "/trade-api/ws/v2"
PRODUCTION_WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
DEMO_WS_URL = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"


@dataclass(frozen=True)
class SequenceGap:
    sid: int
    expected_sequence: int
    received_sequence: int
    market_ticker: str


def websocket_auth_headers(api_key_id: str, private_key: Any, *, timestamp_ms: str | None = None) -> dict[str, str]:
    timestamp = timestamp_ms or str(int(time.time() * 1000))
    signature = create_kalshi_signature(private_key, timestamp, "GET", WS_PATH)
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


class OrderbookReplayState:
    """Exact fixed-point book state with fail-closed sequence-gap recovery."""

    def __init__(self) -> None:
        self.books: dict[str, dict[str, dict[Decimal, Decimal]]] = {}
        self.last_sequence_by_sid: dict[int, int] = {}
        self.recovering_sids: set[int] = set()

    def process(self, payload: Mapping[str, Any]) -> SequenceGap | None:
        message_type = str(payload.get("type", ""))
        if message_type not in {"orderbook_snapshot", "orderbook_delta"}:
            return None
        sid = int(payload.get("sid", 0))
        sequence = int(payload.get("seq", 0))
        message = payload.get("msg", {})
        if not isinstance(message, Mapping):
            raise ValueError("orderbook message body must be an object")
        ticker = str(message.get("market_ticker", ""))
        if not ticker:
            raise ValueError("orderbook message missing market_ticker")
        if message_type == "orderbook_snapshot":
            self.books[ticker] = {
                "yes": _parse_snapshot_levels(message.get("yes_dollars_fp", message.get("yes_dollars", []))),
                "no": _parse_snapshot_levels(message.get("no_dollars_fp", message.get("no_dollars", []))),
            }
            self.last_sequence_by_sid[sid] = sequence
            self.recovering_sids.discard(sid)
            return None
        previous = self.last_sequence_by_sid.get(sid)
        expected = sequence if previous is None else previous + 1
        if previous is None or sid in self.recovering_sids or sequence != expected:
            self.recovering_sids.add(sid)
            return SequenceGap(sid, expected, sequence, ticker)
        side = str(message.get("side", "")).lower()
        if side not in {"yes", "no"}:
            raise ValueError("orderbook delta side must be yes or no")
        price = _decimal(message.get("price_dollars"), "price_dollars")
        delta = _decimal(message.get("delta_fp", message.get("delta")), "delta_fp")
        if not Decimal("0") <= price <= Decimal("1"):
            raise ValueError("orderbook delta price must be in [0, 1]")
        book = self.books.get(ticker)
        if book is None:
            self.recovering_sids.add(sid)
            return SequenceGap(sid, expected, sequence, ticker)
        updated = book[side].get(price, Decimal("0")) + delta
        if updated < 0:
            self.recovering_sids.add(sid)
            return SequenceGap(sid, expected, sequence, ticker)
        if updated == 0:
            book[side].pop(price, None)
        else:
            book[side][price] = updated
        self.last_sequence_by_sid[sid] = sequence
        return None

    def market_snapshot(
        self,
        ticker: str,
        *,
        sequence_number: int,
        timestamp_ms: int,
        raw_payload_hash: str,
        lifecycle_state: str = "active",
    ) -> MarketSnapshot:
        book = self.books.get(ticker)
        if book is None:
            raise KeyError(ticker)
        yes = tuple(FixedPointLevel(price, quantity) for price, quantity in sorted(book["yes"].items(), reverse=True))
        no = tuple(FixedPointLevel(price, quantity) for price, quantity in sorted(book["no"].items(), reverse=True))
        return MarketSnapshot(
            ticker=ticker,
            event_ticker=ticker.rsplit("-", 1)[0],
            timestamp_ms=int(timestamp_ms),
            sequence_number=int(sequence_number),
            yes_bids=yes,
            no_bids=no,
            lifecycle_state=lifecycle_state,
            raw_payload_hash=raw_payload_hash,
        )


class OrderbookDepthStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    received_at_utc TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    sid INTEGER,
                    sequence_number INTEGER,
                    market_ticker TEXT,
                    payload_hash TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    sequence_status TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_replay_order
                ON messages(sid, sequence_number, received_at_utc);
                CREATE TABLE IF NOT EXISTS snapshots (
                    received_at_utc TEXT NOT NULL,
                    market_ticker TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    price_dollars TEXT NOT NULL,
                    quantity_fp TEXT NOT NULL,
                    payload_hash TEXT NOT NULL
                );
                """
            )

    def persist(
        self,
        raw_message: str,
        payload: Mapping[str, Any],
        *,
        sequence_status: str,
        state: OrderbookReplayState,
        received_at: datetime | None = None,
    ) -> str:
        timestamp = (received_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        payload_hash = hashlib.sha256(raw_message.encode("utf-8")).hexdigest()
        message = payload.get("msg", {}) if isinstance(payload.get("msg", {}), Mapping) else {}
        ticker = str(message.get("market_ticker", "")) or None
        sid = int(payload["sid"]) if payload.get("sid") is not None else None
        sequence = int(payload["seq"]) if payload.get("seq") is not None else None
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (timestamp, str(payload.get("type", "unknown")), sid, sequence, ticker, payload_hash, raw_message, sequence_status),
            )
            if str(payload.get("type")) == "orderbook_snapshot" and ticker in state.books and sequence is not None:
                records = [
                    (timestamp, ticker, sequence, side, str(price), str(quantity), payload_hash)
                    for side in ("yes", "no")
                    for price, quantity in state.books[ticker][side].items()
                ]
                connection.executemany("INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", records)
            connection.commit()
        return payload_hash


class ShadowOrderbookStream:
    """Authenticated depth recorder. It contains no order-submission methods."""

    def __init__(
        self,
        *,
        ws_url: str,
        api_key_id: str,
        private_key: Any,
        market_tickers: Sequence[str],
        store: OrderbookDepthStore,
    ) -> None:
        self.ws_url = ws_url
        self.api_key_id = api_key_id
        self.private_key = private_key
        self.market_tickers = tuple(dict.fromkeys(str(item) for item in market_tickers if str(item)))
        if not self.market_tickers:
            raise ValueError("at least one market ticker is required")
        self.store = store
        self.state = OrderbookReplayState()
        self._command_id = 1

    async def run(self, *, duration_seconds: float | None = None) -> None:
        headers = websocket_auth_headers(self.api_key_id, self.private_key)
        started = time.monotonic()
        async with websockets.connect(self.ws_url, additional_headers=headers, ping_interval=20, ping_timeout=20) as websocket:
            await websocket.send(json.dumps({
                "id": self._next_command_id(),
                "cmd": "subscribe",
                "params": {"channels": ["orderbook_delta"], "market_tickers": list(self.market_tickers)},
            }))
            async for raw in websocket:
                payload = json.loads(raw)
                gap = self.state.process(payload)
                status = "sequence_gap" if gap else "ok"
                self.store.persist(raw, payload, sequence_status=status, state=self.state)
                if gap:
                    await websocket.send(json.dumps({
                        "id": self._next_command_id(),
                        "cmd": "update_subscription",
                        "params": {
                            "sids": [gap.sid],
                            "market_tickers": [gap.market_ticker],
                            "action": "get_snapshot",
                        },
                    }))
                if duration_seconds is not None and time.monotonic() - started >= duration_seconds:
                    break

    def _next_command_id(self) -> int:
        result = self._command_id
        self._command_id += 1
        return result


def _parse_snapshot_levels(raw: Any) -> dict[Decimal, Decimal]:
    if not isinstance(raw, list):
        raise ValueError("snapshot levels must be a list")
    result: dict[Decimal, Decimal] = {}
    for level in raw:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            raise ValueError("snapshot levels must contain price/quantity pairs")
        price = _decimal(level[0], "price")
        quantity = _decimal(level[1], "quantity")
        if not Decimal("0") <= price <= Decimal("1") or quantity < 0:
            raise ValueError("invalid snapshot fixed-point level")
        if quantity:
            result[price] = quantity
    return result


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid fixed-point {name}") from exc
    if not result.is_finite():
        raise ValueError(f"fixed-point {name} must be finite")
    return result
