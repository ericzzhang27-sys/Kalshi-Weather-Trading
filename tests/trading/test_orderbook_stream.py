from __future__ import annotations

from decimal import Decimal
import json

from src.trading.orderbook_stream import OrderbookDepthStore, OrderbookReplayState


def _snapshot(seq: int = 10) -> dict:
    return {
        "type": "orderbook_snapshot", "sid": 2, "seq": seq,
        "msg": {
            "market_ticker": "KXHIGHNY-26SEP01-B80.5",
            "yes_dollars_fp": [["0.4000", "10.00"]],
            "no_dollars_fp": [["0.5500", "7.00"]],
        },
    }


def test_orderbook_state_applies_exact_fixed_point_delta() -> None:
    state = OrderbookReplayState()
    assert state.process(_snapshot()) is None
    gap = state.process({
        "type": "orderbook_delta", "sid": 2, "seq": 11,
        "msg": {"market_ticker": "KXHIGHNY-26SEP01-B80.5", "price_dollars": "0.4000", "delta_fp": "-3.25", "side": "yes"},
    })
    assert gap is None
    assert state.books["KXHIGHNY-26SEP01-B80.5"]["yes"][Decimal("0.4000")] == Decimal("6.75")


def test_orderbook_state_detects_gap_and_waits_for_snapshot() -> None:
    state = OrderbookReplayState()
    state.process(_snapshot())
    gap = state.process({
        "type": "orderbook_delta", "sid": 2, "seq": 12,
        "msg": {"market_ticker": "KXHIGHNY-26SEP01-B80.5", "price_dollars": "0.4000", "delta_fp": "1.00", "side": "yes"},
    })
    assert gap is not None and gap.expected_sequence == 11
    assert 2 in state.recovering_sids
    assert state.process(_snapshot(seq=13)) is None
    assert 2 not in state.recovering_sids


def test_depth_store_persists_raw_hash_and_snapshot_levels(tmp_path) -> None:
    state = OrderbookReplayState()
    payload = _snapshot()
    state.process(payload)
    raw = json.dumps(payload, sort_keys=True)
    store = OrderbookDepthStore(tmp_path / "depth.sqlite")
    payload_hash = store.persist(raw, payload, sequence_status="ok", state=state)
    assert len(payload_hash) == 64
    import sqlite3

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 2
