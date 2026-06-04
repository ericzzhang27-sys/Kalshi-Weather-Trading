from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.trading.orderbook import (
    fetch_orderbooks,
    normalize_orderbook,
    summarize_orderbook,
)


FETCHED_AT = datetime(2026, 6, 3, 15, 30, tzinfo=timezone.utc)


def test_normalize_orderbook_infers_opposite_side_asks() -> None:
    payload = {
        "orderbook_fp": {
            "yes_dollars": [[0.40, 10], [0.35, 5]],
            "no_dollars": [[0.55, 7], [0.50, 3]],
        }
    }

    orderbook = normalize_orderbook("KXHIGHNY-26JUN02-B72.5", payload, FETCHED_AT)

    assert set(orderbook["outcome_side"]) == {"YES", "NO"}
    assert set(orderbook["quote_type"]) == {"bid", "ask"}

    yes_ask = orderbook[(orderbook["outcome_side"] == "YES") & (orderbook["quote_type"] == "ask")]
    no_ask = orderbook[(orderbook["outcome_side"] == "NO") & (orderbook["quote_type"] == "ask")]

    assert yes_ask.iloc[0]["price_dollars"] == pytest.approx(0.45)
    assert yes_ask.iloc[0]["source_book_side"] == "no_dollars"
    assert no_ask.iloc[0]["price_dollars"] == pytest.approx(0.60)
    assert no_ask.iloc[0]["source_book_side"] == "yes_dollars"
    assert yes_ask.iloc[0]["cumulative_size"] == 7


def test_summarize_orderbook_computes_best_prices_and_spreads() -> None:
    orderbook = normalize_orderbook(
        "KXHIGHNY-26JUN02-B72.5",
        {
            "yes_dollars": [[0.40, 10]],
            "no_dollars": [[0.55, 7]],
        },
        FETCHED_AT,
    )

    summary = summarize_orderbook(orderbook, "KXHIGHNY-26JUN02-B72.5", FETCHED_AT)

    row = summary.iloc[0]
    assert row["orderbook_status"] == "OK"
    assert row["best_yes_bid"] == pytest.approx(0.40)
    assert row["best_yes_ask"] == pytest.approx(0.45)
    assert row["yes_spread"] == pytest.approx(0.05)
    assert row["yes_midpoint"] == pytest.approx(0.425)
    assert row["yes_bid_depth"] == 10
    assert row["yes_ask_depth"] == 7


def test_summarize_orderbook_rejects_stale_book_when_threshold_is_supplied() -> None:
    orderbook = normalize_orderbook(
        "KXHIGHNY-26JUN02-B72.5",
        {
            "yes_dollars": [[0.40, 10]],
            "no_dollars": [[0.55, 7]],
        },
        FETCHED_AT,
    )

    summary = summarize_orderbook(
        orderbook,
        "KXHIGHNY-26JUN02-B72.5",
        FETCHED_AT,
        evaluated_at=datetime(2026, 6, 3, 15, 40, tzinfo=timezone.utc),
        max_staleness_seconds=300,
    )

    assert summary.iloc[0]["orderbook_status"] == "NO_TRADE"
    assert summary.iloc[0]["orderbook_reason"] == "stale_orderbook"


def test_summarize_orderbook_rejects_unusually_wide_spread() -> None:
    orderbook = normalize_orderbook(
        "KXHIGHNY-26JUN02-B72.5",
        {
            "yes_dollars": [[0.10, 10]],
            "no_dollars": [[0.55, 7]],
        },
        FETCHED_AT,
    )

    summary = summarize_orderbook(
        orderbook,
        "KXHIGHNY-26JUN02-B72.5",
        FETCHED_AT,
        max_spread_dollars=0.25,
    )

    assert summary.iloc[0]["orderbook_status"] == "NO_TRADE"
    assert summary.iloc[0]["orderbook_reason"] == "unusually_wide_orderbook"


def test_empty_orderbook_has_empty_status() -> None:
    orderbook = normalize_orderbook("KXHIGHNY-26JUN02-B72.5", {"yes_dollars": [], "no_dollars": []}, FETCHED_AT)
    summary = summarize_orderbook(orderbook, "KXHIGHNY-26JUN02-B72.5", FETCHED_AT)

    assert isinstance(orderbook, pd.DataFrame)
    assert orderbook.empty
    assert summary.iloc[0]["orderbook_status"] == "EMPTY"


def test_malformed_ladder_rejected() -> None:
    with pytest.raises(ValueError, match=r"yes_dollars levels must be \[price, size\] pairs"):
        normalize_orderbook("KXHIGHNY-26JUN02-B72.5", {"yes_dollars": [0.40], "no_dollars": []}, FETCHED_AT)


def test_fetch_orderbooks_uses_client_and_builds_snapshot() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, int], bool]] = []

        def get(self, path: str, params: dict[str, int], auth: bool = False) -> dict[str, object]:
            self.calls.append((path, params, auth))
            return {"yes_dollars": [[0.40, 10]], "no_dollars": [[0.55, 7]]}

    client = FakeClient()
    snapshot = fetch_orderbooks(client, ["KXHIGHNY-26JUN02-B72.5"], depth=3, auth=True, fetched_at=FETCHED_AT)

    assert client.calls == [("/markets/KXHIGHNY-26JUN02-B72.5/orderbook", {"depth": 3}, True)]
    assert len(snapshot.orderbook) == 4
    assert snapshot.summary.iloc[0]["orderbook_status"] == "OK"
