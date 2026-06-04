from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.trading.kalshi_client import KalshiClient


ORDERBOOK_COLUMNS = [
    "fetched_at",
    "ticker",
    "outcome_side",
    "quote_type",
    "price_dollars",
    "size_contracts",
    "level",
    "cumulative_size",
    "source_book_side",
]

ORDERBOOK_SUMMARY_COLUMNS = [
    "fetched_at",
    "evaluated_at",
    "ticker",
    "staleness_seconds",
    "max_staleness_seconds",
    "best_yes_bid",
    "best_yes_bid_size",
    "best_yes_ask",
    "best_yes_ask_size",
    "yes_spread",
    "yes_midpoint",
    "best_no_bid",
    "best_no_bid_size",
    "best_no_ask",
    "best_no_ask_size",
    "no_spread",
    "no_midpoint",
    "yes_bid_depth",
    "yes_ask_depth",
    "no_bid_depth",
    "no_ask_depth",
    "max_spread_dollars",
    "orderbook_status",
    "orderbook_reason",
]


@dataclass(frozen=True)
class OrderbookSnapshot:
    fetched_at: datetime
    orderbook: pd.DataFrame
    summary: pd.DataFrame


def fetch_orderbooks(
    client: KalshiClient,
    tickers: Sequence[str],
    depth: int = 20,
    auth: bool = False,
    fetched_at: datetime | None = None,
    evaluated_at: datetime | None = None,
    max_staleness_seconds: float | None = None,
    max_spread_dollars: float | None = None,
) -> OrderbookSnapshot:
    """
    Fetch and normalize full order books for a list of market tickers.
    """
    unique_tickers = [ticker for ticker in dict.fromkeys(str(item) for item in tickers) if ticker]
    fetched_at_dt = fetched_at or datetime.now(timezone.utc)
    evaluated_at_dt = evaluated_at or fetched_at_dt
    frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []

    for ticker in unique_tickers:
        payload = client.get(
            f"/markets/{ticker}/orderbook",
            params={"depth": int(depth)},
            auth=auth,
        )
        frame = normalize_orderbook(ticker, payload, fetched_at=fetched_at_dt)
        frames.append(frame)
        summaries.append(
            summarize_orderbook(
                frame,
                ticker=ticker,
                fetched_at=fetched_at_dt,
                evaluated_at=evaluated_at_dt,
                max_staleness_seconds=max_staleness_seconds,
                max_spread_dollars=max_spread_dollars,
            )
        )

    orderbook = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ORDERBOOK_COLUMNS)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=ORDERBOOK_SUMMARY_COLUMNS)
    return OrderbookSnapshot(fetched_at=fetched_at_dt, orderbook=orderbook, summary=summary)


def normalize_orderbook(
    ticker: str,
    payload: Mapping[str, Any],
    fetched_at: datetime | None = None,
) -> pd.DataFrame:
    """
    Normalize Kalshi YES/NO bid arrays and infer complementary asks.
    """
    fetched_at_dt = fetched_at or datetime.now(timezone.utc)
    if not isinstance(payload, Mapping):
        raise ValueError("Orderbook payload must be a mapping")

    raw_book = payload.get("orderbook_fp", payload)
    if not isinstance(raw_book, Mapping):
        raise ValueError("Orderbook payload missing orderbook_fp object")

    yes_bids = _parse_levels(raw_book.get("yes_dollars", []), "yes_dollars")
    no_bids = _parse_levels(raw_book.get("no_dollars", []), "no_dollars")
    records: list[dict[str, Any]] = []

    for price, size in yes_bids:
        records.append(_record(fetched_at_dt, ticker, "YES", "bid", price, size, "yes_dollars"))
        records.append(_record(fetched_at_dt, ticker, "NO", "ask", 1.0 - price, size, "yes_dollars"))

    for price, size in no_bids:
        records.append(_record(fetched_at_dt, ticker, "NO", "bid", price, size, "no_dollars"))
        records.append(_record(fetched_at_dt, ticker, "YES", "ask", 1.0 - price, size, "no_dollars"))

    if not records:
        return pd.DataFrame(columns=ORDERBOOK_COLUMNS)

    frame = pd.DataFrame.from_records(records)
    pieces: list[pd.DataFrame] = []
    for (outcome_side, quote_type), group in frame.groupby(["outcome_side", "quote_type"], sort=False):
        ascending = quote_type == "ask"
        ordered = group.sort_values("price_dollars", ascending=ascending, kind="stable").copy()
        ordered["level"] = range(1, len(ordered) + 1)
        ordered["cumulative_size"] = ordered["size_contracts"].cumsum()
        pieces.append(ordered)
    result = pd.concat(pieces, ignore_index=True)
    return result.reindex(columns=ORDERBOOK_COLUMNS)


def summarize_orderbook(
    orderbook: pd.DataFrame,
    ticker: str,
    fetched_at: datetime | None = None,
    evaluated_at: datetime | None = None,
    max_staleness_seconds: float | None = None,
    max_spread_dollars: float | None = None,
) -> pd.DataFrame:
    fetched_at_dt = fetched_at or datetime.now(timezone.utc)
    evaluated_at_dt = evaluated_at or fetched_at_dt
    staleness_seconds = _seconds_between(fetched_at_dt, evaluated_at_dt)
    summary = {
        "fetched_at": fetched_at_dt.isoformat(),
        "evaluated_at": evaluated_at_dt.isoformat(),
        "ticker": ticker,
        "staleness_seconds": staleness_seconds,
        "max_staleness_seconds": _optional_positive_float(max_staleness_seconds),
        "best_yes_bid": None,
        "best_yes_bid_size": None,
        "best_yes_ask": None,
        "best_yes_ask_size": None,
        "yes_spread": None,
        "yes_midpoint": None,
        "best_no_bid": None,
        "best_no_bid_size": None,
        "best_no_ask": None,
        "best_no_ask_size": None,
        "no_spread": None,
        "no_midpoint": None,
        "yes_bid_depth": 0.0,
        "yes_ask_depth": 0.0,
        "no_bid_depth": 0.0,
        "no_ask_depth": 0.0,
        "max_spread_dollars": _optional_positive_float(max_spread_dollars),
        "orderbook_status": "EMPTY",
        "orderbook_reason": "empty_orderbook",
    }
    if orderbook.empty:
        return pd.DataFrame([summary], columns=ORDERBOOK_SUMMARY_COLUMNS)

    for outcome in ["YES", "NO"]:
        for quote_type in ["bid", "ask"]:
            side = orderbook[
                (orderbook["outcome_side"] == outcome)
                & (orderbook["quote_type"] == quote_type)
            ].copy()
            prefix = f"{outcome.lower()}_{quote_type}"
            if side.empty:
                continue
            side = side.sort_values("level", kind="stable")
            summary[f"best_{prefix}"] = float(side.iloc[0]["price_dollars"])
            summary[f"best_{prefix}_size"] = float(side.iloc[0]["size_contracts"])
            summary[f"{outcome.lower()}_{quote_type}_depth"] = float(side["size_contracts"].sum())

    for outcome in ["yes", "no"]:
        bid = summary[f"best_{outcome}_bid"]
        ask = summary[f"best_{outcome}_ask"]
        if bid is not None and ask is not None:
            summary[f"{outcome}_spread"] = float(ask - bid)
            summary[f"{outcome}_midpoint"] = float((bid + ask) / 2.0)

    crossed = any(
        summary[f"{outcome}_spread"] is not None and summary[f"{outcome}_spread"] < -1e-9
        for outcome in ["yes", "no"]
    )
    stale = (
        summary["max_staleness_seconds"] is not None
        and summary["staleness_seconds"] is not None
        and summary["staleness_seconds"] > summary["max_staleness_seconds"]
    )
    wide = any(
        summary[f"{outcome}_spread"] is not None
        and summary["max_spread_dollars"] is not None
        and summary[f"{outcome}_spread"] > summary["max_spread_dollars"] + 1e-9
        for outcome in ["yes", "no"]
    )
    if crossed:
        summary["orderbook_status"] = "NO_TRADE"
        summary["orderbook_reason"] = "crossed_orderbook"
    elif stale:
        summary["orderbook_status"] = "NO_TRADE"
        summary["orderbook_reason"] = "stale_orderbook"
    elif wide:
        summary["orderbook_status"] = "NO_TRADE"
        summary["orderbook_reason"] = "unusually_wide_orderbook"
    else:
        summary["orderbook_status"] = "OK"
        summary["orderbook_reason"] = ""

    return pd.DataFrame([summary], columns=ORDERBOOK_SUMMARY_COLUMNS)


def save_orderbook_snapshot(snapshot: OrderbookSnapshot, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.orderbook.to_csv(path, index=False)


def save_orderbook_summary(snapshot: OrderbookSnapshot, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.summary.to_csv(path, index=False)


def _parse_levels(raw_levels: Any, field_name: str) -> list[tuple[float, float]]:
    if raw_levels is None:
        return []
    if not isinstance(raw_levels, list):
        raise ValueError(f"{field_name} must be a list")
    levels: list[tuple[float, float]] = []
    for raw_level in raw_levels:
        if not isinstance(raw_level, (list, tuple)) or len(raw_level) < 2:
            raise ValueError(f"{field_name} levels must be [price, size] pairs")
        price = float(raw_level[0])
        size = float(raw_level[1])
        if price < 0.0 or price > 1.0:
            raise ValueError(f"{field_name} price outside [0, 1]: {price}")
        if size < 0.0:
            raise ValueError(f"{field_name} size cannot be negative: {size}")
        if size > 0.0:
            levels.append((price, size))
    return levels


def _record(
    fetched_at: datetime,
    ticker: str,
    outcome_side: str,
    quote_type: str,
    price: float,
    size: float,
    source_book_side: str,
) -> dict[str, Any]:
    return {
        "fetched_at": fetched_at.isoformat(),
        "ticker": ticker,
        "outcome_side": outcome_side,
        "quote_type": quote_type,
        "price_dollars": round(float(price), 6),
        "size_contracts": float(size),
        "level": None,
        "cumulative_size": None,
        "source_book_side": source_book_side,
    }


def _seconds_between(start: datetime, end: datetime) -> float:
    start_dt = _ensure_aware_utc(start)
    end_dt = _ensure_aware_utc(end)
    return max(0.0, float((end_dt - start_dt).total_seconds()))


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"Expected a positive finite value, got {value!r}")
    return numeric
