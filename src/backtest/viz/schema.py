"""Standardized backtest result containers for the visualization suite.

A `BacktestResult` is the single input type consumed by every plot, metric,
report and comparison function in `src.backtest.viz`. It accepts rich inputs
(equity/cash/positions/prices/probabilities/benchmark) but every optional
field degrades gracefully: plots that need missing data simply return None
and are skipped in reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - tzdata missing fallback
    NY_TZ = "America/New_York"

UTC = "UTC"

_COLUMN_ALIASES = {
    "entry_time": "signal_timestamp",
    "open_timestamp": "signal_timestamp",
    "timestamp": "signal_timestamp",
    "exit_time": "exit_timestamp",
    "close_timestamp": "exit_timestamp",
    "profit": "net_pnl",
    "pnl": "net_pnl",
    "realized_pnl": "net_pnl",
    "ticker": "market_ticker",
}

_TOD_BOUNDS = [(8, "before 8 AM"), (10, "8-10 AM"), (12, "10 AM-12 PM"),
               (14, "12-2 PM"), (16, "2-4 PM"), (18, "4-6 PM")]
_TOD_ORDER = [label for _, label in _TOD_BOUNDS] + ["after 6 PM"]

_EDGE_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, np.inf]
_EDGE_LABELS = ["0-2%", "2-5%", "5-10%", "10-15%", "15%+"]

_PRICE_BINS = [0.0, 0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
_PRICE_LABELS = ["1-5c", "5-15c", "15-30c", "30-45c", "45-60c", "60-75c", "75-90c", "90c+"]


def tod_bucket(hour: int) -> str:
    for bound, label in _TOD_BOUNDS:
        if hour < bound:
            return label
    return "after 6 PM"


TOD_ORDER = _TOD_ORDER
EDGE_BINS = _EDGE_BINS
EDGE_LABELS = _EDGE_LABELS
PRICE_BINS = _PRICE_BINS
PRICE_LABELS = _PRICE_LABELS


def to_utc(ts_like) -> pd.Series | pd.DatetimeIndex:
    parsed = pd.to_datetime(ts_like, utc=True, errors="coerce")
    return parsed


@dataclass
class BacktestResult:
    """Container for one backtest run.

    Required: at least one of `equity` or `trades` (equity is reconstructed).
    All other fields are optional and handled gracefully when missing.
    """

    name: str = "backtest"
    initial_capital: float = 1000.0
    equity: pd.Series | None = None
    cash: pd.Series | None = None
    benchmark: pd.Series | None = None
    trades: pd.DataFrame | None = None
    positions: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    probabilities: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ core
    def __post_init__(self) -> None:
        if self.equity is not None:
            self.equity = _as_equity_series(self.equity, self.initial_capital)
        if self.benchmark is not None:
            self.benchmark = _as_equity_series(self.benchmark, self.benchmark.iloc[0] if len(self.benchmark) else 1.0)
        if self.cash is not None:
            self.cash = _as_equity_series(self.cash, np.nan)
        if self.trades is not None and not isinstance(self.trades, pd.DataFrame):
            self.trades = pd.DataFrame(self.trades)
        if self.positions is not None and not isinstance(self.positions, pd.DataFrame):
            self.positions = pd.DataFrame(self.positions)
        if self.prices is not None and not isinstance(self.prices, pd.DataFrame):
            self.prices = pd.DataFrame(self.prices)
        if self.probabilities is not None and not isinstance(self.probabilities, pd.DataFrame):
            self.probabilities = pd.DataFrame(self.probabilities)

    @classmethod
    def from_ledger(
        cls,
        ledger: pd.DataFrame,
        initial_capital: float = 1000.0,
        name: str = "backtest",
        meta: dict[str, Any] | None = None,
    ) -> "BacktestResult":
        """Build a result directly from an engine ledger (`outputs/backtests/trades.csv`)."""
        return cls(trades=ledger.copy(), initial_capital=initial_capital, name=name,
                   meta=dict(meta or {}))

    # ------------------------------------------------------------ availability
    @property
    def has_trades(self) -> bool:
        return self.trades is not None and len(self.trades) > 0

    @property
    def has_prices(self) -> bool:
        return self.prices is not None and len(self.prices) > 0

    @property
    def has_benchmark(self) -> bool:
        return self.benchmark is not None and len(self.benchmark) > 1

    @property
    def has_cash(self) -> bool:
        return self.cash is not None and len(self.cash) > 0

    @property
    def has_positions(self) -> bool:
        return self.positions is not None and len(self.positions) > 0

    def has_column(self, col: str) -> bool:
        return self.has_trades and col in self.trades.columns and self.trades[col].notna().any()

    # ---------------------------------------------------------------- caching
    def _cached(self, key: str, builder):
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    # ------------------------------------------------------------- accessors
    def prepared_trades(self) -> pd.DataFrame:
        """Trades normalized + enriched with derived analysis columns."""
        return self._cached("prepared", lambda: prepare_trades(self))

    def equity_series(self) -> pd.Series:
        """Equity curve; reconstructed from trades when not supplied."""
        return self._cached("equity", lambda: build_equity(self))

    def cash_series(self) -> pd.Series | None:
        if self.has_cash:
            return self.cash
        return self._cached("cash_derived", lambda: _build_cash_from_trades(self)) if self.has_trades else None

    def returns_series(self, freq: str = "D") -> pd.Series:
        return self._cached(("returns", freq), lambda: daily_returns(self.equity_series(), freq))

    def benchmark_series(self) -> pd.Series | None:
        return self.benchmark if self.has_benchmark else None

    # ------------------------------------------------------------- filtering
    def filter(
        self,
        start=None,
        end=None,
        city: str | list[str] | None = None,
        market: str | list[str] | None = None,
        event: str | list[str] | None = None,
        strategy: str | list[str] | None = None,
        model: str | list[str] | None = None,
        side: str | list[str] | None = None,
        min_edge: float | None = None,
        max_edge: float | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        win_only: bool = False,
        loss_only: bool = False,
        **column_filters: Any,
    ) -> "BacktestResult":
        """Filter by date range and/or trade characteristics.

        Equity/exposure views are recomputed from the filtered trades so all
        downstream plots stay consistent.
        """
        t = self.prepared_trades()
        mask = pd.Series(True, index=t.index)
        if start is not None:
            mask &= t["signal_ts"] >= to_utc(start)
        if end is not None:
            mask &= t["signal_ts"] <= to_utc(end)
        for col, value in (
            ("city", city), ("market_ticker", market), ("event_ticker", event),
            ("strategy", strategy), ("model_name", model), ("side_norm", side),
        ):
            if value is not None:
                values = [value] if isinstance(value, str) else list(value)
                mask &= t[col].astype(str).isin([str(v) for v in values])
        if min_edge is not None:
            mask &= t["predicted_edge"].fillna(-np.inf) >= min_edge
        if max_edge is not None:
            mask &= t["predicted_edge"].fillna(np.inf) <= max_edge
        if min_price is not None:
            mask &= t["entry_price"].fillna(-np.inf) >= min_price
        if max_price is not None:
            mask &= t["entry_price"].fillna(np.inf) <= max_price
        if win_only:
            mask &= t["win"]
        if loss_only:
            mask &= ~t["win"]
        for col, value in column_filters.items():
            col = col[:-3] if col.endswith("_in") else col
            if col in t.columns:
                values = [value] if isinstance(value, (str, int, float)) else list(value)
                mask &= t[col].isin(values)
        filtered = t[mask]
        out = BacktestResult(
            name=self.name + " (filtered)",
            initial_capital=self.initial_capital,
            equity=self._slice_equity(start, end),
            benchmark=self._slice_benchmark(start, end),
            trades=filtered.drop(columns=[c for c in filtered.columns if c.startswith("_src")], errors="ignore"),
            meta=dict(self.meta),
        )
        label_bits = []
        if start: label_bits.append(f"from {start}")
        if end: label_bits.append(f"to {end}")
        if city: label_bits.append(f"city={city}")
        if market: label_bits.append(f"market={market}")
        if strategy: label_bits.append(f"strategy={strategy}")
        if side: label_bits.append(f"side={side}")
        if min_edge is not None: label_bits.append(f"edge>={min_edge:g}")
        if label_bits:
            out.name = f"{self.name} [{', '.join(label_bits)}]"
        return out

    def _slice_equity(self, start, end) -> pd.Series | None:
        eq = self.equity_series()
        if start is None and end is None:
            return None
        lo = to_utc(start) if start is not None else None
        hi = to_utc(end) if end is not None else None
        mask = pd.Series(True, index=eq.index)
        if lo is not None:
            mask &= eq.index >= lo
        if hi is not None:
            mask &= eq.index <= hi
        sliced = eq[mask]
        return sliced if len(sliced) > 1 else None

    def _slice_benchmark(self, start, end) -> pd.Series | None:
        bench = self.benchmark
        if bench is None:
            return None
        if start is None and end is None:
            return bench
        lo = to_utc(start) if start is not None else None
        hi = to_utc(end) if end is not None else None
        mask = pd.Series(True, index=bench.index)
        if lo is not None:
            mask &= bench.index >= lo
        if hi is not None:
            mask &= bench.index <= hi
        sliced = bench[mask]
        return sliced if len(sliced) > 1 else None


# --------------------------------------------------------------------- builders
def _as_equity_series(obj, first_value: float) -> pd.Series:
    if isinstance(obj, pd.Series):
        s = obj.astype(float).copy()
    else:
        s = pd.Series(np.asarray(obj, dtype=float))
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.DatetimeIndex(to_utc(s.index))
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def build_equity(result: BacktestResult) -> pd.Series:
    if result.equity is not None and len(result.equity) > 0:
        eq = result.equity.copy()
    elif result.has_trades:
        t = result.prepared_trades().sort_values("signal_ts")
        ts_index = pd.DatetimeIndex(t["signal_ts"])
        if "bankroll_after" in t.columns and t["bankroll_after"].notna().any():
            eq = pd.Series(t["bankroll_after"].astype(float).to_numpy(), index=ts_index)
        else:
            cum = t["net_pnl"].astype(float).cumsum()
            eq = pd.Series(result.initial_capital + cum.to_numpy(), index=ts_index)
    else:
        raise ValueError("BacktestResult needs `equity` or non-empty `trades`.")
    eq = pd.Series(eq.to_numpy(), index=pd.DatetimeIndex(eq.index), name=eq.name)
    eq = eq.groupby(level=0).last().sort_index()
    eq.name = result.name
    eq.name = result.name
    first_ts = eq.index.min() - pd.Timedelta(seconds=1)
    head = pd.Series([result.initial_capital], index=pd.DatetimeIndex([first_ts]), name=eq.name)
    return pd.concat([head, eq])


def _build_cash_from_trades(result: BacktestResult) -> pd.Series | None:
    if not result.has_trades:
        return None
    t = result.prepared_trades().sort_values("signal_ts")
    deployed = t["gross_cost"].astype(float)
    returned = deployed + t["net_pnl"].astype(float)
    events = []
    step = result.initial_capital
    idx, vals = [], []
    for ts_in, ts_out, cost, back in zip(t["signal_ts"], t["exit_ts"], deployed, returned):
        idx.append(ts_in); vals.append(step - cost)
        idx.append(ts_out); vals.append(step - cost + back)
        step = step - cost + back
    if not idx:
        return None
    s = pd.Series(vals, index=pd.DatetimeIndex(idx)).groupby(level=0).last().sort_index()
    return s.resample("h").last().ffill()


def daily_returns(equity: pd.Series, freq: str = "D") -> pd.Series:
    eod = equity.resample(freq).last().dropna()
    rets = eod.pct_change().dropna()
    rets.name = "returns"
    return rets


def prepare_trades(result: BacktestResult) -> pd.DataFrame:
    """Normalize raw ledger into the canonical enriched trade frame used everywhere."""
    df = (result.trades if result.trades is not None else pd.DataFrame()).copy()
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})
    if df.empty:
        return _empty_prepared()
    for col, default in (
        ("net_pnl", np.nan), ("gross_pnl", np.nan), ("fees", 0.0), ("contracts", np.nan),
        ("entry_price", np.nan), ("gross_cost", np.nan), ("predicted_edge", np.nan),
        ("model_probability", np.nan), ("settlement", np.nan), ("city", "unknown"),
        ("market_ticker", "unknown"), ("event_ticker", "unknown"), ("bucket_label", "unknown"),
        ("model_name", "unknown"), ("strategy", "unknown"), ("side", "BUY_YES"),
        ("target_date", None), ("entry_bid", np.nan), ("entry_ask", np.nan),
        ("volume", np.nan), ("exit_price", np.nan),
    ):
        if col not in df.columns:
            df[col] = default
    df["signal_ts"] = pd.DatetimeIndex(to_utc(df["signal_timestamp"]))
    if "exit_timestamp" in df.columns:
        exit_parsed = pd.DatetimeIndex(to_utc(df["exit_timestamp"]))
    else:
        exit_parsed = pd.DatetimeIndex([pd.NaT] * len(df))
    target_dt = pd.to_datetime(df.get("target_date"), errors="coerce")
    settle_est = target_dt + pd.Timedelta(hours=19)
    try:
        settle_est = settle_est.dt.tz_localize(NY_TZ).dt.tz_convert("UTC")
    except Exception:
        settle_est = settle_est.dt.tz_localize("UTC")
    settle_est = pd.DatetimeIndex(settle_est)
    fallback = np.where(pd.isna(exit_parsed), settle_est, exit_parsed)
    df["exit_ts"] = pd.DatetimeIndex(pd.to_datetime(fallback, utc=True))
    df["holding_days"] = (df["exit_ts"] - df["signal_ts"]).dt.total_seconds() / 86400.0
    df["holding_days"] = df["holding_days"].clip(lower=0)
    df["win"] = df["net_pnl"] > 0
    df["side_norm"] = df["side"].astype(str).str.upper().replace({"SELL_YES": "SELL_YES"})
    direction = np.where(df["side_norm"].isin(["BUY_NO", "SELL_YES"]), -1.0, 1.0)
    df["direction"] = direction
    mid = (pd.to_numeric(df["entry_bid"], errors="coerce") + pd.to_numeric(df["entry_ask"], errors="coerce")) / 2.0
    entry = pd.to_numeric(df["entry_price"], errors="coerce")
    df["mid_entry"] = mid
    df["spread_entry"] = pd.to_numeric(df["entry_ask"], errors="coerce") - pd.to_numeric(df["entry_bid"], errors="coerce")
    slip_pc = (entry - mid * direction).abs()
    slip_pc = slip_pc.where(mid.notna())
    df["slippage_per_contract"] = slip_pc
    df["slippage_usd"] = slip_pc * pd.to_numeric(df["contracts"], errors="coerce")
    df["notional"] = entry * pd.to_numeric(df["contracts"], errors="coerce")
    df["signal_date"] = df["signal_ts"].dt.tz_convert(NY_TZ).dt.date
    ny_hour = df["signal_ts"].dt.tz_convert(NY_TZ).dt.hour
    df["hour_ny"] = ny_hour
    df["tod_bucket"] = ny_hour.map(lambda h: tod_bucket(int(h)) if pd.notna(h) else "unknown")
    df["weekday_ny"] = df["signal_ts"].dt.tz_convert(NY_TZ).dt.day_name()
    df["month_period"] = df["signal_ts"].dt.tz_convert(NY_TZ).dt.strftime("%Y-%m")
    df["days_to_settle"] = (df["exit_ts"].dt.tz_convert(NY_TZ).dt.normalize()
                            - df["signal_ts"].dt.tz_convert(NY_TZ).dt.normalize()).dt.days.clip(lower=0)
    df["price_bin"] = pd.cut(entry, bins=_PRICE_BINS, labels=_PRICE_LABELS, right=False)
    edge = pd.to_numeric(df["predicted_edge"], errors="coerce")
    df["edge_bin"] = pd.cut(edge, bins=_EDGE_BINS, labels=_EDGE_LABELS, right=False)
    df["edge_bin"] = df["edge_bin"].cat.add_categories(["unknown"]).fillna("unknown")
    df["temp_bucket"] = _temp_bucket_series(df)
    return df


def _temp_bucket_series(df: pd.DataFrame) -> pd.Series:
    label = df["bucket_label"].astype(str) if "bucket_label" in df.columns else \
        pd.Series("unknown", index=df.index)
    known = label.str.lower().ne("nan") & label.ne("unknown") & label.str.strip().ne("")
    lower = df.get("bucket_lower")
    upper = df.get("bucket_upper")
    if lower is None or upper is None:
        return label.where(known, "unknown")
    lo = pd.to_numeric(lower, errors="coerce").reindex(df.index)
    up = pd.to_numeric(upper, errors="coerce").reindex(df.index)

    def fmt(lo_, up_):
        if pd.isna(lo_) and pd.isna(up_):
            return "unknown"
        if pd.isna(up_):
            return f">= {lo_:g}"
        if pd.isna(lo_):
            return f"< {up_:g}"
        return f"{lo_:g}-{up_:g}"

    formatted = pd.Series([fmt(a, b) for a, b in zip(lo, up)], index=df.index)
    return label.where(known, formatted)


def _empty_prepared() -> pd.DataFrame:
    cols = ["net_pnl", "gross_pnl", "fees", "contracts", "entry_price", "gross_cost",
            "predicted_edge", "model_probability", "settlement", "city", "market_ticker",
            "event_ticker", "bucket_label", "model_name", "strategy", "side", "target_date",
            "entry_bid", "entry_ask", "volume", "exit_price", "signal_ts", "exit_ts",
            "holding_days", "win", "side_norm", "direction", "mid_entry", "spread_entry",
            "slippage_per_contract", "slippage_usd", "notional", "signal_date", "hour_ny",
            "tod_bucket", "weekday_ny", "month_period", "days_to_settle", "price_bin",
            "edge_bin", "temp_bucket"]
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in []}).reindex(columns=cols)


def load_result_from_dir(directory, name: str | None = None, initial_capital: float = 1000.0) -> BacktestResult:
    """Load a BacktestResult from a directory containing trades.csv / trades.parquet."""
    import logging
    from pathlib import Path
    logger = logging.getLogger(__name__)
    directory = Path(directory)
    path = None
    for candidate in ("trades.parquet", "trades.csv"):
        if (directory / candidate).exists():
            path = directory / candidate
            break
    if path is None:
        raise FileNotFoundError(f"No trades file found in {directory}")
    if path.suffix == ".parquet":
        ledger = pd.read_parquet(path)
    else:
        ledger = pd.read_csv(path)
    return BacktestResult.from_ledger(ledger, initial_capital=initial_capital,
                                      name=name or directory.name)
