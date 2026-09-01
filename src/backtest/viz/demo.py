"""Demo + verification runner for the visualization suite, built entirely on
real pipeline artifacts (no fabricated market data).

Run from the repo root::

    python -m src.backtest.viz.demo

Inputs (produced by the normal pipeline; see BACKTEST_README.md):
  - outputs/backtests/trades.csv             engine ledger, base sizing
  - outputs/backtests/trades_high_risk.parquet  engine ledger, high-risk sizing
  - data/kalshi/processed/historical_candles_processed.csv
        real Kalshi hourly candlesticks -> price paths for MAE/MFE and
        trade-on-price overlays

Writes standalone HTML reports plus a comparison dashboard to
`outputs/backtests/viz_demo/`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.viz import BacktestResult, get_registered_plots, run_plot
from src.backtest.viz.compare import create_comparison_report
from src.backtest.viz.report import create_backtest_report

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGERS = {
    "base_sizing": REPO_ROOT / "outputs" / "backtests" / "trades.csv",
    "high_risk_sizing": REPO_ROOT / "outputs" / "backtests" / "trades_high_risk.parquet",
}


def _available_default_ledgers() -> dict[str, Path]:
    """Only keep default ledgers that actually exist (optional variants degrade gracefully)."""
    return {name: path for name, path in DEFAULT_LEDGERS.items() if path.exists()}
DEFAULT_CANDLES = REPO_ROOT / "data" / "kalshi" / "processed" / "historical_candles_processed.csv"


def load_candle_prices(candles_path: str | Path | None = None,
                       market_tickers: list[str] | None = None,
                       price_col: tuple[str, str] = ("yes_bid_close", "yes_ask_close")) -> pd.DataFrame | None:
    """Real Kalshi candlesticks -> long frame [timestamp, market_ticker, price].

    Price is the mid of the hourly yes bid/ask closes; rows without a
    two-sided quote are dropped.
    """
    path = Path(candles_path) if candles_path else DEFAULT_CANDLES
    if not path.exists():
        logger.warning("Candles file not found (%s); price-dependent plots will be skipped", path)
        return None
    usecols = ["timestamp", "market_ticker", *price_col]
    px = pd.read_csv(path, usecols=usecols)
    if market_tickers is not None:
        px = px[px["market_ticker"].isin(set(market_tickers))]
    bid = pd.to_numeric(px[price_col[0]], errors="coerce")
    ask = pd.to_numeric(px[price_col[1]], errors="coerce")
    px["price"] = (bid + ask) / 2.0
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True, errors="coerce")
    px = px.dropna(subset=["timestamp", "price"])
    return px[["timestamp", "market_ticker", "price"]].reset_index(drop=True)


def load_real_backtest(name: str, ledger_path: str | Path,
                       candles_path: str | Path | None = None,
                       initial_capital: float = 1000.0) -> BacktestResult:
    """Build a BacktestResult from a real engine ledger + real candle history."""
    path = Path(ledger_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first with the standard pipeline, "
            f"e.g.: python scripts/run_kalshi_backtest.py --strategy A --threshold 0.02 "
            f"(see BACKTEST_README.md)")
    result = BacktestResult.from_ledger(
        pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path),
        initial_capital=initial_capital,
        name=name,
        meta={"fee_rate": 0.07, "ledger_source": str(path), "candles_source": str(candles_path or DEFAULT_CANDLES)},
    )
    prices = load_candle_prices(candles_path, market_tickers=result.trades["market_ticker"].astype(str).unique().tolist())
    if prices is not None and len(prices):
        result.prices = prices
    return result


def verify_all_plots(result: BacktestResult) -> dict[str, object]:
    """Build every registered plot against a real result; report status."""
    built, skipped, failed = [], [], []
    log_lines: list[str] = []
    for spec in get_registered_plots():
        fig = run_plot(spec, result)
        label = f"{spec.section:>11}/{spec.name}"
        if fig is None:
            skipped.append(spec.name)
            log_lines.append(f"SKIP   {label}")
        elif len(getattr(fig, "data", [])) == 0:
            failed.append(spec.name)
            log_lines.append(f"EMPTY  {label}")
        else:
            built.append(spec.name)
            log_lines.append(f"OK     {label:<40} traces={len(fig.data)}")
    return {"built": built, "skipped": skipped, "empty_or_failed": failed, "log": log_lines}


def main(output_dir: str | Path | None = None,
         ledgers: dict[str, Path] | None = None,
         candles_path: str | Path | None = None,
         filter_example: dict | None = None) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(output_dir) if output_dir else REPO_ROOT / "outputs" / "backtests" / "viz_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_map = ledgers or _available_default_ledgers()

    runs: dict[str, BacktestResult] = {}
    for name, path in ledger_map.items():
        runs[name] = load_real_backtest(name, path, candles_path=candles_path)
        n_prices = len(runs[name].prices) if runs[name].prices is not None else 0
        logger.info("loaded '%s': %d trades, %d candle rows",
                    name, len(runs[name].trades), n_prices)

    primary_name, primary = next(iter(runs.items()))
    verification = verify_all_plots(primary)
    for line in verification["log"]:
        logger.info("%s", line)
    logger.info("plots built=%d skipped=%d empty/failed=%d",
                len(verification["built"]), len(verification["skipped"]),
                len(verification["empty_or_failed"]))

    figures = create_backtest_report(primary, output_path=out_dir / f"report_{primary_name}.html")

    filter_args = filter_example if filter_example is not None \
        else dict(min_edge=0.10, start="2025-01-01")
    filtered = primary.filter(**filter_args)
    create_backtest_report(filtered,
                           output_path=out_dir / "report_filtered_subset.html",
                           title=f"Filtered backtest report - {primary.name}")

    if len(runs) >= 2:
        table, cmp_figs = create_comparison_report(
            {name: res for name, res in runs.items()},
            output_path=out_dir / "comparison_report.html")
        table.to_csv(out_dir / "comparison_table.csv", index=False)
    else:
        table = None

    summary = {
        "output_dir": str(out_dir),
        "runs": {n: int(len(r.trades)) for n, r in runs.items()},
        "verification": {k: v for k, v in verification.items() if k != "log"},
        "figures_in_report": {sec: list(figs.keys()) for sec, figs in figures.items()},
        "filter_example": filter_args,
    }
    print("\n=== demo outputs ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nOpen {out_dir / f'report_{primary_name}.html'} "
          f"and {out_dir / 'comparison_report.html'} in a browser.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest viz demo on real pipeline artifacts")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--candles", default=None, help="override historical_candles_processed.csv path")
    args = parser.parse_args()
    main(output_dir=args.output_dir, candles_path=args.candles)
