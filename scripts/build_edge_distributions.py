from __future__ import annotations

"""Build PnL and win-rate distributions conditioned on predicted edge.

Usage (from repo root):
    python scripts/build_edge_distributions.py --backtest-dir outputs/backtests_real
    python scripts/build_edge_distributions.py --trades path/to/trades.csv --out-dir outputs/edge_distributions
    python scripts/build_edge_distributions.py --trades a.csv b.csv --labels runA runB

Outputs (written to <backtest-dir> or --out-dir):
    edge_pnl_distribution.csv        per-contract PnL stats per edge bucket
    edge_pnl_histogram.csv           histogram counts of PnL per edge bucket
    edge_winrate_distribution.csv    realized vs implied win rate + Wilson CI
    bootstrap_pnl_samples.csv        bootstrap distribution of mean PnL/contract
    bootstrap_winrate_samples.csv    bootstrap distribution of win rate
    edge_distributions_report.md     human-readable summary
    edge_pnl_distribution.png        histogram + ECDF plot
    edge_winrate_vs_edge.png         win-rate vs implied-probability plot
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.edge_distributions import (
    EDGE_BINS,
    EDGE_LABELS,
    build_all_distributions,
    pnl_histogram_by_edge,
    plot_pnl_distribution,
    plot_winrate_vs_edge,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--backtest-dir", type=Path,
                        help="Backtest run directory containing trades.csv")
    inputs.add_argument("--trades", nargs="+", type=Path,
                        help="One or more trade ledger CSVs")
    parser.add_argument("--out-dir", type=Path,
                        help="Output directory (default: backtest dir, or outputs/edge_distributions)")
    parser.add_argument("--labels", nargs="+",
                        help="Optional names for each input ledger in combined tables")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_ledgers(args: argparse.Namespace) -> list[tuple[str, pd.DataFrame]]:
    ledgers: list[tuple[str, pd.DataFrame]] = []
    if args.backtest_dir is not None:
        trades_path = args.backtest_dir / "trades.csv"
        if not trades_path.exists():
            raise FileNotFoundError(f"No trades.csv found in {args.backtest_dir}")
        ledgers.append((args.backtest_dir.name, pd.read_csv(trades_path)))
    else:
        for i, path in enumerate(args.trades or []):
            name = (args.labels[i] if args.labels and i < len(args.labels)
                    else Path(path).stem)
            ledgers.append((name, pd.read_csv(path)))
    for name, ledger in ledgers:
        required = {"predicted_edge", "net_pnl"}
        missing = required - set(ledger.columns)
        if missing:
            raise ValueError(f"Ledger '{name}' is missing columns: {sorted(missing)}")
    return ledgers


def write_markdown_report(out_dir: Path, source_name: str, tables: dict[str, pd.DataFrame]) -> None:
    pnl = tables["pnl_by_edge"]
    wr = tables["winrate_by_edge"]
    lines = [
        "# Edge-conditional outcome distributions",
        "",
        f"Source: `{source_name}`",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        "",
        "Edge buckets are on the gross predicted edge at signal time "
        "(model probability minus executable price, fraction of $1).",
        "",
        "## PnL distribution by predicted edge",
        "",
        "Per-contract net PnL (net of entry cost and taker fees), $ units.",
        "",
        pnl.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Win rate by predicted edge",
        "",
        "Realized win rate with 95% Wilson interval vs model-implied side win "
        "probability; `calibration_gap_pp` = realized minus implied in percentage "
        "points (positive means realized outcomes beat the model's claim).",
        "",
        wr.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Bootstrap distributions (2000 resamples by default) are in "
        "`bootstrap_pnl_samples.csv` / `bootstrap_winrate_samples.csv`; their "
        "2.5/97.5 percentiles appear as `bootstrap_*_lo/hi` columns above.",
        "",
    ]
    (out_dir / "edge_distributions_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ledgers = load_ledgers(args)
    out_dir = args.out_dir or (
        args.backtest_dir if args.backtest_dir is not None
        else REPO_ROOT / "outputs" / "edge_distributions")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, ledger in ledgers:
        tables = build_all_distributions(
            ledger, n_resamples=args.n_resamples, seed=args.seed)
        suffix = "" if len(ledgers) == 1 else f"_{name}"
        tables["pnl_by_edge"].to_csv(out_dir / f"edge_pnl_distribution{suffix}.csv", index=False)
        pnl_histogram_by_edge(ledger).to_csv(out_dir / f"edge_pnl_histogram{suffix}.csv", index=False)
        tables["winrate_by_edge"].to_csv(out_dir / f"edge_winrate_distribution{suffix}.csv", index=False)
        tables["pnl_samples"].to_csv(out_dir / f"bootstrap_pnl_samples{suffix}.csv", index=False)
        tables["winrate_samples"].to_csv(out_dir / f"bootstrap_winrate_samples{suffix}.csv", index=False)

        ledger_plot = ledger.copy()
        plot_pnl_distribution(ledger_plot, out_dir / f"edge_pnl_distribution{suffix}.png",
                              bins=EDGE_BINS, labels=EDGE_LABELS)
        plot_winrate_vs_edge(tables["winrate_by_edge"],
                             out_dir / f"edge_winrate_vs_edge{suffix}.png")
        write_markdown_report(out_dir, name, tables)

        print(f"[{name}] wrote edge distribution outputs to {out_dir}")
        print(f"[{name}] n_trades={len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
