"""Simple visualizations for scraped Kalshi orderbook data.

Two plots per run:
  1. Depth chart  - cumulative YES bid/ask size vs price at one snapshot.
  2. Price history - YES midpoint (implied probability) across all hourly
     snapshots collected so far.

Usage (from repo root):
    python scripts/visualize_orderbook.py                          # pick most liquid active market, latest snapshot
    python scripts/visualize_orderbook.py --ticker KXHIGHNY-26AUG23-T87
    python scripts/visualize_orderbook.py --ticker KXHIGHNY-26AUG23-T87 --fetched-at 2026-08-22T22:00:04
    python scripts/visualize_orderbook.py --show                   # open interactive windows instead of only saving PNGs

Outputs are saved to outputs/orderbook_viz/.
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")  # safe for saving without a display; --show re-enables GUI
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw", "kalshi_orderbooks")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "orderbook_viz")


def load_levels() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "orderbook_levels_*.csv")))
    if not files:
        sys.exit(f"No orderbook level files found in {DATA_DIR}. Run the scraper first.")
    return pd.concat(map(pd.read_csv, files), ignore_index=True)


def load_summary() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "orderbook_summary_*.csv")))
    if not files:
        sys.exit(f"No orderbook summary files found in {DATA_DIR}. Run the scraper first.")
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    return df


def pick_ticker(summary: pd.DataFrame) -> str:
    """Most liquid active market in the latest snapshot (largest total depth)."""
    latest_ts = summary["fetched_at"].max()
    snap = summary[(summary["fetched_at"] == latest_ts) & (summary["market_status"] == "active")]
    if snap.empty:
        snap = summary[summary["fetched_at"] == latest_ts]
    depth_cols = ["yes_bid_depth", "yes_ask_depth"]
    snap = snap.assign(total_depth=snap[depth_cols].sum(axis=1))
    return snap.sort_values("total_depth", ascending=False).iloc[0]["ticker"]


def plot_depth_chart(levels: pd.DataFrame, ticker: str, fetched_at: pd.Timestamp, path: str) -> None:
    """Cumulative YES-side book: bids below mid, asks above."""
    snap = levels[
        (levels["ticker"] == ticker)
        & (pd.to_datetime(levels["fetched_at"], utc=True) == fetched_at)
        & (levels["outcome_side"] == "YES")
    ]
    if snap.empty:
        sys.exit(f"No YES levels found for {ticker} at {fetched_at}.")

    fig, ax = plt.subplots(figsize=(9, 5))
    bids = snap[snap["quote_type"] == "bid"].sort_values("price_dollars")
    asks = snap[snap["quote_type"] == "ask"].sort_values("price_dollars")

    ax.step(bids["price_dollars"], bids["cumulative_size"], where="post",
            color="tab:green", label="YES bids (buy support)")
    ax.fill_between(bids["price_dollars"], bids["cumulative_size"], step="post",
                    color="tab:green", alpha=0.2)
    ax.step(asks["price_dollars"], asks["cumulative_size"], where="pre",
            color="tab:red", label="YES asks (sell pressure)")
    ax.fill_between(asks["price_dollars"], asks["cumulative_size"], step="pre",
                    color="tab:red", alpha=0.2)

    best_bid = bids["price_dollars"].max() if not bids.empty else float("nan")
    best_ask = asks["price_dollars"].min() if not asks.empty else float("nan")
    if bids.empty or asks.empty:
        mid = float("nan")
    else:
        mid = (best_bid + best_ask) / 2
    ax.axvline(mid, color="k", linestyle="--", linewidth=1,
               label=f"midpoint = {mid:.3f}")

    ax.set_xlabel("YES price ($ = implied probability)")
    ax.set_ylabel("Cumulative contracts available")
    ax.set_title(f"Orderbook depth — {ticker} "
                 f"({fetched_at:%Y-%m-%d %H:%M} UTC)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_price_history(summary: pd.DataFrame, ticker: str, path: str) -> None:
    hist = summary[summary["ticker"] == ticker].sort_values("fetched_at")
    if hist.empty:
        sys.exit(f"No summary rows found for {ticker}.")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist["fetched_at"], hist["yes_midpoint"], marker="o",
            color="tab:blue", label="YES midpoint")
    ax.fill_between(hist["fetched_at"],
                    hist["best_yes_bid"], hist["best_yes_ask"],
                    color="tab:blue", alpha=0.15, label="bid-ask range")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Implied probability of YES")
    ax.set_title(f"Implied probability over time — {ticker}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", help="Market ticker (default: most liquid active market).")
    parser.add_argument("--fetched-at", help="Snapshot timestamp (UTC, prefix match ok). Default: latest.")
    parser.add_argument("--show", action="store_true", help="Also open interactive plot windows.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    levels = load_levels()
    summary = load_summary()

    ticker = args.ticker or pick_ticker(summary)
    print(f"Ticker: {ticker}")

    if args.fetched_at:
        matches = sorted(set(
            ts for ts in levels.loc[levels["ticker"] == ticker, "fetched_at"]
            if str(ts).startswith(args.fetched_at)
        ))
        if not matches:
            sys.exit(f"No snapshot for {ticker} matching --fetched-at '{args.fetched_at}'.")
        fetched_at = pd.Timestamp(matches[-1])
    else:
        fetched_at = pd.to_datetime(
            levels.loc[levels["ticker"] == ticker, "fetched_at"], utc=True
        ).max()
    print(f"Snapshot: {fetched_at}")

    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    depth_path = os.path.join(OUT_DIR, f"depth_{ticker}_{stamp}.png".replace("/", "_"))
    hist_path = os.path.join(OUT_DIR, f"history_{ticker}.png".replace("/", "_"))

    plot_depth_chart(levels, ticker, fetched_at, depth_path)
    plot_price_history(summary, ticker, hist_path)
    print(f"Saved: {depth_path}")
    print(f"Saved: {hist_path}")

    if args.show:
        matplotlib.use("TkAgg", force=True)
        import importlib
        importlib.reload(plt)
        plt.figure(1)
        plt.imshow(plt.imread(depth_path))
        plt.axis("off")
        plt.title(f"Depth — {ticker}")
        plt.figure(2)
        plt.imshow(plt.imread(hist_path))
        plt.axis("off")
        plt.title(f"History — {ticker}")
        plt.show()


if __name__ == "__main__":
    main()