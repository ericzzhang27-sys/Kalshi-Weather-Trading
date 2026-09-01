from __future__ import annotations

"""
Distributions of realized outcomes conditioned on predicted edge.

Two views of the trade ledger (`trades.csv` from any backtest run dir):

1. PnL distribution by predicted-edge bucket
   - per-contract net PnL (net_pnl / contracts) summarized per edge bucket
   - empirical quantiles + histogram counts + bootstrap CI of the mean

2. Win-rate distribution by predicted-edge bucket
   - realized win rate per bucket with Wilson score interval
   - model-implied win probability (q = model_prob for BUY_YES, 1-model_prob
     for BUY_NO) so buckets can be checked for edge calibration
   - bootstrap distribution of the bucket win rate

All functions are pure (no I/O); see scripts/build_edge_distributions.py for
the CLI that writes CSV/PNG/markdown outputs next to a backtest run.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

EDGE_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, np.inf]
EDGE_LABELS = ["0-2%", "2-5%", "5-10%", "10-15%", "15%+"]

_PNL_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class EdgeDistributionResult:
    """Container for the two conditional-distribution tables."""

    pnl_by_edge: pd.DataFrame
    winrate_by_edge: pd.DataFrame


def _edge_bin_series(ledger: pd.DataFrame, bins: list[float], labels: list[str]) -> pd.Series:
    edge = pd.to_numeric(ledger["predicted_edge"], errors="coerce")
    binned = pd.cut(edge, bins=bins, labels=labels, right=False)
    return binned.cat.add_categories(["unknown"]).fillna("unknown")


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def pnl_per_contract(ledger: pd.DataFrame) -> pd.Series:
    """Per-contract net PnL in $ (net of entry cost and taker fees)."""
    pnl = _finite_numeric(ledger.get("net_pnl", pd.Series(dtype=float)))
    contracts = _finite_numeric(ledger.get("contracts", pd.Series(dtype=float)))
    contracts = contracts.where(contracts > 0)
    return pnl / contracts


def wilson_interval(wins: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def implied_win_probability(ledger: pd.DataFrame) -> pd.Series:
    """Model-implied probability the traded side wins.

    BUY_YES wins iff YES settles 1 -> q = model_probability.
    BUY_NO wins iff YES settles 0 -> q = 1 - model_probability.
    """
    prob = _finite_numeric(ledger.get("model_probability", pd.Series(dtype=float)))
    side = ledger.get("side", pd.Series(index=ledger.index, dtype=object))
    is_no = side.astype(str).str.upper().eq("BUY_NO")
    return prob.where(~is_no, 1.0 - prob).clip(0.0, 1.0)


def _bucket_order(bins: list[float], labels: list[str]) -> list[str]:
    ordered = [label for label in labels]
    if "unknown" not in ordered:
        ordered.append("unknown")
    return ordered


def pnl_distribution_by_edge(
    ledger: pd.DataFrame,
    *,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Empirical distribution of per-contract net PnL grouped by edge bucket."""
    if ledger.empty or "predicted_edge" not in ledger.columns:
        return pd.DataFrame()
    bins = bins or EDGE_BINS
    labels = labels or EDGE_LABELS
    df = ledger.copy()
    df["pnl_per_contract"] = pnl_per_contract(df)
    df["_edge_bin"] = _edge_bin_series(df, bins, labels)
    valid = df.dropna(subset=["pnl_per_contract"])

    rows = []
    for bucket in _bucket_order(bins, labels):
        group = valid[valid["_edge_bin"] == bucket]["pnl_per_contract"].to_numpy(float)
        if len(group) == 0:
            rows.append({
                "edge_bucket": bucket, "n_trades": 0,
                **{key: np.nan for key in [
                    "mean_pnl_per_contract", "std_pnl_per_contract",
                    *[f"pnl_q{int(q*100)}" for q in _PNL_QUANTILES],
                    "min_pnl_per_contract", "max_pnl_per_contract",
                    "total_net_pnl", "win_rate", "loss_rate",
                    "avg_predicted_edge", "bootstrap_mean_lo", "bootstrap_mean_hi",
                ]},
            })
            continue
        quantiles = {f"pnl_q{int(q*100)}": float(np.quantile(group, q)) for q in _PNL_QUANTILES}
        boot_lo, boot_hi = _bootstrap_mean_ci(group)
        rows.append({
            "edge_bucket": bucket,
            "n_trades": int(len(group)),
            "mean_pnl_per_contract": float(group.mean()),
            "std_pnl_per_contract": float(group.std(ddof=1)) if len(group) > 1 else np.nan,
            **quantiles,
            "min_pnl_per_contract": float(group.min()),
            "max_pnl_per_contract": float(group.max()),
            "total_net_pnl": float(group.sum()),
            "win_rate": float((group > 0).mean()),
            "loss_rate": float((group < 0).mean()),
            "avg_predicted_edge": float(
                pd.to_numeric(valid[valid["_edge_bin"] == bucket]["predicted_edge"], errors="coerce").mean()
            ),
            "bootstrap_mean_lo": boot_lo,
            "bootstrap_mean_hi": boot_hi,
        })
    table = pd.DataFrame(rows)
    zero_mask = table["n_trades"] == 0
    table.loc[zero_mask, "total_net_pnl"] = 0.0
    return table


def pnl_histogram_by_edge(
    ledger: pd.DataFrame,
    *,
    bin_edges: np.ndarray | None = None,
    edge_bins: list[float] | None = None,
    edge_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Histogram counts of per-contract net PnL, one column per edge bucket."""
    if ledger.empty or "predicted_edge" not in ledger.columns:
        return pd.DataFrame()
    edge_bins = edge_bins or EDGE_BINS
    edge_labels = edge_labels or EDGE_LABELS
    if bin_edges is None:
        pnl = pnl_per_contract(ledger).dropna()
        if pnl.empty:
            return pd.DataFrame()
        lo = math.floor(pnl.min() * 20) / 20.0
        hi = math.ceil(pnl.max() * 20) / 20.0
        step = 0.05 if hi - lo <= 3 else 0.10
        bin_edges = np.round(np.arange(lo, hi + step, step), 4)

    df = ledger.copy()
    df["pnl_per_contract"] = pnl_per_contract(df)
    df["_edge_bin"] = _edge_bin_series(df, edge_bins, edge_labels)
    columns: dict[str, pd.Series] = {}
    index_name = "pnl_bin"
    counts = {}
    for bucket in _bucket_order(edge_bins, edge_labels):
        group = df[(df["_edge_bin"] == bucket)]["pnl_per_contract"].dropna()
        hist, edges = np.histogram(group.to_numpy(float), bins=bin_edges)
        counts[bucket] = hist
    table = pd.DataFrame(counts, index=pd.Index(edges[:-1], name=index_name))
    table.index = [f"[{lo:.2f}, {hi:.2f})" for lo, hi in zip(edges[:-1], edges[1:-1])] + \
        [f"[{edges[-2]:.2f}, {edges[-1]:.2f}]"]
    table.index.name = index_name
    return table.reset_index()


def winrate_by_edge(
    ledger: pd.DataFrame,
    *,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Realized vs model-implied win rate grouped by edge bucket."""
    if ledger.empty or "predicted_edge" not in ledger.columns:
        return pd.DataFrame()
    bins = bins or EDGE_BINS
    labels = labels or EDGE_LABELS
    df = ledger.copy()
    df["_edge_bin"] = _edge_bin_series(df, bins, labels)
    df["_implied_q"] = implied_win_probability(df)
    df["_won"] = (_finite_numeric(df["net_pnl"]) > 0).astype(float)

    rows = []
    for bucket in _bucket_order(bins, labels):
        group = df[df["_edge_bin"] == bucket]
        n = int(len(group))
        if n == 0:
            rows.append({
                "edge_bucket": bucket, "n_trades": 0,
                **{key: np.nan for key in [
                    "wins", "win_rate", "wilson_lo", "wilson_hi",
                    "avg_implied_win_prob", "calibration_gap_pp",
                    "avg_predicted_edge", "bootstrap_wr_lo", "bootstrap_wr_hi",
                ]},
            })
            continue
        won = group["_won"].to_numpy(float)
        implied = group["_implied_q"].dropna()
        wins = int(won.sum())
        wr_lo, wr_hi = wilson_interval(wins, n)
        boot_lo, boot_hi = _bootstrap_mean_ci(won) if len(np.unique(won)) > 1 else (
            float(won.mean()), float(won.mean()))
        avg_implied = float(implied.mean()) if len(implied) else np.nan
        rows.append({
            "edge_bucket": bucket,
            "n_trades": n,
            "wins": wins,
            "win_rate": float(won.mean()),
            "wilson_lo": wr_lo,
            "wilson_hi": wr_hi,
            "avg_implied_win_prob": avg_implied,
            # realized minus implied, percentage points (positive => edge underestimated)
            "calibration_gap_pp": (float(won.mean()) - avg_implied) * 100.0
                if np.isfinite(avg_implied) else np.nan,
            "avg_predicted_edge": float(
                pd.to_numeric(group["predicted_edge"], errors="coerce").mean()),
            "bootstrap_wr_lo": boot_lo,
            "bootstrap_wr_hi": boot_hi,
        })
    table = pd.DataFrame(rows)
    zero_mask = table["n_trades"] == 0
    table.loc[zero_mask, "wins"] = 0
    return table


def _bootstrap_mean_ci(values: np.ndarray, *, n_resamples: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def bootstrap_outcome_distributions(
    ledger: pd.DataFrame,
    *,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Full bootstrap distributions of (per-contract PnL, win rate) per edge bucket.

    Returns {"pnl_samples": long DataFrame, "winrate_samples": long DataFrame}.
    """
    bins = bins or EDGE_BINS
    labels = labels or EDGE_LABELS
    df = ledger.copy()
    df["pnl_per_contract"] = pnl_per_contract(df)
    df["_edge_bin"] = _edge_bin_series(df, bins, labels)
    df["_implied_q"] = implied_win_probability(df)
    df["_won"] = (_finite_numeric(df["net_pnl"]) > 0).astype(float)

    rng = np.random.default_rng(seed)
    pnl_frames = []
    wr_frames = []
    for bucket in _bucket_order(bins, labels):
        group = df[df["_edge_bin"] == bucket].dropna(subset=["pnl_per_contract"])
        if group.empty:
            continue
        pnl_values = group["pnl_per_contract"].to_numpy(float)
        won_values = group["_won"].to_numpy(float)
        draws = rng.integers(0, len(group), size=(n_resamples, len(group)))
        sample_means = pnl_values[draws].mean(axis=1)
        sample_wrs = won_values[draws].mean(axis=1)
        pnl_frames.append(pd.DataFrame({
            "edge_bucket": bucket,
            "resample": np.arange(n_resamples),
            "mean_pnl_per_contract": sample_means,
        }))
        wr_frames.append(pd.DataFrame({
            "edge_bucket": bucket,
            "resample": np.arange(n_resamples),
            "win_rate": sample_wrs,
        }))
    pnl_samples = pd.concat(pnl_frames, ignore_index=True) if pnl_frames else pd.DataFrame(
        columns=["edge_bucket", "resample", "mean_pnl_per_contract"])
    wr_samples = pd.concat(wr_frames, ignore_index=True) if wr_frames else pd.DataFrame(
        columns=["edge_bucket", "resample", "win_rate"])
    return {"pnl_samples": pnl_samples, "winrate_samples": wr_samples}


# --------------------------------------------------------------------- plots
def plot_pnl_distribution(
    ledger: pd.DataFrame,
    output_path,
    *,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> object | None:
    """Histogram + ECDF of per-contract net PnL split by edge bucket."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    bins = bins or EDGE_BINS
    labels = labels or EDGE_LABELS
    df = ledger.copy()
    df["pnl_per_contract"] = pnl_per_contract(df)
    df["_edge_bin"] = _edge_bin_series(df, bins, labels)
    df = df.dropna(subset=["pnl_per_contract"])
    if df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_hist, ax_ecdf = axes
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))
    for color, label in zip(colors, labels):
        group = df[df["_edge_bin"] == label]["pnl_per_contract"]
        if group.empty:
            continue
        ax_hist.hist(group, bins=30, alpha=0.55, label=f"{label} (n={len(group)})",
                     color=color, density=True)
    ax_hist.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_hist.set_xlabel("Net PnL per contract ($)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("PnL distribution by predicted edge")
    ax_hist.legend(fontsize=8)

    for color, label in zip(colors, labels):
        group = np.sort(df[df["_edge_bin"] == label]["pnl_per_contract"].to_numpy(float))
        if len(group) == 0:
            continue
        ax_ecdf.step(group, np.arange(1, len(group) + 1) / len(group), where="post",
                     color=color, label=label)
    ax_ecdf.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_ecdf.set_xlabel("Net PnL per contract ($)")
    ax_ecdf.set_ylabel("ECDF")
    ax_ecdf.set_title("Cumulative PnL distribution by predicted edge")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return fig


def plot_winrate_vs_edge(
    winrate_table: pd.DataFrame,
    output_path,
) -> object | None:
    """Realized win rate (with Wilson CI) vs model-implied, per edge bucket."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    table = winrate_table[winrate_table["n_trades"] > 0]
    if table.empty:
        return None
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9, 5))
    lower_err = np.clip(table["win_rate"].to_numpy() - table["wilson_lo"].to_numpy(), 0, None)
    upper_err = np.clip(table["wilson_hi"].to_numpy() - table["win_rate"].to_numpy(), 0, None)
    ax.errorbar(x, table["win_rate"], yerr=[lower_err, upper_err], fmt="o",
                capsize=4, label="Realized win rate (95% Wilson CI)", color="#1f77b4")
    if table["avg_implied_win_prob"].notna().any():
        ax.plot(x, table["avg_implied_win_prob"], "s--", color="#d62728",
                label="Model-implied win prob")
    ax.set_xticks(x, table["edge_bucket"].astype(str), rotation=20)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted edge bucket")
    ax.set_ylabel("Win probability")
    ax.set_title("Win rate by predicted edge")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return fig


def build_all_distributions(
    ledger: pd.DataFrame,
    *,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """One-call entry point returning all tables used by the CLI script."""
    bins = bins or EDGE_BINS
    labels = labels or EDGE_LABELS
    result: dict[str, pd.DataFrame] = {
        "pnl_by_edge": pnl_distribution_by_edge(ledger, bins=bins, labels=labels),
        "winrate_by_edge": winrate_by_edge(ledger, bins=bins, labels=labels),
    }
    result.update(bootstrap_outcome_distributions(
        ledger, bins=bins, labels=labels, n_resamples=n_resamples, seed=seed))
    return result
