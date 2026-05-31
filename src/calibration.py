from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.distributional_model import distribution_cdf
from src.evaluation import (
    negative_log_likelihood,
    prediction_interval_coverage,
    validate_bucket_probabilities,
)


DEFAULT_SIGMA_ALPHA_GRID = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.40,
    1.50,
    1.75,
    2.00,
]


def apply_sigma_scaling(
    sigma: pd.Series | np.ndarray | list[float],
    alpha: float,
) -> np.ndarray:
    alpha_value = _validate_alpha(alpha)
    values = np.asarray(sigma, dtype=float)
    if values.ndim != 1:
        raise ValueError("sigma must be one-dimensional")
    if len(values) == 0:
        raise ValueError("sigma cannot be empty")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("sigma must be finite and greater than 0 before calibration")

    calibrated = values * alpha_value
    if not np.isfinite(calibrated).all() or (calibrated <= 0.0).any():
        raise ValueError("calibrated sigma must be finite and greater than 0")
    return calibrated


def fit_global_sigma_scale(
    y_true: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    alpha_grid: list[float] | tuple[float, ...] = tuple(DEFAULT_SIGMA_ALPHA_GRID),
    coverage_levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    dist_type: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
    coverage_penalty_weight: float = 0.25,
) -> tuple[float, pd.DataFrame]:
    """
    Fit a single post-hoc scale factor for predictive sigma.

    This helper is intentionally split-agnostic. Callers should pass validation
    rows only; the final test set must not be used to choose alpha.
    """
    if not math.isfinite(float(coverage_penalty_weight)) or float(coverage_penalty_weight) < 0.0:
        raise ValueError("coverage_penalty_weight must be finite and nonnegative")

    parsed_grid = [_validate_alpha(alpha) for alpha in alpha_grid]
    if not parsed_grid:
        raise ValueError("alpha_grid must contain at least one value")

    rows: list[dict[str, float]] = []
    for alpha in parsed_grid:
        calibrated_sigma = apply_sigma_scaling(sigma, alpha)
        nll = negative_log_likelihood(
            y_true,
            mu,
            calibrated_sigma,
            dist_type=dist_type,
            df=df,
        )
        coverage = prediction_interval_coverage(
            y_true,
            mu,
            calibrated_sigma,
            levels=coverage_levels,
            dist_type=dist_type,
            df=df,
        )
        row: dict[str, float] = {
            "alpha": float(alpha),
            "nll": float(nll),
        }
        coverage_penalty = 0.0
        for _, coverage_row in coverage.iterrows():
            suffix = _coverage_suffix(float(coverage_row["level"]))
            gap = float(coverage_row["coverage_error"])
            row[f"coverage_{suffix}"] = float(coverage_row["actual_coverage"])
            row[f"coverage_gap_{suffix}"] = gap
            coverage_penalty += abs(gap)
        row["coverage_penalty"] = float(coverage_penalty)
        row["selection_score"] = float(nll + float(coverage_penalty_weight) * coverage_penalty)
        rows.append(row)

    search = pd.DataFrame(rows).sort_values(
        ["selection_score", "nll", "coverage_penalty", "alpha"],
        kind="stable",
    )
    selected_alpha = float(search.iloc[0]["alpha"])
    return selected_alpha, pd.DataFrame(rows)


def cdf_reliability_table(
    y_true: pd.Series | np.ndarray | list[float],
    mu: pd.Series | np.ndarray | list[float],
    sigma: pd.Series | np.ndarray | list[float],
    thresholds: tuple[float, ...] = (-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0),
    n_bins: int = 10,
    split: str | None = None,
    method: str | None = None,
    dist_type: str = "normal",
    df: pd.Series | np.ndarray | list[float] | float | None = None,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    if y.ndim != 1 or mu_array.ndim != 1 or sigma_array.ndim != 1:
        raise ValueError("y_true, mu, and sigma must be one-dimensional")
    if len({len(y), len(mu_array), len(sigma_array)}) != 1:
        raise ValueError("y_true, mu, and sigma must have the same length")
    if len(y) == 0:
        raise ValueError("CDF reliability inputs cannot be empty")
    if not np.isfinite(y).all():
        raise ValueError("y_true contains non-finite values")

    frames: list[pd.DataFrame] = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        if not math.isfinite(threshold_value):
            raise ValueError(f"threshold must be finite, got {threshold!r}")
        predicted = distribution_cdf(
            np.full(len(y), threshold_value, dtype=float),
            mu=mu_array,
            sigma=sigma_array,
            distribution=dist_type,
            df=df,
        )
        actual = (y <= threshold_value).astype(int)
        table = make_calibration_table(predicted, actual, n_bins=n_bins)
        table["threshold"] = threshold_value
        table["prob_bin"] = table.apply(
            lambda row: _format_probability_bin(float(row["bin_lower"]), float(row["bin_upper"])),
            axis=1,
        )
        table["calibration_error"] = (
            table["empirical_frequency"] - table["mean_predicted_probability"]
        )
        frames.append(table)

    result = pd.concat(frames, ignore_index=True)
    if method is not None:
        result.insert(0, "method", str(method))
    if split is not None:
        result.insert(0, "split", str(split))

    ordered = [
        column
        for column in [
            "split",
            "method",
            "threshold",
            "prob_bin",
            "mean_predicted_probability",
            "empirical_frequency",
            "count",
            "calibration_error",
            "bin_lower",
            "bin_upper",
        ]
        if column in result.columns
    ]
    return result[ordered]


def make_calibration_table(
    pred_probs: pd.Series | np.ndarray | list[float],
    actual_indicators: pd.Series | np.ndarray | list[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    probabilities, actual = _validate_calibration_inputs(
        pred_probs,
        actual_indicators,
        n_bins=n_bins,
    )

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(int(n_bins)):
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        if bin_index == int(n_bins) - 1:
            in_bin = (probabilities >= lower) & (probabilities <= upper)
        else:
            in_bin = (probabilities >= lower) & (probabilities < upper)

        count = int(np.count_nonzero(in_bin))
        if count:
            mean_predicted = float(np.mean(probabilities[in_bin]))
            empirical_frequency = float(np.mean(actual[in_bin]))
            calibration_gap = mean_predicted - empirical_frequency
        else:
            mean_predicted = math.nan
            empirical_frequency = math.nan
            calibration_gap = math.nan

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted_probability": mean_predicted,
                "empirical_frequency": empirical_frequency,
                "calibration_gap": calibration_gap,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "bin_lower",
            "bin_upper",
            "count",
            "mean_predicted_probability",
            "empirical_frequency",
            "calibration_gap",
        ],
    )


def calibration_tables_by_bucket(
    bucket_probs: pd.DataFrame,
    realized_bucket_labels: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    probabilities = validate_bucket_probabilities(bucket_probs)
    labels = pd.Series(realized_bucket_labels).reset_index(drop=True)
    if len(labels) != len(probabilities):
        raise ValueError(
            "realized_bucket_labels length must match bucket_probs rows: "
            f"labels={len(labels)}, probabilities={len(probabilities)}"
        )
    if labels.isna().any():
        raise ValueError("realized_bucket_labels contains missing values")
    missing = sorted({str(label) for label in labels.unique() if label not in probabilities.columns})
    if missing:
        raise ValueError(f"Realized bucket labels not in probability columns: {missing}")

    tables: list[pd.DataFrame] = []
    for bucket in probabilities.columns:
        actual = (labels == bucket).astype(int)
        table = make_calibration_table(probabilities[bucket], actual, n_bins=n_bins)
        table.insert(0, "bucket", str(bucket))
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def choose_buckets_for_plots(
    brier_df: pd.DataFrame,
    max_buckets: int = 4,
) -> list[str]:
    if brier_df.empty:
        raise ValueError("brier_df is empty")
    required = ["bucket", "brier_score", "empirical_frequency"]
    missing = [column for column in required if column not in brier_df.columns]
    if missing:
        raise ValueError(f"brier_df is missing required column(s): {missing}")
    if int(max_buckets) < 1:
        raise ValueError("max_buckets must be at least 1")

    working = brier_df.copy()
    working["brier_score"] = pd.to_numeric(working["brier_score"], errors="coerce")
    working["empirical_frequency"] = pd.to_numeric(
        working["empirical_frequency"],
        errors="coerce",
    )
    if working[["brier_score", "empirical_frequency"]].isna().any().any():
        raise ValueError("brier_df brier_score and empirical_frequency must be numeric")

    buckets_in_order = [str(bucket) for bucket in working["bucket"]]
    selected: list[str] = []

    most_common = str(working.sort_values("empirical_frequency", ascending=False).iloc[0]["bucket"])
    worst_brier = str(working.sort_values("brier_score", ascending=False).iloc[0]["bucket"])
    lower_tail = _find_tail_bucket(buckets_in_order, lower=True)
    upper_tail = _find_tail_bucket(buckets_in_order, lower=False)

    for bucket in [most_common, worst_brier, lower_tail, upper_tail]:
        if bucket not in selected:
            selected.append(bucket)
        if len(selected) >= int(max_buckets):
            break
    return selected


def plot_calibration_curve(
    calibration_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    required = ["mean_predicted_probability", "empirical_frequency", "count"]
    missing = [column for column in required if column not in calibration_table.columns]
    if missing:
        raise ValueError(f"calibration_table is missing required column(s): {missing}")

    plot_df = calibration_table[calibration_table["count"] > 0].copy()
    if plot_df.empty:
        raise ValueError("No non-empty calibration bins to plot")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, linestyle="--")
    ax.plot(
        plot_df["mean_predicted_probability"],
        plot_df["empirical_frequency"],
        marker="o",
        linewidth=1.8,
        color="#2f6f8f",
    )
    if "bucket" in calibration_table.columns and calibration_table["bucket"].nunique() == 1:
        bucket_name = str(calibration_table["bucket"].iloc[0])
        ax.set_title(f"Calibration: {bucket_name}")
    else:
        ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_pit_histogram(
    pit_values: pd.Series | np.ndarray | list[float],
    output_path: str | Path,
    bins: int = 10,
) -> None:
    values = np.asarray(pit_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("pit_values must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("pit_values must be finite and between 0 and 1")
    if int(bins) < 1:
        raise ValueError("bins must be at least 1")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=int(bins), range=(0.0, 1.0), color="#4c78a8", edgecolor="white")
    ax.set_title("PIT Histogram")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_coverage_by_group(
    coverage_df: pd.DataFrame,
    group_col: str,
    output_path: str | Path,
) -> None:
    required = [group_col, "actual_coverage", "expected_coverage", "count"]
    missing = [column for column in required if column not in coverage_df.columns]
    if missing:
        raise ValueError(f"coverage_df is missing required column(s): {missing}")
    if coverage_df.empty:
        raise ValueError("coverage_df is empty")

    plot_df = coverage_df.copy()
    plot_df["_group_label"] = plot_df[group_col].astype(str)
    expected = float(pd.to_numeric(plot_df["expected_coverage"], errors="raise").iloc[0])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(7.0, min(14.0, 0.38 * len(plot_df) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    colors = np.where(plot_df["enough_sample"].astype(bool), "#4c78a8", "#bab0ab")
    ax.bar(plot_df["_group_label"], plot_df["actual_coverage"], color=colors)
    ax.axhline(expected, color="black", linewidth=1.0, linestyle="--")
    ax.set_title(f"Coverage by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Actual coverage")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45 if len(plot_df) > 8 else 0)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_coverage_before_after(
    coverage_df: pd.DataFrame,
    output_path: str | Path,
    split: str | None = None,
) -> None:
    required = ["method", "nominal_coverage", "empirical_coverage"]
    missing = [column for column in required if column not in coverage_df.columns]
    if missing:
        raise ValueError(f"coverage_df is missing required column(s): {missing}")
    if coverage_df.empty:
        raise ValueError("coverage_df is empty")

    plot_df = coverage_df.copy()
    if split is not None and "split" in plot_df.columns:
        plot_df = plot_df[plot_df["split"].astype(str) == str(split)].copy()
    if plot_df.empty:
        raise ValueError("No coverage rows remain after split filtering")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, linestyle="--")
    group_columns = ["method"]
    if "split" in plot_df.columns:
        group_columns.insert(0, "split")
    for group_values, group_df in plot_df.groupby(group_columns, sort=False):
        group_df = group_df.sort_values("nominal_coverage")
        label = (
            " / ".join(str(value) for value in group_values)
            if isinstance(group_values, tuple)
            else str(group_values)
        )
        ax.plot(
            group_df["nominal_coverage"],
            group_df["empirical_coverage"],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_title("Coverage Before/After")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_sigma_scaling_validation_nll(
    alpha_search_df: pd.DataFrame,
    output_path: str | Path,
    selected_alpha: float | None = None,
) -> None:
    nll_col = "validation_nll" if "validation_nll" in alpha_search_df.columns else "nll"
    required = ["alpha", nll_col]
    missing = [column for column in required if column not in alpha_search_df.columns]
    if missing:
        raise ValueError(f"alpha_search_df is missing required column(s): {missing}")
    if alpha_search_df.empty:
        raise ValueError("alpha_search_df is empty")

    plot_df = alpha_search_df.copy()
    plot_df["alpha"] = pd.to_numeric(plot_df["alpha"], errors="raise")
    plot_df[nll_col] = pd.to_numeric(plot_df[nll_col], errors="raise")
    plot_df = plot_df.sort_values("alpha")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plot_df["alpha"], plot_df[nll_col], marker="o", linewidth=1.8, color="#4c78a8")
    if selected_alpha is not None:
        ax.axvline(float(selected_alpha), color="#d55e00", linestyle="--", linewidth=1.3)
    ax.set_title("Validation NLL by Sigma Scale")
    ax.set_xlabel("Sigma scale alpha")
    ax.set_ylabel("Validation NLL")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _validate_calibration_inputs(
    pred_probs: pd.Series | np.ndarray | list[float],
    actual_indicators: pd.Series | np.ndarray | list[int],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(n_bins) < 1:
        raise ValueError("n_bins must be at least 1")

    probabilities = np.asarray(pred_probs, dtype=float)
    actual = np.asarray(actual_indicators, dtype=float)
    if probabilities.ndim != 1 or actual.ndim != 1:
        raise ValueError("pred_probs and actual_indicators must be one-dimensional")
    if len(probabilities) != len(actual):
        raise ValueError(
            "pred_probs and actual_indicators must have the same length: "
            f"pred_probs={len(probabilities)}, actual_indicators={len(actual)}"
        )
    if len(probabilities) == 0:
        raise ValueError("Calibration inputs cannot be empty")
    if not np.isfinite(probabilities).all():
        raise ValueError("pred_probs contains non-finite values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("pred_probs must be between 0 and 1")
    if not np.isfinite(actual).all():
        raise ValueError("actual_indicators contains non-finite values")
    if not np.isin(actual, [0.0, 1.0]).all():
        raise ValueError("actual_indicators must contain only 0/1 values")
    return probabilities, actual


def _find_tail_bucket(buckets: list[str], lower: bool) -> str:
    candidates = (
        ["lower", "le_", "le ", "-inf", "bucket_0", "market_bucket_0"]
        if lower
        else ["higher", "gt_", "gt ", "inf", "bucket_5", "market_bucket_5"]
    )
    for bucket in buckets:
        text = bucket.lower()
        if any(candidate in text for candidate in candidates):
            return bucket
    return buckets[0] if lower else buckets[-1]


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"alpha must be finite and greater than 0, got {alpha!r}")
    return value


def _coverage_suffix(level: float) -> str:
    return f"{int(round(float(level) * 100)):02d}"


def _format_probability_bin(lower: float, upper: float) -> str:
    left = f"{lower:.1f}"
    right = f"{upper:.1f}"
    return f"{left}-{right}"
