from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (  # noqa: E402
    calibration_tables_by_bucket,
    choose_buckets_for_plots,
    plot_calibration_curve,
    plot_coverage_by_group,
    plot_pit_histogram,
)
from src.distribution_pricing import (  # noqa: E402
    validate_bucket_probabilities as validate_long_bucket_probabilities,
)
from src.evaluation import (  # noqa: E402
    bucket_brier_scores,
    compute_pit_values,
    coverage_by_group,
    interval_log_loss,
    negative_log_likelihood,
    prediction_interval_coverage,
    residual_summary,
    standardized_residuals,
    validate_bucket_probabilities,
    validate_distribution_params,
)
from src.splits import chronological_train_validation_test_split  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]

# Input mapping from previous days:
# - Day 11 NGBoost distribution parameters: one row per prediction state, with
#   mu/sigma and validation/test split in outputs/ngboost_distribution_params_v0.csv.
# - Day 11 NGBoost bucket probabilities: long format, one row per row_id/bucket
#   in outputs/ngboost_bucket_probs_v0.csv. Bucket names are final-temperature
#   labels that move with forecast_high, so Day 13 evaluates fixed bucket_index
#   positions for this market-style diagnostic.
# - Day 9 empirical baseline: fixed forecast-error interval probabilities in
#   outputs/day9_empirical_baseline/empirical_baseline_predictions.csv. The
#   empirical comparison uses these same forecast-error intervals for NGBoost.
DEFAULT_PARAMS_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_params_v0.csv"
DEFAULT_BUCKET_PROBS_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_probs_v0.csv"
DEFAULT_MODELING_ROWS_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"
DEFAULT_BASELINE_PATH = (
    REPO_ROOT / "outputs" / "day9_empirical_baseline" / "empirical_baseline_predictions.csv"
)
DEFAULT_NGBOOST_METRICS_PATH = REPO_ROOT / "outputs" / "ngboost_nll_v0.json"

COVERAGE_REPORT_PATH = REPO_ROOT / "outputs" / "coverage_report.csv"
RESIDUAL_SUMMARY_PATH = REPO_ROOT / "outputs" / "standardized_residual_summary.csv"
BUCKET_BRIER_PATH = REPO_ROOT / "outputs" / "bucket_brier_scores.csv"
CALIBRATION_TABLES_PATH = REPO_ROOT / "outputs" / "calibration_tables.csv"
COVERAGE_BY_GROUP_PATH = REPO_ROOT / "outputs" / "coverage_by_group.csv"
EVALUATION_REPORT_PATH = REPO_ROOT / "outputs" / "ngboost_evaluation_report.csv"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

KEY_COLUMNS = ["date_key", "station_id", "prediction_hour", "forecast_horizon_hours"]
OOS_SPLITS = ["validation", "test"]

ERROR_INTERVAL_SCHEMA = [
    {"label": "(-inf, -3]", "lower": None, "upper": -3.0, "prob_col": "prob_error_le_-3"},
    {"label": "(-3, -1]", "lower": -3.0, "upper": -1.0, "prob_col": "prob_error_-3_to_-1"},
    {"label": "(-1, 1]", "lower": -1.0, "upper": 1.0, "prob_col": "prob_error_-1_to_1"},
    {"label": "(1, 3]", "lower": 1.0, "upper": 3.0, "prob_col": "prob_error_1_to_3"},
    {"label": "(3, inf)", "lower": 3.0, "upper": None, "prob_col": "prob_error_gt_3"},
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    params = load_ngboost_params(args.params_path)
    params = enrich_ngboost_params(params, args.modeling_rows_path)
    params = ensure_out_of_sample_splits(params)
    validate_chronological_evaluation(params, args.ngboost_metrics_path)

    market_probs, market_labels = load_ngboost_market_bucket_probabilities(
        args.bucket_probs_path,
        params["row_id"],
    )

    coverage_report, residual_report, pit_by_split = compute_distribution_reports(params)
    write_csv(coverage_report, COVERAGE_REPORT_PATH)
    write_csv(residual_report, RESIDUAL_SUMMARY_PATH)
    plot_pit_histogram(
        pit_by_split["combined_out_of_sample"],
        FIGURES_DIR / "pit_histogram.png",
        bins=10,
    )

    bucket_brier_report, market_log_loss = compute_market_bucket_reports(
        params,
        market_probs,
        market_labels,
    )
    write_csv(bucket_brier_report, BUCKET_BRIER_PATH)

    calibration_tables = compute_and_plot_calibration_tables(
        params,
        market_probs,
        market_labels,
        bucket_brier_report,
    )

    coverage_group_report = compute_and_plot_group_coverage(params)
    write_csv(coverage_group_report, COVERAGE_BY_GROUP_PATH)

    comparison_report, comparison_calibration = compute_model_comparison(
        params=params,
        baseline_path=args.baseline_path,
        coverage_report=coverage_report,
        residual_report=residual_report,
    )
    write_csv(comparison_report, EVALUATION_REPORT_PATH)

    if not comparison_calibration.empty:
        calibration_tables = pd.concat(
            [calibration_tables, comparison_calibration],
            ignore_index=True,
        )
    write_csv(calibration_tables, CALIBRATION_TABLES_PATH)

    print_summary(
        params=params,
        coverage_report=coverage_report,
        residual_report=residual_report,
        bucket_brier_report=bucket_brier_report,
        market_log_loss=market_log_loss,
        comparison_report=comparison_report,
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Day 13 NGBoost probability quality on chronological OOS rows."
    )
    parser.add_argument("--params-path", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--bucket-probs-path", default=str(DEFAULT_BUCKET_PROBS_PATH))
    parser.add_argument("--modeling-rows-path", default=str(DEFAULT_MODELING_ROWS_PATH))
    parser.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--ngboost-metrics-path", default=str(DEFAULT_NGBOOST_METRICS_PATH))
    return parser.parse_args(argv)


def load_ngboost_params(path: str | Path) -> pd.DataFrame:
    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"NGBoost distribution parameter file not found: {input_path}")
    df = pd.read_csv(input_path)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    required = ["date", "forecast_error", "mu", "sigma"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"NGBoost parameter file is missing required columns: {missing}")
    validate_distribution_params(df)
    df = add_stable_keys(df)
    return df


def enrich_ngboost_params(params: pd.DataFrame, modeling_rows_path: str | Path) -> pd.DataFrame:
    path = resolve_path(modeling_rows_path)
    if not path.exists():
        warnings.warn(
            f"Modeling rows file not found; season diagnostics will use date-derived season: {path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return add_date_derived_season(params)

    columns = [
        "date",
        "location",
        "prediction_time",
        "forecast_horizon_hours",
        "season",
        "month",
    ]
    modeling = pd.read_csv(path, usecols=lambda column: column in columns)
    modeling = add_stable_keys(modeling)
    keep = [*KEY_COLUMNS, *[column for column in ["season", "month"] if column in modeling.columns]]
    modeling = modeling[keep].drop_duplicates(KEY_COLUMNS)
    if modeling.duplicated(KEY_COLUMNS).any():
        raise ValueError("Modeling rows have duplicate stable keys after deduplication")

    enriched = params.merge(
        modeling,
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
        suffixes=("", "_modeling"),
    )
    if len(enriched) != len(params):
        raise AssertionError("Joining modeling diagnostics changed NGBoost row count")
    if "season" not in enriched.columns or enriched["season"].isna().any():
        enriched = add_date_derived_season(enriched)
    return enriched


def add_date_derived_season(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    dates = pd.to_datetime(working["date"], errors="raise")
    month = dates.dt.month
    derived = np.select(
        [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11]),
        ],
        [0, 1, 2, 3],
    )
    working["season"] = working.get("season", pd.Series(index=working.index, dtype=float))
    working["season"] = pd.to_numeric(working["season"], errors="coerce").fillna(pd.Series(derived))
    return working


def add_stable_keys(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "date" not in working.columns:
        raise ValueError("A date column is required for stable chronological keys")
    dates = pd.to_datetime(working["date"], errors="coerce")
    if dates.isna().any():
        bad_count = int(dates.isna().sum())
        raise ValueError(f"date contains {bad_count} unparsable values")
    working["date_key"] = dates.dt.normalize().dt.date.astype(str)

    if "station_id" not in working.columns:
        source = first_existing_column(working, ["location", "station"])
        if source is None:
            raise ValueError("Stable keys require station_id, location, or station")
        working["station_id"] = working[source].astype(str)
    else:
        working["station_id"] = working["station_id"].astype(str)

    if "prediction_hour" not in working.columns:
        source = first_existing_column(
            working,
            ["prediction_time", "prediction_timestamp", "timestamp", "prediction_clock_time"],
        )
        if source is None:
            raise ValueError("Stable keys require prediction_hour or a prediction timestamp")
        parsed = pd.to_datetime(working[source], errors="coerce")
        if parsed.isna().any():
            raw = working[source].astype(str).str.strip()
            parsed_hour = pd.to_numeric(raw.str.split(":").str[0], errors="coerce")
            if parsed_hour.isna().any():
                raise ValueError(f"Could not derive prediction_hour from {source!r}")
            working["prediction_hour"] = parsed_hour.astype(int)
        else:
            working["prediction_hour"] = parsed.dt.hour.astype(int)
    else:
        working["prediction_hour"] = pd.to_numeric(
            working["prediction_hour"],
            errors="coerce",
        ).astype(int)

    if "forecast_horizon_hours" not in working.columns:
        raise ValueError("Stable keys require forecast_horizon_hours")
    working["forecast_horizon_hours"] = pd.to_numeric(
        working["forecast_horizon_hours"],
        errors="coerce",
    )
    if working["forecast_horizon_hours"].isna().any():
        raise ValueError("forecast_horizon_hours contains missing or non-numeric values")

    if working.duplicated(KEY_COLUMNS).any():
        duplicate_count = int(working.duplicated(KEY_COLUMNS).sum())
        raise ValueError(f"Stable keys are not unique; duplicate count={duplicate_count}")
    return working


def ensure_out_of_sample_splits(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "split" not in working.columns:
        warnings.warn(
            "NGBoost parameter file has no split column; inferring chronological validation/test splits.",
            RuntimeWarning,
            stacklevel=2,
        )
        split_result = chronological_train_validation_test_split(working)
        validation = split_result.validation.copy()
        validation["split"] = "validation"
        test = split_result.test.copy()
        test["split"] = "test"
        working = pd.concat([validation, test], ignore_index=True)

    split_values = set(working["split"].dropna().astype(str))
    train_like = sorted(split_values - set(OOS_SPLITS))
    if train_like:
        warnings.warn(
            f"Dropping non-OOS split rows from evaluation: {train_like}",
            RuntimeWarning,
            stacklevel=2,
        )
        working = working[working["split"].isin(OOS_SPLITS)].copy()
    if working.empty:
        raise ValueError("No validation/test rows remain for NGBoost evaluation")
    return working.sort_values(["date_key", "station_id", "prediction_hour"]).reset_index(drop=True)


def validate_chronological_evaluation(
    params: pd.DataFrame,
    metrics_path: str | Path,
) -> None:
    for split in OOS_SPLITS:
        if split not in set(params["split"]):
            raise ValueError(f"Required OOS split is missing from NGBoost params: {split!r}")

    dates = pd.to_datetime(params["date_key"], errors="raise")
    validation_max = dates[params["split"] == "validation"].max()
    test_min = dates[params["split"] == "test"].min()
    if validation_max >= test_min:
        raise AssertionError(
            "Chronological evaluation overlap: validation max date must precede test min date "
            f"({validation_max.date()} >= {test_min.date()})"
        )

    metrics_file = resolve_path(metrics_path)
    if not metrics_file.exists():
        warnings.warn(
            f"NGBoost metrics JSON not found; skipping train-date chronology check: {metrics_file}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    train_date_max = pd.to_datetime(
        payload.get("train_date_range", {}).get("date_max"),
        errors="coerce",
    )
    if pd.isna(train_date_max):
        return
    eval_min = dates.min()
    if train_date_max >= eval_min:
        raise AssertionError(
            "Evaluation rows are not strictly after the training period: "
            f"train max={train_date_max.date()}, eval min={eval_min.date()}"
        )


def load_ngboost_market_bucket_probabilities(
    path: str | Path,
    row_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"NGBoost bucket probability file not found: {input_path}")
    long_df = pd.read_csv(input_path)
    validate_long_bucket_probabilities(long_df)
    required = [
        "row_id",
        "bucket_index",
        "probability",
        "forecast_error",
        "error_lower",
        "error_upper",
    ]
    missing = [column for column in required if column not in long_df.columns]
    if missing:
        raise ValueError(f"NGBoost long bucket probability file is missing columns: {missing}")

    selected_ids = pd.Index(pd.to_numeric(row_ids, errors="raise").astype(int))
    working = long_df[long_df["row_id"].isin(selected_ids)].copy()
    if working.empty:
        raise ValueError("No NGBoost bucket probability rows match distribution parameter row_ids")
    working["row_id"] = pd.to_numeric(working["row_id"], errors="raise").astype(int)
    working["bucket_index"] = pd.to_numeric(
        working["bucket_index"],
        errors="raise",
    ).astype(int)
    working["bucket_label"] = "market_bucket_" + working["bucket_index"].astype(str)

    lower = pd.to_numeric(working["error_lower"], errors="coerce")
    upper = pd.to_numeric(working["error_upper"], errors="coerce")
    error = pd.to_numeric(working["forecast_error"], errors="raise")
    in_bucket = (lower.isna() | (error > lower)) & (upper.isna() | (error <= upper))
    matches = working[in_bucket].copy()
    counts = matches.groupby("row_id").size()
    missing_ids = sorted(set(selected_ids) - set(counts.index))
    bad_counts = counts[counts != 1]
    if missing_ids or not bad_counts.empty:
        raise ValueError(
            "Could not assign exactly one realized market bucket per row. "
            f"missing_rows={len(missing_ids)}, bad_count_rows={len(bad_counts)}"
        )

    labels = matches.set_index("row_id")["bucket_label"].sort_index()
    probabilities = working.pivot(
        index="row_id",
        columns="bucket_label",
        values="probability",
    )
    ordered_columns = [
        f"market_bucket_{bucket_index}"
        for bucket_index in sorted(working["bucket_index"].unique())
    ]
    probabilities = probabilities[ordered_columns].sort_index()
    probabilities = validate_bucket_probabilities(probabilities, allow_renormalize=True)

    if not selected_ids.isin(probabilities.index).all():
        raise ValueError("Some selected row_ids are missing market bucket probabilities")
    probabilities = probabilities.loc[selected_ids]
    labels = labels.loc[selected_ids]
    return probabilities.reset_index(drop=True), labels.reset_index(drop=True)


def compute_distribution_reports(
    params: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    coverage_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    pit_by_split: dict[str, pd.Series] = {}

    for split_name, split_df in iter_evaluation_splits(params):
        validate_distribution_params(split_df)
        coverage = prediction_interval_coverage(
            split_df["forecast_error"],
            split_df["mu"],
            split_df["sigma"],
            levels=(0.5, 0.8, 0.9),
        )
        coverage.insert(0, "split", split_name)
        coverage_frames.append(coverage)

        z = standardized_residuals(split_df["forecast_error"], split_df["mu"], split_df["sigma"])
        summary = residual_summary(z)
        summary.insert(0, "split", split_name)
        residual_frames.append(summary)

        pit_by_split[split_name] = compute_pit_values(
            split_df["forecast_error"],
            split_df["mu"],
            split_df["sigma"],
        )

    return (
        pd.concat(coverage_frames, ignore_index=True),
        pd.concat(residual_frames, ignore_index=True),
        pit_by_split,
    )


def compute_market_bucket_reports(
    params: pd.DataFrame,
    market_probs: pd.DataFrame,
    market_labels: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frames: list[pd.DataFrame] = []
    log_loss_by_split: dict[str, float] = {}
    for split_name, split_df in iter_evaluation_splits(params):
        positions = split_df.index.to_numpy()
        probs = market_probs.iloc[positions].reset_index(drop=True)
        labels = market_labels.iloc[positions].reset_index(drop=True)
        brier = bucket_brier_scores(probs, labels)
        split_log_loss = interval_log_loss(probs, labels)
        brier["interval_log_loss"] = split_log_loss
        brier["mean_bucket_brier"] = float(brier["brier_score"].mean())
        brier.insert(0, "split", split_name)
        brier.insert(1, "bucket_schema", "kalshi_around_forecast_bucket_index")
        frames.append(brier)
        log_loss_by_split[split_name] = split_log_loss
    return pd.concat(frames, ignore_index=True), log_loss_by_split


def compute_and_plot_calibration_tables(
    params: pd.DataFrame,
    market_probs: pd.DataFrame,
    market_labels: pd.Series,
    bucket_brier_report: pd.DataFrame,
) -> pd.DataFrame:
    combined_brier = bucket_brier_report[
        bucket_brier_report["split"] == "combined_out_of_sample"
    ]
    selected_buckets = choose_buckets_for_plots(combined_brier, max_buckets=4)

    tables: list[pd.DataFrame] = []
    for split_name, split_df in iter_evaluation_splits(params):
        positions = split_df.index.to_numpy()
        probs = market_probs.iloc[positions].reset_index(drop=True)
        labels = market_labels.iloc[positions].reset_index(drop=True)
        table = calibration_tables_by_bucket(probs, labels, n_bins=10)
        table = table[table["bucket"].isin(selected_buckets)].reset_index(drop=True)
        table.insert(0, "split", split_name)
        table.insert(1, "model", "ngboost_normal_v0")
        table.insert(2, "bucket_schema", "kalshi_around_forecast_bucket_index")
        tables.append(table)

        if split_name == "combined_out_of_sample":
            for bucket in selected_buckets:
                bucket_table = table[table["bucket"] == bucket]
                plot_calibration_curve(
                    bucket_table,
                    FIGURES_DIR / f"calibration_bucket_{sanitize_filename(bucket)}.png",
                )

    return pd.concat(tables, ignore_index=True)


def compute_and_plot_group_coverage(params: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["prediction_hour"]
    if "season" in params.columns and params["season"].notna().any():
        group_columns.append("season")
    if "forecast_horizon_hours" in params.columns:
        group_columns.append("forecast_horizon_hours")

    frames: list[pd.DataFrame] = []
    for split_name, split_df in iter_evaluation_splits(params):
        for group_col in group_columns:
            grouped = coverage_by_group(split_df, group_col, level=0.8, min_count=30)
            output = grouped.rename(columns={group_col: "group_value"})
            output.insert(0, "split", split_name)
            output.insert(1, "group_col", group_col)
            frames.append(output)

            if split_name == "combined_out_of_sample" and group_col == "prediction_hour":
                plot_coverage_by_group(grouped, group_col, FIGURES_DIR / "coverage_by_hour.png")
            if split_name == "combined_out_of_sample" and group_col == "season":
                plot_coverage_by_group(grouped, group_col, FIGURES_DIR / "coverage_by_season.png")

    return pd.concat(frames, ignore_index=True)


def compute_model_comparison(
    params: pd.DataFrame,
    baseline_path: str | Path,
    coverage_report: pd.DataFrame,
    residual_report: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    comparison_calibration: list[pd.DataFrame] = []

    for split_name in OOS_SPLITS:
        split_df = params[params["split"] == split_name].copy()
        probs = normal_error_interval_probabilities(split_df)
        labels = assign_error_interval_labels(split_df["forecast_error"])
        brier = bucket_brier_scores(probs, labels)
        rows.append(
            build_comparison_row(
                model="ngboost_normal_v0",
                split=split_name,
                n_rows=len(split_df),
                nll=negative_log_likelihood(
                    split_df["forecast_error"],
                    split_df["mu"],
                    split_df["sigma"],
                ),
                interval_log_loss_value=interval_log_loss(probs, labels),
                mean_bucket_brier=float(brier["brier_score"].mean()),
                coverage_report=coverage_report,
                residual_report=residual_report,
            )
        )

    baseline_file = resolve_path(baseline_path)
    if baseline_file.exists():
        baseline = load_empirical_baseline(baseline_file)
        ngboost_test, baseline_aligned = align_ngboost_and_baseline(
            params[params["split"] == "test"],
            baseline,
        )

        ngboost_probs = normal_error_interval_probabilities(ngboost_test)
        ngboost_labels = assign_error_interval_labels(ngboost_test["forecast_error"])
        ngboost_calibration = calibration_tables_by_bucket(ngboost_probs, ngboost_labels, n_bins=10)
        ngboost_calibration.insert(0, "split", "test")
        ngboost_calibration.insert(1, "model", "ngboost_normal_v0")
        ngboost_calibration.insert(2, "bucket_schema", "day9_forecast_error_intervals")
        comparison_calibration.append(ngboost_calibration)

        baseline_probs = baseline_probability_frame(baseline_aligned)
        baseline_labels = baseline_aligned["true_interval"].reset_index(drop=True)
        baseline_brier = bucket_brier_scores(baseline_probs, baseline_labels)
        baseline_calibration = calibration_tables_by_bucket(
            baseline_probs,
            baseline_labels,
            n_bins=10,
        )
        baseline_calibration.insert(0, "split", "test")
        baseline_calibration.insert(1, "model", "empirical_baseline_day9")
        baseline_calibration.insert(2, "bucket_schema", "day9_forecast_error_intervals")
        comparison_calibration.append(baseline_calibration)

        rows.append(
            {
                "model": "empirical_baseline_day9",
                "split": "test",
                "bucket_schema": "day9_forecast_error_intervals",
                "n_rows": int(len(baseline_aligned)),
                "nll": math.nan,
                "interval_log_loss": interval_log_loss(baseline_probs, baseline_labels),
                "mean_bucket_brier": float(baseline_brier["brier_score"].mean()),
                "coverage_error_50": math.nan,
                "coverage_error_80": math.nan,
                "coverage_error_90": math.nan,
                "residual_mean": math.nan,
                "residual_std": math.nan,
            }
        )
    else:
        warnings.warn(
            f"Empirical baseline file not found; skipping baseline comparison: {baseline_file}",
            RuntimeWarning,
            stacklevel=2,
        )

    calibration = (
        pd.concat(comparison_calibration, ignore_index=True)
        if comparison_calibration
        else pd.DataFrame()
    )
    return pd.DataFrame(rows), calibration


def build_comparison_row(
    model: str,
    split: str,
    n_rows: int,
    nll: float,
    interval_log_loss_value: float,
    mean_bucket_brier: float,
    coverage_report: pd.DataFrame,
    residual_report: pd.DataFrame,
) -> dict[str, Any]:
    coverage = coverage_report[coverage_report["split"] == split].set_index("level")
    residual = residual_report[residual_report["split"] == split].iloc[0]
    return {
        "model": model,
        "split": split,
        "bucket_schema": "day9_forecast_error_intervals",
        "n_rows": int(n_rows),
        "nll": float(nll),
        "interval_log_loss": float(interval_log_loss_value),
        "mean_bucket_brier": float(mean_bucket_brier),
        "coverage_error_50": float(coverage.loc[0.5, "coverage_error"]),
        "coverage_error_80": float(coverage.loc[0.8, "coverage_error"]),
        "coverage_error_90": float(coverage.loc[0.9, "coverage_error"]),
        "residual_mean": float(residual["mean"]),
        "residual_std": float(residual["std"]),
    }


def normal_error_interval_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    mu = pd.to_numeric(df["mu"], errors="raise").to_numpy(dtype=float)
    sigma = pd.to_numeric(df["sigma"], errors="raise").to_numpy(dtype=float)
    probabilities: dict[str, np.ndarray] = {}
    for spec in ERROR_INTERVAL_SCHEMA:
        lower = spec["lower"]
        upper = spec["upper"]
        lower_cdf = 0.0 if lower is None else norm.cdf(float(lower), loc=mu, scale=sigma)
        upper_cdf = 1.0 if upper is None else norm.cdf(float(upper), loc=mu, scale=sigma)
        probabilities[spec["label"]] = np.asarray(upper_cdf - lower_cdf, dtype=float)
    probs = pd.DataFrame(probabilities).reset_index(drop=True)
    return validate_bucket_probabilities(probs, allow_renormalize=True)


def assign_error_interval_labels(errors: pd.Series) -> pd.Series:
    labels: list[str] = []
    for raw_value in pd.to_numeric(errors, errors="raise"):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"forecast_error must be finite, got {raw_value!r}")
        matched_label: str | None = None
        for spec in ERROR_INTERVAL_SCHEMA:
            lower = spec["lower"]
            upper = spec["upper"]
            lower_ok = lower is None or value > float(lower)
            upper_ok = upper is None or value <= float(upper)
            if lower_ok and upper_ok:
                matched_label = str(spec["label"])
                break
        if matched_label is None:
            raise ValueError(f"No forecast-error interval matched value {value:g}")
        labels.append(matched_label)
    return pd.Series(labels, name="realized_error_interval")


def load_empirical_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "date",
        "station_id",
        "prediction_hour",
        "forecast_horizon_hours",
        "forecast_error",
        "true_interval",
        *[str(spec["prob_col"]) for spec in ERROR_INTERVAL_SCHEMA],
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Empirical baseline file is missing required columns: {missing}")
    df = add_stable_keys(df)
    probs = baseline_probability_frame(df)
    validate_bucket_probabilities(probs)
    labels = df["true_interval"].reset_index(drop=True)
    assigned = assign_error_interval_labels(df["forecast_error"])
    if not labels.equals(assigned):
        mismatch_count = int((labels != assigned).sum())
        raise ValueError(
            "Empirical baseline true_interval does not match forecast_error intervals: "
            f"{mismatch_count} mismatches"
        )
    return df


def baseline_probability_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = {
        str(spec["label"]): pd.to_numeric(df[str(spec["prob_col"])], errors="raise")
        for spec in ERROR_INTERVAL_SCHEMA
    }
    return pd.DataFrame(data).reset_index(drop=True)


def align_ngboost_and_baseline(
    ngboost_test: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ng = ngboost_test.set_index(KEY_COLUMNS, verify_integrity=True).sort_index()
    base = baseline.set_index(KEY_COLUMNS, verify_integrity=True).sort_index()
    missing_from_baseline = ng.index.difference(base.index)
    extra_baseline = base.index.difference(ng.index)
    if len(missing_from_baseline) or len(extra_baseline):
        raise ValueError(
            "NGBoost and empirical baseline rows do not align on stable keys: "
            f"missing_from_baseline={len(missing_from_baseline)}, "
            f"extra_baseline={len(extra_baseline)}"
        )

    base_aligned = base.loc[ng.index].reset_index()
    ng_aligned = ng.reset_index()
    if not np.allclose(
        pd.to_numeric(ng_aligned["forecast_error"], errors="raise"),
        pd.to_numeric(base_aligned["forecast_error"], errors="raise"),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Aligned NGBoost and empirical baseline forecast_error values differ")
    return ng_aligned, base_aligned


def iter_evaluation_splits(params: pd.DataFrame):
    for split_name in OOS_SPLITS:
        split_df = params[params["split"] == split_name].copy()
        if split_df.empty:
            raise ValueError(f"Evaluation split is empty: {split_name}")
        yield split_name, split_df
    yield "combined_out_of_sample", params[params["split"].isin(OOS_SPLITS)].copy()


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    if not output.exists() or output.stat().st_size == 0:
        raise AssertionError(f"Expected CSV output was not written: {output}")


def print_summary(
    params: pd.DataFrame,
    coverage_report: pd.DataFrame,
    residual_report: pd.DataFrame,
    bucket_brier_report: pd.DataFrame,
    market_log_loss: dict[str, float],
    comparison_report: pd.DataFrame,
) -> None:
    print("Day 13 NGBoost probability-quality evaluation complete.")
    print(f"Evaluated rows: {len(params):,}")
    for split in OOS_SPLITS:
        split_rows = int((params["split"] == split).sum())
        coverage_80 = coverage_report[
            (coverage_report["split"] == split) & (coverage_report["level"] == 0.8)
        ]["actual_coverage"].iloc[0]
        residual_std = residual_report[residual_report["split"] == split]["std"].iloc[0]
        mean_market_brier = bucket_brier_report[
            bucket_brier_report["split"] == split
        ]["brier_score"].mean()
        print(
            f"{split}: rows={split_rows:,}, 80% coverage={coverage_80:.4f}, "
            f"residual std={residual_std:.4f}, market mean Brier={mean_market_brier:.4f}, "
            f"market log loss={market_log_loss[split]:.4f}"
        )
    print(f"Saved coverage report: {COVERAGE_REPORT_PATH}")
    print(f"Saved residual summary: {RESIDUAL_SUMMARY_PATH}")
    print(f"Saved bucket Brier scores: {BUCKET_BRIER_PATH}")
    print(f"Saved calibration tables: {CALIBRATION_TABLES_PATH}")
    print(f"Saved group coverage: {COVERAGE_BY_GROUP_PATH}")
    print(f"Saved model comparison: {EVALUATION_REPORT_PATH}")
    print(f"Saved figures in: {FIGURES_DIR}")
    if not comparison_report.empty:
        print("Model comparison:")
        print(comparison_report.to_string(index=False))


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "bucket"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


if __name__ == "__main__":
    main(sys.argv[1:])
