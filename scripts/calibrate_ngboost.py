from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (  # noqa: E402
    DEFAULT_SIGMA_ALPHA_GRID,
    apply_sigma_scaling,
    calibration_tables_by_bucket,
    cdf_reliability_table,
    fit_global_sigma_scale,
    plot_coverage_before_after,
    plot_sigma_scaling_validation_nll,
)
from src.distribution_pricing import (  # noqa: E402
    validate_bucket_probabilities as validate_long_bucket_probabilities,
)
from src.distributional_model import distribution_cdf, normalize_distribution_name  # noqa: E402
from src.evaluation import (  # noqa: E402
    bucket_brier_scores,
    grouped_interval_coverage_report,
    interval_coverage_report,
    interval_log_loss,
    negative_log_likelihood,
    prediction_interval_coverage,
    validate_bucket_probabilities,
    validate_distribution_params,
)
from src.ngboost_predict import apply_sigma_scaling_to_predictions  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_params_v0.csv"
DEFAULT_BUCKET_PROBS_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_probs_v0.csv"
DEFAULT_MODELING_ROWS_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"

RAW_COVERAGE_PATH = REPO_ROOT / "outputs" / "ngboost_interval_coverage_raw.csv"
COVERAGE_BY_HOUR_PATH = REPO_ROOT / "outputs" / "ngboost_coverage_by_hour.csv"
COVERAGE_BY_SEASON_PATH = REPO_ROOT / "outputs" / "ngboost_coverage_by_season.csv"
COVERAGE_BY_HORIZON_PATH = REPO_ROOT / "outputs" / "ngboost_coverage_by_horizon.csv"
ALPHA_SEARCH_PATH = REPO_ROOT / "outputs" / "ngboost_sigma_scaling_alpha_search.csv"
CALIBRATION_REPORT_PATH = REPO_ROOT / "outputs" / "ngboost_calibration_report.csv"
METHOD_COVERAGE_PATH = REPO_ROOT / "outputs" / "ngboost_interval_coverage_before_after.csv"
CALIBRATED_BUCKETS_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_probabilities_calibrated.csv"
CDF_RELIABILITY_PATH = REPO_ROOT / "outputs" / "ngboost_cdf_reliability_table.csv"
BUCKET_RELIABILITY_PATH = REPO_ROOT / "outputs" / "ngboost_bucket_reliability_table.csv"
CALIBRATION_CONFIG_PATH = REPO_ROOT / "models" / "calibration_config.json"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

OOS_SPLITS = ["validation", "test"]
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
GROUP_COVERAGE_LEVELS = (0.5, 0.8, 0.9)
CDF_THRESHOLDS = (-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    params = enrich_prediction_metadata(load_prediction_params(args.params_path), args.modeling_rows_path)
    dist_type = infer_params_distribution_type(params, args.dist_type)
    split_summary = validate_split_discipline(params)
    bucket_template = load_bucket_template(args.bucket_probs_path, params)

    raw_coverage = build_raw_interval_coverage(params, dist_type=dist_type)
    write_csv(raw_coverage, RAW_COVERAGE_PATH)
    write_group_coverage_outputs(params, min_group_n=args.min_group_n, dist_type=dist_type)

    alpha_search = build_alpha_search(params, bucket_template, args, dist_type=dist_type)
    selected_alpha, selection_details = choose_selected_alpha(alpha_search, args.nll_tolerance)
    alpha_search["selected"] = np.isclose(alpha_search["alpha"], selected_alpha)
    write_csv(alpha_search, ALPHA_SEARCH_PATH)

    raw_buckets = recompute_bucket_probabilities(
        bucket_template,
        alpha=1.0,
        method="raw_ngboost",
        dist_type=dist_type,
    )
    calibrated_buckets = recompute_bucket_probabilities(
        bucket_template,
        alpha=selected_alpha,
        method="global_sigma_scaled",
        dist_type=dist_type,
    )
    write_csv(calibrated_buckets, CALIBRATED_BUCKETS_PATH)

    methods = {
        "raw_ngboost": {"alpha": 1.0, "bucket_probs": raw_buckets},
        "global_sigma_scaled": {"alpha": selected_alpha, "bucket_probs": calibrated_buckets},
    }
    report = build_calibration_report(params, methods, dist_type=dist_type)
    write_csv(report, CALIBRATION_REPORT_PATH)

    method_coverage = build_method_coverage_table(report)
    write_csv(method_coverage, METHOD_COVERAGE_PATH)

    cdf_reliability = build_cdf_reliability_tables(params, methods, dist_type=dist_type)
    write_csv(cdf_reliability, CDF_RELIABILITY_PATH)

    bucket_reliability = build_bucket_reliability_tables(params, methods)
    write_csv(bucket_reliability, BUCKET_RELIABILITY_PATH)

    write_calibration_config(
        selected_alpha=selected_alpha,
        selection_details=selection_details,
        split_summary=split_summary,
        alpha_search=alpha_search,
        args=args,
        dist_type=dist_type,
    )
    write_plots(method_coverage, alpha_search, selected_alpha)

    print("Day 18 NGBoost calibration complete.")
    print(f"Selected alpha from validation only: {selected_alpha:.4f}")
    print(f"Saved calibration report: {CALIBRATION_REPORT_PATH}")
    print(f"Saved calibrated bucket probabilities: {CALIBRATED_BUCKETS_PATH}")
    print(f"Saved calibration config: {CALIBRATION_CONFIG_PATH}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate NGBoost predictive sigma using validation-only global scaling."
    )
    parser.add_argument("--params-path", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--bucket-probs-path", default=str(DEFAULT_BUCKET_PROBS_PATH))
    parser.add_argument("--modeling-rows-path", default=str(DEFAULT_MODELING_ROWS_PATH))
    parser.add_argument(
        "--dist-type",
        default="auto",
        help="Distribution to use for calibration. Defaults to auto-infer from params.",
    )
    parser.add_argument("--min-group-n", type=int, default=30)
    parser.add_argument(
        "--nll-tolerance",
        type=float,
        default=0.02,
        help="Maximum validation NLL degradation from raw alpha=1.0 allowed for selection.",
    )
    parser.add_argument(
        "--coverage-penalty-weight",
        type=float,
        default=0.25,
        help="Weight for absolute coverage gaps in validation alpha scoring.",
    )
    return parser.parse_args(argv)


def load_prediction_params(path: str | Path) -> pd.DataFrame:
    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"NGBoost parameter file not found: {input_path}")
    df = pd.read_csv(input_path)
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=int))
    required = ["row_id", "split", "date", "forecast_error", "forecast_high", "mu", "sigma"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"NGBoost parameter file is missing required columns: {missing}")
    if "distribution_type" not in df.columns:
        df["distribution_type"] = "normal"
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    validate_distribution_params(df)
    return df


def infer_params_distribution_type(params: pd.DataFrame, requested_dist_type: str = "auto") -> str:
    requested = str(requested_dist_type).strip().lower()
    if requested not in {"", "auto", "infer"}:
        return normalize_distribution_name(str(requested_dist_type))
    if "distribution_type" not in params.columns:
        return "normal"
    values = params["distribution_type"].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "normal"
    normalized = {normalize_distribution_name(value) for value in values.unique()}
    if len(normalized) != 1:
        raise ValueError(f"Prediction params contain multiple distribution types: {sorted(normalized)}")
    return next(iter(normalized))


def enrich_prediction_metadata(params: pd.DataFrame, modeling_rows_path: str | Path) -> pd.DataFrame:
    working = params.copy()
    dates = pd.to_datetime(working["date"], errors="raise")
    if "month" not in working.columns or working["month"].isna().any():
        working["month"] = dates.dt.month
    if "season" not in working.columns or working["season"].isna().any():
        working["season"] = season_from_month(dates.dt.month)
    if "prediction_hour" not in working.columns:
        working["prediction_hour"] = derive_prediction_hour(working)

    modeling_path = resolve_path(modeling_rows_path)
    if modeling_path.exists() and {"season", "month"}.difference(working.columns):
        modeling = pd.read_csv(modeling_path)
        keep = [
            column
            for column in ["date", "prediction_time", "forecast_horizon_hours", "season", "month"]
            if column in modeling.columns
        ]
        if {"date", "prediction_time", "forecast_horizon_hours"}.issubset(keep):
            modeling = modeling[keep].drop_duplicates(
                ["date", "prediction_time", "forecast_horizon_hours"]
            )
            working = working.merge(
                modeling,
                on=["date", "prediction_time", "forecast_horizon_hours"],
                how="left",
                suffixes=("", "_modeling"),
            )
    return working


def season_from_month(month: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(month, errors="raise").astype(int)
    values = np.select(
        [
            numeric.isin([12, 1, 2]),
            numeric.isin([3, 4, 5]),
            numeric.isin([6, 7, 8]),
            numeric.isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "fall"],
        default="unknown",
    )
    return pd.Series(values, index=month.index)


def derive_prediction_hour(df: pd.DataFrame) -> pd.Series:
    for column in ["prediction_time", "prediction_timestamp", "timestamp"]:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        if not parsed.isna().any():
            return parsed.dt.hour.astype(int)
        raw = df[column].astype(str).str.strip()
        hour = pd.to_numeric(raw.str.split(":").str[0], errors="coerce")
        if not hour.isna().any():
            return hour.astype(int)
    raise ValueError("Could not derive prediction_hour from prediction timestamp columns")


def validate_split_discipline(params: pd.DataFrame) -> dict[str, Any]:
    splits = set(params["split"].dropna().astype(str))
    missing = [split for split in OOS_SPLITS if split not in splits]
    if missing:
        raise ValueError(f"Required validation/test split(s) are missing: {missing}")

    dates = pd.to_datetime(params["date"], errors="raise").dt.normalize()
    summary: dict[str, Any] = {
        "fit_split": "validation",
        "test_used_for_alpha_selection": False,
        "splits": {},
    }
    for split in sorted(splits):
        mask = params["split"].astype(str) == split
        split_dates = dates[mask]
        summary["splits"][split] = {
            "row_count": int(mask.sum()),
            "date_min": split_dates.min().date().isoformat(),
            "date_max": split_dates.max().date().isoformat(),
        }

    validation_max = dates[params["split"].astype(str) == "validation"].max()
    test_min = dates[params["split"].astype(str) == "test"].min()
    if validation_max >= test_min:
        raise AssertionError(
            "Calibration split leakage: validation max date must be before test min date "
            f"({validation_max.date()} >= {test_min.date()})"
        )
    if "train" in splits:
        train_max = dates[params["split"].astype(str) == "train"].max()
        validation_min = dates[params["split"].astype(str) == "validation"].min()
        if train_max >= validation_min:
            raise AssertionError(
                "Calibration split leakage: train max date must be before validation min date "
                f"({train_max.date()} >= {validation_min.date()})"
            )
    return summary


def load_bucket_template(path: str | Path, params: pd.DataFrame) -> pd.DataFrame:
    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"NGBoost bucket probability file not found: {input_path}")
    long_df = pd.read_csv(input_path)
    if "row_id" not in long_df.columns:
        raise ValueError("Bucket probability file must include row_id")
    required = ["row_id", "bucket_index", "probability", "error_lower", "error_upper"]
    missing = [column for column in required if column not in long_df.columns]
    if missing:
        raise ValueError(f"Bucket probability file is missing required columns: {missing}")
    long_df["row_id"] = pd.to_numeric(long_df["row_id"], errors="raise").astype(int)

    param_columns = [
        "row_id",
        "mu",
        "sigma",
        "forecast_error",
        "forecast_high",
        "distribution_type",
    ]
    if "split" not in long_df.columns:
        param_columns.append("split")
    if "df" in params.columns:
        param_columns.append("df")
    merged = long_df.drop(
        columns=[column for column in ["mu", "sigma", "forecast_error", "forecast_high"] if column in long_df],
        errors="ignore",
    ).merge(
        params[param_columns],
        on="row_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_params"),
    )
    expected_rows = len(params) * int(long_df["bucket_index"].nunique())
    if len(merged) != expected_rows:
        raise ValueError(
            "Bucket template does not align to prediction params: "
            f"got {len(merged)} rows, expected {expected_rows}"
        )
    validate_long_bucket_probabilities(merged)
    return merged.sort_values(["row_id", "bucket_index"], kind="stable").reset_index(drop=True)


def build_raw_interval_coverage(params: pd.DataFrame, dist_type: str) -> pd.DataFrame:
    frames = []
    for split, split_df in iter_oos_splits(params):
        coverage = interval_coverage_report(
            split_df["forecast_error"],
            split_df["mu"],
            split_df["sigma"],
            split=split,
            levels=COVERAGE_LEVELS,
            dist_type=dist_type,
            df=split_df["df"] if normalize_distribution_name(dist_type) == "student_t" else None,
            skew=split_df["skew"] if normalize_distribution_name(dist_type) == "skew_normal" else None,
        )
        frames.append(coverage.drop(columns=["avg_interval_width"]))
    return pd.concat(frames, ignore_index=True)


def write_group_coverage_outputs(params: pd.DataFrame, min_group_n: int, dist_type: str) -> None:
    outputs = {
        "prediction_hour": COVERAGE_BY_HOUR_PATH,
        "season": COVERAGE_BY_SEASON_PATH,
        "forecast_horizon_hours": COVERAGE_BY_HORIZON_PATH,
    }
    for group_col, path in outputs.items():
        if group_col not in params.columns:
            continue
        frames = []
        for split, split_df in iter_oos_splits(params):
            frames.append(
                grouped_interval_coverage_report(
                    split_df,
                    group_col=group_col,
                    split=split,
                    levels=GROUP_COVERAGE_LEVELS,
                    dist_type=dist_type,
                    min_group_n=min_group_n,
                )
            )
        write_csv(pd.concat(frames, ignore_index=True), path)


def build_alpha_search(
    params: pd.DataFrame,
    bucket_template: pd.DataFrame,
    args: argparse.Namespace,
    dist_type: str,
) -> pd.DataFrame:
    validation = params[params["split"].astype(str) == "validation"].copy()
    # Safeguard: alpha search is fit on validation only. Test rows are scored later for reporting only.
    _selected, search = fit_global_sigma_scale(
        validation["forecast_error"],
        validation["mu"],
        validation["sigma"],
        alpha_grid=tuple(DEFAULT_SIGMA_ALPHA_GRID),
        coverage_levels=COVERAGE_LEVELS,
        dist_type=dist_type,
        df=validation["df"] if normalize_distribution_name(dist_type) == "student_t" else None,
        skew=validation["skew"] if normalize_distribution_name(dist_type) == "skew_normal" else None,
        coverage_penalty_weight=args.coverage_penalty_weight,
    )
    search = search.rename(columns={"nll": "validation_nll"})

    validation_ids = validation["row_id"].to_numpy(dtype=int)
    validation_template = bucket_template[bucket_template["row_id"].isin(validation_ids)].copy()
    bucket_brier: list[float] = []
    bucket_log_loss: list[float] = []
    for alpha in search["alpha"]:
        priced = recompute_bucket_probabilities(
            validation_template,
            alpha=float(alpha),
            method="alpha_search",
            dist_type=dist_type,
        )
        probs, labels = bucket_probability_frame_and_labels(priced, row_ids=validation_ids)
        brier = bucket_brier_scores(probs, labels)
        bucket_brier.append(float(brier["brier_score"].mean()))
        bucket_log_loss.append(float(interval_log_loss(probs, labels)))

    search["bucket_brier"] = bucket_brier
    search["bucket_log_loss"] = bucket_log_loss
    search["distance_from_raw_alpha"] = (search["alpha"] - 1.0).abs()
    search["selection_score"] = (
        search["validation_nll"] + args.coverage_penalty_weight * search["coverage_penalty"]
    )
    return search.sort_values("alpha").reset_index(drop=True)


def choose_selected_alpha(alpha_search: pd.DataFrame, nll_tolerance: float) -> tuple[float, dict[str, Any]]:
    if alpha_search.empty:
        raise ValueError("alpha_search is empty")
    if not math.isfinite(float(nll_tolerance)) or float(nll_tolerance) < 0.0:
        raise ValueError("nll_tolerance must be finite and nonnegative")

    raw_rows = alpha_search[np.isclose(alpha_search["alpha"], 1.0)]
    if raw_rows.empty:
        raise ValueError("alpha grid must include 1.0 for raw comparison")
    raw_nll = float(raw_rows.iloc[0]["validation_nll"])
    eligible = alpha_search[alpha_search["validation_nll"] <= raw_nll + float(nll_tolerance)].copy()
    if eligible.empty:
        eligible = alpha_search.copy()
    selected = eligible.sort_values(
        [
            "coverage_penalty",
            "validation_nll",
            "bucket_log_loss",
            "distance_from_raw_alpha",
            "alpha",
        ],
        kind="stable",
    ).iloc[0]
    details = {
        "selection_rule": (
            "validation-only alpha search; eligible alphas must be within "
            f"{float(nll_tolerance):.4f} NLL of raw alpha=1.0, then minimize "
            "absolute validation coverage gaps, with validation NLL and bucket log loss "
            "as tie-breakers"
        ),
        "raw_validation_nll": raw_nll,
        "eligible_alpha_count": int(len(eligible)),
        "selected_validation_nll": float(selected["validation_nll"]),
        "selected_coverage_penalty": float(selected["coverage_penalty"]),
    }
    return float(selected["alpha"]), details


def recompute_bucket_probabilities(
    bucket_template: pd.DataFrame,
    alpha: float,
    method: str,
    dist_type: str = "normal",
) -> pd.DataFrame:
    working = bucket_template.copy()
    working = apply_sigma_scaling_to_predictions(
        working,
        alpha=alpha,
        sigma_col="sigma",
        output_sigma_col="sigma",
        raw_sigma_col="raw_sigma",
    )
    dist = normalize_distribution_name(dist_type)
    working["distribution_type"] = dist

    mu = pd.to_numeric(working["mu"], errors="raise").to_numpy(dtype=float)
    sigma = pd.to_numeric(working["sigma"], errors="raise").to_numpy(dtype=float)
    lower = pd.to_numeric(working["error_lower"], errors="coerce")
    upper = pd.to_numeric(working["error_upper"], errors="coerce")
    df_values = (
        pd.to_numeric(working["df"], errors="raise").to_numpy(dtype=float)
        if dist == "student_t" and "df" in working.columns
        else None
    )
    skew_values = (
        pd.to_numeric(working["skew"], errors="raise").to_numpy(dtype=float)
        if dist == "skew_normal" and "skew" in working.columns
        else None
    )

    lower_cdf = np.zeros(len(working), dtype=float)
    lower_mask = lower.notna().to_numpy()
    if lower_mask.any():
        lower_cdf[lower_mask] = distribution_cdf(
            lower[lower_mask].to_numpy(dtype=float),
            mu=mu[lower_mask],
            sigma=sigma[lower_mask],
            distribution=dist,
            df=df_values[lower_mask] if df_values is not None else None,
            skew=skew_values[lower_mask] if skew_values is not None else None,
        )

    upper_cdf = np.ones(len(working), dtype=float)
    upper_mask = upper.notna().to_numpy()
    if upper_mask.any():
        upper_cdf[upper_mask] = distribution_cdf(
            upper[upper_mask].to_numpy(dtype=float),
            mu=mu[upper_mask],
            sigma=sigma[upper_mask],
            distribution=dist,
            df=df_values[upper_mask] if df_values is not None else None,
            skew=skew_values[upper_mask] if skew_values is not None else None,
        )

    probability = np.clip(upper_cdf - lower_cdf, 0.0, 1.0)
    if not np.isfinite(probability).all():
        raise ValueError("Calibrated bucket probabilities contain non-finite values")
    working["probability"] = probability
    working["method"] = str(method)
    working["alpha"] = float(alpha)
    validate_long_bucket_probabilities(working)
    return working


def build_calibration_report(
    params: pd.DataFrame,
    methods: dict[str, dict[str, Any]],
    dist_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, spec in methods.items():
        alpha = float(spec["alpha"])
        bucket_probs = spec["bucket_probs"]
        for split, split_df in iter_oos_splits(params):
            sigma = apply_sigma_scaling(split_df["sigma"], alpha)
            coverage = prediction_interval_coverage(
                split_df["forecast_error"],
                split_df["mu"],
                sigma,
                levels=COVERAGE_LEVELS,
                dist_type=dist_type,
                df=split_df["df"] if normalize_distribution_name(dist_type) == "student_t" else None,
                skew=split_df["skew"] if normalize_distribution_name(dist_type) == "skew_normal" else None,
            ).set_index("level")
            row_ids = split_df["row_id"].to_numpy(dtype=int)
            probs, labels = bucket_probability_frame_and_labels(bucket_probs, row_ids=row_ids)
            brier = bucket_brier_scores(probs, labels)
            row = {
                "method": method,
                "alpha": alpha,
                "split": split,
                "nll": negative_log_likelihood(
                    split_df["forecast_error"],
                    split_df["mu"],
                    sigma,
                    dist_type=dist_type,
                    df=split_df["df"] if normalize_distribution_name(dist_type) == "student_t" else None,
                    skew=split_df["skew"] if normalize_distribution_name(dist_type) == "skew_normal" else None,
                ),
                "bucket_brier": float(brier["brier_score"].mean()),
                "bucket_log_loss": interval_log_loss(probs, labels),
                "mean_sigma": float(np.mean(sigma)),
                "median_sigma": float(np.median(sigma)),
            }
            for level in COVERAGE_LEVELS:
                suffix = coverage_suffix(level)
                row[f"coverage_{suffix}"] = float(coverage.loc[level, "actual_coverage"])
                row[f"coverage_gap_{suffix}"] = float(coverage.loc[level, "coverage_error"])
            rows.append(row)
    columns = [
        "method",
        "alpha",
        "split",
        "nll",
        "bucket_brier",
        "bucket_log_loss",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "coverage_95",
        "coverage_gap_50",
        "coverage_gap_80",
        "coverage_gap_90",
        "coverage_gap_95",
        "mean_sigma",
        "median_sigma",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_method_coverage_table(report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in report.iterrows():
        for level in COVERAGE_LEVELS:
            suffix = coverage_suffix(level)
            rows.append(
                {
                    "method": row["method"],
                    "alpha": float(row["alpha"]),
                    "split": row["split"],
                    "nominal_coverage": float(level),
                    "empirical_coverage": float(row[f"coverage_{suffix}"]),
                    "coverage_gap": float(row[f"coverage_gap_{suffix}"]),
                }
            )
    return pd.DataFrame(rows)


def build_cdf_reliability_tables(
    params: pd.DataFrame,
    methods: dict[str, dict[str, Any]],
    dist_type: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, spec in methods.items():
        alpha = float(spec["alpha"])
        for split, split_df in iter_oos_splits(params):
            sigma = apply_sigma_scaling(split_df["sigma"], alpha)
            frames.append(
                cdf_reliability_table(
                    split_df["forecast_error"],
                    split_df["mu"],
                    sigma,
                    thresholds=CDF_THRESHOLDS,
                    split=split,
                    method=method,
                    dist_type=dist_type,
                    df=split_df["df"] if normalize_distribution_name(dist_type) == "student_t" else None,
                    skew=split_df["skew"] if normalize_distribution_name(dist_type) == "skew_normal" else None,
                )
            )
    return pd.concat(frames, ignore_index=True)


def build_bucket_reliability_tables(
    params: pd.DataFrame,
    methods: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, spec in methods.items():
        for split, split_df in iter_oos_splits(params):
            probs, labels = bucket_probability_frame_and_labels(
                spec["bucket_probs"],
                row_ids=split_df["row_id"].to_numpy(dtype=int),
            )
            table = calibration_tables_by_bucket(probs, labels, n_bins=10)
            table["prob_bin"] = table.apply(
                lambda row: f"{float(row['bin_lower']):.1f}-{float(row['bin_upper']):.1f}",
                axis=1,
            )
            table["calibration_error"] = (
                table["empirical_frequency"] - table["mean_predicted_probability"]
            )
            table.insert(0, "method", method)
            table.insert(0, "split", split)
            frames.append(
                table[
                    [
                        "split",
                        "method",
                        "bucket",
                        "prob_bin",
                        "mean_predicted_probability",
                        "empirical_frequency",
                        "count",
                        "calibration_error",
                        "bin_lower",
                        "bin_upper",
                    ]
                ]
            )
    return pd.concat(frames, ignore_index=True)


def bucket_probability_frame_and_labels(
    long_df: pd.DataFrame,
    row_ids: np.ndarray | list[int] | pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    selected_ids = pd.Index(pd.Series(row_ids, dtype=int), name="row_id")
    working = long_df[long_df["row_id"].isin(selected_ids)].copy()
    if working.empty:
        raise ValueError("No bucket probability rows matched selected row ids")
    working["bucket"] = "market_bucket_" + pd.to_numeric(
        working["bucket_index"],
        errors="raise",
    ).astype(int).astype(str)

    lower = pd.to_numeric(working["error_lower"], errors="coerce")
    upper = pd.to_numeric(working["error_upper"], errors="coerce")
    error = pd.to_numeric(working["forecast_error"], errors="raise")
    in_bucket = (lower.isna() | (error > lower)) & (upper.isna() | (error <= upper))
    matches = working[in_bucket].copy()
    counts = matches.groupby("row_id").size()
    missing_ids = selected_ids.difference(pd.Index(counts.index))
    bad_counts = counts[counts != 1]
    if len(missing_ids) or len(bad_counts):
        raise ValueError(
            "Could not assign exactly one realized bucket per row: "
            f"missing={len(missing_ids)}, bad_counts={len(bad_counts)}"
        )

    probs = working.pivot(index="row_id", columns="bucket", values="probability")
    ordered_columns = [
        f"market_bucket_{bucket_index}"
        for bucket_index in sorted(pd.to_numeric(working["bucket_index"], errors="raise").unique())
    ]
    probs = probs[ordered_columns].loc[selected_ids]
    labels = matches.set_index("row_id")["bucket"].loc[selected_ids]
    return (
        validate_bucket_probabilities(probs.reset_index(drop=True), allow_renormalize=True),
        labels.reset_index(drop=True),
    )


def write_calibration_config(
    selected_alpha: float,
    selection_details: dict[str, Any],
    split_summary: dict[str, Any],
    alpha_search: pd.DataFrame,
    args: argparse.Namespace,
    dist_type: str,
) -> None:
    selected_row = alpha_search[np.isclose(alpha_search["alpha"], selected_alpha)].iloc[0].to_dict()
    payload = {
        "model_name": "ngboost_calibrated_v3",
        "base_model": Path(args.params_path).name,
        "base_distribution_type": normalize_distribution_name(dist_type),
        "calibration_method": "global_sigma_scaling",
        "alpha": float(selected_alpha),
        "fit_split": "validation",
        "selection_metric": "validation_nll_with_coverage_check",
        "selection_details": to_jsonable(selection_details),
        "selected_validation_metrics": to_jsonable(selected_row),
        "split_summary": split_summary,
        "alpha_grid": DEFAULT_SIGMA_ALPHA_GRID,
        "test_set_used_for_alpha_selection": False,
        "notes": (
            "Alpha selected using validation only. Test set used only for final before/after "
            "reporting; do not tune alpha or group-specific calibration from test results."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    CALIBRATION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_plots(
    method_coverage: pd.DataFrame,
    alpha_search: pd.DataFrame,
    selected_alpha: float,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_coverage_before_after(
        method_coverage,
        FIGURES_DIR / "coverage_before_after.png",
    )
    plot_sigma_scaling_validation_nll(
        alpha_search,
        FIGURES_DIR / "sigma_scaling_validation_nll.png",
        selected_alpha=selected_alpha,
    )


def iter_oos_splits(params: pd.DataFrame):
    for split in OOS_SPLITS:
        split_df = params[params["split"].astype(str) == split].copy()
        if split_df.empty:
            raise ValueError(f"Split {split!r} is empty")
        yield split, split_df


def coverage_suffix(level: float) -> str:
    return f"{int(round(float(level) * 100)):02d}"


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    if not output.exists() or output.stat().st_size == 0:
        raise AssertionError(f"Expected output was not written: {output}")


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main(sys.argv[1:])
