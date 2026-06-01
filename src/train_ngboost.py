from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import yaml


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import make_calibration_table, plot_calibration_curve, plot_pit_histogram  # noqa: E402
from src.distribution_pricing import price_buckets_for_dataframe  # noqa: E402
from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
    distribution_nll,
    get_feature_columns,
    normal_nll,
    normalize_distribution_name,
    predict_distribution_details,
    predict_distribution_params,
    train_ngboost_distribution,
    train_ngboost_normal,
    validate_no_leakage_feature_columns,
)
from src.evaluation import (  # noqa: E402
    compute_pit_values,
    interval_log_loss,
    prediction_interval_coverage,
    validate_bucket_probabilities,
)
from src.features import load_feature_list, validate_feature_columns_exist  # noqa: E402
from src.splits import chronological_train_validation_test_split  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
DEFAULT_MODELING_TABLE_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"
DEFAULT_FEATURE_COLUMNS_PATH = REPO_ROOT / "outputs" / "day8_features" / "feature_columns.json"
DEFAULT_FINAL_FEATURE_LIST_PATH = REPO_ROOT / "outputs" / "final_feature_list.json"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_normal_v0.pkl"
FEATURE_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_normal_v0_features.json"
PARAMS_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_params_v0.csv"
METRICS_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_nll_v0.json"
TUNING_SEARCH_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_hyperparameter_search.csv"
BEST_MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_best_v2.pkl"
BEST_METADATA_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_best_v2_metadata.json"
BEST_NOTES_OUTPUT_PATH = REPO_ROOT / "outputs" / "best_ngboost_v2_notes.md"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
EMPIRICAL_BASELINE_PATH = (
    REPO_ROOT / "outputs" / "day9_empirical_baseline" / "empirical_baseline_predictions.csv"
)

SEARCH_RESULT_COLUMNS = [
    "run_id",
    "timestamp",
    "status",
    "error_message",
    "seed",
    "distribution",
    "features_version",
    "target_column",
    "date_column",
    "n_train",
    "n_val",
    "n_test",
    "train_start",
    "train_end",
    "val_start",
    "val_end",
    "test_start",
    "test_end",
    "candidate_name",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_samples_leaf",
    "minibatch_frac",
    "natural_gradient",
    "train_nll",
    "val_nll",
    "val_mae",
    "val_rmse",
    "val_bias",
    "val_50_coverage",
    "val_80_coverage",
    "val_90_coverage",
    "val_50_coverage_error",
    "val_80_coverage_error",
    "val_90_coverage_error",
    "val_bucket_brier",
    "val_interval_log_loss",
    "val_calibration_error",
    "val_pit_mean",
    "val_pit_std",
    "val_pit_min",
    "val_pit_max",
    "bucket_prob_min",
    "bucket_prob_max",
    "bucket_prob_sum_mean",
    "bucket_prob_sum_max_abs_error",
    "bucket_prob_normalization_rate",
    "notes",
]

DATETIME_COLUMNS = [
    "date",
    "target_date",
    "prediction_time",
    "prediction_timestamp",
    "current_temp_source_time",
    "max_temp_so_far_source_time",
    "forecast_temp_source_valid_time",
    "forecast_max_so_far_source_valid_time",
]

METADATA_COLUMNS = [
    "date",
    "timestamp",
    "prediction_time",
    "prediction_timestamp",
    "location",
    "station",
    "station_id",
    "forecast_high",
    "actual_high",
    "official_high",
    "forecast_horizon_hours",
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "tune":
        run_tuning(args)
        return
    run_standard_training(args)


def run_standard_training(args: argparse.Namespace) -> None:
    dataset_path = _resolve_path(args.dataset_path)
    feature_list_path = _resolve_path(args.feature_list)
    config = load_model_config(_resolve_path(args.config_path))
    standard_config = standard_training_config(config, args)
    distribution = str(standard_config["distribution"])
    model_name = f"ngboost_{distribution}_v0"

    df = load_modeling_table(dataset_path)
    validate_target_column(df)

    split_result = chronological_train_validation_test_split(
        df,
        train_end_date=args.train_end_date,
        validation_end_date=args.validation_end_date,
    )
    train_df = split_result.train
    validation_df = split_result.validation
    test_df = split_result.test
    split_summary = split_result.summary

    validate_target_column(train_df, split_name="train")
    validate_target_column(validation_df, split_name="validation")
    validate_target_column(test_df, split_name="test")

    feature_columns = load_explicit_feature_list(df, feature_list_path)
    feature_source_path = feature_list_path

    X_train, X_validation, X_test, imputer, preprocessing_notes = build_imputed_feature_frames(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = validation_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model = train_ngboost_distribution(
        X_train=X_train,
        y_train=y_train,
        X_val=X_validation,
        y_val=y_validation,
        distribution=distribution,
        n_estimators=int(standard_config["n_estimators"]),
        learning_rate=float(standard_config["learning_rate"]),
        max_depth=int(standard_config["max_depth"]),
        min_samples_leaf=int(standard_config["min_samples_leaf"]),
        minibatch_frac=float(standard_config["minibatch_frac"]),
        natural_gradient=bool(standard_config["natural_gradient"]),
        random_state=int(standard_config["random_state"]),
        early_stopping_rounds=standard_config["early_stopping_rounds"],
    )

    validation_details = predict_distribution_details(model, X_validation, distribution)
    test_details = predict_distribution_details(model, X_test, distribution)
    validation_mu = np.asarray(validation_details["mu"], dtype=float)
    raw_validation_sigma = np.asarray(validation_details["sigma"], dtype=float)
    test_mu = np.asarray(test_details["mu"], dtype=float)
    raw_test_sigma = np.asarray(test_details["sigma"], dtype=float)
    sigma_scale = float(standard_config["sigma_scale"])
    validation_sigma = raw_validation_sigma * sigma_scale
    test_sigma = raw_test_sigma * sigma_scale

    validation_nll = distribution_nll(
        y_validation,
        validation_mu,
        validation_sigma,
        distribution=distribution,
        df=validation_details.get("df"),
    )
    test_nll = distribution_nll(
        y_test,
        test_mu,
        test_sigma,
        distribution=distribution,
        df=test_details.get("df"),
    )
    validate_distribution_outputs(
        split_name="validation",
        expected_count=len(validation_df),
        mu=validation_mu,
        sigma=validation_sigma,
        nll=validation_nll,
    )
    validate_distribution_outputs(
        split_name="test",
        expected_count=len(test_df),
        mu=test_mu,
        sigma=test_sigma,
        nll=test_nll,
    )

    baseline_mu = float(np.mean(y_train))
    baseline_sigma = float(np.std(y_train, ddof=1))
    if not np.isfinite(baseline_sigma) or baseline_sigma <= 0:
        raise ValueError("Train target standard deviation is not positive; cannot score baseline")
    validation_baseline_nll = normal_nll(
        y_validation,
        np.full_like(y_validation, baseline_mu, dtype=float),
        np.full_like(y_validation, baseline_sigma, dtype=float),
    )
    test_baseline_nll = normal_nll(
        y_test,
        np.full_like(y_test, baseline_mu, dtype=float),
        np.full_like(y_test, baseline_sigma, dtype=float),
    )

    predictions = pd.concat(
        [
            build_prediction_frame(
                split_name="validation",
                split_df=validation_df,
                mu=validation_mu,
                sigma=validation_sigma,
                nll=validation_nll,
                distribution_type=distribution,
                df=validation_details.get("df"),
            ),
            build_prediction_frame(
                split_name="test",
                split_df=test_df,
                mu=test_mu,
                sigma=test_sigma,
                nll=test_nll,
                distribution_type=distribution,
                df=test_details.get("df"),
            ),
        ],
        ignore_index=True,
    )
    predictions.insert(0, "row_id", np.arange(len(predictions), dtype=int))
    predictions["sigma_scale"] = sigma_scale
    if not np.isclose(sigma_scale, 1.0):
        predictions["raw_sigma"] = np.concatenate([raw_validation_sigma, raw_test_sigma])
    if len(predictions) != len(validation_df) + len(test_df):
        raise AssertionError("Distribution prediction count does not match validation/test rows")

    sigma_summary = {
        "validation": summarize_sigma(validation_sigma),
        "test": summarize_sigma(test_sigma),
    }
    metrics = build_metrics(
        model_name=model_name,
        distribution_type=distribution,
        sigma_scale=sigma_scale,
        split_summary=split_summary,
        feature_columns=feature_columns,
        validation_nll=validation_nll,
        test_nll=test_nll,
        validation_baseline_nll=validation_baseline_nll,
        test_baseline_nll=test_baseline_nll,
        baseline_mu=baseline_mu,
        baseline_sigma=baseline_sigma,
        sigma_summary=sigma_summary,
        preprocessing_notes=preprocessing_notes,
        empirical_baseline_note=build_empirical_baseline_note(EMPIRICAL_BASELINE_PATH),
    )

    write_outputs(
        model=model,
        imputer=imputer,
        feature_columns=feature_columns,
        model_name=model_name,
        distribution_type=distribution,
        sigma_scale=sigma_scale,
        split_summary=split_summary,
        preprocessing_notes=preprocessing_notes,
        predictions=predictions,
        metrics=metrics,
        feature_columns_path=feature_source_path,
    )
    print_report(
        dataset_path=dataset_path,
        total_rows=len(df),
        split_summary=split_summary,
        feature_columns=feature_columns,
        distribution_type=distribution,
        standard_config=standard_config,
        validation_nll=validation_nll,
        test_nll=test_nll,
        validation_baseline_nll=validation_baseline_nll,
        test_baseline_nll=test_baseline_nll,
        sigma_summary=sigma_summary,
        preprocessing_notes=preprocessing_notes,
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Day 11 NGBoost Normal forecast-error model.")
    parser.add_argument(
        "--mode",
        choices=["train", "tune"],
        default="train",
        help="Run the existing training path or the Day 16 tuning workflow.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the model configuration YAML.",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_MODELING_TABLE_PATH),
        help="Path to the modeling rows CSV.",
    )
    parser.add_argument(
        "--feature-columns-path",
        default=str(DEFAULT_FEATURE_COLUMNS_PATH),
        help="Path to the leakage-safe Day 8 feature columns JSON used by --mode tune.",
    )
    parser.add_argument(
        "--feature-list",
        default=str(DEFAULT_FINAL_FEATURE_LIST_PATH),
        help=(
            "Explicit feature-list JSON for --mode train. Defaults to the Day 17 frozen "
            "production feature list."
        ),
    )
    parser.add_argument(
        "--train-end-date",
        default=None,
        help="Inclusive train end date. Defaults to repo-aware chronological split.",
    )
    parser.add_argument(
        "--validation-end-date",
        default=None,
        help="Inclusive validation end date. Defaults to repo-aware chronological split.",
    )
    parser.add_argument(
        "--distribution",
        default=None,
        help="Override the configured NGBoost distribution for --mode train.",
    )
    parser.add_argument(
        "--overwrite-search-results",
        action="store_true",
        help="Overwrite the Day 16 search CSV instead of appending this run.",
    )
    return parser.parse_args(argv)


def load_model_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be a mapping: {path}")
    return config


def standard_training_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ngboost_config = config.get("ngboost", {})
    if not isinstance(ngboost_config, dict):
        ngboost_config = {}
    standard = ngboost_config.get("standard_training", {})
    if not isinstance(standard, dict):
        standard = {}

    distribution_source = (
        args.distribution
        or standard.get("distribution")
        or ngboost_config.get("distribution")
        or "Normal"
    )
    early_stopping = standard.get("early_stopping_rounds", 20)
    if early_stopping is not None:
        early_stopping = int(early_stopping)
        if early_stopping <= 0:
            early_stopping = None

    sigma_scale = float(standard.get("sigma_scale", 1.0))
    if not math.isfinite(sigma_scale) or sigma_scale <= 0.0:
        raise ValueError(f"ngboost.standard_training.sigma_scale must be positive, got {sigma_scale!r}")

    return {
        "distribution": normalize_distribution_name(str(distribution_source)),
        "sigma_scale": sigma_scale,
        "n_estimators": int(standard.get("n_estimators", 120)),
        "learning_rate": float(standard.get("learning_rate", 0.05)),
        "max_depth": int(standard.get("max_depth", 2)),
        "min_samples_leaf": int(standard.get("min_samples_leaf", 50)),
        "minibatch_frac": float(standard.get("minibatch_frac", 1.0)),
        "natural_gradient": bool(standard.get("natural_gradient", ngboost_config.get("natural_gradient", True))),
        "random_state": int(standard.get("random_state", ngboost_config.get("random_state", 11))),
        "early_stopping_rounds": early_stopping,
    }


def load_feature_columns_from_spec(df: pd.DataFrame, feature_columns_path: Path) -> list[str]:
    if not feature_columns_path.exists():
        raise FileNotFoundError(
            "Feature columns JSON is required; refusing to infer every numeric column: "
            f"{feature_columns_path}"
        )
    feature_columns = get_feature_columns(df, feature_columns_path)
    validate_no_leakage_feature_columns(feature_columns)
    return validate_feature_columns_exist(df, feature_columns)


def load_explicit_feature_list(df: pd.DataFrame, feature_list_path: Path) -> list[str]:
    feature_columns = load_feature_list(feature_list_path)
    validate_no_leakage_feature_columns(feature_columns)
    return validate_feature_columns_exist(df, feature_columns)


def run_tuning(args: argparse.Namespace) -> None:
    config_path = _resolve_path(args.config_path)
    config = load_model_config(config_path)
    data_config = config.get("data", {})
    dataset_path = _resolve_path(data_config.get("modeling_table_path", args.dataset_path))
    feature_columns_path = _resolve_path(
        data_config.get("feature_columns_path", args.feature_columns_path)
    )

    configured_target = str(config.get("target_column", TARGET_COLUMN))
    if configured_target != TARGET_COLUMN:
        raise ValueError(
            f"Day 16 tuning expects target_column={TARGET_COLUMN!r}; got {configured_target!r}"
        )

    df = load_modeling_table(dataset_path)
    validate_target_column(df)
    split_result = _configured_chronological_split(df, config, args)
    train_df = split_result.train
    validation_df = split_result.validation
    test_df = split_result.test
    split_summary = split_result.summary
    split_flat = _flatten_split_summary(split_summary)
    _print_split_summary(split_summary)
    print("Test split is reserved and will not be predicted or scored during tuning.", flush=True)

    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        validate_target_column(split_df, split_name=split_name)

    feature_columns = load_feature_columns_from_spec(df, feature_columns_path)

    X_train, X_validation, _X_test, train_imputer, preprocessing_notes = (
        build_imputed_feature_frames(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            feature_columns=feature_columns,
        )
    )
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = validation_df[TARGET_COLUMN].to_numpy(dtype=float)

    ngboost_config = config.get("ngboost", {})
    tuning_config = ngboost_config.get("tuning", {})
    if not tuning_config.get("enabled", True):
        raise ValueError("ngboost.tuning.enabled is false; refusing to run tuning mode")
    grid = list(tuning_config.get("grid", []))
    if not grid:
        raise ValueError("ngboost.tuning.grid must contain at least one candidate")

    distribution = normalize_distribution_name(str(ngboost_config.get("distribution", "Normal")))
    seed = int(ngboost_config.get("random_state", 42))
    natural_gradient = bool(ngboost_config.get("natural_gradient", True))
    features_version = str(config.get("features_version", feature_columns_path.name))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(
        f"Day 16 NGBoost tuning started with {len(grid)} candidates, "
        f"distribution={distribution}, features={len(feature_columns)}.",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    diagnostics_by_run_id: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for index, raw_candidate in enumerate(grid, start=1):
        candidate = dict(raw_candidate)
        candidate_name = str(candidate.get("name", f"candidate_{index}"))
        run_id = f"ngb16_{timestamp}_{index:02d}_{_slugify(candidate_name)}"
        elapsed = time.perf_counter() - started
        eta = _estimated_remaining(elapsed, completed=index - 1, total=len(grid))
        print(
            f"[{index}/{len(grid)}] Starting {candidate_name}. "
            f"Elapsed {_format_duration(elapsed)}; estimated remaining {eta}.",
            flush=True,
        )
        candidate_started = time.perf_counter()
        row, diagnostics = _run_tuning_candidate(
            run_id=run_id,
            candidate=candidate,
            index=index,
            distribution=distribution,
            seed=seed,
            natural_gradient=natural_gradient,
            features_version=features_version,
            feature_columns=feature_columns,
            train_df=train_df,
            validation_df=validation_df,
            X_train=X_train,
            X_validation=X_validation,
            y_train=y_train,
            y_validation=y_validation,
            split_flat=split_flat,
        )
        rows.append(row)
        if diagnostics is not None:
            diagnostics_by_run_id[run_id] = diagnostics
        candidate_elapsed = time.perf_counter() - candidate_started
        total_elapsed = time.perf_counter() - started
        eta = _estimated_remaining(total_elapsed, completed=index, total=len(grid))
        print(
            f"[{index}/{len(grid)}] {candidate_name} {row['status']} in "
            f"{_format_duration(candidate_elapsed)}. Estimated remaining {eta}.",
            flush=True,
        )

    current_run_results = pd.DataFrame(rows, columns=SEARCH_RESULT_COLUMNS)
    _write_search_results(
        current_run_results,
        TUNING_SEARCH_OUTPUT_PATH,
        overwrite=bool(args.overwrite_search_results),
    )

    tolerance = float(tuning_config.get("selection_tolerance_nll", 0.02))
    selected_row, selection_details = _select_best_candidate(current_run_results, tolerance)
    selected_candidate = next(
        candidate for candidate in grid if str(candidate.get("name")) == selected_row["candidate_name"]
    )
    refit_on_train_val = bool(tuning_config.get("refit_best_on_train_val", True))

    final_model, final_imputer, final_training_rows = _fit_final_model(
        selected_candidate=selected_candidate,
        distribution=distribution,
        seed=seed,
        natural_gradient=natural_gradient,
        refit_on_train_val=refit_on_train_val,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )
    _write_best_model_artifact(
        model=final_model,
        imputer=final_imputer,
        feature_columns=feature_columns,
        split_summary=split_summary,
        preprocessing_notes=preprocessing_notes,
        selected_row=selected_row,
        selected_candidate=selected_candidate,
        distribution=distribution,
        refit_on_train_val=refit_on_train_val,
        final_training_rows=final_training_rows,
    )
    _write_best_metadata(
        selected_row=selected_row,
        selected_candidate=selected_candidate,
        feature_columns=feature_columns,
        split_summary=split_summary,
        distribution=distribution,
        refit_on_train_val=refit_on_train_val,
        final_training_rows=final_training_rows,
        selection_details=selection_details,
    )
    _write_best_diagnostics_plots(
        selected_run_id=str(selected_row["run_id"]),
        diagnostics_by_run_id=diagnostics_by_run_id,
        results=current_run_results,
    )
    _write_selection_notes(
        selected_row=selected_row,
        selected_candidate=selected_candidate,
        grid=grid,
        results=current_run_results,
        split_summary=split_summary,
        selection_details=selection_details,
        refit_on_train_val=refit_on_train_val,
    )

    print("Day 16 NGBoost tuning complete.", flush=True)
    print(f"Selected candidate: {selected_row['candidate_name']}", flush=True)
    print(f"Validation NLL: {float(selected_row['val_nll']):.6f}", flush=True)
    print(f"Saved search results: {TUNING_SEARCH_OUTPUT_PATH}", flush=True)
    print(f"Saved best model: {BEST_MODEL_OUTPUT_PATH}", flush=True)
    print(f"Saved metadata: {BEST_METADATA_OUTPUT_PATH}", flush=True)
    print(f"Saved notes: {BEST_NOTES_OUTPUT_PATH}", flush=True)


def _configured_chronological_split(
    df: pd.DataFrame,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    split_config = config.get("splits", {})
    strategy = str(split_config.get("strategy", "chronological")).lower()
    if strategy != "chronological":
        raise ValueError(f"Only chronological splits are supported, got {strategy!r}")

    validation_start = split_config.get("validation_start")
    test_start = split_config.get("test_start")
    date_column = config.get("date_column")
    if validation_start is not None or test_start is not None:
        return chronological_train_validation_test_split(
            df,
            validation_start_date=validation_start,
            test_start_date=test_start,
            date_column=date_column,
        )

    return chronological_train_validation_test_split(
        df,
        train_end_date=args.train_end_date,
        validation_end_date=args.validation_end_date,
        date_column=date_column,
    )


def _flatten_split_summary(summary: dict[str, Any]) -> dict[str, Any]:
    splits = summary["splits"]
    return {
        "date_column": summary["date_column"],
        "n_train": splits["train"]["row_count"],
        "n_val": splits["validation"]["row_count"],
        "n_test": splits["test"]["row_count"],
        "train_start": splits["train"]["date_min"],
        "train_end": splits["train"]["date_max"],
        "val_start": splits["validation"]["date_min"],
        "val_end": splits["validation"]["date_max"],
        "test_start": splits["test"]["date_min"],
        "test_end": splits["test"]["date_max"],
    }


def _print_split_summary(summary: dict[str, Any]) -> None:
    print("Chronological split summary:", flush=True)
    for split in ["train", "validation", "test"]:
        item = summary["splits"][split]
        print(
            f"  {split}: {item['row_count']:,} rows, "
            f"{item['date_min']} to {item['date_max']}",
            flush=True,
        )


def _run_tuning_candidate(
    run_id: str,
    candidate: dict[str, Any],
    index: int,
    distribution: str,
    seed: int,
    natural_gradient: bool,
    features_version: str,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    split_flat: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row = _base_search_row(
        run_id=run_id,
        candidate=candidate,
        index=index,
        distribution=distribution,
        seed=seed,
        natural_gradient=natural_gradient,
        features_version=features_version,
        target_column=TARGET_COLUMN,
        split_flat=split_flat,
    )

    params = _candidate_hyperparameters(candidate, seed, natural_gradient)
    try:
        model = train_ngboost_distribution(
            X_train=X_train,
            y_train=y_train,
            distribution=distribution,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            minibatch_frac=params["minibatch_frac"],
            natural_gradient=params["natural_gradient"],
            random_state=params["random_state"],
            early_stopping_rounds=None,
        )
    except Exception as exc:
        row["status"] = "failed_fit"
        row["error_message"] = _compact_error(exc)
        return row, None

    try:
        train_details = predict_distribution_details(model, X_train, distribution)
        validation_details = predict_distribution_details(model, X_validation, distribution)
    except Exception as exc:
        row["status"] = "failed_predict"
        row["error_message"] = _compact_error(exc)
        return row, None

    try:
        metrics, diagnostics = _evaluate_validation_predictions(
            train_df=train_df,
            validation_df=validation_df,
            y_train=y_train,
            y_validation=y_validation,
            train_details=train_details,
            validation_details=validation_details,
            distribution=distribution,
        )
        row.update(metrics)
        row["status"] = "success"
        row["notes"] = _candidate_warning_notes(row, train_details, validation_details)
        diagnostics["model"] = model
        diagnostics["feature_columns"] = feature_columns
        return row, diagnostics
    except Exception as exc:
        row["status"] = "failed_eval"
        row["error_message"] = _compact_error(exc)
        return row, None


def _candidate_hyperparameters(
    candidate: dict[str, Any],
    seed: int,
    natural_gradient: bool,
) -> dict[str, Any]:
    return {
        "n_estimators": int(candidate.get("n_estimators", 300)),
        "learning_rate": float(candidate.get("learning_rate", 0.03)),
        "max_depth": int(candidate.get("max_depth", 2)),
        "min_samples_leaf": int(candidate.get("min_samples_leaf", 20)),
        "minibatch_frac": float(candidate.get("minibatch_frac", 1.0)),
        "natural_gradient": bool(candidate.get("natural_gradient", natural_gradient)),
        "random_state": int(candidate.get("random_state", seed)),
    }


def _base_search_row(
    run_id: str,
    candidate: dict[str, Any],
    index: int,
    distribution: str,
    seed: int,
    natural_gradient: bool,
    features_version: str,
    target_column: str,
    split_flat: dict[str, Any],
) -> dict[str, Any]:
    params = _candidate_hyperparameters(candidate, seed, natural_gradient)
    row = {column: math.nan for column in SEARCH_RESULT_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "error_message": "",
            "seed": params["random_state"],
            "distribution": distribution,
            "features_version": features_version,
            "target_column": target_column,
            "date_column": split_flat["date_column"],
            "candidate_name": str(candidate.get("name", f"candidate_{index}")),
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "minibatch_frac": params["minibatch_frac"],
            "natural_gradient": params["natural_gradient"],
            "notes": "",
        }
    )
    for key in [
        "n_train",
        "n_val",
        "n_test",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ]:
        row[key] = split_flat[key]
    return row


def _evaluate_validation_predictions(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    train_details: dict[str, Any],
    validation_details: dict[str, Any],
    distribution: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    train_mu = np.asarray(train_details["mu"], dtype=float)
    train_sigma = np.asarray(train_details["sigma"], dtype=float)
    train_df_values = train_details["df"] if train_details.get("df") is not None else None
    val_mu = np.asarray(validation_details["mu"], dtype=float)
    val_sigma = np.asarray(validation_details["sigma"], dtype=float)
    val_df_values = validation_details["df"] if validation_details.get("df") is not None else None

    train_nll = distribution_nll(
        y_train,
        mu=train_mu,
        sigma=train_sigma,
        distribution=distribution,
        df=train_df_values,
    )
    val_nll = distribution_nll(
        y_validation,
        mu=val_mu,
        sigma=val_sigma,
        distribution=distribution,
        df=val_df_values,
    )
    if not np.isfinite(train_nll).all() or not np.isfinite(val_nll).all():
        raise ValueError("NLL contains non-finite values")

    coverage = prediction_interval_coverage(
        y_validation,
        val_mu,
        val_sigma,
        levels=(0.5, 0.8, 0.9),
        dist_type=distribution,
        df=val_df_values,
    ).set_index("level")

    prediction_frame = _build_distribution_prediction_frame(
        validation_df,
        validation_details,
        val_nll,
        distribution,
    )
    bucket_probs, realized_labels, bucket_diag = _market_bucket_probabilities(
        prediction_frame,
        distribution,
    )
    brier = _multiclass_bucket_brier(bucket_probs, realized_labels)
    log_loss = interval_log_loss(bucket_probs, realized_labels)
    calibration_table, calibration_error = _weighted_calibration_error(
        bucket_probs,
        realized_labels,
    )
    pit = compute_pit_values(
        y_validation,
        val_mu,
        val_sigma,
        dist_type=distribution,
        df=val_df_values,
    )
    point = _point_prediction_metrics(y_validation, val_mu)

    metrics = {
        "train_nll": float(np.mean(train_nll)),
        "val_nll": float(np.mean(val_nll)),
        "val_mae": point["mae"],
        "val_rmse": point["rmse"],
        "val_bias": point["bias"],
        "val_50_coverage": float(coverage.loc[0.5, "actual_coverage"]),
        "val_80_coverage": float(coverage.loc[0.8, "actual_coverage"]),
        "val_90_coverage": float(coverage.loc[0.9, "actual_coverage"]),
        "val_50_coverage_error": abs(float(coverage.loc[0.5, "actual_coverage"]) - 0.5),
        "val_80_coverage_error": abs(float(coverage.loc[0.8, "actual_coverage"]) - 0.8),
        "val_90_coverage_error": abs(float(coverage.loc[0.9, "actual_coverage"]) - 0.9),
        "val_bucket_brier": brier,
        "val_interval_log_loss": log_loss,
        "val_calibration_error": calibration_error,
        "val_pit_mean": float(pit.mean()),
        "val_pit_std": float(pit.std(ddof=1)),
        "val_pit_min": float(pit.min()),
        "val_pit_max": float(pit.max()),
        "bucket_prob_min": bucket_diag["bucket_prob_min"],
        "bucket_prob_max": bucket_diag["bucket_prob_max"],
        "bucket_prob_sum_mean": bucket_diag["bucket_prob_sum_mean"],
        "bucket_prob_sum_max_abs_error": bucket_diag["bucket_prob_sum_max_abs_error"],
        "bucket_prob_normalization_rate": bucket_diag["bucket_prob_normalization_rate"],
    }
    diagnostics = {
        "pit": pit,
        "calibration_table": calibration_table,
        "bucket_probs": bucket_probs,
        "realized_labels": realized_labels,
        "prediction_frame": prediction_frame,
        "coverage": coverage.reset_index(),
    }
    return metrics, diagnostics


def _build_distribution_prediction_frame(
    split_df: pd.DataFrame,
    details: dict[str, Any],
    nll: np.ndarray,
    distribution: str,
) -> pd.DataFrame:
    frame = build_prediction_frame(
        split_name="validation",
        split_df=split_df,
        mu=np.asarray(details["mu"], dtype=float),
        sigma=np.asarray(details["sigma"], dtype=float),
        nll=np.asarray(nll, dtype=float),
    )
    if "row_id" not in frame.columns:
        frame.insert(0, "row_id", np.arange(len(frame), dtype=int))
    frame["scale"] = frame["sigma"]
    frame["distribution_type"] = distribution
    if details.get("df") is not None:
        frame["df"] = np.asarray(details["df"], dtype=float)
    else:
        frame["df"] = np.nan
    return frame


def _market_bucket_probabilities(
    prediction_frame: pd.DataFrame,
    distribution: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    long = price_buckets_for_dataframe(prediction_frame, dist_type=distribution)
    long["market_bucket"] = "market_bucket_" + long["bucket_index"].astype(int).astype(str)

    actual_column = _actual_temperature_column(long)
    actual = pd.to_numeric(long[actual_column], errors="raise")
    lower = pd.to_numeric(long["bucket_lower_temp"], errors="coerce")
    upper = pd.to_numeric(long["bucket_upper_temp"], errors="coerce")
    in_bucket = (lower.isna() | (actual > lower)) & (upper.isna() | (actual <= upper))
    labels = long[in_bucket][["row_id", "market_bucket"]]
    if labels.duplicated("row_id").any():
        raise ValueError("A validation row matched multiple realized market buckets")
    expected_rows = set(pd.to_numeric(prediction_frame["row_id"], errors="raise").astype(int))
    matched_rows = set(pd.to_numeric(labels["row_id"], errors="raise").astype(int))
    if expected_rows != matched_rows:
        raise ValueError(
            "Could not assign exactly one realized bucket per validation row: "
            f"missing={len(expected_rows - matched_rows)}, extra={len(matched_rows - expected_rows)}"
        )
    labels = labels.set_index("row_id")["market_bucket"]

    probs = long.pivot(index="row_id", columns="market_bucket", values="probability")
    ordered_columns = [
        f"market_bucket_{bucket_index}"
        for bucket_index in sorted(long["bucket_index"].astype(int).unique())
    ]
    ordered_row_ids = pd.Index(pd.to_numeric(prediction_frame["row_id"], errors="raise").astype(int))
    probs = probs[ordered_columns].reindex(ordered_row_ids)
    labels = labels.reindex(ordered_row_ids)
    if probs.isna().any().any() or labels.isna().any():
        raise ValueError("Bucket probabilities or realized labels failed row alignment")

    probs = probs.astype(float)
    raw_values = probs.to_numpy(dtype=float)
    if not np.isfinite(raw_values).all():
        raise ValueError("Bucket probabilities contain non-finite values")
    if raw_values.min() < -1e-10 or raw_values.max() > 1.0 + 1e-10:
        raise ValueError(
            "Bucket probabilities are outside [0, 1] beyond numerical tolerance: "
            f"min={raw_values.min():.12g}, max={raw_values.max():.12g}"
        )

    probs = probs.clip(lower=0.0, upper=1.0)
    row_sums = probs.sum(axis=1)
    needs_normalization = (row_sums - 1.0).abs() > 1e-8
    normalization_rate = float(needs_normalization.mean())
    if needs_normalization.any():
        probs.loc[needs_normalization] = probs.loc[needs_normalization].div(
            row_sums.loc[needs_normalization],
            axis=0,
        )

    probs = validate_bucket_probabilities(probs, allow_renormalize=False)
    final_row_sums = probs.sum(axis=1)
    diagnostics = {
        "bucket_prob_min": float(probs.min().min()),
        "bucket_prob_max": float(probs.max().max()),
        "bucket_prob_sum_mean": float(final_row_sums.mean()),
        "bucket_prob_sum_max_abs_error": float((final_row_sums - 1.0).abs().max()),
        "bucket_prob_normalization_rate": normalization_rate,
    }
    return probs.reset_index(drop=True), labels.reset_index(drop=True), diagnostics


def _actual_temperature_column(df: pd.DataFrame) -> str:
    for column in ["actual_high", "official_high", "actual_official_high"]:
        if column in df.columns:
            return column
    raise ValueError("Bucket evaluation requires actual_high or official_high")


def _multiclass_bucket_brier(
    bucket_probs: pd.DataFrame,
    realized_labels: pd.Series,
) -> float:
    probs = validate_bucket_probabilities(bucket_probs)
    labels = pd.Series(realized_labels).reset_index(drop=True)
    if len(labels) != len(probs):
        raise ValueError("Realized bucket labels length does not match probabilities")
    one_hot = pd.DataFrame(0.0, index=probs.index, columns=probs.columns)
    positions = pd.Index(probs.columns).get_indexer(labels)
    if (positions < 0).any():
        raise ValueError("Realized bucket label not present in probability columns")
    one_hot_values = one_hot.to_numpy(dtype=float)
    one_hot_values[np.arange(len(labels)), positions] = 1.0
    score = float(np.mean(np.sum((probs.to_numpy(dtype=float) - one_hot_values) ** 2, axis=1)))
    if not math.isfinite(score):
        raise ValueError("Bucket Brier score is not finite")
    return score


def _weighted_calibration_error(
    bucket_probs: pd.DataFrame,
    realized_labels: pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    probs = validate_bucket_probabilities(bucket_probs)
    labels = pd.Series(realized_labels).reset_index(drop=True)
    positions = pd.Index(probs.columns).get_indexer(labels)
    if (positions < 0).any():
        raise ValueError("Realized bucket label not present in probability columns")

    probabilities = probs.to_numpy(dtype=float)
    actual = np.zeros_like(probabilities, dtype=int)
    actual[np.arange(len(labels)), positions] = 1
    table = make_calibration_table(
        probabilities.ravel(),
        actual.ravel(),
        n_bins=n_bins,
    )
    nonempty = table[table["count"] > 0].copy()
    total = float(nonempty["count"].sum())
    if total <= 0.0:
        raise ValueError("Calibration table has no non-empty bins")
    gaps = (
        nonempty["mean_predicted_probability"] - nonempty["empirical_frequency"]
    ).abs()
    weighted_error = float(np.sum(gaps * nonempty["count"]) / total)
    return table, weighted_error


def _point_prediction_metrics(y_true: np.ndarray, mu: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_true, dtype=float) - np.asarray(mu, dtype=float)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
    }


def _candidate_warning_notes(
    row: dict[str, Any],
    train_details: dict[str, Any],
    validation_details: dict[str, Any],
) -> str:
    notes: list[str] = []
    train_nll = float(row["train_nll"])
    val_nll = float(row["val_nll"])
    if math.isfinite(train_nll) and math.isfinite(val_nll) and val_nll - train_nll > 0.25:
        notes.append("train NLL is much better than validation NLL")
    for details, split in [(train_details, "train"), (validation_details, "validation")]:
        sigma = np.asarray(details["sigma"], dtype=float)
        if np.isfinite(sigma).all() and float(np.min(sigma)) < 1e-4:
            notes.append(f"{split} sigma/scale has near-zero values")
    if float(row["bucket_prob_normalization_rate"]) > 0.01:
        notes.append("bucket probabilities frequently needed normalization")
    if abs(float(row["val_80_coverage"]) - 0.8) > 0.06:
        notes.append("80% coverage far from nominal")
    if abs(float(row["val_90_coverage"]) - 0.9) > 0.06:
        notes.append("90% coverage far from nominal")
    return "; ".join(notes)


def _write_search_results(results: pd.DataFrame, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        existing = pd.read_csv(path)
        columns = list(dict.fromkeys([*existing.columns, *SEARCH_RESULT_COLUMNS]))
        combined = pd.concat(
            [existing.reindex(columns=columns), results.reindex(columns=columns)],
            ignore_index=True,
        )
        combined.to_csv(path, index=False)
    else:
        results.to_csv(path, index=False)
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Expected search output was not written: {path}")


def _select_best_candidate(
    results: pd.DataFrame,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    success = results[results["status"] == "success"].copy()
    if success.empty:
        raise ValueError("No successful NGBoost candidates were available for selection")
    numeric_columns = [
        "val_nll",
        "val_80_coverage_error",
        "val_90_coverage_error",
        "val_interval_log_loss",
        "val_bucket_brier",
        "val_calibration_error",
        "max_depth",
        "min_samples_leaf",
        "n_estimators",
    ]
    for column in numeric_columns:
        success[column] = pd.to_numeric(success[column], errors="raise")
    best_nll = float(success["val_nll"].min())
    eligible = success[success["val_nll"] <= best_nll + float(tolerance)].copy()
    eligible["_neg_min_samples_leaf"] = -eligible["min_samples_leaf"]
    selected = eligible.sort_values(
        [
            "val_80_coverage_error",
            "val_90_coverage_error",
            "val_interval_log_loss",
            "val_bucket_brier",
            "val_calibration_error",
            "max_depth",
            "_neg_min_samples_leaf",
            "n_estimators",
            "candidate_name",
        ],
        kind="stable",
    ).iloc[0]
    success_sorted = success.sort_values("val_nll", kind="stable").reset_index(drop=True)
    tiny_improvement = (
        len(success_sorted) > 1
        and float(success_sorted.loc[1, "val_nll"] - success_sorted.loc[0, "val_nll"]) < 0.005
    )
    notes: list[str] = []
    if tiny_improvement:
        notes.append("validation NLL improvements are tiny between the top candidates")
    nll_winner = str(success_sorted.loc[0, "candidate_name"])
    if str(selected["candidate_name"]) != nll_winner:
        notes.append(
            "best validation NLL candidate was not selected because coverage/tie-breakers favored "
            f"{selected['candidate_name']}"
        )
    return (
        selected.drop(labels=[label for label in ["_neg_min_samples_leaf"] if label in selected]).to_dict(),
        {
            "best_val_nll": best_nll,
            "selection_tolerance_nll": float(tolerance),
            "eligible_candidate_count": int(len(eligible)),
            "successful_candidate_count": int(len(success)),
            "selection_rule": (
                "min val_nll within tolerance, then lower 80/90 coverage error, "
                "interval log loss, bucket Brier, calibration error, then simpler trees"
            ),
            "notes": notes,
        },
    )


def _fit_final_model(
    selected_candidate: dict[str, Any],
    distribution: str,
    seed: int,
    natural_gradient: bool,
    refit_on_train_val: bool,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[Any, SimpleImputer, int]:
    params = _candidate_hyperparameters(selected_candidate, seed, natural_gradient)
    fit_df = (
        pd.concat([train_df, validation_df], ignore_index=True)
        if refit_on_train_val
        else train_df.copy()
    )
    X_fit, _X_validation, _X_test, imputer, _notes = build_imputed_feature_frames(
        train_df=fit_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )
    y_fit = fit_df[TARGET_COLUMN].to_numpy(dtype=float)
    model = train_ngboost_distribution(
        X_train=X_fit,
        y_train=y_fit,
        distribution=distribution,
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        minibatch_frac=params["minibatch_frac"],
        natural_gradient=params["natural_gradient"],
        random_state=params["random_state"],
        early_stopping_rounds=None,
    )
    return model, imputer, int(len(fit_df))


def _write_best_model_artifact(
    model: Any,
    imputer: SimpleImputer,
    feature_columns: list[str],
    split_summary: dict[str, Any],
    preprocessing_notes: dict[str, Any],
    selected_row: dict[str, Any],
    selected_candidate: dict[str, Any],
    distribution: str,
    refit_on_train_val: bool,
    final_training_rows: int,
) -> None:
    BEST_MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_columns": feature_columns,
        "target": TARGET_COLUMN,
        "model_name": "ngboost_best_v2",
        "distribution_type": distribution,
        "selected_run_id": selected_row["run_id"],
        "selected_candidate_name": selected_row["candidate_name"],
        "hyperparameters": _candidate_hyperparameters(
            selected_candidate,
            int(selected_row["seed"]),
            bool(selected_row["natural_gradient"]),
        ),
        "split_summary": split_summary,
        "preprocessing_notes": preprocessing_notes,
        "refit_on_train_val": bool(refit_on_train_val),
        "final_training_rows": final_training_rows,
        "test_set_used_for_tuning": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with BEST_MODEL_OUTPUT_PATH.open("wb") as file:
        pickle.dump(artifact, file)
    if not BEST_MODEL_OUTPUT_PATH.exists() or BEST_MODEL_OUTPUT_PATH.stat().st_size == 0:
        raise AssertionError(f"Expected best model was not written: {BEST_MODEL_OUTPUT_PATH}")


def _write_best_metadata(
    selected_row: dict[str, Any],
    selected_candidate: dict[str, Any],
    feature_columns: list[str],
    split_summary: dict[str, Any],
    distribution: str,
    refit_on_train_val: bool,
    final_training_rows: int,
    selection_details: dict[str, Any],
) -> None:
    BEST_METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_metrics = {
        key: selected_row[key]
        for key in selected_row
        if key.startswith("val_") or key in {"train_nll"}
    }
    payload = {
        "selected_run_id": selected_row["run_id"],
        "selected_candidate_name": selected_row["candidate_name"],
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "distribution": distribution,
        "target_column": TARGET_COLUMN,
        "date_column": split_summary["date_column"],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "split_dates": split_summary,
        "n_train": int(selected_row["n_train"]),
        "n_val": int(selected_row["n_val"]),
        "n_test": int(selected_row["n_test"]),
        "final_training_rows": int(final_training_rows),
        "hyperparameters": _candidate_hyperparameters(
            selected_candidate,
            int(selected_row["seed"]),
            bool(selected_row["natural_gradient"]),
        ),
        "validation_metrics": validation_metrics,
        "selection_rule": selection_details["selection_rule"],
        "selection_details": selection_details,
        "refit_on_train_val": bool(refit_on_train_val),
        "test_set_used_for_tuning": False,
        "package_versions": _package_versions(),
    }
    BEST_METADATA_OUTPUT_PATH.write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )


def _write_best_diagnostics_plots(
    selected_run_id: str,
    diagnostics_by_run_id: dict[str, dict[str, Any]],
    results: pd.DataFrame,
) -> None:
    diagnostics = diagnostics_by_run_id.get(selected_run_id)
    if diagnostics is None:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_pit_histogram(
        diagnostics["pit"],
        FIGURES_DIR / "pit_histogram_ngboost_v2.png",
        bins=10,
    )
    plot_calibration_curve(
        diagnostics["calibration_table"],
        FIGURES_DIR / "calibration_curve_ngboost_v2.png",
    )
    _plot_tuning_metric(
        results,
        metric="val_nll",
        output_path=FIGURES_DIR / "ngboost_tuning_val_nll.png",
        ylabel="Validation NLL",
    )
    _plot_tuning_coverage(results, FIGURES_DIR / "ngboost_tuning_coverage.png")
    _plot_tuning_metric(
        results,
        metric="val_interval_log_loss",
        output_path=FIGURES_DIR / "ngboost_tuning_interval_log_loss.png",
        ylabel="Validation Interval Log Loss",
    )


def _plot_tuning_metric(
    results: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = results[results["status"] == "success"].copy()
    if plot_df.empty or metric not in plot_df.columns:
        return
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * len(plot_df)), 4.8))
    ax.bar(plot_df["candidate_name"], plot_df[metric], color="#4c78a8")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Candidate")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_tuning_coverage(results: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = results[results["status"] == "success"].copy()
    if plot_df.empty:
        return
    for column in ["val_50_coverage", "val_80_coverage", "val_90_coverage"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["val_50_coverage", "val_80_coverage", "val_90_coverage"])
    if plot_df.empty:
        return
    x = np.arange(len(plot_df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(plot_df)), 4.8))
    for offset, level in [(-width, "50"), (0.0, "80"), (width, "90")]:
        ax.bar(x + offset, plot_df[f"val_{level}_coverage"], width=width, label=f"{level}%")
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.0)
    ax.axhline(0.8, color="#444444", linestyle="--", linewidth=1.0)
    ax.axhline(0.9, color="#111111", linestyle="-.", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["candidate_name"], rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Observed validation coverage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_selection_notes(
    selected_row: dict[str, Any],
    selected_candidate: dict[str, Any],
    grid: list[dict[str, Any]],
    results: pd.DataFrame,
    split_summary: dict[str, Any],
    selection_details: dict[str, Any],
    refit_on_train_val: bool,
) -> None:
    BEST_NOTES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    v1 = _load_v1_validation_metrics()
    success = results[results["status"] == "success"].copy()
    lines = [
        "# Best NGBoost v2 Selection Notes",
        "",
        "## Summary",
        "",
        (
            f"Selected `{selected_row['candidate_name']}` as `ngboost_best_v2` using "
            "chronological validation probability metrics only."
        ),
        "",
        "## Chronological Split",
        "",
        (
            f"Train dates: {split_summary['splits']['train']['date_min']} to "
            f"{split_summary['splits']['train']['date_max']}"
        ),
        (
            f"Validation dates: {split_summary['splits']['validation']['date_min']} to "
            f"{split_summary['splits']['validation']['date_max']}"
        ),
        (
            f"Test dates: {split_summary['splits']['test']['date_min']} to "
            f"{split_summary['splits']['test']['date_max']}"
        ),
        "Test set used during tuning: No",
        "",
        "## Search Space",
        "",
    ]
    for candidate in grid:
        params = _candidate_hyperparameters(
            candidate,
            int(selected_row["seed"]),
            bool(selected_row["natural_gradient"]),
        )
        lines.append(
            "- `{name}`: n_estimators={n_estimators}, learning_rate={learning_rate}, "
            "max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, "
            "minibatch_frac={minibatch_frac}".format(
                name=candidate.get("name"),
                **params,
            )
        )
    lines.extend(
        [
            "",
            "## Metrics Used",
            "",
            "Validation NLL/log score was the primary metric.",
            "Coverage, bucket Brier score, interval log loss, and calibration error were used as safeguards.",
            "MAE, RMSE, and bias were tracked as secondary diagnostics only.",
            "",
            "## Best Candidate",
            "",
        ]
    )
    best_params = _candidate_hyperparameters(
        selected_candidate,
        int(selected_row["seed"]),
        bool(selected_row["natural_gradient"]),
    )
    for key, value in best_params.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Validation Results",
            "",
            f"- validation NLL: {float(selected_row['val_nll']):.6f}",
            f"- 50% coverage: {float(selected_row['val_50_coverage']):.4f}",
            f"- 80% coverage: {float(selected_row['val_80_coverage']):.4f}",
            f"- 90% coverage: {float(selected_row['val_90_coverage']):.4f}",
            f"- bucket Brier score: {float(selected_row['val_bucket_brier']):.6f}",
            f"- interval log loss: {float(selected_row['val_interval_log_loss']):.6f}",
            f"- calibration error: {float(selected_row['val_calibration_error']):.6f}",
            f"- MAE: {float(selected_row['val_mae']):.6f}",
            f"- RMSE: {float(selected_row['val_rmse']):.6f}",
            f"- bias: {float(selected_row['val_bias']):.6f}",
            "",
            "## Why This Candidate Was Selected",
            "",
            (
                f"The best validation NLL was {selection_details['best_val_nll']:.6f}. "
                f"Candidates within {selection_details['selection_tolerance_nll']:.3f} NLL "
                "were tie-broken by 80% and 90% coverage error, interval log loss, bucket "
                "Brier score, calibration error, and simpler tree settings."
            ),
        ]
    )
    if selection_details["notes"]:
        lines.extend(["", *[f"- {note}" for note in selection_details["notes"]]])

    lines.extend(["", "## Comparison to v1", ""])
    if v1 is None:
        lines.append("Existing v1 validation metrics were not available in a comparable summary file.")
    else:
        lines.extend(
            [
                "Existing v1 metrics were loaded from `outputs/ngboost_distribution_comparison.csv`.",
                "",
                "| metric | v1 validation | v2 validation |",
                "|---|---:|---:|",
                f"| NLL | {v1['val_nll']:.6f} | {float(selected_row['val_nll']):.6f} |",
                f"| 80% coverage | {v1['val_80_coverage']:.4f} | {float(selected_row['val_80_coverage']):.4f} |",
                f"| 90% coverage | {v1['val_90_coverage']:.4f} | {float(selected_row['val_90_coverage']):.4f} |",
                (
                    f"| interval log loss | {v1['val_interval_log_loss']:.6f} | "
                    f"{float(selected_row['val_interval_log_loss']):.6f} |"
                ),
                (
                    f"| bucket Brier | {v1['val_bucket_brier']:.6f} prior mean-per-bucket | "
                    f"{float(selected_row['val_bucket_brier']):.6f} multiclass sum |"
                ),
                "| calibration error | not available | "
                f"{float(selected_row['val_calibration_error']):.6f} |",
            ]
        )

    failed_count = int((results["status"] != "success").sum())
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is one chronological validation period, so validation overfitting is still possible.",
            "- Weather forecast errors can be seasonal, and a single validation year may not represent every regime.",
            "- Heavy-tail behavior may remain imperfect under the configured NGBoost distribution.",
            "- Calibration can still be imperfect even when NLL improves.",
            "- The test set remains untouched and is reserved for final evaluation.",
            "",
            "## Test Set Status",
            "",
            "The test set remains untouched for final evaluation.",
            "",
            "## Run Accounting",
            "",
            f"- Successful candidates: {len(success)}",
            f"- Failed candidates logged: {failed_count}",
            f"- Refit selected model on train + validation: {'Yes' if refit_on_train_val else 'No'}",
        ]
    )
    BEST_NOTES_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_v1_validation_metrics() -> dict[str, float] | None:
    path = REPO_ROOT / "outputs" / "ngboost_distribution_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = df[df["model_name"].astype(str) == "ngboost_normal_v1"]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "val_nll": float(row["val_nll"]),
        "val_80_coverage": float(row["coverage_80"]),
        "val_90_coverage": float(row["coverage_90"]),
        "val_bucket_brier": float(row["market_mean_bucket_brier"]),
        "val_interval_log_loss": float(row["bucket_interval_log_loss"]),
    }


def _package_versions() -> dict[str, str]:
    packages = {
        "ngboost": "ngboost",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
        "pyyaml": "PyYAML",
    }
    versions: dict[str, str] = {}
    for key, package in packages.items():
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "not_installed"
    return versions


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _estimated_remaining(elapsed: float, completed: int, total: int) -> str:
    if completed <= 0:
        return "calculating after first candidate"
    remaining = max(0, int(total) - int(completed))
    return _format_duration((float(elapsed) / float(completed)) * remaining)


def _compact_error(exc: Exception, max_len: int = 500) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message[:max_len]


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text[:80] or "candidate"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def load_modeling_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Modeling table not found: {path}")
    df = pd.read_csv(path)
    for column in DATETIME_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
            if df[column].isna().any():
                bad_count = int(df[column].isna().sum())
                raise ValueError(f"{column!r} has {bad_count} unparsable timestamp values")
    return df


def validate_target_column(df: pd.DataFrame, split_name: str | None = None) -> None:
    label = f" in {split_name}" if split_name else ""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    missing = int(target.isna().sum())
    if missing:
        raise ValueError(f"{TARGET_COLUMN!r} has {missing} missing/non-numeric values{label}")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(f"{TARGET_COLUMN!r} has non-finite values{label}")
    df[TARGET_COLUMN] = target.astype(float)


def build_imputed_feature_frames(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer, dict[str, Any]]:
    raw_frames = {
        "train": _numeric_feature_frame(train_df, feature_columns),
        "validation": _numeric_feature_frame(validation_df, feature_columns),
        "test": _numeric_feature_frame(test_df, feature_columns),
    }
    counts = {
        split: _missing_and_infinite_counts(frame)
        for split, frame in raw_frames.items()
    }
    cleaned_frames = {
        split: frame.replace([np.inf, -np.inf], np.nan)
        for split, frame in raw_frames.items()
    }

    all_missing_train = [
        column
        for column in feature_columns
        if cleaned_frames["train"][column].isna().all()
    ]
    if all_missing_train:
        raise ValueError(f"Train feature columns are entirely missing: {all_missing_train}")

    imputer = SimpleImputer(strategy="median")
    imputer.fit(cleaned_frames["train"])

    transformed: dict[str, pd.DataFrame] = {}
    for split, frame in cleaned_frames.items():
        values = imputer.transform(frame)
        if values.shape[1] != len(feature_columns):
            raise AssertionError(
                f"Imputer changed feature width for {split}: {values.shape[1]} vs {len(feature_columns)}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values remain after imputation for {split}")
        transformed[split] = pd.DataFrame(values, columns=feature_columns, index=frame.index)

    notes = {
        "rows_dropped": 0,
        "imputation_strategy": "median fit on train only",
        "missing_and_infinite_values_by_split": counts,
        "train_imputer_medians": {
            column: float(value)
            for column, value in zip(feature_columns, imputer.statistics_, strict=True)
        },
    }
    return (
        transformed["train"],
        transformed["validation"],
        transformed["test"],
        imputer,
        notes,
    )


def _numeric_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns: {missing}")
    frame = df[feature_columns].copy()
    for column in feature_columns:
        if not (
            pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        ):
            raise ValueError(f"Selected feature {column!r} is not numeric")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _missing_and_infinite_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.to_numpy(dtype=float)
    missing = int(np.isnan(values).sum())
    infinite = int(np.isinf(values).sum())
    return {
        "missing_values": missing,
        "infinite_values": infinite,
        "total_values_imputed_or_replaced": missing + infinite,
    }


def validate_distribution_outputs(
    split_name: str,
    expected_count: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    nll: np.ndarray,
) -> None:
    if len(mu) != expected_count or len(sigma) != expected_count or len(nll) != expected_count:
        raise AssertionError(
            f"{split_name} prediction lengths do not match rows: "
            f"mu={len(mu)}, sigma={len(sigma)}, nll={len(nll)}, rows={expected_count}"
        )
    if not np.isfinite(mu).all():
        raise ValueError(f"{split_name} mu contains non-finite values")
    if not np.isfinite(sigma).all():
        raise ValueError(f"{split_name} sigma contains non-finite values")
    if (sigma <= 0).any():
        raise ValueError(f"{split_name} sigma contains non-positive values")
    near_zero_share = float(np.mean(sigma < 1e-4))
    if near_zero_share > 0.5:
        raise ValueError(
            f"{split_name} sigma is nearly zero for {near_zero_share:.1%} of rows"
        )
    if not np.isfinite(nll).all():
        raise ValueError(f"{split_name} NLL contains non-finite values")


def build_prediction_frame(
    split_name: str,
    split_df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    nll: np.ndarray,
    distribution_type: str = "normal",
    df: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "split": split_name,
            "forecast_error": split_df[TARGET_COLUMN].to_numpy(dtype=float),
            "mu": mu,
            "sigma": sigma,
            "distribution_type": normalize_distribution_name(distribution_type),
            "nll": nll,
        }
    )
    if df is not None:
        frame["df"] = np.asarray(df, dtype=float)

    metadata = pd.DataFrame(index=split_df.index)
    for column in METADATA_COLUMNS:
        if column == "timestamp":
            source = _first_existing_column(split_df, ["timestamp", "prediction_timestamp"])
            if source is not None:
                metadata[column] = split_df[source].values
            continue
        if column in split_df.columns:
            metadata[column] = split_df[column].values

    if "date" not in metadata.columns and "target_date" in split_df.columns:
        metadata["date"] = split_df["target_date"].values
    if "timestamp" not in metadata.columns and "prediction_time" in split_df.columns:
        metadata["timestamp"] = split_df["prediction_time"].values

    metadata = metadata.reset_index(drop=True)
    result = pd.concat([metadata, frame], axis=1)
    ordered = [
        column
        for column in [
            "split",
            "date",
            "timestamp",
            "prediction_time",
            "prediction_timestamp",
            "location",
            "station",
            "station_id",
            "forecast_high",
            "actual_high",
            "official_high",
            "forecast_horizon_hours",
            "forecast_error",
            "mu",
            "sigma",
            "nll",
        ]
        if column in result.columns
    ]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining]


def build_metrics(
    model_name: str,
    distribution_type: str,
    sigma_scale: float,
    split_summary: dict[str, Any],
    feature_columns: list[str],
    validation_nll: np.ndarray,
    test_nll: np.ndarray,
    validation_baseline_nll: np.ndarray,
    test_baseline_nll: np.ndarray,
    baseline_mu: float,
    baseline_sigma: float,
    sigma_summary: dict[str, dict[str, float]],
    preprocessing_notes: dict[str, Any],
    empirical_baseline_note: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "target_name": TARGET_COLUMN,
        "distribution_type": normalize_distribution_name(distribution_type),
        "sigma_scale": float(sigma_scale),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_summary": split_summary,
        "train_date_range": split_summary["splits"]["train"],
        "validation_date_range": split_summary["splits"]["validation"],
        "test_date_range": split_summary["splits"]["test"],
        "train_row_count": split_summary["splits"]["train"]["row_count"],
        "validation_row_count": split_summary["splits"]["validation"]["row_count"],
        "test_row_count": split_summary["splits"]["test"]["row_count"],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "validation_ngboost_nll": float(np.mean(validation_nll)),
        "test_ngboost_nll": float(np.mean(test_nll)),
        "validation_constant_normal_baseline_nll": float(np.mean(validation_baseline_nll)),
        "test_constant_normal_baseline_nll": float(np.mean(test_baseline_nll)),
        "constant_normal_baseline": {
            "mu_train": baseline_mu,
            "sigma_train": baseline_sigma,
        },
        "sigma_summary": sigma_summary,
        "notes": {
            "preprocessing": preprocessing_notes,
            "empirical_baseline_comparability": empirical_baseline_note,
            "scope": "Probability-signal layer only; no trading logic or bucket conversion was run.",
        },
    }


def summarize_sigma(sigma: np.ndarray) -> dict[str, float]:
    sigma_array = np.asarray(sigma, dtype=float)
    return {
        "min": float(np.min(sigma_array)),
        "median": float(np.median(sigma_array)),
        "mean": float(np.mean(sigma_array)),
        "max": float(np.max(sigma_array)),
        "near_zero_share_lt_1e-4": float(np.mean(sigma_array < 1e-4)),
    }


def build_empirical_baseline_note(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_available",
            "note": "No empirical baseline prediction file was found.",
        }
    columns = list(pd.read_csv(path, nrows=1).columns)
    continuous_nll_columns = [
        column
        for column in columns
        if column in {"continuous_nll", "density_nll", "normal_density_nll"}
    ]
    if continuous_nll_columns:
        return {
            "status": "available_not_loaded",
            "note": (
                "An empirical baseline file has a possible continuous NLL column, "
                "but Day 11 metrics keep the comparison to the train-only constant Normal baseline."
            ),
            "candidate_columns": continuous_nll_columns,
        }
    return {
        "status": "not_directly_comparable",
        "note": (
            "Existing empirical baseline outputs bucket/interval log scores. "
            "Those are not directly comparable to continuous Normal density NLL."
        ),
        "available_columns": columns,
    }


def write_outputs(
    model: Any,
    imputer: SimpleImputer,
    feature_columns: list[str],
    model_name: str,
    distribution_type: str,
    sigma_scale: float,
    split_summary: dict[str, Any],
    preprocessing_notes: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    feature_columns_path: Path,
) -> None:
    for path in [MODEL_OUTPUT_PATH, FEATURE_OUTPUT_PATH, PARAMS_OUTPUT_PATH, METRICS_OUTPUT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": model,
        "imputer": imputer,
        "feature_columns": feature_columns,
        "target": TARGET_COLUMN,
        "model_name": model_name,
        "distribution_type": normalize_distribution_name(distribution_type),
        "sigma_scale": float(sigma_scale),
        "split_summary": split_summary,
        "preprocessing_notes": preprocessing_notes,
    }
    with MODEL_OUTPUT_PATH.open("wb") as file:
        pickle.dump(model_artifact, file)

    feature_payload = {
        "model_name": model_name,
        "target": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "distribution_type": normalize_distribution_name(distribution_type),
        "sigma_scale": float(sigma_scale),
        "source_feature_spec": str(feature_columns_path),
        "imputation_strategy": preprocessing_notes["imputation_strategy"],
    }
    FEATURE_OUTPUT_PATH.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")
    predictions.to_csv(PARAMS_OUTPUT_PATH, index=False)
    METRICS_OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for path in [MODEL_OUTPUT_PATH, FEATURE_OUTPUT_PATH, PARAMS_OUTPUT_PATH, METRICS_OUTPUT_PATH]:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Expected output was not written: {path}")


def print_report(
    dataset_path: Path,
    total_rows: int,
    split_summary: dict[str, Any],
    feature_columns: list[str],
    distribution_type: str,
    standard_config: dict[str, Any],
    validation_nll: np.ndarray,
    test_nll: np.ndarray,
    validation_baseline_nll: np.ndarray,
    test_baseline_nll: np.ndarray,
    sigma_summary: dict[str, dict[str, float]],
    preprocessing_notes: dict[str, Any],
) -> None:
    print(f"Day 11 NGBoost {normalize_distribution_name(distribution_type)} training complete.")
    print(f"Loaded dataset: {dataset_path}")
    print(f"Total rows: {total_rows:,}")
    for split in ["train", "validation", "test"]:
        item = split_summary["splits"][split]
        print(
            f"{split.title()}: {item['row_count']:,} rows, "
            f"{item['date_min']} to {item['date_max']}"
        )
    print(f"Selected feature count: {len(feature_columns):,}")
    print(f"First 10 selected features: {feature_columns[:10]}")
    print(f"Configured distribution: {normalize_distribution_name(distribution_type)}")
    print(
        "Training params: "
        f"n_estimators={standard_config['n_estimators']}, "
        f"learning_rate={standard_config['learning_rate']}, "
        f"max_depth={standard_config['max_depth']}, "
        f"min_samples_leaf={standard_config['min_samples_leaf']}, "
        f"minibatch_frac={standard_config['minibatch_frac']}, "
        f"sigma_scale={standard_config['sigma_scale']}, "
        f"natural_gradient={standard_config['natural_gradient']}, "
        f"random_state={standard_config['random_state']}, "
        f"early_stopping_rounds={standard_config['early_stopping_rounds']}"
    )
    print(
        "Imputed/replaced values: "
        f"{preprocessing_notes['missing_and_infinite_values_by_split']}"
    )
    print(f"Validation NGBoost NLL: {float(np.mean(validation_nll)):.6f}")
    print(f"Validation constant Normal NLL: {float(np.mean(validation_baseline_nll)):.6f}")
    print(f"Test NGBoost NLL: {float(np.mean(test_nll)):.6f}")
    print(f"Test constant Normal NLL: {float(np.mean(test_baseline_nll)):.6f}")
    for split in ["validation", "test"]:
        summary = sigma_summary[split]
        print(
            f"{split.title()} sigma min/median/mean/max: "
            f"{summary['min']:.6f}, {summary['median']:.6f}, "
            f"{summary['mean']:.6f}, {summary['max']:.6f}"
        )
    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved feature list: {FEATURE_OUTPUT_PATH}")
    print(f"Saved distribution params: {PARAMS_OUTPUT_PATH}")
    print(f"Saved metrics: {METRICS_OUTPUT_PATH}")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


if __name__ == "__main__":
    main()
