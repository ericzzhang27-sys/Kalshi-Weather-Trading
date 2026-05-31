from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import plot_pit_histogram  # noqa: E402
from src.distribution_pricing import price_buckets_for_dataframe  # noqa: E402
from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
    distribution_cdf,
    distribution_nll,
    get_feature_columns,
    normalize_distribution_name,
    predict_distribution_details,
    train_ngboost_distribution,
    validate_no_leakage_feature_columns,
)
from src.evaluation import (  # noqa: E402
    bucket_brier_scores,
    compute_pit_values,
    coverage_by_group,
    interval_log_loss,
    prediction_interval_coverage,
    residual_summary,
    standardized_residuals,
    validate_bucket_probabilities,
)
from src.features import load_feature_list, validate_feature_columns_exist  # noqa: E402
from src.splits import chronological_train_validation_test_split  # noqa: E402
from src.train_ngboost import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS_PATH,
    DEFAULT_FINAL_FEATURE_LIST_PATH,
    DEFAULT_MODELING_TABLE_PATH,
    build_imputed_feature_frames,
    build_prediction_frame,
    load_modeling_table,
    validate_target_column,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMAL_MODEL_PATH = REPO_ROOT / "models" / "ngboost_normal_v1.pkl"
HEAVYTAIL_MODEL_PATH = REPO_ROOT / "models" / "ngboost_heavytail_attempt.pkl"
STUDENT_T_MODEL_PATH = REPO_ROOT / "models" / "ngboost_student_t_attempt.pkl"
LAPLACE_MODEL_PATH = REPO_ROOT / "models" / "ngboost_laplace_attempt.pkl"
COMPARISON_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_comparison.csv"
GROUP_COVERAGE_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_group_coverage.csv"
CANDIDATE_PARAMS_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_candidate_params.csv"
NOTES_PATH = REPO_ROOT / "outputs" / "distribution_choice_notes.md"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

ERROR_INTERVAL_SCHEMA = [
    {"label": "(-inf, -3]", "lower": None, "upper": -3.0},
    {"label": "(-3, -1]", "lower": -3.0, "upper": -1.0},
    {"label": "(-1, 1]", "lower": -1.0, "upper": 1.0},
    {"label": "(1, 3]", "lower": 1.0, "upper": 3.0},
    {"label": "(3, inf)", "lower": 3.0, "upper": None},
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_path = resolve_path(args.dataset_path)
    feature_columns_path = resolve_path(args.feature_columns_path)
    feature_list_path = resolve_path(args.feature_list)

    prepared = prepare_training_data(
        dataset_path=dataset_path,
        feature_columns_path=feature_columns_path,
        feature_list_path=feature_list_path,
        train_end_date=args.train_end_date,
        validation_end_date=args.validation_end_date,
    )

    trained_candidates: dict[str, pd.DataFrame] = {}
    training_notes: list[str] = []

    normal_params = train_and_predict_candidate(
        model_name="ngboost_normal_v1",
        distribution="normal",
        model_output_path=NORMAL_MODEL_PATH,
        prepared=prepared,
    )
    trained_candidates["ngboost_normal_v1"] = normal_params

    heavy_tail_specs = [("ngboost_laplace_attempt", "laplace", HEAVYTAIL_MODEL_PATH)]
    if args.force_student_t:
        heavy_tail_specs.insert(0, ("ngboost_student_t_attempt", "student_t", STUDENT_T_MODEL_PATH))
    else:
        training_notes.append(
            "ngboost_student_t_attempt: not retrained by default because NGBoost 0.5.10 "
            "produced overflow/NaN failures on this project; use --force-student-t to retry."
        )

    for spec in heavy_tail_specs:
        model_name, distribution, output_path = spec
        try:
            trained_candidates[model_name] = train_and_predict_candidate(
                model_name=model_name,
                distribution=distribution,
                model_output_path=output_path,
                prepared=prepared,
            )
            if distribution == "laplace":
                write_model_artifact(
                    model=trained_candidates[model_name].attrs["model"],
                    model_name=model_name,
                    distribution=distribution,
                    model_output_path=LAPLACE_MODEL_PATH,
                    prepared=prepared,
                )
                trained_candidates[model_name].attrs.pop("model", None)
            training_notes.append(
                f"{model_name}: trained successfully with stable prediction parameters."
            )
        except Exception as exc:
            training_notes.append(f"{model_name}: failed or unstable: {type(exc).__name__}: {exc}")

    candidate_frames: list[pd.DataFrame] = [normal_params]
    for factor in [1.05, 1.10, 1.15, 1.20]:
        candidate_frames.append(
            sigma_adjusted_candidate(
                normal_params,
                model_name=f"ngboost_normal_sigma_x{round(factor * 100):03d}",
                sigma_adjustment=factor,
            )
        )

    for model_name, params in trained_candidates.items():
        if model_name == "ngboost_normal_v1":
            continue
        candidate_frames.append(params)

    comparison_rows: list[dict[str, Any]] = []
    group_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, dict[str, Any]] = {}

    for params in candidate_frames:
        model_name = str(params["model_name"].iloc[0])
        comparison, groups, diag = evaluate_candidate(params)
        comparison_rows.append(comparison)
        group_frames.extend(groups)
        diagnostics[model_name] = diag

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = mark_selected_model(comparison_df)
    selected_model = str(comparison_df.loc[comparison_df["selected"], "model_name"].iloc[0])

    write_outputs(
        comparison_df=comparison_df,
        group_frames=group_frames,
        candidate_frames=candidate_frames,
        diagnostics=diagnostics,
        selected_model=selected_model,
        training_notes=training_notes,
        prepared=prepared,
    )

    print("NGBoost distribution comparison complete.")
    print(f"Selected validation candidate: {selected_model}")
    print(comparison_df.to_string(index=False))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit NGBoost forecast-error distributions and Normal scale variants."
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_MODELING_TABLE_PATH))
    parser.add_argument("--feature-columns-path", default=str(DEFAULT_FEATURE_COLUMNS_PATH))
    parser.add_argument(
        "--feature-list",
        default=str(DEFAULT_FINAL_FEATURE_LIST_PATH),
        help="Explicit production feature-list JSON. Used by default when present.",
    )
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--validation-end-date", default=None)
    parser.add_argument(
        "--force-student-t",
        action="store_true",
        help="Retry full NGBoost Student-t training despite prior numerical instability.",
    )
    return parser.parse_args(argv)


def prepare_training_data(
    dataset_path: Path,
    feature_columns_path: Path,
    feature_list_path: Path,
    train_end_date: str | None,
    validation_end_date: str | None,
) -> dict[str, Any]:
    df = load_modeling_table(dataset_path)
    validate_target_column(df)
    split_result = chronological_train_validation_test_split(
        df,
        train_end_date=train_end_date,
        validation_end_date=validation_end_date,
    )
    for split_name, split_df in [
        ("train", split_result.train),
        ("validation", split_result.validation),
        ("test", split_result.test),
    ]:
        validate_target_column(split_df, split_name=split_name)

    if feature_list_path.exists():
        feature_columns = load_feature_list(feature_list_path)
        feature_source_path = feature_list_path
    else:
        feature_columns = get_feature_columns(df, feature_columns_path)
        feature_source_path = feature_columns_path
    validate_no_leakage_feature_columns(feature_columns)
    feature_columns = validate_feature_columns_exist(df, feature_columns)
    X_train, X_validation, X_test, imputer, preprocessing_notes = build_imputed_feature_frames(
        train_df=split_result.train,
        validation_df=split_result.validation,
        test_df=split_result.test,
        feature_columns=feature_columns,
    )
    return {
        "df": df,
        "split_result": split_result,
        "train_df": split_result.train,
        "validation_df": split_result.validation,
        "test_df": split_result.test,
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": split_result.train[TARGET_COLUMN].to_numpy(dtype=float),
        "y_validation": split_result.validation[TARGET_COLUMN].to_numpy(dtype=float),
        "y_test": split_result.test[TARGET_COLUMN].to_numpy(dtype=float),
        "imputer": imputer,
        "feature_columns": feature_columns,
        "split_summary": split_result.summary,
        "preprocessing_notes": preprocessing_notes,
        "dataset_path": dataset_path,
        "feature_columns_path": feature_source_path,
    }


def train_and_predict_candidate(
    model_name: str,
    distribution: str,
    model_output_path: Path,
    prepared: dict[str, Any],
) -> pd.DataFrame:
    dist = normalize_distribution_name(distribution)
    model = train_ngboost_distribution(
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
        X_val=prepared["X_validation"],
        y_val=prepared["y_validation"],
        distribution=dist,
    )

    validation_details = predict_distribution_details(model, prepared["X_validation"], dist)
    test_details = predict_distribution_details(model, prepared["X_test"], dist)
    params = build_candidate_params(
        model_name=model_name,
        distribution=dist,
        sigma_adjustment=1.0,
        validation_details=validation_details,
        test_details=test_details,
        prepared=prepared,
    )
    validate_candidate_cdf(params)
    write_model_artifact(
        model=model,
        model_name=model_name,
        distribution=dist,
        model_output_path=model_output_path,
        prepared=prepared,
    )
    params.attrs["model"] = model
    return params


def build_candidate_params(
    model_name: str,
    distribution: str,
    sigma_adjustment: float,
    validation_details: dict[str, Any],
    test_details: dict[str, Any],
    prepared: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name, split_df, details in [
        ("validation", prepared["validation_df"], validation_details),
        ("test", prepared["test_df"], test_details),
    ]:
        y = split_df[TARGET_COLUMN].to_numpy(dtype=float)
        mu = np.asarray(details["mu"], dtype=float)
        sigma = np.asarray(details["sigma"], dtype=float)
        df_values = details.get("df")
        nll = distribution_nll(
            y,
            mu=mu,
            sigma=sigma,
            distribution=distribution,
            df=df_values,
        )
        frame = build_prediction_frame(
            split_name=split_name,
            split_df=split_df,
            mu=mu,
            sigma=sigma,
            nll=nll,
        )
        frame["scale"] = sigma
        frame["distribution_type"] = distribution
        frame["df"] = np.asarray(df_values, dtype=float) if df_values is not None else np.nan
        frame["model_name"] = model_name
        frame["sigma_adjustment"] = float(sigma_adjustment)
        append_diagnostic_columns(frame, split_df)
        frames.append(frame)

    params = pd.concat(frames, ignore_index=True)
    params.insert(0, "row_id", np.arange(len(params), dtype=int))
    return params


def sigma_adjusted_candidate(
    base_params: pd.DataFrame,
    model_name: str,
    sigma_adjustment: float,
) -> pd.DataFrame:
    params = base_params.copy()
    params["model_name"] = model_name
    params["sigma_adjustment"] = float(sigma_adjustment)
    params["distribution_type"] = "normal"
    params["sigma"] = pd.to_numeric(params["sigma"], errors="raise") * float(sigma_adjustment)
    params["scale"] = params["sigma"]
    params["df"] = np.nan
    params["nll"] = distribution_nll(
        params[TARGET_COLUMN],
        mu=params["mu"],
        sigma=params["sigma"],
        distribution="normal",
    )
    validate_candidate_cdf(params)
    return params


def append_diagnostic_columns(frame: pd.DataFrame, split_df: pd.DataFrame) -> None:
    for column in ["season", "month"]:
        if column in split_df.columns:
            frame[column] = split_df[column].to_numpy()

    source = None
    for candidate in ["prediction_time", "prediction_timestamp", "timestamp"]:
        if candidate in frame.columns:
            source = candidate
            break
    if source is not None:
        parsed = pd.to_datetime(frame[source], errors="coerce")
        frame["prediction_hour"] = parsed.dt.hour


def validate_candidate_cdf(params: pd.DataFrame) -> None:
    validation = params[params["split"] == "validation"].head(50)
    if validation.empty:
        raise ValueError("Candidate has no validation rows")
    dist = str(validation["distribution_type"].iloc[0])
    df_values = validation["df"] if normalize_distribution_name(dist) == "student_t" else None
    values = distribution_cdf(
        validation[TARGET_COLUMN],
        mu=validation["mu"],
        sigma=validation["sigma"],
        distribution=dist,
        df=df_values,
    )
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{dist} candidate failed stable CDF validation")


def write_model_artifact(
    model: Any,
    model_name: str,
    distribution: str,
    model_output_path: Path,
    prepared: dict[str, Any],
) -> None:
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "imputer": prepared["imputer"],
        "feature_columns": prepared["feature_columns"],
        "target": TARGET_COLUMN,
        "model_name": model_name,
        "distribution_type": distribution,
        "split_summary": prepared["split_summary"],
        "preprocessing_notes": prepared["preprocessing_notes"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cdf_required_for_bucket_pricing": True,
    }
    with model_output_path.open("wb") as file:
        pickle.dump(artifact, file)


def evaluate_candidate(
    params: pd.DataFrame,
) -> tuple[dict[str, Any], list[pd.DataFrame], dict[str, Any]]:
    model_name = str(params["model_name"].iloc[0])
    distribution = str(params["distribution_type"].iloc[0])
    sigma_adjustment = float(params["sigma_adjustment"].iloc[0])
    split_metrics: dict[str, dict[str, Any]] = {}
    group_frames: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}

    for split in ["validation", "test"]:
        split_df = params[params["split"] == split].reset_index(drop=True)
        metrics, split_diag = evaluate_candidate_split(split_df, distribution)
        split_metrics[split] = metrics
        diagnostics[split] = split_diag

        for group_col in ["prediction_hour", "season", "forecast_horizon_hours"]:
            if group_col not in split_df.columns or split_df[group_col].isna().all():
                continue
            grouped = coverage_by_group(
                split_df,
                group_col=group_col,
                dist_type=distribution,
                level=0.8,
                min_count=30,
            )
            grouped.insert(0, "model_name", model_name)
            grouped.insert(1, "distribution", distribution)
            grouped.insert(2, "sigma_adjustment", sigma_adjustment)
            grouped.insert(3, "split", split)
            group_frames.append(grouped)

    val = split_metrics["validation"]
    test = split_metrics["test"]
    row = {
        "model_name": model_name,
        "distribution": distribution,
        "sigma_adjustment": sigma_adjustment,
        "val_nll": val["nll"],
        "test_nll": test["nll"],
        "bucket_interval_log_loss": val["market_interval_log_loss"],
        "test_bucket_interval_log_loss": test["market_interval_log_loss"],
        "day9_interval_log_loss": val["day9_interval_log_loss"],
        "test_day9_interval_log_loss": test["day9_interval_log_loss"],
        "market_mean_bucket_brier": val["market_mean_bucket_brier"],
        "test_market_mean_bucket_brier": test["market_mean_bucket_brier"],
        "day9_mean_bucket_brier": val["day9_mean_bucket_brier"],
        "coverage_50": val["coverage_50"],
        "coverage_80": val["coverage_80"],
        "coverage_90": val["coverage_90"],
        "test_coverage_50": test["coverage_50"],
        "test_coverage_80": test["coverage_80"],
        "test_coverage_90": test["coverage_90"],
        "avg_width_80": val["avg_width_80"],
        "avg_width_90": val["avg_width_90"],
        "residual_mean": val["residual_mean"],
        "residual_std": val["residual_std"],
        "pit_mean": val["pit_mean"],
        "pit_shape_note": val["pit_shape_note"],
        "tail_issue_note": val["tail_issue_note"],
        "cdf_stable": True,
        "selected": False,
    }
    if "df_median" in val:
        row["df_median"] = val["df_median"]
    return row, group_frames, diagnostics


def evaluate_candidate_split(
    split_df: pd.DataFrame,
    distribution: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    df_values = split_df["df"] if normalize_distribution_name(distribution) == "student_t" else None
    coverage = prediction_interval_coverage(
        split_df[TARGET_COLUMN],
        split_df["mu"],
        split_df["sigma"],
        levels=(0.5, 0.8, 0.9),
        dist_type=distribution,
        df=df_values,
    ).set_index("level")
    pit = compute_pit_values(
        split_df[TARGET_COLUMN],
        split_df["mu"],
        split_df["sigma"],
        dist_type=distribution,
        df=df_values,
    )
    z = standardized_residuals(
        split_df[TARGET_COLUMN],
        split_df["mu"],
        split_df["sigma"],
        dist_type=distribution,
        df=df_values,
    )
    residual = residual_summary(z).iloc[0]

    market_probs, market_labels = market_bucket_probabilities(split_df, distribution)
    market_brier = bucket_brier_scores(market_probs, market_labels)
    day9_probs, day9_labels = error_interval_probabilities(split_df, distribution)
    day9_brier = bucket_brier_scores(day9_probs, day9_labels)

    metrics = {
        "nll": float(pd.to_numeric(split_df["nll"], errors="raise").mean()),
        "market_interval_log_loss": interval_log_loss(market_probs, market_labels),
        "day9_interval_log_loss": interval_log_loss(day9_probs, day9_labels),
        "market_mean_bucket_brier": float(market_brier["brier_score"].mean()),
        "day9_mean_bucket_brier": float(day9_brier["brier_score"].mean()),
        "coverage_50": float(coverage.loc[0.5, "actual_coverage"]),
        "coverage_80": float(coverage.loc[0.8, "actual_coverage"]),
        "coverage_90": float(coverage.loc[0.9, "actual_coverage"]),
        "avg_width_80": float(coverage.loc[0.8, "avg_interval_width"]),
        "avg_width_90": float(coverage.loc[0.9, "avg_interval_width"]),
        "residual_mean": float(residual["mean"]),
        "residual_std": float(residual["std"]),
        "pit_mean": float(pit.mean()),
        "pit_shape_note": describe_pit_shape(pit),
    }
    metrics["tail_issue_note"] = describe_tail_issue(metrics)
    if normalize_distribution_name(distribution) == "student_t":
        metrics["df_median"] = float(pd.to_numeric(split_df["df"], errors="raise").median())

    diagnostics = {
        "pit": pit,
        "standardized_residual": z,
        "coverage": coverage.reset_index(),
        "market_brier": market_brier,
        "day9_brier": day9_brier,
    }
    return metrics, diagnostics


def market_bucket_probabilities(
    split_df: pd.DataFrame,
    distribution: str,
) -> tuple[pd.DataFrame, pd.Series]:
    long = price_buckets_for_dataframe(split_df, dist_type=distribution)
    long["market_bucket"] = "market_bucket_" + long["bucket_index"].astype(int).astype(str)

    actual = pd.to_numeric(long["actual_high"], errors="raise")
    lower = pd.to_numeric(long["bucket_lower_temp"], errors="coerce")
    upper = pd.to_numeric(long["bucket_upper_temp"], errors="coerce")
    in_bucket = (lower.isna() | (actual > lower)) & (upper.isna() | (actual <= upper))
    labels = long[in_bucket][["row_id", "market_bucket"]]
    if labels.duplicated("row_id").any():
        raise ValueError("A market row matched multiple realized buckets")
    labels = labels.set_index("row_id")["market_bucket"]

    probs = long.pivot(index="row_id", columns="market_bucket", values="probability")
    ordered_row_ids = split_df["row_id"].to_numpy()
    probs = probs.reindex(ordered_row_ids)
    labels = labels.reindex(ordered_row_ids)
    if probs.isna().any().any() or labels.isna().any():
        raise ValueError("Market bucket probabilities or labels failed row alignment")
    probs = validate_bucket_probabilities(probs.reset_index(drop=True))
    return probs, labels.reset_index(drop=True)


def error_interval_probabilities(
    split_df: pd.DataFrame,
    distribution: str,
) -> tuple[pd.DataFrame, pd.Series]:
    df_values = split_df["df"] if normalize_distribution_name(distribution) == "student_t" else None
    probabilities: dict[str, np.ndarray] = {}
    for spec in ERROR_INTERVAL_SCHEMA:
        lower = spec["lower"]
        upper = spec["upper"]
        lower_cdf = (
            np.zeros(len(split_df), dtype=float)
            if lower is None
            else distribution_cdf(
                float(lower),
                mu=split_df["mu"],
                sigma=split_df["sigma"],
                distribution=distribution,
                df=df_values,
            )
        )
        upper_cdf = (
            np.ones(len(split_df), dtype=float)
            if upper is None
            else distribution_cdf(
                float(upper),
                mu=split_df["mu"],
                sigma=split_df["sigma"],
                distribution=distribution,
                df=df_values,
            )
        )
        probabilities[str(spec["label"])] = np.asarray(upper_cdf - lower_cdf, dtype=float)

    probs = validate_bucket_probabilities(pd.DataFrame(probabilities), allow_renormalize=True)
    labels = assign_error_interval_labels(split_df[TARGET_COLUMN])
    return probs, labels


def assign_error_interval_labels(errors: pd.Series) -> pd.Series:
    labels: list[str] = []
    for raw_value in pd.to_numeric(errors, errors="raise"):
        value = float(raw_value)
        matched = None
        for spec in ERROR_INTERVAL_SCHEMA:
            lower = spec["lower"]
            upper = spec["upper"]
            lower_ok = lower is None or value > float(lower)
            upper_ok = upper is None or value <= float(upper)
            if lower_ok and upper_ok:
                matched = str(spec["label"])
                break
        if matched is None:
            raise ValueError(f"No forecast-error interval matched value {value:g}")
        labels.append(matched)
    return pd.Series(labels, name="realized_error_interval")


def describe_pit_shape(pit: pd.Series) -> str:
    values = np.asarray(pit, dtype=float)
    counts, _ = np.histogram(values, bins=10, range=(0.0, 1.0))
    shares = counts / counts.sum()
    edge_share = float((shares[0] + shares[-1]) / 2.0)
    center_share = float((shares[4] + shares[5]) / 2.0)
    pit_mean = float(values.mean())
    skew_note = ""
    if pit_mean < 0.47:
        skew_note = " with low-PIT skew"
    elif pit_mean > 0.53:
        skew_note = " with high-PIT skew"
    if edge_share > center_share * 1.2:
        return "U-shaped / too narrow" + skew_note
    if center_share > edge_share * 1.2:
        return "hump-shaped / too wide" + skew_note
    if skew_note:
        return "roughly flat" + skew_note
    return "roughly flat"


def describe_tail_issue(metrics: dict[str, Any]) -> str:
    notes: list[str] = []
    if metrics["coverage_80"] < 0.78 or metrics["coverage_90"] < 0.88:
        notes.append("undercovered / overconfident")
    elif metrics["coverage_80"] > 0.82 or metrics["coverage_90"] > 0.92:
        notes.append("wide on validation")
    else:
        notes.append("coverage near nominal")
    if abs(metrics["residual_mean"]) > 0.10:
        notes.append("directional residual shift")
    if metrics["residual_std"] > 1.10:
        notes.append("residuals too dispersed")
    if metrics["residual_std"] < 0.90:
        notes.append("residuals compressed")
    return "; ".join(notes)


def mark_selected_model(comparison_df: pd.DataFrame) -> pd.DataFrame:
    df = comparison_df.copy()
    df["selection_score"] = (
        df["val_nll"].rank(method="min")
        + df["bucket_interval_log_loss"].rank(method="min")
        + (df["coverage_80"] - 0.8).abs().rank(method="min")
        + (df["coverage_90"] - 0.9).abs().rank(method="min")
        + (df["pit_mean"] - 0.5).abs().rank(method="min")
    )
    selected_index = df.sort_values(
        ["selection_score", "val_nll", "bucket_interval_log_loss"],
        kind="stable",
    ).index[0]
    df["selected"] = False
    df.loc[selected_index, "selected"] = True
    return df


def write_outputs(
    comparison_df: pd.DataFrame,
    group_frames: list[pd.DataFrame],
    candidate_frames: list[pd.DataFrame],
    diagnostics: dict[str, dict[str, Any]],
    selected_model: str,
    training_notes: list[str],
    prepared: dict[str, Any],
) -> None:
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    if group_frames:
        pd.concat(group_frames, ignore_index=True).to_csv(GROUP_COVERAGE_PATH, index=False)
    serializable_frames = []
    for frame in candidate_frames:
        output_frame = frame.copy()
        output_frame.attrs.clear()
        serializable_frames.append(output_frame)
    pd.concat(serializable_frames, ignore_index=True).to_csv(CANDIDATE_PARAMS_PATH, index=False)

    plot_main_diagnostics(comparison_df, diagnostics, selected_model)
    write_notes(comparison_df, selected_model, training_notes, prepared)


def plot_main_diagnostics(
    comparison_df: pd.DataFrame,
    diagnostics: dict[str, dict[str, Any]],
    selected_model: str,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    main_models = [
        "ngboost_normal_v1",
        "ngboost_student_t_attempt",
        "ngboost_laplace_attempt",
        selected_model,
    ]
    main_models = [model for index, model in enumerate(main_models) if model in diagnostics and model not in main_models[:index]]

    for model in main_models:
        plot_pit_histogram(
            diagnostics[model]["validation"]["pit"],
            FIGURES_DIR / f"pit_histogram_{model}.png",
            bins=10,
        )
    plot_coverage_comparison(
        comparison_df[comparison_df["model_name"].isin(main_models)],
        FIGURES_DIR / "ngboost_distribution_coverage_validation.png",
    )
    plot_residual_histograms(
        {model: diagnostics[model]["validation"]["standardized_residual"] for model in main_models},
        FIGURES_DIR / "ngboost_distribution_standardized_residuals_validation.png",
    )


def plot_coverage_comparison(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.copy()
    labels = plot_df["model_name"].astype(str).tolist()
    x = np.arange(len(plot_df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8.0, len(plot_df) * 1.8), 5.0))
    for offset, level in [(-width, "50"), (0.0, "80"), (width, "90")]:
        ax.bar(x + offset, plot_df[f"coverage_{level}"], width=width, label=f"{level}%")
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.0)
    ax.axhline(0.8, color="#444444", linestyle="--", linewidth=1.0)
    ax.axhline(0.9, color="#111111", linestyle="-.", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Actual validation coverage")
    ax.set_title("Distribution Candidate Interval Coverage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residual_histograms(residuals: dict[str, pd.Series], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bins = np.linspace(-4.0, 4.0, 41)
    for model_name, values in residuals.items():
        ax.hist(
            np.asarray(values, dtype=float),
            bins=bins,
            alpha=0.35,
            density=True,
            label=model_name,
        )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title("Validation Standardized Residuals")
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_notes(
    comparison_df: pd.DataFrame,
    selected_model: str,
    training_notes: list[str],
    prepared: dict[str, Any],
) -> None:
    selected = comparison_df[comparison_df["model_name"] == selected_model].iloc[0]
    normal = comparison_df[comparison_df["model_name"] == "ngboost_normal_v1"].iloc[0]
    t_rows = comparison_df[comparison_df["model_name"] == "ngboost_student_t_attempt"]
    laplace_rows = comparison_df[comparison_df["model_name"] == "ngboost_laplace_attempt"]

    day13 = load_day13_summary()
    student_t_text = (
        "Student-t trained successfully and produced finite CDF, PPF, log-score, and bucket "
        "probabilities."
        if not t_rows.empty
        else "Student-t was attempted but was not usable in this environment."
    )
    laplace_text = (
        "Laplace also trained successfully as a signed heavier-tailed candidate."
        if not laplace_rows.empty
        else "Laplace was not available as a stable trained candidate."
    )

    lines = [
        "# NGBoost Distribution Choice Notes",
        "",
        "## What Day 13 Suggested",
        "",
        day13,
        "",
        "The practical read is mixed: validation Normal was slightly wide, while the later test period was undercovered and left too little mass in the tails. Distribution choice is therefore selected on validation probability metrics, with test-period diagnostics treated as robustness context.",
        "",
        "## Candidates Tested",
        "",
        "- `ngboost_normal_v1`: explicit Normal NGBoost baseline, same chronological split, features, seed, and hyperparameters as the prior Normal path.",
        "- `ngboost_student_t_attempt`: NGBoost `T` distribution for signed forecast errors.",
        "- `ngboost_laplace_attempt`: NGBoost `Laplace` signed-error candidate, saved to `models/ngboost_heavytail_attempt.pkl` when stable.",
        "- Normal post-hoc sigma inflation variants at 1.05, 1.10, 1.15, and 1.20, with the mean unchanged.",
        "",
        "Positive-only NGBoost distributions such as Gamma and LogNormal were excluded because raw `forecast_error` can be negative.",
        "",
        "## Heavy-Tail Support",
        "",
        student_t_text,
        laplace_text,
        "",
        *[f"- {note}" for note in training_notes],
        "",
        "The mitigation is to use a supported signed heavy-tailed distribution with stable CDF values (`Laplace`) and to keep Normal sigma-inflation variants as conservative calibration checks rather than forcing a fragile custom Student-t implementation.",
        "",
        "## Validation Result",
        "",
        f"Selected model: `{selected_model}`.",
        "",
        "| model | dist | sigma x | val NLL | bucket log loss | cov80 | cov90 | PIT note | selected |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in comparison_df.sort_values("selection_score").iterrows():
        lines.append(
            "| {model} | {dist} | {sig:.2f} | {nll:.4f} | {bucket:.4f} | {cov80:.3f} | {cov90:.3f} | {pit} | {selected} |".format(
                model=row["model_name"],
                dist=row["distribution"],
                sig=float(row["sigma_adjustment"]),
                nll=float(row["val_nll"]),
                bucket=float(row["bucket_interval_log_loss"]),
                cov80=float(row["coverage_80"]),
                cov90=float(row["coverage_90"]),
                pit=row["pit_shape_note"],
                selected="yes" if bool(row["selected"]) else "",
            )
        )

    lines.extend(
        [
            "",
            "The selected row had the best validation balance across continuous NLL, final-temperature bucket interval log loss, interval coverage, and PIT behavior. The Normal baseline remains the clean reference: "
            f"validation NLL {float(normal['val_nll']):.4f}, bucket log loss {float(normal['bucket_interval_log_loss']):.4f}, "
            f"80% coverage {float(normal['coverage_80']):.3f}.",
            "",
            "## Bucket-CDF Compatibility",
            "",
            "Every successful candidate was evaluated through CDF differences using the same final-temperature bucket conversion. The selected model is compatible with downstream bucket probability generation: probabilities were finite, nonnegative within tolerance, and row-normalized by construction over the open-ended bucket set.",
            "",
            "## Limitations",
            "",
            "- This comparison still uses one chronological validation year. A walk-forward validation pass would be a stronger guard against the 2025-2026 undercoverage seen on Day 13.",
            "- Sigma inflation can improve tail coverage but may degrade validation log score when the validation period is already slightly wide.",
            "- Student-t introduces an additional degrees-of-freedom parameter; even when its CDF is stable, it should be monitored for very low df values and over-wide intervals.",
            "- No residual clipping was used for final training or evaluation. Extreme residuals remain in the validation metrics.",
            "",
            "## Reproducibility",
            "",
            f"- Dataset: `{Path(prepared['dataset_path']).relative_to(REPO_ROOT)}`",
            f"- Feature spec: `{Path(prepared['feature_columns_path']).relative_to(REPO_ROOT)}`",
            f"- Split: train through {prepared['split_summary']['train_end_date']}, validation through {prepared['split_summary']['validation_end_date']}",
            f"- Feature count: {len(prepared['feature_columns'])}",
        ]
    )
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_day13_summary() -> str:
    coverage_path = REPO_ROOT / "outputs" / "coverage_report.csv"
    residual_path = REPO_ROOT / "outputs" / "standardized_residual_summary.csv"
    eval_path = REPO_ROOT / "outputs" / "ngboost_evaluation_report.csv"
    if not coverage_path.exists() or not residual_path.exists() or not eval_path.exists():
        return "Prior Day 13 diagnostics were not fully available."

    coverage = pd.read_csv(coverage_path)
    residual = pd.read_csv(residual_path)
    eval_report = pd.read_csv(eval_path)
    val_eval = eval_report[(eval_report["model"] == "ngboost_normal_v0") & (eval_report["split"] == "validation")].iloc[0]
    test_eval = eval_report[(eval_report["model"] == "ngboost_normal_v0") & (eval_report["split"] == "test")].iloc[0]
    val_cov = coverage[coverage["split"] == "validation"].set_index("level")
    test_cov = coverage[coverage["split"] == "test"].set_index("level")
    val_res = residual[residual["split"] == "validation"].iloc[0]
    test_res = residual[residual["split"] == "test"].iloc[0]
    return (
        f"Day 13 Normal `ngboost_normal_v0` had validation NLL {float(val_eval['nll']):.4f} "
        f"and test NLL {float(test_eval['nll']):.4f}. Validation coverage was "
        f"50/80/90% = {float(val_cov.loc[0.5, 'actual_coverage']):.3f}/"
        f"{float(val_cov.loc[0.8, 'actual_coverage']):.3f}/"
        f"{float(val_cov.loc[0.9, 'actual_coverage']):.3f}; test coverage was "
        f"{float(test_cov.loc[0.5, 'actual_coverage']):.3f}/"
        f"{float(test_cov.loc[0.8, 'actual_coverage']):.3f}/"
        f"{float(test_cov.loc[0.9, 'actual_coverage']):.3f}. Standardized residual std moved "
        f"from {float(val_res['std']):.3f} on validation to {float(test_res['std']):.3f} on test, "
        f"with test residual mean {float(test_res['mean']):.3f}."
    )


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


if __name__ == "__main__":
    main(sys.argv[1:])
