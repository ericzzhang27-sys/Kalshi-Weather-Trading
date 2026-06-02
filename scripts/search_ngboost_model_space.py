from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distribution_pricing import price_buckets_for_dataframe  # noqa: E402
from src.distributional_model import (  # noqa: E402
    distribution_nll,
    normalize_distribution_name,
    predict_distribution_details,
    train_ngboost_distribution,
)
from src.evaluation import interval_log_loss, validate_bucket_probabilities  # noqa: E402
from src.splits import chronological_train_validation_test_split  # noqa: E402
from src.train_ngboost import (  # noqa: E402
    TARGET_COLUMN,
    build_imputed_feature_frames,
    build_prediction_frame,
    load_modeling_table,
    validate_target_column,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"
DEFAULT_FINAL_FEATURE_LIST = REPO_ROOT / "outputs" / "final_feature_list.json"
DEFAULT_SAFE_FEATURE_LIST = REPO_ROOT / "outputs" / "day8_features" / "feature_columns.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_search"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_best_validation_search.pkl"


BASE_HYPERPARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_samples_leaf": 50,
    "minibatch_frac": 1.0,
    "natural_gradient": True,
    "random_state": 11,
    "early_stopping_rounds": 20,
}

REFINEMENT_HYPERPARAMS = [
    {
        "name": "standard_120_lr005_leaf50",
        **BASE_HYPERPARAMS,
    },
    {
        "name": "standard_120_lr005_leaf20",
        **BASE_HYPERPARAMS,
        "min_samples_leaf": 20,
    },
    {
        "name": "medium_300_lr003_leaf50",
        **BASE_HYPERPARAMS,
        "n_estimators": 300,
        "learning_rate": 0.03,
    },
    {
        "name": "more_500_lr002_leaf50",
        **BASE_HYPERPARAMS,
        "n_estimators": 500,
        "learning_rate": 0.02,
    },
]

SCALE_FACTORS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search leakage-safe NGBoost feature, distribution, hyperparameter, and sigma-scale combinations."
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--final-feature-list", default=str(DEFAULT_FINAL_FEATURE_LIST))
    parser.add_argument("--safe-feature-list", default=str(DEFAULT_SAFE_FEATURE_LIST))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--model-output-path", default=str(MODEL_OUTPUT_PATH))
    parser.add_argument("--top-refine", type=int, default=4)
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["normal", "laplace", "cauchy"],
        help="Distribution families to search in the broad stage.",
    )
    parser.add_argument(
        "--include-student-t",
        action="store_true",
        help="Include Student-t in the broad stage. It has been numerically fragile in prior runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_modeling_table(Path(args.dataset_path))
    validate_target_column(data)
    split = chronological_train_validation_test_split(data)
    for split_name, frame in [
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ]:
        validate_target_column(frame, split_name=split_name)

    safe_features = load_feature_columns(args.safe_feature_list)
    final_features = load_feature_columns(args.final_feature_list)
    feature_sets = build_feature_sets(final_features, safe_features)
    distributions = [normalize_distribution_name(value) for value in args.distributions]
    if args.include_student_t:
        distributions.append("student_t")
    distributions = list(dict.fromkeys(distributions))

    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    broad_specs = [
        (feature_name, distribution, REFINEMENT_HYPERPARAMS[0])
        for feature_name in feature_sets
        for distribution in distributions
    ]
    print(
        f"Stage 1: {len(broad_specs)} broad feature/distribution runs.",
        flush=True,
    )
    for index, (feature_name, distribution, params) in enumerate(broad_specs, start=1):
        result = train_and_score_candidate(
            data_splits=split,
            feature_name=feature_name,
            feature_columns=feature_sets[feature_name],
            distribution=distribution,
            hyperparams=params,
        )
        record_candidate_result(
            result,
            run_rows,
            metric_rows,
            trained_cache,
        )
        write_checkpoint(output_dir, run_rows, metric_rows)
        print(
            f"[broad {index}/{len(broad_specs)}] {feature_name} / {distribution}: {result['status']}",
            flush=True,
        )

    broad_metrics = pd.DataFrame(metric_rows)
    top_pairs = select_top_feature_distribution_pairs(
        broad_metrics,
        top_n=int(args.top_refine),
    )
    print(f"Stage 2: refining {len(top_pairs)} validation-selected pairs.", flush=True)
    refinement_specs = [
        (feature_name, distribution, params)
        for feature_name, distribution in top_pairs
        for params in REFINEMENT_HYPERPARAMS
    ]
    seen = {
        (
            str(row["feature_set"]),
            str(row["distribution"]),
            str(row["hyperparams_name"]),
        )
        for row in run_rows
    }
    for index, (feature_name, distribution, params) in enumerate(refinement_specs, start=1):
        key = (feature_name, distribution, str(params["name"]))
        if key in seen:
            continue
        result = train_and_score_candidate(
            data_splits=split,
            feature_name=feature_name,
            feature_columns=feature_sets[feature_name],
            distribution=distribution,
            hyperparams=params,
        )
        record_candidate_result(
            result,
            run_rows,
            metric_rows,
            trained_cache,
        )
        write_checkpoint(output_dir, run_rows, metric_rows)
        print(
            f"[refine {index}/{len(refinement_specs)}] {feature_name} / {distribution} / {params['name']}: {result['status']}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    runs = pd.DataFrame(run_rows)
    metrics = add_selection_ranks(metrics)
    selected = select_validation_winner(metrics)
    write_outputs(
        metrics=metrics,
        runs=runs,
        selected=selected,
        split_summary=split.summary,
        feature_sets=feature_sets,
        safe_features=safe_features,
        output_dir=output_dir,
        elapsed_seconds=time.perf_counter() - started,
    )
    save_selected_model(
        selected=selected,
        trained_cache=trained_cache,
        feature_sets=feature_sets,
        split_summary=split.summary,
        model_output_path=Path(args.model_output_path),
    )
    print(
        "Selected validation winner: "
        f"{selected['feature_set']} / {selected['distribution']} / "
        f"{selected['hyperparams_name']} / sigma x{selected['sigma_factor']}.",
        flush=True,
    )


def load_feature_columns(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("features", payload.get("feature_columns"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"No feature list found in {path}")
    features = [str(column) for column in raw]
    if len(set(features)) != len(features):
        raise ValueError(f"Feature list has duplicates: {path}")
    return features


def build_feature_sets(
    final_features: list[str],
    safe_features: list[str],
) -> dict[str, list[str]]:
    safe = set(safe_features)
    for feature in final_features:
        if feature not in safe:
            raise ValueError(f"Final feature is not in safe Day 8 feature list: {feature}")

    candidates: dict[str, list[str]] = {
        "current36": final_features,
        "full_safe39": safe_features,
    }
    addable = [
        feature for feature in safe_features
        if feature not in final_features
    ]
    for feature in addable:
        candidates[f"current36_plus_{feature}"] = append_features(final_features, [feature], safe_features)

    if len(addable) >= 2:
        candidates["current36_plus_forecast_high_day_cos"] = append_features(
            final_features,
            ["forecast_high", "day_of_year_cos"],
            safe_features,
        )
        candidates["current36_plus_forecast_high_temp_range"] = append_features(
            final_features,
            ["forecast_high", "temp_range_so_far"],
            safe_features,
        )

    return candidates


def append_features(
    base_features: list[str],
    additions: list[str],
    order_reference: list[str],
) -> list[str]:
    selected = set(base_features) | set(additions)
    missing = [feature for feature in additions if feature not in order_reference]
    if missing:
        raise ValueError(f"Cannot add features outside safe reference list: {missing}")
    return [feature for feature in order_reference if feature in selected]


def train_and_score_candidate(
    *,
    data_splits: Any,
    feature_name: str,
    feature_columns: list[str],
    distribution: str,
    hyperparams: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    dist = normalize_distribution_name(distribution)
    run_id = stable_run_id(feature_name, dist, str(hyperparams["name"]))
    try:
        X_train, X_validation, X_test, imputer, preprocessing_notes = build_imputed_feature_frames(
            train_df=data_splits.train,
            validation_df=data_splits.validation,
            test_df=data_splits.test,
            feature_columns=feature_columns,
        )
        y_train = data_splits.train[TARGET_COLUMN].to_numpy(dtype=float)
        model = train_ngboost_distribution(
            X_train=X_train,
            y_train=y_train,
            X_val=X_validation,
            y_val=data_splits.validation[TARGET_COLUMN].to_numpy(dtype=float),
            distribution=dist,
            n_estimators=int(hyperparams["n_estimators"]),
            learning_rate=float(hyperparams["learning_rate"]),
            max_depth=int(hyperparams["max_depth"]),
            min_samples_leaf=int(hyperparams["min_samples_leaf"]),
            minibatch_frac=float(hyperparams["minibatch_frac"]),
            natural_gradient=bool(hyperparams["natural_gradient"]),
            random_state=int(hyperparams["random_state"]),
            early_stopping_rounds=hyperparams.get("early_stopping_rounds"),
        )
        validation_details = predict_distribution_details(model, X_validation, dist)
        test_details = predict_distribution_details(model, X_test, dist)
        return {
            "status": "success",
            "run_id": run_id,
            "feature_set": feature_name,
            "feature_count": len(feature_columns),
            "distribution": dist,
            "hyperparams": dict(hyperparams),
            "elapsed_seconds": time.perf_counter() - started,
            "model": model,
            "imputer": imputer,
            "preprocessing_notes": preprocessing_notes,
            "validation_frame": data_splits.validation,
            "test_frame": data_splits.test,
            "validation_details": validation_details,
            "test_details": test_details,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "run_id": run_id,
            "feature_set": feature_name,
            "feature_count": len(feature_columns),
            "distribution": dist,
            "hyperparams": dict(hyperparams),
            "elapsed_seconds": time.perf_counter() - started,
            "error_message": f"{type(exc).__name__}: {exc}",
        }


def record_candidate_result(
    result: dict[str, Any],
    run_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    trained_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    hyperparams = result["hyperparams"]
    base_row = {
        "run_id": result["run_id"],
        "status": result["status"],
        "feature_set": result["feature_set"],
        "feature_count": result["feature_count"],
        "distribution": result["distribution"],
        "hyperparams_name": hyperparams["name"],
        "n_estimators": hyperparams["n_estimators"],
        "learning_rate": hyperparams["learning_rate"],
        "max_depth": hyperparams["max_depth"],
        "min_samples_leaf": hyperparams["min_samples_leaf"],
        "minibatch_frac": hyperparams["minibatch_frac"],
        "natural_gradient": hyperparams["natural_gradient"],
        "random_state": hyperparams["random_state"],
        "early_stopping_rounds": hyperparams.get("early_stopping_rounds"),
        "elapsed_seconds": result["elapsed_seconds"],
        "error_message": result.get("error_message", ""),
    }
    run_rows.append(base_row)
    if result["status"] != "success":
        return

    cache_key = (
        result["feature_set"],
        result["distribution"],
        str(hyperparams["name"]),
    )
    trained_cache[cache_key] = result
    for sigma_factor in SCALE_FACTORS:
        for split_name in ["validation", "test"]:
            metric_rows.append(
                score_split(
                    result=result,
                    split_name=split_name,
                    sigma_factor=float(sigma_factor),
                )
            )


def score_split(
    *,
    result: dict[str, Any],
    split_name: str,
    sigma_factor: float,
) -> dict[str, Any]:
    frame = result[f"{split_name}_frame"]
    details = result[f"{split_name}_details"]
    dist = result["distribution"]
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=float)
    mu = np.asarray(details["mu"], dtype=float)
    sigma = np.asarray(details["sigma"], dtype=float) * float(sigma_factor)
    df_values = details.get("df")
    nll = distribution_nll(
        y_true,
        mu=mu,
        sigma=sigma,
        distribution=dist,
        df=df_values,
    )
    prediction_frame = build_prediction_frame(
        split_name=split_name,
        split_df=frame,
        mu=mu,
        sigma=sigma,
        nll=np.asarray(nll, dtype=float),
        distribution_type=dist,
        df=df_values,
    )
    if "row_id" not in prediction_frame.columns:
        prediction_frame.insert(0, "row_id", np.arange(len(prediction_frame), dtype=int))
    prediction_frame["distribution_type"] = dist
    long = price_buckets_for_dataframe(prediction_frame, dist_type=dist)
    bucket_probs, labels = long_bucket_probabilities_and_labels(long, prediction_frame)
    brier = multiclass_bucket_brier(bucket_probs, labels)
    log_loss = interval_log_loss(bucket_probs, labels)
    return {
        "run_id": result["run_id"],
        "feature_set": result["feature_set"],
        "feature_count": result["feature_count"],
        "distribution": dist,
        "hyperparams_name": result["hyperparams"]["name"],
        "sigma_factor": float(sigma_factor),
        "split": split_name,
        "nll": float(np.mean(nll)),
        "bucket_log_loss": float(log_loss),
        "bucket_brier": float(brier),
        "mean_sigma": float(np.mean(sigma)),
        "median_sigma": float(np.median(sigma)),
    }


def long_bucket_probabilities_and_labels(
    long: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    working = long.copy()
    working["market_bucket"] = "market_bucket_" + working["bucket_index"].astype(int).astype(str)
    actual = pd.to_numeric(working["actual_high"], errors="raise")
    lower = pd.to_numeric(working["bucket_lower_temp"], errors="coerce")
    upper = pd.to_numeric(working["bucket_upper_temp"], errors="coerce")
    in_bucket = (lower.isna() | (actual > lower)) & (upper.isna() | (actual <= upper))
    labels = working[in_bucket][["row_id", "market_bucket"]]
    if labels.duplicated("row_id").any():
        raise ValueError("A row matched multiple realized buckets")
    expected = pd.Index(pd.to_numeric(prediction_frame["row_id"], errors="raise").astype(int))
    labels = labels.set_index("row_id")["market_bucket"].reindex(expected)
    if labels.isna().any():
        raise ValueError("Some rows did not match a realized bucket")

    probs = working.pivot(index="row_id", columns="market_bucket", values="probability")
    ordered_columns = [
        f"market_bucket_{bucket_index}"
        for bucket_index in sorted(working["bucket_index"].astype(int).unique())
    ]
    probs = probs[ordered_columns].reindex(expected)
    if probs.isna().any().any():
        raise ValueError("Bucket probability frame has missing values")
    probs = validate_bucket_probabilities(probs.astype(float), allow_renormalize=False)
    return probs.reset_index(drop=True), labels.reset_index(drop=True)


def multiclass_bucket_brier(
    bucket_probs: pd.DataFrame,
    labels: pd.Series,
) -> float:
    probabilities = validate_bucket_probabilities(bucket_probs)
    realized = pd.Series(labels).reset_index(drop=True)
    positions = pd.Index(probabilities.columns).get_indexer(realized)
    if (positions < 0).any():
        raise ValueError("Realized label missing from bucket probability columns")
    actual = np.zeros_like(probabilities.to_numpy(dtype=float))
    actual[np.arange(len(realized)), positions] = 1.0
    return float(np.mean(np.sum((probabilities.to_numpy(dtype=float) - actual) ** 2, axis=1)))


def select_top_feature_distribution_pairs(
    metrics: pd.DataFrame,
    *,
    top_n: int,
) -> list[tuple[str, str]]:
    if metrics.empty:
        raise ValueError("No metrics available for refinement selection")
    validation = add_selection_ranks(metrics[metrics["split"] == "validation"].copy())
    winners = (
        validation.sort_values("validation_rank_sum", kind="stable")
        .drop_duplicates(["feature_set", "distribution"], keep="first")
        .head(top_n)
    )
    return [
        (str(row["feature_set"]), str(row["distribution"]))
        for _, row in winners.iterrows()
    ]


def add_selection_ranks(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    result = metrics.copy()
    validation_mask = result["split"] == "validation"
    validation = result.loc[validation_mask].copy()
    for metric in ["nll", "bucket_log_loss", "bucket_brier"]:
        validation[f"val_{metric}_rank"] = validation[metric].rank(method="min", ascending=True)
    validation["validation_rank_sum"] = (
        validation["val_nll_rank"]
        + validation["val_bucket_log_loss_rank"]
        + validation["val_bucket_brier_rank"]
    )
    rank_cols = [
        "run_id",
        "sigma_factor",
        "val_nll_rank",
        "val_bucket_log_loss_rank",
        "val_bucket_brier_rank",
        "validation_rank_sum",
    ]
    result = result.merge(
        validation[rank_cols],
        on=["run_id", "sigma_factor"],
        how="left",
    )
    return result


def select_validation_winner(metrics: pd.DataFrame) -> pd.Series:
    validation = metrics[metrics["split"] == "validation"].copy()
    if validation.empty:
        raise ValueError("No validation metrics were produced")
    return validation.sort_values(
        ["validation_rank_sum", "nll", "bucket_log_loss", "bucket_brier"],
        kind="stable",
    ).iloc[0]


def write_outputs(
    *,
    metrics: pd.DataFrame,
    runs: pd.DataFrame,
    selected: pd.Series,
    split_summary: dict[str, Any],
    feature_sets: dict[str, list[str]],
    safe_features: list[str],
    output_dir: Path,
    elapsed_seconds: float,
) -> None:
    metrics_path = output_dir / "ngboost_model_space_search.csv"
    runs_path = output_dir / "ngboost_model_space_train_runs.csv"
    feature_path = output_dir / "best_validation_feature_list.json"
    report_path = output_dir / "ngboost_model_space_best_summary.md"

    metrics.sort_values(
        ["split", "validation_rank_sum", "nll", "bucket_log_loss", "bucket_brier"],
        na_position="last",
        kind="stable",
    ).to_csv(metrics_path, index=False)
    runs.to_csv(runs_path, index=False)
    feature_path.write_text(
        json.dumps(
            {
                "version": "model_space_search_validation_winner",
                "selection_rule": "Validation-only rank sum across NLL, bucket log loss, and bucket Brier",
                "features": feature_sets[str(selected["feature_set"])],
                "feature_count": len(feature_sets[str(selected["feature_set"])]),
                "safe_feature_reference_count": len(safe_features),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        build_report(
            metrics=metrics,
            runs=runs,
            selected=selected,
            split_summary=split_summary,
            elapsed_seconds=elapsed_seconds,
            metrics_path=metrics_path,
            runs_path=runs_path,
            feature_path=feature_path,
        ),
        encoding="utf-8",
    )


def write_checkpoint(
    output_dir: Path,
    run_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_rows:
        pd.DataFrame(run_rows).to_csv(
            output_dir / "ngboost_model_space_train_runs_partial.csv",
            index=False,
        )
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(
            output_dir / "ngboost_model_space_search_partial.csv",
            index=False,
        )


def build_report(
    *,
    metrics: pd.DataFrame,
    runs: pd.DataFrame,
    selected: pd.Series,
    split_summary: dict[str, Any],
    elapsed_seconds: float,
    metrics_path: Path,
    runs_path: Path,
    feature_path: Path,
) -> str:
    selected_rows = metrics[
        (metrics["run_id"] == selected["run_id"])
        & (metrics["sigma_factor"] == selected["sigma_factor"])
    ].sort_values("split")
    validation_winners = {}
    for metric in ["nll", "bucket_log_loss", "bucket_brier"]:
        row = metrics[metrics["split"] == "validation"].sort_values(metric, kind="stable").iloc[0]
        validation_winners[metric] = row

    lines = [
        "# NGBoost Model Space Search",
        "",
        f"- Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Elapsed seconds: {elapsed_seconds:.1f}",
        f"- Successful training runs: {int((runs['status'] == 'success').sum())}",
        f"- Failed training runs: {int((runs['status'] != 'success').sum())}",
        f"- Metrics rows: {len(metrics)}",
        f"- Metrics CSV: `{metrics_path.relative_to(REPO_ROOT)}`",
        f"- Training-run CSV: `{runs_path.relative_to(REPO_ROOT)}`",
        f"- Selected feature list: `{feature_path.relative_to(REPO_ROOT)}`",
        "",
        "## Leakage and Selection Guardrails",
        "",
        "- All candidate features come from the Day 8 leakage-safe feature list.",
        "- Excluded forecast-lookahead columns were not searched because their issue/run timestamps cannot prove availability at prediction time.",
        "- Model selection uses validation metrics only; test metrics are reported after selection and are not used for choosing features, distributions, hyperparameters, or sigma scale.",
        f"- Chronological split: train through {split_summary['train_end_date']}, validation through {split_summary['validation_end_date']}, test starts {split_summary['test_start_date']}.",
        "",
        "## Selected Validation Winner",
        "",
        markdown_table(selected_rows),
        "",
        "## Individual Validation Metric Winners",
        "",
    ]
    for metric, row in validation_winners.items():
        lines.extend(
            [
                f"### Best Validation {metric}",
                "",
                markdown_table(row.to_frame().T),
                "",
            ]
        )
    failed = runs[runs["status"] != "success"]
    if len(failed) > 0:
        lines.extend(["## Failed Runs", "", markdown_table(failed), ""])
    return "\n".join(lines)


def save_selected_model(
    *,
    selected: pd.Series,
    trained_cache: dict[tuple[str, str, str], dict[str, Any]],
    feature_sets: dict[str, list[str]],
    split_summary: dict[str, Any],
    model_output_path: Path,
) -> None:
    key = (
        str(selected["feature_set"]),
        str(selected["distribution"]),
        str(selected["hyperparams_name"]),
    )
    result = trained_cache[key]
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": result["model"],
        "imputer": result["imputer"],
        "feature_columns": feature_sets[str(selected["feature_set"])],
        "target": TARGET_COLUMN,
        "model_name": "ngboost_best_validation_search",
        "distribution_type": str(selected["distribution"]),
        "sigma_scale": float(selected["sigma_factor"]),
        "selection_rule": "validation-only rank sum across NLL, bucket log loss, and bucket Brier",
        "selected_validation_row": selected.to_dict(),
        "split_summary": split_summary,
        "preprocessing_notes": result["preprocessing_notes"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set_used_for_selection": False,
    }
    with model_output_path.open("wb") as handle:
        pickle.dump(artifact, handle)


def stable_run_id(feature_name: str, distribution: str, hyperparams_name: str) -> str:
    return f"{slug(feature_name)}__{slug(distribution)}__{slug(hyperparams_name)}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    visible = df.copy()
    for column in visible.columns:
        if pd.api.types.is_float_dtype(visible[column]):
            visible[column] = visible[column].map(lambda value: f"{float(value):.6g}")
    columns = [str(column) for column in visible.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in visible.to_numpy(dtype=object)
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


if __name__ == "__main__":
    main(sys.argv[1:])
