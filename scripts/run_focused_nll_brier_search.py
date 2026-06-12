from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.search_ngboost_model_space import (  # noqa: E402
    append_features,
    stable_run_id,
)
from scripts.evaluate_ngboost import (  # noqa: E402
    assign_error_interval_labels,
    error_interval_probabilities,
)
from src.distributional_model import TARGET_COLUMN, normalize_distribution_name  # noqa: E402
from src.distributional_model import distribution_nll  # noqa: E402
from src.evaluation import bucket_brier_scores, interval_log_loss  # noqa: E402
from src.features import load_feature_list  # noqa: E402
from src.splits import chronological_train_validation_test_split  # noqa: E402
from src.train_ngboost import (  # noqa: E402
    DEFAULT_FINAL_FEATURE_LIST_PATH,
    DEFAULT_MODELING_TABLE_PATH,
    build_imputed_feature_frames,
    load_modeling_table,
    validate_target_column,
)
from src.distributional_model import predict_distribution_details, train_ngboost_distribution  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_FEATURE_LIST_PATH = REPO_ROOT / "outputs" / "day8_features" / "feature_columns.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "focused_nll_brier_search_20260611"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_best_focused_nll_brier_search.pkl"

SIGMA_FACTORS = [
    0.60,
    0.65,
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
]

HYPERPARAMS = [
    {
        "name": "fast_120_lr005_depth2_leaf50_sub1_direct",
        "n_estimators": 120,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "minibatch_frac": 1.0,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 20,
    },
    {
        "name": "mid_300_lr002_depth3_leaf50_sub08_direct",
        "n_estimators": 300,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "minibatch_frac": 0.8,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 10,
    },
    {
        "name": "prod_500_lr001_depth3_leaf50_sub08_direct",
        "n_estimators": 500,
        "learning_rate": 0.01,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "minibatch_frac": 0.8,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 5,
    },
]

DISTRIBUTIONS = ["normal", "laplace", "skew_normal"]


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_modeling_table(DEFAULT_MODELING_TABLE_PATH)
    validate_target_column(df)
    split = chronological_train_validation_test_split(df)
    for split_name, frame in [
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ]:
        validate_target_column(frame, split_name=split_name)

    final_features = load_feature_list(DEFAULT_FINAL_FEATURE_LIST_PATH)
    safe_features = load_feature_list(SAFE_FEATURE_LIST_PATH)
    feature_sets = build_focused_feature_sets(final_features, safe_features)

    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    completed_keys: set[tuple[str, str, str]] = set()

    stage1_specs = [
        ("current36", distribution, hyperparams)
        for distribution in DISTRIBUTIONS
        for hyperparams in HYPERPARAMS
    ]
    print(f"Stage 1: {len(stage1_specs)} current36 distribution/hyperparameter fits.", flush=True)
    for index, (feature_name, distribution, hyperparams) in enumerate(stage1_specs, start=1):
        result = train_and_score(
            split=split,
            feature_name=feature_name,
            feature_columns=feature_sets[feature_name],
            distribution=distribution,
            hyperparams=hyperparams,
        )
        record_result(result, run_rows, metric_rows, trained_cache)
        completed_keys.add((feature_name, normalize_distribution_name(distribution), hyperparams["name"]))
        write_checkpoint(run_rows, metric_rows)
        print(
            f"[stage1 {index}/{len(stage1_specs)}] {feature_name} / {distribution} / "
            f"{hyperparams['name']}: {result['status']}",
            flush=True,
        )

    stage1_metrics = pd.DataFrame(metric_rows)
    selected_model_specs = select_stage2_specs(stage1_metrics, top_n=3)
    stage2_specs = [
        (feature_name, distribution, hyperparams)
        for distribution, hyperparams_name in selected_model_specs
        for feature_name in feature_sets
        if feature_name != "current36"
        for hyperparams in HYPERPARAMS
        if hyperparams["name"] == hyperparams_name
    ]
    print(f"Stage 2: {len(stage2_specs)} validation-selected feature probes.", flush=True)
    for index, (feature_name, distribution, hyperparams) in enumerate(stage2_specs, start=1):
        key = (feature_name, normalize_distribution_name(distribution), hyperparams["name"])
        if key in completed_keys:
            continue
        result = train_and_score(
            split=split,
            feature_name=feature_name,
            feature_columns=feature_sets[feature_name],
            distribution=distribution,
            hyperparams=hyperparams,
        )
        record_result(result, run_rows, metric_rows, trained_cache)
        completed_keys.add(key)
        write_checkpoint(run_rows, metric_rows)
        print(
            f"[stage2 {index}/{len(stage2_specs)}] {feature_name} / {distribution} / "
            f"{hyperparams['name']}: {result['status']}",
            flush=True,
        )

    metrics = add_validation_selection_columns(pd.DataFrame(metric_rows))
    runs = pd.DataFrame(run_rows)
    metrics.to_csv(OUTPUT_DIR / "focused_search_metrics.csv", index=False)
    runs.to_csv(OUTPUT_DIR / "focused_search_runs.csv", index=False)

    winners = select_winners(metrics)
    write_report(metrics, runs, winners, elapsed_seconds=time.perf_counter() - started)
    save_winner_model(winners["rank_sum"], trained_cache, feature_sets, split.summary)
    print_summary(metrics, winners)


def build_focused_feature_sets(
    final_features: list[str],
    safe_features: list[str],
) -> dict[str, list[str]]:
    return {
        "current36": final_features,
        "plus_forecast_high_day_cos": append_features(
            final_features,
            ["forecast_high", "day_of_year_cos"],
            safe_features,
        ),
        "plus_forecast_high_temp_range": append_features(
            final_features,
            ["forecast_high", "temp_range_so_far"],
            safe_features,
        ),
        "plus_ndfd_metadata": append_features(
            final_features,
            ["ndfd_lead_hours", "ndfd_grid_distance_km"],
            safe_features,
        ),
        "curated_weather_plus": append_features(
            final_features,
            [
                "forecast_high",
                "nws_forecast_high_f",
                "ndfd_lead_hours",
                "ndfd_grid_distance_km",
                "day_of_year_cos",
                "nws_relative_humidity",
                "nws_wind_gust_kt",
                "temp_range_so_far",
                "min_temp_so_far",
                "recent_forecast_revision",
            ],
            safe_features,
        ),
    }


def train_and_score(
    *,
    split: Any,
    feature_name: str,
    feature_columns: list[str],
    distribution: str,
    hyperparams: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    dist = normalize_distribution_name(distribution)
    run_id = stable_run_id(feature_name, dist, hyperparams["name"])
    try:
        X_train, X_validation, X_test, imputer, preprocessing_notes = build_imputed_feature_frames(
            train_df=split.train,
            validation_df=split.validation,
            test_df=split.test,
            feature_columns=feature_columns,
        )
        model = train_ngboost_distribution(
            X_train=X_train,
            y_train=split.train[TARGET_COLUMN].to_numpy(dtype=float),
            X_val=X_validation,
            y_val=split.validation[TARGET_COLUMN].to_numpy(dtype=float),
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
            "validation_frame": split.validation,
            "test_frame": split.test,
            "validation_details": predict_distribution_details(model, X_validation, dist),
            "test_details": predict_distribution_details(model, X_test, dist),
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


def record_result(
    result: dict[str, Any],
    run_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    trained_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    hyperparams = result["hyperparams"]
    run_rows.append(
        {
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
    )
    if result["status"] != "success":
        return
    trained_cache[
        (
            result["feature_set"],
            result["distribution"],
            str(hyperparams["name"]),
        )
    ] = result
    for sigma_factor in SIGMA_FACTORS:
        for split_name in ["validation", "test"]:
            metric_rows.append(
                score_split_fast(
                    result=result,
                    split_name=split_name,
                    sigma_factor=float(sigma_factor),
                )
            )


def score_split_fast(
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
    skew_values = details.get("skew")
    nll = distribution_nll(
        y_true,
        mu=mu,
        sigma=sigma,
        distribution=dist,
        df=df_values,
        skew=skew_values,
    )

    prediction_frame = frame.copy()
    prediction_frame["mu"] = mu
    prediction_frame["sigma"] = sigma
    prediction_frame["distribution_type"] = dist
    if df_values is not None:
        prediction_frame["df"] = np.asarray(df_values, dtype=float)
    if skew_values is not None:
        prediction_frame["skew"] = np.asarray(skew_values, dtype=float)
    probs = error_interval_probabilities(prediction_frame, dist_type=dist)
    labels = assign_error_interval_labels(prediction_frame[TARGET_COLUMN])
    brier = bucket_brier_scores(probs, labels)
    log_loss = interval_log_loss(probs, labels)
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
        "bucket_brier": float(brier["brier_score"].mean()),
        "mean_sigma": float(np.mean(sigma)),
        "median_sigma": float(np.median(sigma)),
    }


def select_stage2_specs(metrics: pd.DataFrame, top_n: int) -> list[tuple[str, str]]:
    validation = add_validation_selection_columns(metrics)
    top = (
        validation[validation["split"] == "validation"]
        .sort_values(["rank_sum", "nll", "bucket_brier"], kind="stable")
        .drop_duplicates(["distribution", "hyperparams_name"], keep="first")
        .head(top_n)
    )
    return [
        (str(row["distribution"]), str(row["hyperparams_name"]))
        for _, row in top.iterrows()
    ]


def add_validation_selection_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    result = metrics.copy()
    validation = result[result["split"] == "validation"].copy()
    validation["nll_rank"] = validation["nll"].rank(method="min", ascending=True)
    validation["brier_rank"] = validation["bucket_brier"].rank(method="min", ascending=True)
    validation["log_loss_rank"] = validation["bucket_log_loss"].rank(method="min", ascending=True)
    validation["rank_sum"] = validation["nll_rank"] + validation["brier_rank"] + validation["log_loss_rank"]
    rank_cols = [
        "run_id",
        "sigma_factor",
        "nll_rank",
        "brier_rank",
        "log_loss_rank",
        "rank_sum",
    ]
    return result.merge(validation[rank_cols], on=["run_id", "sigma_factor"], how="left")


def select_winners(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    validation = metrics[metrics["split"] == "validation"].copy()
    if validation.empty:
        raise ValueError("No validation metrics available")
    return {
        "rank_sum": validation.sort_values(["rank_sum", "nll", "bucket_brier"], kind="stable").iloc[0],
        "nll": validation.sort_values(["nll", "bucket_brier"], kind="stable").iloc[0],
        "brier": validation.sort_values(["bucket_brier", "nll"], kind="stable").iloc[0],
    }


def write_checkpoint(run_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "focused_search_runs_partial.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "focused_search_metrics_partial.csv", index=False)


def write_report(
    metrics: pd.DataFrame,
    runs: pd.DataFrame,
    winners: dict[str, pd.Series],
    *,
    elapsed_seconds: float,
) -> None:
    lines = [
        "# Focused NGBoost NLL/Brier Search",
        "",
        f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Elapsed seconds: {elapsed_seconds:.1f}",
        f"- Successful runs: {int((runs['status'] == 'success').sum())}",
        f"- Failed runs: {int((runs['status'] != 'success').sum())}",
        "- Selection uses validation only; test rows are reported for the selected validation winners.",
        "",
    ]
    for name, row in winners.items():
        selected = metrics[
            (metrics["run_id"] == row["run_id"])
            & (metrics["sigma_factor"] == row["sigma_factor"])
        ].sort_values("split")
        lines.extend([f"## Validation Winner: {name}", "", markdown_table(selected), ""])
    lines.extend(["## Best Validation Rows", ""])
    lines.append(markdown_table(metrics[metrics["split"] == "validation"].sort_values("rank_sum").head(20)))
    (OUTPUT_DIR / "focused_search_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_winner_model(
    selected: pd.Series,
    trained_cache: dict[tuple[str, str, str], dict[str, Any]],
    feature_sets: dict[str, list[str]],
    split_summary: dict[str, Any],
) -> None:
    key = (
        str(selected["feature_set"]),
        str(selected["distribution"]),
        str(selected["hyperparams_name"]),
    )
    result = trained_cache[key]
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": result["model"],
        "imputer": result["imputer"],
        "feature_columns": feature_sets[str(selected["feature_set"])],
        "target": TARGET_COLUMN,
        "model_name": "ngboost_best_focused_nll_brier_search",
        "distribution_type": str(selected["distribution"]),
        "sigma_scale": float(selected["sigma_factor"]),
        "selection_rule": "validation-only rank sum across NLL, bucket Brier, and bucket log loss",
        "selected_validation_row": selected.to_dict(),
        "split_summary": split_summary,
        "preprocessing_notes": result["preprocessing_notes"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set_used_for_selection": False,
    }
    with MODEL_OUTPUT_PATH.open("wb") as handle:
        pickle.dump(artifact, handle)


def print_summary(metrics: pd.DataFrame, winners: dict[str, pd.Series]) -> None:
    print("Done.", flush=True)
    for name, row in winners.items():
        selected = metrics[
            (metrics["run_id"] == row["run_id"])
            & (metrics["sigma_factor"] == row["sigma_factor"])
        ].sort_values("split")
        print(f"Winner {name}:", flush=True)
        for _, item in selected.iterrows():
            print(
                f"  {item['split']}: {item['feature_set']} / {item['distribution']} / "
                f"{item['hyperparams_name']} / sigma x{float(item['sigma_factor']):.2f} "
                f"NLL={float(item['nll']):.6f} Brier={float(item['bucket_brier']):.6f} "
                f"LogLoss={float(item['bucket_log_loss']):.6f}",
                flush=True,
            )
    print(f"Metrics: {OUTPUT_DIR / 'focused_search_metrics.csv'}", flush=True)
    print(f"Report: {OUTPUT_DIR / 'focused_search_report.md'}", flush=True)
    print(f"Selected model: {MODEL_OUTPUT_PATH}", flush=True)


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


if __name__ == "__main__":
    main()
