from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
    normalize_distribution_name,
    predict_distribution_details,
    train_ngboost_distribution,
)
from src.evaluation import bucket_brier_scores  # noqa: E402
from src.features import (  # noqa: E402
    get_all_model_features,
    get_feature_groups,
    get_features_without_group,
    get_minimal_feature_set,
    save_feature_list,
    validate_feature_columns_exist,
)
from src.train_ngboost import (  # noqa: E402
    BEST_METADATA_OUTPUT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODELING_TABLE_PATH,
    _configured_chronological_split,
    _evaluate_validation_predictions,
    _flatten_split_summary,
    _json_safe,
    build_imputed_feature_frames,
    load_model_config,
    load_modeling_table,
    validate_target_column,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_feature_ablation.csv"
BRIER_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_feature_ablation_brier_by_bucket.csv"
RUN_FEATURE_LIST_DIR = REPO_ROOT / "outputs" / "day17_feature_lists"
METADATA_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_feature_ablation_metadata.json"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_path = resolve_path(args.dataset_path)
    config_path = resolve_path(args.config_path)
    metadata_path = resolve_path(args.best_metadata_path)

    config = load_model_config(config_path)
    best_metadata = load_best_metadata(metadata_path)
    prepared = prepare_data(dataset_path, config)
    feature_groups = get_feature_groups()

    reference_features = get_all_model_features()
    metadata_features = list(best_metadata.get("feature_columns", []))
    if metadata_features and metadata_features != reference_features:
        raise ValueError(
            "Day 17 full feature reference does not match Day 16 metadata feature order."
        )

    experiments = build_experiments(feature_groups)
    rows: list[dict[str, Any]] = []
    brier_rows: list[pd.DataFrame] = []

    for index, experiment in enumerate(experiments, start=1):
        print(
            f"[{index}/{len(experiments)}] {experiment['run_name']} "
            f"({len(experiment['features'])} features)",
            flush=True,
        )
        row, brier = run_experiment(
            experiment=experiment,
            prepared=prepared,
            best_metadata=best_metadata,
        )
        rows.append(row)
        if brier is not None:
            brier_rows.append(brier)

    output = pd.DataFrame(rows)
    ABLATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(ABLATION_OUTPUT_PATH, index=False)
    if brier_rows:
        pd.concat(brier_rows, ignore_index=True).to_csv(BRIER_OUTPUT_PATH, index=False)
    write_metadata(
        prepared=prepared,
        best_metadata=best_metadata,
        experiments=experiments,
        output=output,
    )
    print(f"Saved ablation results: {ABLATION_OUTPUT_PATH}", flush=True)
    if brier_rows:
        print(f"Saved per-bucket Brier scores: {BRIER_OUTPUT_PATH}", flush=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Day 17 NGBoost feature robustness and ablation experiments."
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_MODELING_TABLE_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--best-metadata-path", default=str(BEST_METADATA_OUTPUT_PATH))
    return parser.parse_args(argv)


def load_best_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Day 16 best metadata JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["distribution", "hyperparameters", "split_dates", "feature_columns"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Day 16 best metadata is missing required keys: {missing}")
    return payload


def prepare_data(dataset_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    df = load_modeling_table(dataset_path)
    validate_target_column(df)
    split_result = _configured_chronological_split(
        df,
        config,
        SimpleNamespace(train_end_date=None, validation_end_date=None),
    )
    for split_name, split_df in [
        ("train", split_result.train),
        ("validation", split_result.validation),
        ("test", split_result.test),
    ]:
        validate_target_column(split_df, split_name=split_name)
    split_flat = _flatten_split_summary(split_result.summary)
    return {
        "df": df,
        "train_df": split_result.train,
        "validation_df": split_result.validation,
        "split_summary": split_result.summary,
        "split_flat": split_flat,
        "dataset_path": dataset_path,
        "config": config,
    }


def build_experiments(feature_groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = [
        {
            "run_name": "full_feature_set",
            "removed_group": "",
            "features": get_all_model_features(),
            "notes": "Reference Day 16 full feature set.",
        }
    ]
    for group_name in feature_groups:
        experiments.append(
            {
                "run_name": f"remove_{group_name}",
                "removed_group": group_name,
                "features": get_features_without_group(group_name),
                "notes": f"Removed feature group: {group_name}.",
            }
        )
    experiments.append(
        {
            "run_name": "minimal_feature_set",
            "removed_group": "minimal_feature_set",
            "features": get_minimal_feature_set(),
            "notes": (
                "Minimal robustness check. forecast_high_minus_current_temp is not an existing "
                "column, so forecast_high and current_temp are included separately; "
                "max_so_far_minus_forecast_high is the available sign-inverted equivalent "
                "of forecast_high_minus_max_so_far."
            ),
        }
    )
    return experiments


def run_experiment(
    experiment: dict[str, Any],
    prepared: dict[str, Any],
    best_metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    run_name = str(experiment["run_name"])
    features = list(experiment["features"])
    base_row = {
        "run_name": run_name,
        "removed_group": experiment["removed_group"],
        "num_features": len(features),
        "validation_nll": math.nan,
        "bucket_interval_log_loss": math.nan,
        "mean_brier_score": math.nan,
        "multiclass_bucket_brier": math.nan,
        "brier_by_bucket": "",
        "coverage_50": math.nan,
        "coverage_80": math.nan,
        "coverage_90": math.nan,
        "coverage_error_80": math.nan,
        "coverage_error_90": math.nan,
        "avg_predicted_sigma": math.nan,
        "feature_list": json.dumps(features),
        "feature_list_path": str((RUN_FEATURE_LIST_DIR / f"{run_name}.json").relative_to(REPO_ROOT)),
        "notes": experiment["notes"],
        "status": "started",
    }

    try:
        validate_feature_columns_exist(prepared["df"], features)
        save_feature_list(
            RUN_FEATURE_LIST_DIR / f"{run_name}.json",
            {
                "run_name": run_name,
                "target": TARGET_COLUMN,
                "features": features,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection_data": "chronological_validation_only",
            },
        )

        X_train, X_validation, _unused, _imputer, _notes = build_imputed_feature_frames(
            train_df=prepared["train_df"],
            validation_df=prepared["validation_df"],
            test_df=prepared["validation_df"],
            feature_columns=features,
        )
        y_train = prepared["train_df"][TARGET_COLUMN].to_numpy(dtype=float)
        y_validation = prepared["validation_df"][TARGET_COLUMN].to_numpy(dtype=float)

        distribution = normalize_distribution_name(str(best_metadata["distribution"]))
        params = dict(best_metadata["hyperparameters"])
        model = train_ngboost_distribution(
            X_train=X_train,
            y_train=y_train,
            distribution=distribution,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            minibatch_frac=float(params["minibatch_frac"]),
            natural_gradient=bool(params["natural_gradient"]),
            random_state=int(params["random_state"]),
            early_stopping_rounds=None,
        )
        train_details = predict_distribution_details(model, X_train, distribution)
        validation_details = predict_distribution_details(model, X_validation, distribution)
        metrics, diagnostics = _evaluate_validation_predictions(
            train_df=prepared["train_df"],
            validation_df=prepared["validation_df"],
            y_train=y_train,
            y_validation=y_validation,
            train_details=train_details,
            validation_details=validation_details,
            distribution=distribution,
        )

        brier = bucket_brier_scores(
            diagnostics["bucket_probs"],
            diagnostics["realized_labels"],
        )
        brier.insert(0, "run_name", run_name)
        brier.insert(1, "removed_group", experiment["removed_group"])
        brier_by_bucket = {
            str(row["bucket"]): float(row["brier_score"])
            for _, row in brier.iterrows()
        }
        sigma = np.asarray(validation_details["sigma"], dtype=float)
        base_row.update(
            {
                "validation_nll": metrics["val_nll"],
                "bucket_interval_log_loss": metrics["val_interval_log_loss"],
                "mean_brier_score": float(brier["brier_score"].mean()),
                "multiclass_bucket_brier": metrics["val_bucket_brier"],
                "brier_by_bucket": json.dumps(brier_by_bucket, sort_keys=True),
                "coverage_50": metrics["val_50_coverage"],
                "coverage_80": metrics["val_80_coverage"],
                "coverage_90": metrics["val_90_coverage"],
                "coverage_error_80": metrics["val_80_coverage_error"],
                "coverage_error_90": metrics["val_90_coverage_error"],
                "avg_predicted_sigma": float(np.mean(sigma)),
                "status": "success",
            }
        )
        return base_row, brier
    except Exception as exc:
        base_row["status"] = "failed"
        base_row["notes"] = f"{experiment['notes']} Error: {type(exc).__name__}: {exc}"
        return base_row, None


def write_metadata(
    prepared: dict[str, Any],
    best_metadata: dict[str, Any],
    experiments: list[dict[str, Any]],
    output: pd.DataFrame,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(prepared["dataset_path"]).relative_to(REPO_ROOT)),
        "selection_data": "chronological_validation_only",
        "test_set_used_for_feature_selection": False,
        "split_summary": prepared["split_summary"],
        "day16_selected_run_id": best_metadata.get("selected_run_id"),
        "day16_selected_candidate_name": best_metadata.get("selected_candidate_name"),
        "day16_hyperparameters": best_metadata.get("hyperparameters"),
        "run_count": len(experiments),
        "successful_run_count": int((output["status"] == "success").sum()),
        "ablation_output_path": str(ABLATION_OUTPUT_PATH.relative_to(REPO_ROOT)),
        "brier_output_path": str(BRIER_OUTPUT_PATH.relative_to(REPO_ROOT)),
    }
    METADATA_OUTPUT_PATH.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


if __name__ == "__main__":
    main(sys.argv[1:])
