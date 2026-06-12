from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.search_ngboost_model_space import append_features  # noqa: E402
from scripts.run_focused_nll_brier_search import (  # noqa: E402
    add_validation_selection_columns,
    markdown_table,
    record_result,
    select_winners,
    train_and_score,
)
import scripts.run_focused_nll_brier_search as focused_search  # noqa: E402
from src.distributional_model import TARGET_COLUMN  # noqa: E402
from src.features import load_feature_list  # noqa: E402
from src.splits import chronological_train_validation_test_split  # noqa: E402
from src.train_ngboost import (  # noqa: E402
    DEFAULT_FINAL_FEATURE_LIST_PATH,
    DEFAULT_MODELING_TABLE_PATH,
    load_modeling_table,
    validate_target_column,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_FEATURE_LIST_PATH = REPO_ROOT / "outputs" / "day8_features" / "feature_columns.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "refined_skew_search_20260611"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_best_refined_skew_search.pkl"

SIGMA_FACTORS = [
    0.90,
    0.95,
    1.00,
    1.03,
    1.05,
    1.08,
    1.10,
    1.12,
    1.15,
    1.18,
    1.20,
    1.25,
    1.30,
]

HYPERPARAMS = [
    {
        "name": "control_120_lr005_depth2_leaf50_sub1_direct",
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
        "name": "quick_100_lr006_depth2_leaf50_sub1_direct",
        "n_estimators": 100,
        "learning_rate": 0.06,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "minibatch_frac": 1.0,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 20,
    },
    {
        "name": "regularized_160_lr004_depth2_leaf75_sub1_direct",
        "n_estimators": 160,
        "learning_rate": 0.04,
        "max_depth": 2,
        "min_samples_leaf": 75,
        "minibatch_frac": 1.0,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 20,
    },
    {
        "name": "stump_160_lr004_depth1_leaf50_sub1_direct",
        "n_estimators": 160,
        "learning_rate": 0.04,
        "max_depth": 1,
        "min_samples_leaf": 50,
        "minibatch_frac": 1.0,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 20,
    },
]


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    focused_search.SIGMA_FACTORS = SIGMA_FACTORS

    df = add_refinement_features(load_modeling_table(DEFAULT_MODELING_TABLE_PATH))
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
    feature_sets = build_refined_feature_sets(final_features, safe_features, df.columns)

    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    specs = [
        (feature_name, hyperparams)
        for feature_name in feature_sets
        for hyperparams in HYPERPARAMS
    ]

    print(f"Refined skew-normal search: {len(specs)} fits.", flush=True)
    for index, (feature_name, hyperparams) in enumerate(specs, start=1):
        result = train_and_score(
            split=split,
            feature_name=feature_name,
            feature_columns=feature_sets[feature_name],
            distribution="skew_normal",
            hyperparams=hyperparams,
        )
        record_result(result, run_rows, metric_rows, trained_cache)
        write_checkpoint(run_rows, metric_rows)
        print(
            f"[{index}/{len(specs)}] {feature_name} / {hyperparams['name']}: "
            f"{result['status']} ({result['elapsed_seconds']:.1f}s)",
            flush=True,
        )

    metrics = add_validation_selection_columns(pd.DataFrame(metric_rows))
    runs = pd.DataFrame(run_rows)
    metrics.to_csv(OUTPUT_DIR / "refined_skew_metrics.csv", index=False)
    runs.to_csv(OUTPUT_DIR / "refined_skew_runs.csv", index=False)

    winners = select_winners(metrics)
    save_winner_model(winners["rank_sum"], trained_cache, feature_sets, split.summary)
    write_report(metrics, runs, winners, elapsed_seconds=time.perf_counter() - started)
    print_summary(metrics, winners)


def add_refinement_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if {"ndfd_lead_hours", "ndfd_grid_distance_km"}.issubset(result.columns):
        result["ndfd_lead_x_grid_distance"] = (
            result["ndfd_lead_hours"] * result["ndfd_grid_distance_km"]
        )
    if {"forecast_high", "day_of_year_cos"}.issubset(result.columns):
        result["forecast_high_x_day_cos"] = result["forecast_high"] * result["day_of_year_cos"]
    if {"temp_range_so_far", "forecast_horizon_hours"}.issubset(result.columns):
        result["temp_range_x_forecast_horizon"] = (
            result["temp_range_so_far"] * result["forecast_horizon_hours"]
        )
    if {"temp_range_so_far", "ndfd_lead_hours"}.issubset(result.columns):
        result["temp_range_x_ndfd_lead"] = result["temp_range_so_far"] * result["ndfd_lead_hours"]
    if {"max_so_far_minus_forecast_high", "forecast_horizon_hours"}.issubset(result.columns):
        result["max_error_x_forecast_horizon"] = (
            result["max_so_far_minus_forecast_high"] * result["forecast_horizon_hours"]
        )
    return result


def build_refined_feature_sets(
    final_features: list[str],
    safe_features: list[str],
    available_columns: pd.Index,
) -> dict[str, list[str]]:
    available = set(str(column) for column in available_columns)

    def extend(extra_columns: list[str]) -> list[str]:
        allowed = list(safe_features) + [
            "ndfd_lead_x_grid_distance",
            "forecast_high_x_day_cos",
            "temp_range_x_forecast_horizon",
            "temp_range_x_ndfd_lead",
            "max_error_x_forecast_horizon",
        ]
        columns = append_features(final_features, extra_columns, allowed)
        return [column for column in columns if column in available]

    return {
        "plus_ndfd_metadata": extend(["ndfd_lead_hours", "ndfd_grid_distance_km"]),
        "plus_day_cos_forecast_high": extend(["forecast_high", "day_of_year_cos"]),
        "plus_ndfd_day_cos_forecast_high": extend(
            ["ndfd_lead_hours", "ndfd_grid_distance_km", "forecast_high", "day_of_year_cos"]
        ),
        "plus_ndfd_temp_range": extend(
            ["ndfd_lead_hours", "ndfd_grid_distance_km", "temp_range_so_far"]
        ),
        "plus_ndfd_temp_day": extend(
            [
                "ndfd_lead_hours",
                "ndfd_grid_distance_km",
                "temp_range_so_far",
                "forecast_high",
                "day_of_year_cos",
            ]
        ),
        "plus_ndfd_interactions": extend(
            [
                "ndfd_lead_hours",
                "ndfd_grid_distance_km",
                "ndfd_lead_x_grid_distance",
                "forecast_high_x_day_cos",
            ]
        ),
        "plus_compact_all": extend(
            [
                "forecast_high",
                "day_of_year_cos",
                "ndfd_lead_hours",
                "ndfd_grid_distance_km",
                "temp_range_so_far",
                "recent_forecast_revision",
                "ndfd_lead_x_grid_distance",
                "forecast_high_x_day_cos",
                "temp_range_x_forecast_horizon",
                "temp_range_x_ndfd_lead",
                "max_error_x_forecast_horizon",
            ]
        ),
    }


def write_checkpoint(run_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "refined_skew_runs_partial.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "refined_skew_metrics_partial.csv", index=False)


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
        "model_name": "ngboost_best_refined_skew_search",
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


def write_report(
    metrics: pd.DataFrame,
    runs: pd.DataFrame,
    winners: dict[str, pd.Series],
    *,
    elapsed_seconds: float,
) -> None:
    lines = [
        "# Refined Skew-Normal NLL/Brier Search",
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
    lines.append(markdown_table(metrics[metrics["split"] == "validation"].sort_values("rank_sum").head(25)))
    (OUTPUT_DIR / "refined_skew_report.md").write_text("\n".join(lines), encoding="utf-8")


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
                f"  {item['split']}: {item['feature_set']} / {item['hyperparams_name']} / "
                f"sigma x{float(item['sigma_factor']):.2f} "
                f"NLL={float(item['nll']):.6f} Brier={float(item['bucket_brier']):.6f} "
                f"LogLoss={float(item['bucket_log_loss']):.6f}",
                flush=True,
            )
    print(f"Metrics: {OUTPUT_DIR / 'refined_skew_metrics.csv'}", flush=True)
    print(f"Report: {OUTPUT_DIR / 'refined_skew_report.md'}", flush=True)
    print(f"Selected model: {MODEL_OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
