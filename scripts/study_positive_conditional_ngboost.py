from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conditional_increase_model import (
    conditional_cdf,
    evaluate_conditional_distribution,
    positive_increase_rows,
    predict_conditional_distribution,
    predict_conditional_mean,
    train_conditional_candidate,
)
from src.hurdle_comparison import time_of_day_bucket
from src.hurdle_model import expanding_window_splits, materialize_fold


DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
EXCEEDANCE_BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
HURDLE_FEATURES_PATH = REPO_ROOT / "models" / "hurdle_features.json"
MODEL_PATH = REPO_ROOT / "models" / "remaining_increase_ngboost.pkl"
FEATURES_PATH = REPO_ROOT / "models" / "remaining_increase_features.json"
METADATA_PATH = REPO_ROOT / "models" / "remaining_increase_metadata.json"
BUNDLE_PATH = REPO_ROOT / "models" / "remaining_increase_bundle.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "remaining_increase" / "candidate_study"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_exceedance_bundle() -> dict:
    bundle = json.loads(EXCEEDANCE_BUNDLE_PATH.read_text(encoding="utf-8"))
    if bundle.get("status") not in {"frozen_validated", "frozen_validated_user_override"}:
        raise ValueError("Exceedance model must be frozen before the conditional study")
    for label, filename in bundle["paths"].items():
        path = EXCEEDANCE_BUNDLE_PATH.parent / filename
        if _sha256(path) != bundle["sha256"][label]:
            raise ValueError(f"Frozen exceedance {label} hash mismatch")
    return bundle


def _empirical_outputs(positive: pd.DataFrame) -> dict[str, object]:
    target = positive["remaining_increase"].to_numpy(dtype=float)
    integer_share = float(np.mean(np.isclose(target, np.round(target))))
    counts = positive["remaining_increase"].value_counts().sort_index().rename_axis("delta").reset_index(name="count")
    counts["share"] = counts["count"] / len(positive)
    counts.to_csv(OUTPUT_DIR / "empirical_delta_counts.csv", index=False)

    by_time = positive.assign(time_bucket=time_of_day_bucket(positive["prediction_time"]))
    summary = (
        by_time.groupby("time_bucket", observed=False)["remaining_increase"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    near_zero = by_time["remaining_increase"].le(1).groupby(by_time["time_bucket"], observed=False).mean()
    right_tail = by_time["remaining_increase"].gt(10).groupby(by_time["time_bucket"], observed=False).mean()
    summary["share_delta_eq_1"] = summary["time_bucket"].map(near_zero)
    summary["share_delta_gt_10"] = summary["time_bucket"].map(right_tail)
    summary.to_csv(OUTPUT_DIR / "empirical_delta_by_time.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0.5, float(target.max()) + 1.5, 1.0)
    ax.hist(target, bins=bins, edgecolor="white")
    ax.set(xlabel="Positive remaining increase (deg F)", ylabel="Rows", title="Conditional target distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "empirical_delta_histogram.png", dpi=150)
    plt.close(fig)
    return {
        "rows": int(len(target)),
        "integer_share": integer_share,
        "share_delta_eq_1": float(np.mean(target == 1)),
        "share_delta_le_3": float(np.mean(target <= 3)),
        "share_delta_gt_10": float(np.mean(target > 10)),
        "maximum": float(target.max()),
    }


def _prediction_frame(
    artifact: dict, frame: pd.DataFrame, model_name: str, fold: str
) -> pd.DataFrame:
    target = frame["remaining_increase"].to_numpy(dtype=float)
    distribution = predict_conditional_distribution(artifact, frame)
    upper = conditional_cdf(artifact, frame, target)
    lower = conditional_cdf(artifact, frame, target - 1.0)
    realized_probability = np.clip(upper - lower, 1e-15, 1.0)
    if artifact["type"] == "shifted_poisson_ngboost":
        realized_probability = np.clip(
            distribution.dist.pmf((target - artifact["target_shift"]).astype(int)), 1e-15, 1.0
        )
    result = frame[
        ["_row_id", "target_date", "prediction_time", "remaining_increase"]
    ].copy()
    result["model"] = model_name
    result["fold"] = fold
    result["predicted_mean"] = predict_conditional_mean(artifact, frame)
    result["cdf_lower"] = lower
    result["cdf_upper"] = upper
    result["pit_midpoint"] = (lower + upper) / 2.0
    result["realized_interval_probability"] = realized_probability
    result["time_bucket"] = time_of_day_bucket(result["prediction_time"]).astype(str)
    result["size_bucket"] = np.where(target <= 3, "small_1_to_3", "large_gt_3")
    return result


def _pit_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, fold), group in predictions.groupby(["model", "fold"], sort=False):
        pit = group["pit_midpoint"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "fold": fold,
                "n": len(group),
                "pit_mean": float(pit.mean()),
                "pit_variance": float(pit.var()),
                "pit_ks_uniform": float(np.max(np.abs(np.sort(pit) - (np.arange(len(pit)) + 0.5) / len(pit)))),
            }
        )
    return pd.DataFrame(rows)


def _probability_diagnostics(
    artifact: dict,
    frame: pd.DataFrame,
    model_name: str,
    fold: str,
    max_delta: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    target = frame["remaining_increase"].to_numpy(dtype=float)
    cdf_rows: list[dict] = []
    interval_rows: list[dict] = []
    brier_values: list[float] = []
    for threshold in range(1, max_delta + 1):
        predicted_cdf = conditional_cdf(artifact, frame, float(threshold))
        actual_cdf = (target <= threshold).astype(float)
        cdf_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "threshold": threshold,
                "n": len(frame),
                "mean_predicted_cdf": float(predicted_cdf.mean()),
                "empirical_cdf": float(actual_cdf.mean()),
                "calibration_gap": float(predicted_cdf.mean() - actual_cdf.mean()),
                "cdf_brier": float(np.mean((predicted_cdf - actual_cdf) ** 2)),
            }
        )
        upper = predicted_cdf
        lower = conditional_cdf(artifact, frame, float(threshold - 1))
        probability = np.clip(upper - lower, 0.0, 1.0)
        actual = (target == threshold).astype(float)
        brier = float(np.mean((probability - actual) ** 2))
        brier_values.append(brier)
        interval_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "interval": f"{threshold - 1}<delta<={threshold}",
                "n": len(frame),
                "mean_probability": float(probability.mean()),
                "empirical_frequency": float(actual.mean()),
                "calibration_gap": float(probability.mean() - actual.mean()),
                "brier": brier,
            }
        )
    tail_probability = 1.0 - conditional_cdf(artifact, frame, float(max_delta))
    tail_actual = (target > max_delta).astype(float)
    tail_brier = float(np.mean((tail_probability - tail_actual) ** 2))
    brier_values.append(tail_brier)
    interval_rows.append(
        {
            "model": model_name,
            "fold": fold,
            "interval": f"delta>{max_delta}",
            "n": len(frame),
            "mean_probability": float(tail_probability.mean()),
            "empirical_frequency": float(tail_actual.mean()),
            "calibration_gap": float(tail_probability.mean() - tail_actual.mean()),
            "brier": tail_brier,
        }
    )
    return pd.DataFrame(cdf_rows), pd.DataFrame(interval_rows), float(np.mean(brier_values))


def _group_metrics(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, group_name), group in predictions.groupby(["model", group_col], observed=False):
        target = group["remaining_increase"].to_numpy(dtype=float)
        mean = group["predicted_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                group_col: group_name,
                "n": len(group),
                "interval_nll": float(-np.mean(np.log(group["realized_interval_probability"]))),
                "pit_mean": float(group["pit_midpoint"].mean()),
                "mae_secondary": float(np.mean(np.abs(mean - target))),
            }
        )
    return pd.DataFrame(rows)


def _select_winner(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    selected = metrics.copy()
    selected["coverage_error"] = (
        (selected["coverage_80"] - 0.80).abs() + (selected["coverage_90"] - 0.90).abs()
    )
    proper_metrics = [
        "interval_nll",
        "mean_bucket_brier",
        "cdf_calibration_error",
        "crps",
        "coverage_error",
    ]
    for metric in proper_metrics:
        selected[f"rank_{metric}"] = selected[metric].rank(method="min")
    selected["probability_rank_sum"] = selected[[f"rank_{x}" for x in proper_metrics]].sum(axis=1)
    selected = selected.sort_values(
        ["probability_rank_sum", "interval_nll", "crps", "model"]
    ).reset_index(drop=True)
    return str(selected.iloc[0]["model"]), selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Positive-conditional NGBoost candidate study")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["weibull", "halfnormal", "lognormal", "exponential", "shifted_poisson"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_dir = OUTPUT_DIR / "candidates"
    candidate_dir.mkdir(exist_ok=True)
    exceedance_bundle = _verify_exceedance_bundle()
    features = list(json.loads(HURDLE_FEATURES_PATH.read_text(encoding="utf-8"))["features"])
    dataset = pd.read_csv(DATASET_PATH)
    dataset["target_date"] = pd.to_datetime(dataset["target_date"], errors="raise")
    dataset["prediction_time"] = pd.to_datetime(dataset["prediction_time"], errors="raise")
    dataset["_row_id"] = np.arange(len(dataset))
    positive = positive_increase_rows(dataset)
    empirical = _empirical_outputs(positive)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validation_start = str(config["splits"]["validation_start"])
    test_start = str(config["splits"]["test_start"])
    date = positive["target_date"].dt.normalize()
    pretest = positive.loc[date < pd.Timestamp(test_start)].copy()
    test = positive.loc[date >= pd.Timestamp(test_start)].copy()
    folds = expanding_window_splits(pretest, test_start=test_start, minimum_training_years=2)
    if not folds or folds[-1]["val_start"] != validation_start:
        raise ValueError("Conditional folds do not reproduce the configured validation period")

    fold_metrics: list[dict] = []
    validation_predictions: list[pd.DataFrame] = []
    cdf_tables: list[pd.DataFrame] = []
    interval_tables: list[pd.DataFrame] = []
    for candidate in args.candidates:
        for fold in folds:
            train, validation = materialize_fold(pretest, fold)
            print(f"Training {candidate} {fold['name']}: {len(train):,} -> {len(validation):,}")
            artifact = train_conditional_candidate(
                train,
                features,
                candidate,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
            )
            metrics = evaluate_conditional_distribution(artifact, validation)
            cdf_table, interval_table, mean_bucket_brier = _probability_diagnostics(
                artifact, validation, candidate, fold["name"]
            )
            metrics["mean_bucket_brier"] = mean_bucket_brier
            metrics["cdf_calibration_error"] = float(cdf_table["calibration_gap"].abs().mean())
            fold_metrics.append({"model": candidate, "fold": fold["name"], **metrics})
            validation_predictions.append(_prediction_frame(artifact, validation, candidate, fold["name"]))
            cdf_tables.append(cdf_table)
            interval_tables.append(interval_table)

    fold_table = pd.DataFrame(fold_metrics)
    fold_table.to_csv(OUTPUT_DIR / "validation_fold_metrics.csv", index=False)
    predictions = pd.concat(validation_predictions, ignore_index=True)
    predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    pit = _pit_diagnostics(predictions)
    pit.to_csv(OUTPUT_DIR / "validation_pit_diagnostics.csv", index=False)
    pd.concat(cdf_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_cdf_calibration.csv", index=False
    )
    pd.concat(interval_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_interval_calibration.csv", index=False
    )
    _group_metrics(predictions, "time_bucket").to_csv(
        OUTPUT_DIR / "validation_metrics_by_time.csv", index=False
    )
    _group_metrics(predictions, "size_bucket").to_csv(
        OUTPUT_DIR / "validation_metrics_by_size.csv", index=False
    )

    weighted = []
    for model, group in fold_table.groupby("model"):
        weights = group["n"].to_numpy(dtype=float)
        row = {"model": model, "n": int(weights.sum())}
        for metric in [
            "density_nll", "interval_nll", "crps", "mean_bucket_brier", "cdf_calibration_error", "mae", "rmse", "coverage_80", "coverage_90"
        ]:
            row[metric] = float(np.average(group[metric], weights=weights))
        weighted.append(row)
    winner, comparison = _select_winner(pd.DataFrame(weighted))
    comparison.to_csv(OUTPUT_DIR / "validation_candidate_comparison.csv", index=False)

    test_rows: list[dict] = []
    test_predictions: list[pd.DataFrame] = []
    test_cdf_tables: list[pd.DataFrame] = []
    test_interval_tables: list[pd.DataFrame] = []
    final_artifacts: dict[str, dict] = {}
    for candidate in args.candidates:
        print(f"Refitting {candidate} on all pre-test positive rows")
        artifact = train_conditional_candidate(
            pretest,
            features,
            candidate,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
        )
        final_artifacts[candidate] = artifact
        with (candidate_dir / f"{candidate}.pkl").open("wb") as handle:
            pickle.dump(artifact, handle)
        test_metrics = evaluate_conditional_distribution(artifact, test)
        cdf_table, interval_table, mean_bucket_brier = _probability_diagnostics(
            artifact, test, candidate, "test"
        )
        test_metrics["mean_bucket_brier"] = mean_bucket_brier
        test_metrics["cdf_calibration_error"] = float(cdf_table["calibration_gap"].abs().mean())
        test_rows.append({"model": candidate, **test_metrics})
        test_predictions.append(_prediction_frame(artifact, test, candidate, "test"))
        test_cdf_tables.append(cdf_table)
        test_interval_tables.append(interval_table)
    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "test_candidate_diagnostics.csv", index=False)
    combined_test_predictions = pd.concat(test_predictions, ignore_index=True)
    combined_test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    _group_metrics(combined_test_predictions, "time_bucket").to_csv(
        OUTPUT_DIR / "test_metrics_by_time.csv", index=False
    )
    _group_metrics(combined_test_predictions, "size_bucket").to_csv(
        OUTPUT_DIR / "test_metrics_by_size.csv", index=False
    )
    pd.concat(test_cdf_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "test_cdf_calibration.csv", index=False
    )
    pd.concat(test_interval_tables, ignore_index=True).to_csv(
        OUTPUT_DIR / "test_interval_calibration.csv", index=False
    )

    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_artifacts[winner], handle)
    FEATURES_PATH.write_text(
        json.dumps({"features": features, "feature_count": len(features)}, indent=2), encoding="utf-8"
    )
    metadata = {
        "status": "frozen_after_validation_candidate_study",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "winner": winner,
        "selection_data": "expanding-window validation only",
        "selection_metrics": [
            "integer_interval_nll",
            "mean_bucket_brier",
            "cdf_calibration_error",
            "crps",
            "interval_coverage_error",
        ],
        "rmse_used_for_selection": False,
        "candidate_order_not_a_preference": list(args.candidates),
        "target": "remaining_increase = final_daily_high - current_max_so_far",
        "training_filter": "remaining_increase > 0",
        "integer_probability_definition": "P(k-1 < delta <= k | delta > 0)",
        "empirical_distribution": empirical,
        "features": features,
        "splits": {"validation_start": validation_start, "test_start": test_start, "folds": folds},
        "validation_comparison": comparison.to_dict(orient="records"),
        "test_diagnostics_not_used_for_selection": pd.DataFrame(test_rows).to_dict(orient="records"),
        "exceedance_winner": exceedance_bundle["winner"],
        "exceedance_bundle_sha256": _sha256(EXCEEDANCE_BUNDLE_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    bundle = {
        "status": "frozen_validated",
        "winner": winner,
        "created_at_utc": metadata["generated_at_utc"],
        "paths": {
            "model": MODEL_PATH.name,
            "features": FEATURES_PATH.name,
            "metadata": METADATA_PATH.name,
        },
        "sha256": {
            "model": _sha256(MODEL_PATH),
            "features": _sha256(FEATURES_PATH),
            "metadata": _sha256(METADATA_PATH),
            "dataset": _sha256(DATASET_PATH),
        },
        "depends_on_exceedance_bundle_sha256": _sha256(EXCEEDANCE_BUNDLE_PATH),
    }
    BUNDLE_PATH.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    report = [
        "# Positive-Conditional NGBoost Candidate Study",
        "",
        f"Winner selected without test access: **{winner}**.",
        f"Positive rows: {empirical['rows']:,}; integer-valued share: {empirical['integer_share']:.3%}.",
        f"+1 deg F share: {empirical['share_delta_eq_1']:.3%}; >10 deg F share: {empirical['share_delta_gt_10']:.3%}.",
        "",
        "Selection uses interval NLL, bucket Brier, CDF calibration, CRPS, and coverage error. RMSE is diagnostic only.",
        "Continuous distributions are discretized into one-degree settlement intervals before trading scores.",
        "",
        comparison.to_markdown(index=False),
    ]
    (OUTPUT_DIR / "study_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Frozen conditional winner: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
