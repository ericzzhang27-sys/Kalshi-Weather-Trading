from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hurdle_calibration import apply_calibrator, fit_calibrator
from src.hurdle_dataset import HurdleDatasetConfig, build_hurdle_dataset, validate_hurdle_dataset
from src.hurdle_evaluation import (
    calibration_by_time_of_day,
    calibration_error_metrics,
    forecast_gap_testing,
    reliability_bins,
    tail_failure_report,
    test_by_temperature_state,
)
from src.hurdle_features import add_hurdle_core_features, progressive_feature_sets, select_hurdle_features
from src.hurdle_model import (
    HurdlePredictor,
    climatological_baseline_fit,
    climatological_predict,
    evaluate_classifier,
    expanding_window_splits,
    materialize_fold,
    predict_proba,
    train_boosted_classifier,
    train_logistic_regression,
)


OUTPUT_DIR = REPO_ROOT / "outputs" / "hurdle"
MODELS_DIR = REPO_ROOT / "models"
DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the hurdle probability model")
    parser.add_argument("--reuse-dataset", action="store_true", help="reuse the cached five-minute dataset")
    parser.add_argument("--skip-ngboost", action="store_true", help="skip the optional Bernoulli NGBoost comparison")
    parser.add_argument(
        "--confirm-tail-review",
        action="store_true",
        help="record that the generated <5% false-confidence cases were manually reviewed",
    )
    return parser.parse_args()


def _split_config() -> tuple[str, str]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    splits = payload.get("splits", {})
    return splits.get("validation_start", "2024-01-01"), splits.get("test_start", "2025-01-01")


def _available_features(df: pd.DataFrame, level: str) -> list[str]:
    columns = select_hurdle_features(df, level, available_only=True)
    return [column for column in columns if df[column].notna().mean() >= 0.20]


def _fit_model(kind: str, train: pd.DataFrame, features: list[str]) -> Any:
    if kind == "logistic":
        return train_logistic_regression(train, features)
    return train_boosted_classifier(train, features, kind=kind)


def _cross_validate(
    df: pd.DataFrame,
    folds: list[dict[str, Any]],
    *,
    level: str,
    kind: str,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    features = _available_features(df, level)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in folds:
        train, validation = materialize_fold(df, fold)
        model = _fit_model(kind, train, features)
        probability = predict_proba(model, validation, features)
        metrics = evaluate_classifier(validation["will_increase"].to_numpy(), probability)
        fold_metrics.append({"fold": fold["name"], **metrics})
        predictions.append(
            pd.DataFrame(
                {
                    "row_id": validation["_row_id"].to_numpy(),
                    "validation_year": pd.to_datetime(validation["target_date"]).dt.year.to_numpy(),
                    "actual": validation["will_increase"].to_numpy(dtype=int),
                    "probability": probability,
                }
            )
        )
    oof = pd.concat(predictions, ignore_index=True)
    aggregate = evaluate_classifier(oof["actual"].to_numpy(), oof["probability"].to_numpy())
    return {
        "level": level,
        "model": kind,
        "n_features": len(features),
        **{f"cv_{key}": value for key, value in aggregate.items()},
        "fold_metrics": fold_metrics,
    }, oof, features


def _cross_validate_baseline(
    df: pd.DataFrame, folds: list[dict[str, Any]]
) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in folds:
        train, validation = materialize_fold(df, fold)
        model = climatological_baseline_fit(train)
        probability = climatological_predict(model, validation)
        metrics = evaluate_classifier(validation["will_increase"].to_numpy(), probability)
        fold_metrics.append({"fold": fold["name"], **metrics})
        predictions.append(
            pd.DataFrame(
                {
                    "row_id": validation["_row_id"].to_numpy(),
                    "validation_year": pd.to_datetime(validation["target_date"]).dt.year.to_numpy(),
                    "actual": validation["will_increase"].to_numpy(dtype=int),
                    "probability": probability,
                }
            )
        )
    oof = pd.concat(predictions, ignore_index=True)
    metrics = evaluate_classifier(oof["actual"].to_numpy(), oof["probability"].to_numpy())
    return {
        "level": "A_time_only",
        "model": "climatology",
        "n_features": 1,
        **{f"cv_{key}": value for key, value in metrics.items()},
        "fold_metrics": fold_metrics,
    }, oof


def _select_calibration(oof: pd.DataFrame) -> tuple[str, Any, pd.DataFrame]:
    """Choose calibration from forward-only validation predictions, never test."""
    years = sorted(oof["validation_year"].unique())
    evaluation_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for method in ("raw", "platt", "isotonic"):
        method_frames: list[pd.DataFrame] = []
        for year in years[1:]:
            fit_rows = oof.loc[oof["validation_year"] < year]
            score_rows = oof.loc[oof["validation_year"] == year].copy()
            calibrator = fit_calibrator(method, fit_rows["probability"].to_numpy(), fit_rows["actual"].to_numpy())
            score_rows["calibrated_probability"] = apply_calibrator(
                method, calibrator, score_rows["probability"].to_numpy()
            )
            method_frames.append(score_rows)
        combined = pd.concat(method_frames, ignore_index=True)
        metrics = evaluate_classifier(combined["actual"].to_numpy(), combined["calibrated_probability"].to_numpy())
        rows.append({"method": method, **metrics})
        combined["method"] = method
        evaluation_frames.append(combined)
    comparison = pd.DataFrame(rows).sort_values(["brier", "log_loss"]).reset_index(drop=True)
    selected = str(comparison.iloc[0]["method"])
    final_calibrator = fit_calibrator(selected, oof["probability"].to_numpy(), oof["actual"].to_numpy())
    detailed = pd.concat(evaluation_frames, ignore_index=True)
    return selected, final_calibrator, comparison.merge(
        detailed.groupby("method").size().rename("forward_validation_rows"), on="method"
    )


def _year_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby(pd.to_datetime(frame["target_date"]).dt.year):
        rows.append({"year": int(year), **evaluate_classifier(group["will_increase"], group["p_increase"])})
    return pd.DataFrame(rows)


def _plot_reliability(table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    populated = table.loc[table["count"] > 0]
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="perfect calibration")
    axis.plot(
        populated["mean_predicted"],
        populated["empirical_frequency"],
        marker="o",
        color="#2563eb",
        linewidth=2,
        label="hurdle model",
    )
    for row in populated.itertuples():
        axis.annotate(str(row.count), (row.mean_predicted, row.empirical_frequency), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed increase rate", title="Hurdle-model reliability — untouched test")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _sanity_checks(predictor: HurdlePredictor, training: pd.DataFrame) -> pd.DataFrame:
    template = training[predictor.feature_names].median(numeric_only=True).to_dict()
    cases = [
        ("A", 13, 82.0, 82.0, 88.0, 2.5, 0.0, "very high"),
        ("B", 18, 86.0, 84.0, 86.0, -1.5, 90.0, "very low"),
        ("C", 16, 85.0, 84.8, 87.0, 0.8, 5.0, "meaningful"),
    ]
    rows: list[dict[str, Any]] = []
    for name, hour, current_max, current_temp, forecast_high, change_60m, minutes_since, expected in cases:
        values = dict(template)
        overrides = {
            "current_temp": current_temp,
            "current_max_so_far": current_max,
            "current_temp_minus_max_so_far": current_temp - current_max,
            "minutes_since_max_temp_so_far": minutes_since,
            "forecast_high": forecast_high,
            "forecast_gap": forecast_high - current_max,
            "forecast_high_minus_current_temp": forecast_high - current_temp,
            "max_so_far_minus_forecast_high": current_max - forecast_high,
            "temp_change_60m": change_60m,
            "temp_slope_60m": change_60m,
            "minute_of_day_sin": np.sin(2 * np.pi * hour / 24),
            "minute_of_day_cos": np.cos(2 * np.pi * hour / 24),
        }
        values.update({key: value for key, value in overrides.items() if key in predictor.feature_names})
        probability = float(predictor.predict_proba(values)[0])
        rows.append({"case": name, "expected": expected, "p_increase": probability})
    return pd.DataFrame(rows)


def _write_report(metadata: dict[str, Any], checks: dict[str, bool]) -> None:
    metrics = metadata["test_metrics"]
    lines = [
        "# Hurdle Model — Training & Testing Report",
        "",
        f"Generated: {metadata['generated_at_utc']}",
        f"Deployment eligible: **{metadata['deployment_eligible']}**",
        f"Selected model: `{metadata['model_type']}` / `{metadata['feature_level']}` / `{metadata['calibrator_type']}`",
        f"Dataset: {metadata['dataset_version']}, {metadata['prediction_frequency']} cadence",
        f"Train through {metadata['train_end']}; untouched test begins {metadata['test_start']}",
        "",
        "## Test performance",
        "",
        f"- Brier: {metrics['brier']:.6f} (climatology {metadata['baseline_test_metrics']['brier']:.6f})",
        f"- Log loss: {metrics['log_loss']:.6f} (climatology {metadata['baseline_test_metrics']['log_loss']:.6f})",
        f"- ROC-AUC: {metrics['roc_auc']:.6f}; PR-AUC: {metrics['pr_auc']:.6f}",
        f"- ECE: {metadata['test_calibration']['ece']:.6f}; low-tail gap: {metadata['low_tail']['gap']:.6f}",
        "",
        "## Acceptance checks",
        "",
        *[f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items()],
        "",
        f"Same-feed target invariant violations: {metadata['invariant_violation_dates']}",
        f"Official settlement disagreement days (audit only): {metadata['official_settlement_disagreement_days']}",
        f"Dangerous <5% false-confidence rows: {metadata['tail_failure_count']}",
    ]
    (OUTPUT_DIR / "hurdle_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    validation_start, test_start = _split_config()

    if args.reuse_dataset:
        dataset = pd.read_csv(DATASET_PATH)
        for column in ("target_date", "prediction_time"):
            dataset[column] = pd.to_datetime(dataset[column], errors="coerce")
        for column in (
            "prediction_time_utc",
            "observation_time_utc",
            "forecast_issue_time_utc",
            "atmosphere_time_utc",
            "current_max_time_utc",
        ):
            if column in dataset.columns:
                dataset[column] = pd.to_datetime(dataset[column], utc=True, errors="coerce")
        summary_path = OUTPUT_DIR / "hurdle_dataset_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        dataset, summary = build_hurdle_dataset(config=HurdleDatasetConfig())
        violations = dataset.attrs.get("invariant_violations")
        disagreements = dataset.attrs.get("official_settlement_disagreements")
        violations.to_csv(OUTPUT_DIR / "hurdle_invariant_violations.csv", index=False)
        disagreements.to_csv(OUTPUT_DIR / "hurdle_official_settlement_disagreements.csv", index=False)
        dataset = add_hurdle_core_features(dataset)
        dataset.to_csv(DATASET_PATH, index=False)
        (OUTPUT_DIR / "hurdle_dataset_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    validate_hurdle_dataset(dataset)
    dataset = dataset.reset_index(drop=True)
    dataset["_row_id"] = np.arange(len(dataset))
    dates = pd.to_datetime(dataset["target_date"]).dt.normalize()
    pretest = dataset.loc[dates < pd.Timestamp(test_start)].copy()
    test = dataset.loc[dates >= pd.Timestamp(test_start)].copy()
    if pretest.empty or test.empty:
        raise ValueError("Chronological train/test split is empty")
    if set(pretest["target_date"].dt.date) & set(test["target_date"].dt.date):
        raise AssertionError("Whole-day split invariant failed")
    folds = expanding_window_splits(pretest, test_start=test_start, minimum_training_years=2)

    print(f"Dataset: {len(dataset):,} rows / {dataset['target_date'].nunique():,} days")
    print(f"Pre-test: {len(pretest):,}; untouched test: {len(test):,}; folds: {[f['name'] for f in folds]}")

    results: list[dict[str, Any]] = []
    baseline_result, baseline_oof = _cross_validate_baseline(pretest, folds)
    results.append(baseline_result)
    candidate_oof: dict[tuple[str, str], pd.DataFrame] = {}
    candidate_features: dict[tuple[str, str], list[str]] = {}
    for level in progressive_feature_sets():
        for kind in ("logistic", "histgb"):
            print(f"Cross-validating {kind} / {level}")
            result, oof, features = _cross_validate(pretest, folds, level=level, kind=kind)
            results.append(result)
            candidate_oof[(kind, level)] = oof
            candidate_features[(kind, level)] = features
    if not args.skip_ngboost:
        print("Cross-validating Bernoulli NGBoost / E_plus_atmos")
        result, oof, features = _cross_validate(pretest, folds, level="E_plus_atmos", kind="ngb")
        results.append(result)
        candidate_oof[("ngb", "E_plus_atmos")] = oof
        candidate_features[("ngb", "E_plus_atmos")] = features

    flat_results = [{key: value for key, value in row.items() if key != "fold_metrics"} for row in results]
    ablation = pd.DataFrame(flat_results).sort_values(["cv_brier", "cv_log_loss"])
    ablation.to_csv(OUTPUT_DIR / "feature_ablation_results.csv", index=False)
    (OUTPUT_DIR / "expanding_fold_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    boosted_rows = ablation.loc[ablation["model"].isin(["histgb", "ngb"])]
    selected_row = boosted_rows.sort_values(["cv_brier", "cv_log_loss"]).iloc[0]
    selected_key = (str(selected_row["model"]), str(selected_row["level"]))
    selected_oof = candidate_oof[selected_key]
    features = candidate_features[selected_key]
    best_logistic_row = ablation.loc[ablation["model"].eq("logistic")].sort_values(
        ["cv_brier", "cv_log_loss"]
    ).iloc[0]
    best_logistic_key = ("logistic", str(best_logistic_row["level"]))

    calibrator_type, calibrator, calibration_comparison = _select_calibration(selected_oof)
    calibration_comparison.to_csv(OUTPUT_DIR / "calibration_comparison.csv", index=False)
    print(f"Selected before test: {selected_key}, calibration={calibrator_type}")

    classifier = _fit_model(selected_key[0], pretest, features)
    predictor = HurdlePredictor(classifier, features, calibrator, calibrator_type)
    test_probability = predictor.predict_proba(test)
    test_eval = test.copy()
    test_eval["p_increase"] = test_probability
    test_metrics = evaluate_classifier(test_eval["will_increase"], test_probability)
    test_calibration = calibration_error_metrics(test_eval["will_increase"], test_probability)

    baseline = climatological_baseline_fit(pretest)
    baseline_test_probability = climatological_predict(baseline, test)
    baseline_test_metrics = evaluate_classifier(test["will_increase"], baseline_test_probability)
    logistic_features = candidate_features[best_logistic_key]
    logistic = train_logistic_regression(pretest, logistic_features)
    logistic_test_probability = predict_proba(logistic, test, logistic_features)
    logistic_test_metrics = evaluate_classifier(test["will_increase"], logistic_test_probability)

    reliability = reliability_bins(test_eval["will_increase"], test_probability)
    reliability.to_csv(OUTPUT_DIR / "reliability_bins_test.csv", index=False)
    _plot_reliability(reliability, OUTPUT_DIR / "reliability_diagram_test.png")
    by_time = calibration_by_time_of_day(test_eval, "will_increase", "p_increase")
    by_time.to_csv(OUTPUT_DIR / "calibration_by_time_of_day.csv", index=False)
    for name, table in test_by_temperature_state(test_eval, "will_increase", "p_increase").items():
        table.to_csv(OUTPUT_DIR / f"calibration_{name}.csv", index=False)
    forecast_gap_testing(test_eval, "will_increase", "p_increase").to_csv(
        OUTPUT_DIR / "forecast_gap_testing.csv", index=False
    )
    failures = tail_failure_report(test_eval, "will_increase", "p_increase", threshold=0.05, top_n=len(test_eval))
    failures.to_csv(OUTPUT_DIR / "tail_failures.csv", index=False)
    year_metrics = _year_metrics(test_eval)
    year_metrics.to_csv(OUTPUT_DIR / "metrics_by_test_year.csv", index=False)
    sanity = _sanity_checks(predictor, pretest)
    sanity.to_csv(OUTPUT_DIR / "sanity_tests.csv", index=False)

    low_mask = test_probability < 0.10
    low_predicted = float(test_probability[low_mask].mean()) if low_mask.any() else float("nan")
    low_actual = float(test_eval.loc[low_mask, "will_increase"].mean()) if low_mask.any() else float("nan")
    low_tail = {
        "count": int(low_mask.sum()),
        "mean_predicted": low_predicted,
        "actual_rate": low_actual,
        "gap": float(abs(low_predicted - low_actual)) if low_mask.any() else float("nan"),
    }
    time_gap = (by_time["mean_pred"] - by_time["empirical_rate"]).abs()
    sanity_map = sanity.set_index("case")["p_increase"].to_dict()
    leakage_free = True
    for source_column in ("observation_time_utc", "forecast_issue_time_utc", "atmosphere_time_utc"):
        if source_column in dataset.columns:
            available = dataset[source_column].notna()
            leakage_free = leakage_free and bool(
                (dataset.loc[available, source_column] <= dataset.loc[available, "prediction_time_utc"]).all()
            )
    checks = {
        "beats_climatology_brier": bool(test_metrics["brier"] < baseline_test_metrics["brier"]),
        "beats_climatology_log_loss": bool(test_metrics["log_loss"] < baseline_test_metrics["log_loss"]),
        "beats_logistic_brier": bool(test_metrics["brier"] < logistic_test_metrics["brier"]),
        "reliability_ece_at_most_0_05": bool(test_calibration["ece"] <= 0.05),
        "low_probability_gap_at_most_0_03": bool(low_tail["count"] >= 100 and low_tail["gap"] <= 0.03),
        "stable_across_test_years": bool(year_metrics["brier"].max() - year_metrics["brier"].min() <= 0.04),
        "stable_by_time_of_day": bool(time_gap.max() <= 0.08),
        "no_data_leakage": bool(leakage_free),
        "forecast_inputs_reproducible": bool(dataset["forecast_source"].eq("nws_ndfd_historical_forecast").all()),
        "settlement_invariant_holds": bool((dataset["final_daily_high"] >= dataset["current_max_so_far"]).all()),
        "sanity_scenarios_ordered": bool(
            sanity_map.get("A", 0) > sanity_map.get("C", 0) > sanity_map.get("B", 1)
        ),
        "dangerous_tail_cases_reviewed": bool(args.confirm_tail_review or failures.empty),
    }
    deployment_eligible = all(checks.values())

    with (MODELS_DIR / "hurdle_classifier.pkl").open("wb") as handle:
        pickle.dump(classifier, handle)
    with (MODELS_DIR / "hurdle_calibrator.pkl").open("wb") as handle:
        pickle.dump(calibrator, handle)
    (MODELS_DIR / "hurdle_features.json").write_text(
        json.dumps({"features": features, "feature_count": len(features)}, indent=2), encoding="utf-8"
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_eligible": deployment_eligible,
        "training_dates": f"{pretest['target_date'].min().date()} to {pretest['target_date'].max().date()}",
        "validation_dates": [fold["val_start"] + " to " + fold["val_end"] for fold in folds],
        "test_dates": f"{test['target_date'].min().date()} to {test['target_date'].max().date()}",
        "train_end": str(pretest["target_date"].max().date()),
        "test_start": test_start,
        "test_was_used_for_selection": False,
        "feature_list": features,
        "feature_level": selected_key[1],
        "model_type": selected_key[0],
        "model_hyperparameters": classifier.get_params() if hasattr(classifier, "get_params") else classifier["model"].get_params(),
        "calibrator_type": calibrator_type,
        "calibrator_selection": calibration_comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "test_calibration": test_calibration,
        "baseline_test_metrics": baseline_test_metrics,
        "logistic_test_metrics": logistic_test_metrics,
        "low_tail": low_tail,
        "dataset_version": summary["dataset_version"],
        "dataset_rows": int(len(dataset)),
        "dataset_days": int(dataset["target_date"].nunique()),
        "prediction_frequency": summary["prediction_frequency"],
        "weather_station": summary["weather_station"],
        "settlement_rounding": summary["settlement_rounding"],
        "target_definition": summary["target_definition"],
        "forecast_source": summary["forecast_source"],
        "invariant_violation_dates": summary["n_violation_dates"],
        "invariant_violation_policy": summary["violation_policy"],
        "official_settlement_disagreement_days": summary["official_settlement_disagreement_days"],
        "official_settlement_role": summary["official_settlement_role"],
        "tail_failure_count": int(len(failures)),
        "tail_review_confirmed": bool(args.confirm_tail_review),
        "acceptance_checks": checks,
    }
    (MODELS_DIR / "hurdle_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    test_eval[
        [
            "target_date",
            "prediction_time",
            "current_temp",
            "current_max_so_far",
            "final_daily_high",
            "will_increase",
            "forecast_high",
            "p_increase",
        ]
    ].to_csv(OUTPUT_DIR / "hurdle_test_predictions.csv", index=False)
    _write_report(metadata, checks)

    print(f"Test Brier {test_metrics['brier']:.6f} vs climatology {baseline_test_metrics['brier']:.6f} and logistic {logistic_test_metrics['brier']:.6f}")
    print(f"Test log loss {test_metrics['log_loss']:.6f}; ECE {test_calibration['ece']:.6f}")
    print(f"Low-tail gap {low_tail['gap']:.6f} on {low_tail['count']:,} rows")
    print(f"Dangerous <5% failures: {len(failures)}")
    print(f"Deployment eligible: {deployment_eligible}")
    return 0 if deployment_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
