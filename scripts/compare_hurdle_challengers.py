from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hurdle_calibration import apply_calibrator, fit_calibrator
from src.hurdle_comparison import brier_skill_score, choose_exceedance_winner, time_of_day_bucket
from src.hurdle_evaluation import calibration_error_metrics, reliability_bins
from src.hurdle_model import (
    climatological_baseline_fit,
    climatological_predict,
    evaluate_classifier,
    expanding_window_splits,
    materialize_fold,
    predict_proba,
    train_boosted_classifier,
    train_logistic_regression,
)


DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
DATASET_SUMMARY_PATH = REPO_ROOT / "outputs" / "hurdle" / "hurdle_dataset_summary.json"
FEATURES_PATH = REPO_ROOT / "models" / "hurdle_features.json"
METADATA_PATH = REPO_ROOT / "models" / "hurdle_metadata.json"
MODEL_PATH = REPO_ROOT / "models" / "hurdle_classifier.pkl"
CALIBRATOR_PATH = REPO_ROOT / "models" / "hurdle_calibrator.pkl"
BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "hurdle" / "challenger_study"
CANDIDATE_DIR = OUTPUT_DIR / "candidates"

MODEL_KINDS = {
    "ngboost_bernoulli": "ngb",
    "logistic_regression": "logistic",
    "lightgbm_classifier": "lightgbm",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs() -> tuple[pd.DataFrame, list[str], dict[str, Any], str]:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["target_date"] = pd.to_datetime(dataset["target_date"], errors="raise")
    dataset["prediction_time"] = pd.to_datetime(dataset["prediction_time"], errors="raise")
    dataset["_row_id"] = np.arange(len(dataset))
    feature_payload = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    features = list(feature_payload["features"])
    missing = sorted(set(features).difference(dataset.columns))
    if missing:
        raise ValueError(f"Incumbent feature contract is not reproducible: {missing}")
    incumbent_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    test_start = str(config.get("splits", {}).get("test_start", incumbent_metadata["test_start"]))
    return dataset, features, incumbent_metadata, test_start


def _fit(model_name: str, train: pd.DataFrame, features: list[str]) -> Any:
    kind = MODEL_KINDS[model_name]
    if kind == "logistic":
        return train_logistic_regression(train, features)
    return train_boosted_classifier(train, features, kind=kind)


def _validation_predictions(
    model_name: str,
    pretest: pd.DataFrame,
    features: list[str],
    folds: list[dict[str, Any]],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for fold in folds:
        train, validation = materialize_fold(pretest, fold)
        model = _fit(model_name, train, features)
        probability = predict_proba(model, validation, features)
        pieces.append(
            pd.DataFrame(
                {
                    "row_id": validation["_row_id"].to_numpy(),
                    "validation_year": validation["target_date"].dt.year.to_numpy(),
                    "actual": validation["will_increase"].to_numpy(dtype=int),
                    "raw_probability": probability,
                    "fold": fold["name"],
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def _select_calibrator(
    model_name: str,
    oof: pd.DataFrame,
) -> tuple[str, Any, pd.DataFrame, pd.DataFrame]:
    years = sorted(oof["validation_year"].unique())
    comparisons: list[dict[str, Any]] = []
    forward_predictions: list[pd.DataFrame] = []
    for method in ("raw", "platt", "isotonic"):
        method_pieces: list[pd.DataFrame] = []
        for year in years[1:]:
            fit_rows = oof.loc[oof["validation_year"] < year]
            score_rows = oof.loc[oof["validation_year"] == year].copy()
            calibrator = fit_calibrator(
                method,
                fit_rows["raw_probability"].to_numpy(),
                fit_rows["actual"].to_numpy(),
            )
            score_rows["probability"] = apply_calibrator(
                method, calibrator, score_rows["raw_probability"].to_numpy()
            )
            score_rows["calibration_method"] = method
            method_pieces.append(score_rows)
        combined = pd.concat(method_pieces, ignore_index=True)
        metrics = evaluate_classifier(combined["actual"], combined["probability"])
        comparisons.append({"model": model_name, "calibration_method": method, **metrics})
        forward_predictions.append(combined)
    comparison = pd.DataFrame(comparisons).sort_values(["brier", "log_loss"]).reset_index(drop=True)
    selected_method = str(comparison.iloc[0]["calibration_method"])
    selected_calibrator = fit_calibrator(
        selected_method, oof["raw_probability"].to_numpy(), oof["actual"].to_numpy()
    )
    selected_forward = pd.concat(forward_predictions, ignore_index=True).loc[
        lambda frame: frame["calibration_method"].eq(selected_method)
    ]
    return selected_method, selected_calibrator, comparison, selected_forward


def _low_probability_table(model: str, y: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    bins = [(0.0, 0.05, "<5%"), (0.05, 0.10, "5–10%"), (0.10, 0.20, "10–20%")]
    rows: list[dict[str, Any]] = []
    y_array = np.asarray(y, dtype=int)
    for lower, upper, label in bins:
        mask = (probability >= lower) & (probability < upper)
        rows.append(
            {
                "model": model,
                "probability_bin": label,
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_predicted": float(probability[mask].mean()) if mask.any() else float("nan"),
                "actual_rate": float(y_array[mask].mean()) if mask.any() else float("nan"),
                "calibration_gap": float(probability[mask].mean() - y_array[mask].mean()) if mask.any() else float("nan"),
                "brier": float(np.mean((probability[mask] - y_array[mask]) ** 2)) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _time_diagnostics(
    model: str,
    test: pd.DataFrame,
    probability: np.ndarray,
    baseline_probability: np.ndarray,
) -> pd.DataFrame:
    frame = test[["prediction_time", "will_increase"]].copy()
    frame["probability"] = probability
    frame["baseline_probability"] = baseline_probability
    frame["time_bucket"] = time_of_day_bucket(frame["prediction_time"])
    rows: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("time_bucket", observed=True):
        metrics = evaluate_classifier(group["will_increase"], group["probability"])
        rows.append(
            {
                "model": model,
                "time_bucket": str(bucket),
                "count": int(len(group)),
                "brier": metrics["brier"],
                "baseline_brier": float(
                    np.mean((group["baseline_probability"] - group["will_increase"]) ** 2)
                ),
                "brier_skill_score": brier_skill_score(
                    group["will_increase"], group["probability"], group["baseline_probability"]
                ),
                "log_loss": metrics["log_loss"],
                "mean_predicted": metrics["mean_pred"],
                "actual_rate": metrics["empirical_rate"],
                "calibration_gap": float(metrics["mean_pred"] - metrics["empirical_rate"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_reliability(reliability: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="perfect calibration")
    colors = {
        "ngboost_bernoulli": "#2563eb",
        "logistic_regression": "#059669",
        "lightgbm_classifier": "#dc2626",
    }
    for model, group in reliability.loc[reliability["count"] > 0].groupby("model"):
        axis.plot(
            group["mean_predicted"],
            group["empirical_frequency"],
            marker="o",
            linewidth=2,
            label=model,
            color=colors.get(model),
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed exceedance rate",
        title="Exceedance-model reliability — untouched test",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "reliability_comparison_test.png", dpi=170)
    plt.close(figure)


def _write_report(
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    low_probability: pd.DataFrame,
    winner: str,
    incumbent: str,
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Exceedance Model Challenger Study",
        "",
        "All candidates used the same 40 features, expanding whole-day folds, and untouched test period.",
        "Calibration method selection used forward validation predictions only. AUC was not used for selection.",
        "",
        f"Incumbent entering the corrected-target study: **{incumbent}**",
        f"Winner: **{winner}**",
        f"Decision: `{decision['decision']}` — {decision['reason']}",
        "",
        "## Untouched-test comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Validation-only calibration selection",
        "",
        calibration.to_markdown(index=False),
        "",
        "## Low-probability test calibration",
        "",
        low_probability.to_markdown(index=False),
        "",
        "Replacement requires at least +0.02 absolute BSS or clearly better late-day calibration without material overall degradation.",
    ]
    (OUTPUT_DIR / "challenger_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    dataset, features, incumbent_metadata, test_start = _load_inputs()
    dataset_summary = json.loads(DATASET_SUMMARY_PATH.read_text(encoding="utf-8"))
    incumbent_bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    incumbent = str(incumbent_bundle.get("winner", "lightgbm_classifier"))
    if incumbent not in MODEL_KINDS:
        raise ValueError(f"Frozen incumbent {incumbent!r} is not a configured challenger")
    date = dataset["target_date"].dt.normalize()
    pretest = dataset.loc[date < pd.Timestamp(test_start)].copy()
    test = dataset.loc[date >= pd.Timestamp(test_start)].copy()
    folds = expanding_window_splits(pretest, test_start=test_start, minimum_training_years=2)
    baseline = climatological_baseline_fit(pretest)
    baseline_test_probability = climatological_predict(baseline, test)
    baseline_test_brier = float(
        np.mean((baseline_test_probability - test["will_increase"].to_numpy()) ** 2)
    )
    baseline_oof_pieces: list[pd.DataFrame] = []
    for fold in folds:
        fold_train, fold_validation = materialize_fold(pretest, fold)
        fold_baseline = climatological_baseline_fit(fold_train)
        baseline_oof_pieces.append(
            pd.DataFrame(
                {
                    "row_id": fold_validation["_row_id"].to_numpy(),
                    "baseline_probability": climatological_predict(fold_baseline, fold_validation),
                }
            )
        )
    baseline_oof = pd.concat(baseline_oof_pieces, ignore_index=True).set_index("row_id")

    print(f"Exact feature contract: {len(features)} columns")
    print(f"Folds: {[fold['name'] for fold in folds]}; test: {test['target_date'].min().date()} to {test['target_date'].max().date()}")

    fitted_models: dict[str, Any] = {}
    fitted_calibrators: dict[str, Any] = {}
    selected_calibration: dict[str, str] = {}
    comparison_rows: list[dict[str, Any]] = []
    calibration_tables: list[pd.DataFrame] = []
    low_tables: list[pd.DataFrame] = []
    time_tables: list[pd.DataFrame] = []
    reliability_tables: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name in MODEL_KINDS:
        print(f"Validating {model_name}")
        oof = _validation_predictions(model_name, pretest, features, folds)
        method, calibrator, calibration_comparison, forward_predictions = _select_calibrator(model_name, oof)
        calibration_tables.append(calibration_comparison)
        selected_calibration[model_name] = method
        fitted_calibrators[model_name] = calibrator

        print(f"Refitting {model_name}; validation calibration={method}")
        model = _fit(model_name, pretest, features)
        fitted_models[model_name] = model
        raw_test_probability = predict_proba(model, test, features)
        test_probability = apply_calibrator(method, calibrator, raw_test_probability)
        metrics = evaluate_classifier(test["will_increase"], test_probability)
        calibration_metrics = calibration_error_metrics(test["will_increase"], test_probability)
        late_mask = test["prediction_time"].dt.hour >= 16
        late_y = test.loc[late_mask, "will_increase"].to_numpy(dtype=int)
        late_probability = test_probability[late_mask.to_numpy()]
        late_brier = float(np.mean((late_probability - late_y) ** 2))
        late_gap = float(abs(late_probability.mean() - late_y.mean()))

        forward_baseline_probability = baseline_oof.loc[
            forward_predictions["row_id"], "baseline_probability"
        ].to_numpy()
        validation_bss = brier_skill_score(
            forward_predictions["actual"], forward_predictions["probability"], forward_baseline_probability
        )
        comparison_rows.append(
            {
                "model": model_name,
                "calibration": method,
                "brier": metrics["brier"],
                "brier_skill_score": brier_skill_score(
                    test["will_increase"], test_probability, baseline_test_probability
                ),
                "validation_bss": validation_bss,
                "log_loss": metrics["log_loss"],
                "ece": calibration_metrics["ece"],
                "late_day_brier": late_brier,
                "late_day_calibration_gap": late_gap,
                "roc_auc_secondary": metrics["roc_auc"],
                "pr_auc_secondary": metrics["pr_auc"],
                "test_rows": metrics["n"],
            }
        )
        low_tables.append(_low_probability_table(model_name, test["will_increase"], test_probability))
        time_tables.append(
            _time_diagnostics(model_name, test, test_probability, baseline_test_probability)
        )
        reliability = reliability_bins(test["will_increase"], test_probability)
        reliability["model"] = model_name
        reliability_tables.append(reliability)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "target_date": test["target_date"].to_numpy(),
                    "prediction_time": test["prediction_time"].to_numpy(),
                    "will_increase": test["will_increase"].to_numpy(),
                    "model": model_name,
                    "calibration": method,
                    "p_increase": test_probability,
                }
            )
        )
        with (CANDIDATE_DIR / f"{model_name}_classifier.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        with (CANDIDATE_DIR / f"{model_name}_calibrator.pkl").open("wb") as handle:
            pickle.dump(calibrator, handle)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["brier_skill_score", "log_loss"], ascending=[False, True]
    )
    calibration_table = pd.concat(calibration_tables, ignore_index=True)
    low_probability = pd.concat(low_tables, ignore_index=True)
    time_diagnostics = pd.concat(time_tables, ignore_index=True)
    reliability = pd.concat(reliability_tables, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    winner, decision = choose_exceedance_winner(comparison, incumbent=incumbent)

    comparison["delta_bss_vs_incumbent"] = comparison["brier_skill_score"] - float(
        comparison.loc[comparison["model"].eq(incumbent), "brier_skill_score"].iloc[0]
    )
    comparison.to_csv(OUTPUT_DIR / "overall_test_comparison.csv", index=False)
    calibration_table.to_csv(OUTPUT_DIR / "validation_calibration_comparison.csv", index=False)
    low_probability.to_csv(OUTPUT_DIR / "low_probability_calibration_test.csv", index=False)
    time_diagnostics.to_csv(OUTPUT_DIR / "bss_by_time_of_day_test.csv", index=False)
    reliability.to_csv(OUTPUT_DIR / "reliability_bins_test.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "candidate_test_predictions.csv", index=False)
    _plot_reliability(reliability)
    _write_report(comparison, calibration_table, low_probability, winner, incumbent, decision)

    # Freeze the winner to the stable exceedance artifact names.
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(fitted_models[winner], handle)
    with CALIBRATOR_PATH.open("wb") as handle:
        pickle.dump(fitted_calibrators[winner], handle)
    FEATURES_PATH.write_text(
        json.dumps({"features": features, "feature_count": len(features)}, indent=2), encoding="utf-8"
    )
    winner_row = comparison.loc[comparison["model"].eq(winner)].iloc[0].to_dict()
    winner_calibration_rows = calibration_table.loc[
        calibration_table["model"].eq(winner)
    ].to_dict(orient="records")
    baseline_test_metrics = evaluate_classifier(
        test["will_increase"], baseline_test_probability
    )
    winner_model = fitted_models[winner]
    frozen_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_eligible": bool(
            winner_row["brier_skill_score"] > 0
            and winner_row["ece"] <= 0.05
            and (dataset["final_daily_high"] >= dataset["current_max_so_far"]).all()
        ),
        "training_dates": f"{pretest['target_date'].min().date()} to {pretest['target_date'].max().date()}",
        "validation_dates": [
            fold["val_start"] + " to " + fold["val_end"] for fold in folds
        ],
        "test_dates": f"{test['target_date'].min().date()} to {test['target_date'].max().date()}",
        "train_end": str(pretest["target_date"].max().date()),
        "test_start": test_start,
        "test_was_used_for_selection": False,
        "freeze_status": "frozen_after_challenger_study",
        "model_type": winner,
        "model_hyperparameters": (
            winner_model.get_params() if hasattr(winner_model, "get_params") else {}
        ),
        "feature_list": features,
        "feature_count": len(features),
        "feature_level": incumbent_metadata.get("feature_level", "E_plus_atmos"),
        "calibrator_type": selected_calibration[winner],
        "calibrator_selection": winner_calibration_rows,
        "test_metrics": winner_row,
        "test_calibration": {"ece": winner_row["ece"]},
        "brier_skill_reference": "five-minute time-of-day climatology",
        "baseline_test_brier": baseline_test_brier,
        "baseline_test_metrics": baseline_test_metrics,
        "dataset_version": dataset_summary["dataset_version"],
        "dataset_rows": int(len(dataset)),
        "dataset_days": int(dataset["target_date"].nunique()),
        "prediction_frequency": dataset_summary["prediction_frequency"],
        "decision_window_local": dataset_summary["decision_window_local"],
        "weather_station": dataset_summary["weather_station"],
        "settlement_rounding": dataset_summary["settlement_rounding"],
        "target_source": dataset_summary["target_source"],
        "target_definition": dataset_summary["target_definition"],
        "official_settlement_role": dataset_summary["official_settlement_role"],
        "official_settlement_disagreement_days": dataset_summary[
            "official_settlement_disagreement_days"
        ],
        "complete_day_rules": dataset_summary["complete_day_rules"],
        "forecast_source": dataset_summary["forecast_source"],
        "invariant_violation_dates": dataset_summary["n_violation_dates"],
        "acceptance_checks": {
            "beats_time_climatology_brier": bool(winner_row["brier_skill_score"] > 0),
            "reliability_ece_at_most_0_05": bool(winner_row["ece"] <= 0.05),
            "same_feed_target_invariant_holds": bool(
                (dataset["final_daily_high"] >= dataset["current_max_so_far"]).all()
            ),
            "no_future_observations": bool(
                (pd.to_datetime(dataset["observation_time_utc"], utc=True)
                 <= pd.to_datetime(dataset["prediction_time_utc"], utc=True)).all()
            ),
        },
        "challenger_study": {
            "models": list(MODEL_KINDS),
            "identical_feature_count": len(features),
            "identical_folds": folds,
            "test_start": test_start,
            "test_used_once_after_freezing_candidates": True,
            "winner": winner,
            "incumbent": incumbent,
            "decision": decision,
            "comparison": comparison.to_dict(orient="records"),
        },
    }
    METADATA_PATH.write_text(json.dumps(frozen_metadata, indent=2, default=str), encoding="utf-8")
    bundle = {
        "status": "frozen_validated",
        "winner": winner,
        "calibration": selected_calibration[winner],
        "feature_count": len(features),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "classifier": MODEL_PATH.name,
            "calibrator": CALIBRATOR_PATH.name,
            "features": FEATURES_PATH.name,
            "metadata": METADATA_PATH.name,
        },
        "sha256": {
            "classifier": _sha256(MODEL_PATH),
            "calibrator": _sha256(CALIBRATOR_PATH),
            "features": _sha256(FEATURES_PATH),
            "metadata": _sha256(METADATA_PATH),
            "dataset": _sha256(DATASET_PATH),
        },
        "replacement_decision": decision,
    }
    BUNDLE_PATH.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")

    print(comparison.to_string(index=False))
    print(f"Frozen winner: {winner} ({selected_calibration[winner]})")
    print(f"Decision: {decision['decision']} — {decision['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
