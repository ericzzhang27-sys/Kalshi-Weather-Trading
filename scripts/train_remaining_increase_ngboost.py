from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conditional_increase_model import (
    evaluate_shifted_poisson,
    positive_increase_rows,
    predict_conditional_distribution,
    train_shifted_poisson_ngboost,
)
from src.hurdle_model import expanding_window_splits, materialize_fold


DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
DATASET_SUMMARY_PATH = REPO_ROOT / "outputs" / "hurdle" / "hurdle_dataset_summary.json"
EXCEEDANCE_BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
HURDLE_FEATURES_PATH = REPO_ROOT / "models" / "hurdle_features.json"
MODEL_PATH = REPO_ROOT / "models" / "remaining_increase_ngboost.pkl"
FEATURES_PATH = REPO_ROOT / "models" / "remaining_increase_features.json"
METADATA_PATH = REPO_ROOT / "models" / "remaining_increase_metadata.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "remaining_increase"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_exceedance_bundle() -> dict:
    if not EXCEEDANCE_BUNDLE_PATH.exists():
        raise FileNotFoundError("Freeze the exceedance model before training the conditional stage")
    bundle = json.loads(EXCEEDANCE_BUNDLE_PATH.read_text(encoding="utf-8"))
    if bundle.get("status") not in {"frozen_validated", "frozen_validated_user_override"}:
        raise ValueError("Exceedance bundle is not frozen and validated")
    for label, filename in bundle["paths"].items():
        path = EXCEEDANCE_BUNDLE_PATH.parent / filename
        actual = _sha256(path)
        expected = bundle["sha256"][label]
        if actual != expected:
            raise ValueError(f"Frozen exceedance {label} hash mismatch")
    return bundle


def _baseline_nll(train_target: pd.Series, target: pd.Series) -> float:
    from scipy.stats import poisson

    mean_shifted = float((train_target - 1.0).mean())
    return float(-np.mean(poisson(mu=max(mean_shifted, 1e-9)).logpmf((target - 1.0).astype(int))))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exceedance_bundle = _verify_frozen_exceedance_bundle()
    dataset_summary = json.loads(DATASET_SUMMARY_PATH.read_text(encoding="utf-8"))
    feature_payload = json.loads(HURDLE_FEATURES_PATH.read_text(encoding="utf-8"))
    features = list(feature_payload["features"])
    dataset = pd.read_csv(DATASET_PATH)
    dataset["target_date"] = pd.to_datetime(dataset["target_date"], errors="raise")
    dataset["prediction_time"] = pd.to_datetime(dataset["prediction_time"], errors="raise")
    dataset["_row_id"] = np.arange(len(dataset))
    positive = positive_increase_rows(dataset)
    if not positive["will_increase"].eq(1).all():
        raise AssertionError("remaining_increase > 0 must exactly match will_increase == 1")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    test_start = str(config.get("splits", {}).get("test_start", "2025-01-01"))
    date = positive["target_date"].dt.normalize()
    pretest = positive.loc[date < pd.Timestamp(test_start)].copy()
    test = positive.loc[date >= pd.Timestamp(test_start)].copy()
    folds = expanding_window_splits(pretest, test_start=test_start, minimum_training_years=2)

    fold_rows: list[dict] = []
    validation_predictions: list[pd.DataFrame] = []
    for fold in folds:
        train, validation = materialize_fold(pretest, fold)
        print(f"Training conditional fold {fold['name']}: {len(train):,} -> {len(validation):,}")
        artifact = train_shifted_poisson_ngboost(train, features)
        metrics = evaluate_shifted_poisson(artifact, validation)
        metrics["baseline_nll"] = _baseline_nll(train["remaining_increase"], validation["remaining_increase"])
        metrics["nll_skill"] = 1.0 - metrics["nll"] / metrics["baseline_nll"]
        fold_rows.append({"fold": fold["name"], **metrics})
        distribution = predict_conditional_distribution(artifact, validation)
        validation_predictions.append(
            pd.DataFrame(
                {
                    "target_date": validation["target_date"].to_numpy(),
                    "prediction_time": validation["prediction_time"].to_numpy(),
                    "remaining_increase": validation["remaining_increase"].to_numpy(),
                    "predicted_mean": distribution.params["mu"] + 1.0,
                    "poisson_mu_shifted": distribution.params["mu"],
                    "fold": fold["name"],
                }
            )
        )
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(OUTPUT_DIR / "expanding_fold_metrics.csv", index=False)
    pd.concat(validation_predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_predictions.csv", index=False
    )

    print(f"Refitting conditional model through {pretest['target_date'].max().date()}")
    final_artifact = train_shifted_poisson_ngboost(pretest, features)
    test_metrics = evaluate_shifted_poisson(final_artifact, test)
    test_metrics["baseline_nll"] = _baseline_nll(
        pretest["remaining_increase"], test["remaining_increase"]
    )
    test_metrics["nll_skill"] = 1.0 - test_metrics["nll"] / test_metrics["baseline_nll"]
    distribution = predict_conditional_distribution(final_artifact, test)
    test_predictions = test[
        ["target_date", "prediction_time", "remaining_increase", "current_max_so_far", "final_daily_high"]
    ].copy()
    test_predictions["predicted_mean"] = distribution.params["mu"] + 1.0
    test_predictions["poisson_mu_shifted"] = distribution.params["mu"]
    test_predictions["cdf_at_realized"] = distribution.dist.cdf(
        (test["remaining_increase"].to_numpy() - 1.0).astype(int)
    )
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_artifact, handle)
    FEATURES_PATH.write_text(
        json.dumps({"features": features, "feature_count": len(features)}, indent=2), encoding="utf-8"
    )
    metadata = {
        "status": "trained_not_yet_combined_with_hurdle_cdf",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "NGBoost shifted Poisson",
        "distribution": "1 + Poisson(mu(X))",
        "target": "remaining_increase",
        "training_filter": "remaining_increase > 0 and will_increase == 1",
        "target_transform": "remaining_increase - 1",
        "positive_support": "integer Fahrenheit values >= 1",
        "dataset_version": dataset_summary["dataset_version"],
        "target_source": dataset_summary["target_source"],
        "target_definition": dataset_summary["target_definition"],
        "dataset_sha256": _sha256(DATASET_PATH),
        "feature_count": len(features),
        "training_dates": f"{pretest['target_date'].min().date()} to {pretest['target_date'].max().date()}",
        "test_dates": f"{test['target_date'].min().date()} to {test['target_date'].max().date()}",
        "training_rows": int(len(pretest)),
        "test_rows": int(len(test)),
        "folds": folds,
        "fold_metrics": fold_rows,
        "test_metrics": test_metrics,
        "exceedance_bundle_sha256": _sha256(EXCEEDANCE_BUNDLE_PATH),
        "exceedance_winner": exceedance_bundle["winner"],
        "model_sha256": _sha256(MODEL_PATH),
        "features_sha256": _sha256(FEATURES_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = [
        "# Conditional Remaining-Increase NGBoost",
        "",
        "The exceedance model was frozen and hash-verified before this stage ran.",
        "Only rows with `remaining_increase > 0` were used.",
        "The model is `1 + Poisson(mu(X))`, so its support is the positive integer Fahrenheit values.",
        "",
        f"Test NLL: {test_metrics['nll']:.6f} (baseline {test_metrics['baseline_nll']:.6f})",
        f"Test NLL skill: {test_metrics['nll_skill']:.6f}",
        f"Test MAE: {test_metrics['mae']:.6f}; RMSE: {test_metrics['rmse']:.6f}",
        f"80% coverage: {test_metrics['coverage_80']:.6f}; 90% coverage: {test_metrics['coverage_90']:.6f}",
        "",
        "This artifact has not yet been combined with the frozen binary exceedance probability into the final bucket CDF.",
    ]
    (OUTPUT_DIR / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
