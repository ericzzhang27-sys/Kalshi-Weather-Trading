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
    evaluate_conditional_distribution,
    positive_increase_rows,
    train_shifted_poisson_ngboost,
)
from src.hurdle_comparison import time_of_day_bucket
from src.hurdle_distribution import categorical_scores, integer_delta_probabilities
from src.hurdle_model import load_hurdle_predictor


DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
EXCEEDANCE_BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
FEATURES_PATH = REPO_ROOT / "models" / "remaining_increase_features.json"
MODEL_PATH = REPO_ROOT / "models" / "remaining_increase_5min_single_run.pkl"
METADATA_PATH = REPO_ROOT / "models" / "remaining_increase_5min_single_run_metadata.json"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "remaining_increase" / "single_5min_poisson_run"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exceedance_predictor():
    bundle = json.loads(EXCEEDANCE_BUNDLE_PATH.read_text(encoding="utf-8"))
    if bundle.get("status") not in {"frozen_validated", "frozen_validated_user_override"}:
        raise ValueError("Upgraded exceedance bundle is not frozen and validated")
    for label, filename in bundle["paths"].items():
        artifact = EXCEEDANCE_BUNDLE_PATH.parent / filename
        if _sha256(artifact) != bundle["sha256"][label]:
            raise ValueError(f"Exceedance bundle {label} hash mismatch")
    if _sha256(DATASET_PATH) != bundle["sha256"]["dataset"]:
        raise ValueError("Exceedance model and Poisson run must use the identical dataset")
    predictor = load_hurdle_predictor(
        EXCEEDANCE_BUNDLE_PATH.parent / bundle["paths"]["classifier"],
        EXCEEDANCE_BUNDLE_PATH.parent / bundle["paths"]["features"],
        EXCEEDANCE_BUNDLE_PATH.parent / bundle["paths"]["calibrator"],
        bundle["calibration"],
    )
    return bundle, predictor


def _realized_probability(matrix: np.ndarray, target: np.ndarray, max_delta: int) -> np.ndarray:
    category = np.where(target > max_delta, max_delta + 1, target).astype(int)
    return np.clip(matrix[np.arange(len(target)), category], 1e-15, 1.0)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exceedance_bundle, exceedance = _load_exceedance_predictor()
    features = list(json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["features"])
    dataset = pd.read_csv(DATASET_PATH, low_memory=False)
    dataset["target_date"] = pd.to_datetime(dataset["target_date"], errors="raise")
    dataset["prediction_time"] = pd.to_datetime(dataset["prediction_time"], errors="raise")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    test_start = pd.Timestamp(str(config["splits"]["test_start"]))
    pretest = dataset.loc[dataset["target_date"].dt.normalize() < test_start].copy()
    test = dataset.loc[dataset["target_date"].dt.normalize() >= test_start].copy()
    positive_train = positive_increase_rows(pretest)
    positive_test = positive_increase_rows(test)

    print(f"Single Poisson fit: {len(positive_train):,} positive five-minute rows")
    poisson = train_shifted_poisson_ngboost(
        positive_train,
        features,
        n_estimators=200,
        learning_rate=0.03,
    )
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(poisson, handle)
    conditional_metrics = evaluate_conditional_distribution(poisson, positive_test)

    p_increase = exceedance.predict_proba(test)
    max_delta = 10
    probability = integer_delta_probabilities(
        p_increase,
        poisson,
        test,
        max_delta=max_delta,
    )
    target = test["remaining_increase"].to_numpy(dtype=int)
    full_metrics = categorical_scores(probability, target, max_delta=max_delta)
    realized = _realized_probability(probability, target, max_delta)

    prediction_export = test[
        ["target_date", "prediction_time", "remaining_increase", "current_max_so_far"]
    ].copy()
    prediction_export["p_increase"] = p_increase
    prediction_export["realized_category_probability"] = realized
    prediction_export["row_nll"] = -np.log(realized)
    prediction_export.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    grouped = prediction_export.assign(
        time_bucket=time_of_day_bucket(prediction_export["prediction_time"]),
        outcome=np.where(target == 0, "delta_zero", "delta_positive"),
    )
    time_rows: list[dict] = []
    for bucket, indices in grouped.groupby("time_bucket", observed=False).groups.items():
        idx = test.index.get_indexer(indices)
        time_rows.append(
            {
                "time_bucket": str(bucket),
                "n": len(idx),
                "zero_rate": float(np.mean(target[idx] == 0)),
                **categorical_scores(probability[idx], target[idx], max_delta=max_delta),
            }
        )
    by_time = pd.DataFrame(time_rows)
    by_time.to_csv(OUTPUT_DIR / "full_hurdle_by_time.csv", index=False)

    outcome_rows = []
    for outcome, group in grouped.groupby("outcome"):
        outcome_rows.append(
            {
                "outcome": outcome,
                "n": len(group),
                "share": len(group) / len(grouped),
                "nll": float(group["row_nll"].mean()),
            }
        )
    by_outcome = pd.DataFrame(outcome_rows)
    by_outcome.to_csv(OUTPUT_DIR / "nll_by_outcome.csv", index=False)

    metadata = {
        "status": "completed_single_run_not_promoted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": 1,
        "conditional_model": "NGBoost shifted Poisson",
        "conditional_training_frequency": "5 minutes",
        "conditional_training_rows": len(positive_train),
        "conditional_test_rows": len(positive_test),
        "full_hurdle_test_rows": len(test),
        "feature_count": len(features),
        "training_dates": f"{pretest['target_date'].min().date()} to {pretest['target_date'].max().date()}",
        "test_dates": f"{test['target_date'].min().date()} to {test['target_date'].max().date()}",
        "conditional_test_metrics": conditional_metrics,
        "full_hurdle_test_metrics": full_metrics,
        "exceedance_winner": exceedance_bundle["winner"],
        "exceedance_calibration": exceedance_bundle["calibration"],
        "dataset_sha256": _sha256(DATASET_PATH),
        "exceedance_bundle_sha256": _sha256(EXCEEDANCE_BUNDLE_PATH),
        "poisson_model_sha256": _sha256(MODEL_PATH),
        "promotion": "none",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = [
        "# Single Five-Minute Poisson Hurdle Test",
        "",
        "One shifted-Poisson fit was trained on all pre-test positive five-minute rows.",
        f"It was combined with the frozen upgraded **{exceedance_bundle['winner']}** exceedance model.",
        "No production artifact was replaced.",
        "",
        f"Conditional positive test NLL: {conditional_metrics['interval_nll']:.6f}",
        f"Conditional positive CRPS: {conditional_metrics['crps']:.6f}",
        f"Full hurdle multiclass NLL: {full_metrics['multiclass_nll']:.6f}",
        f"Full hurdle mean bucket Brier: {full_metrics['mean_bucket_brier']:.6f}",
        "",
        "## Outcome decomposition",
        "",
        by_outcome.to_markdown(index=False),
        "",
        "## Time-of-day results",
        "",
        by_time.to_markdown(index=False),
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
