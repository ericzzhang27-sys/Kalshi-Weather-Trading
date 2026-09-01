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
    conditional_cdf,
    evaluate_conditional_distribution,
    load_conditional_increase_model,
    positive_increase_rows,
    train_shifted_poisson_ngboost,
)
from src.hurdle_comparison import time_of_day_bucket
from src.hurdle_model import expanding_window_splits, materialize_fold


DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
FEATURES_PATH = REPO_ROOT / "models" / "remaining_increase_features.json"
INCUMBENT_PATH = REPO_ROOT / "models" / "remaining_increase_ngboost.pkl"
CHALLENGER_PATH = REPO_ROOT / "models" / "remaining_increase_hourly_challenger.pkl"
METADATA_PATH = REPO_ROOT / "models" / "remaining_increase_hourly_challenger_metadata.json"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "remaining_increase" / "hourly_challenger"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hourly_rows(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["prediction_time"], errors="raise")
    result = frame.loc[timestamp.dt.minute.eq(0)].copy()
    if result.empty:
        raise ValueError("No top-of-hour rows are available")
    return result


def _mean_bucket_brier(artifact: dict, frame: pd.DataFrame, max_delta: int = 10) -> float:
    target = frame["remaining_increase"].to_numpy(dtype=float)
    values: list[float] = []
    for delta in range(1, max_delta + 1):
        probability = conditional_cdf(artifact, frame, delta) - conditional_cdf(
            artifact, frame, delta - 1
        )
        values.append(float(np.mean((probability - (target == delta)) ** 2)))
    tail_probability = 1.0 - conditional_cdf(artifact, frame, max_delta)
    values.append(float(np.mean((tail_probability - (target > max_delta)) ** 2)))
    return float(np.mean(values))


def _metrics(artifact: dict, frame: pd.DataFrame) -> dict[str, float | int]:
    result = evaluate_conditional_distribution(artifact, frame)
    result["mean_bucket_brier"] = _mean_bucket_brier(artifact, frame)
    return result


def _group_metrics(
    name: str, artifact: dict, frame: pd.DataFrame, evaluation_set: str
) -> list[dict]:
    working = frame.assign(time_bucket=time_of_day_bucket(frame["prediction_time"]))
    rows: list[dict] = []
    for bucket, group in working.groupby("time_bucket", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "model": name,
                "evaluation_set": evaluation_set,
                "time_bucket": str(bucket),
                **_metrics(artifact, group),
            }
        )
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = list(json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["features"])
    dataset = pd.read_csv(DATASET_PATH, low_memory=False)
    dataset["target_date"] = pd.to_datetime(dataset["target_date"], errors="raise")
    dataset["prediction_time"] = pd.to_datetime(dataset["prediction_time"], errors="raise")
    dataset["_row_id"] = np.arange(len(dataset))
    positive = positive_increase_rows(dataset)
    hourly = _hourly_rows(positive)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    test_start = str(config["splits"]["test_start"])
    date = positive["target_date"].dt.normalize()
    pretest_5m = positive.loc[date < pd.Timestamp(test_start)].copy()
    test_5m = positive.loc[date >= pd.Timestamp(test_start)].copy()
    pretest_hourly = _hourly_rows(pretest_5m)
    test_hourly = _hourly_rows(test_5m)
    folds = expanding_window_splits(pretest_5m, test_start=test_start, minimum_training_years=2)

    fold_rows: list[dict] = []
    for fold in folds:
        train_5m, _ = materialize_fold(pretest_5m, fold)
        train_hourly, validation_hourly = materialize_fold(pretest_hourly, fold)
        for name, training in (
            ("five_minute_control", train_5m),
            ("hourly_challenger", train_hourly),
        ):
            print(
                f"Training {name} {fold['name']}: "
                f"{len(training):,} -> {len(validation_hourly):,} hourly validation rows"
            )
            artifact = train_shifted_poisson_ngboost(
                training,
                features,
                n_estimators=200,
                learning_rate=0.03,
            )
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold["name"],
                    "training_rows": len(training),
                    "validation_frequency": "hourly",
                    **_metrics(artifact, validation_hourly),
                }
            )

    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(OUTPUT_DIR / "validation_fold_metrics.csv", index=False)
    aggregate_rows: list[dict] = []
    for model, group in folds_frame.groupby("model"):
        weights = group["n"].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "model": model,
                "n": int(weights.sum()),
                **{
                    metric: float(np.average(group[metric], weights=weights))
                    for metric in [
                        "interval_nll",
                        "crps",
                        "mean_bucket_brier",
                        "coverage_80",
                        "coverage_90",
                        "mae",
                        "rmse",
                    ]
                },
            }
        )
    validation = pd.DataFrame(aggregate_rows).sort_values("interval_nll")
    validation.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)

    print(f"Refitting hourly challenger on {len(pretest_hourly):,} pre-test hourly rows")
    challenger = train_shifted_poisson_ngboost(
        pretest_hourly,
        features,
        n_estimators=200,
        learning_rate=0.03,
    )
    incumbent = load_conditional_increase_model(INCUMBENT_PATH)
    with CHALLENGER_PATH.open("wb") as handle:
        pickle.dump(challenger, handle)

    test_rows: list[dict] = []
    group_rows: list[dict] = []
    for evaluation_name, evaluation in (
        ("hourly_test", test_hourly),
        ("all_five_minute_test", test_5m),
    ):
        for model_name, artifact in (
            ("five_minute_incumbent", incumbent),
            ("hourly_challenger", challenger),
        ):
            test_rows.append(
                {
                    "model": model_name,
                    "evaluation_set": evaluation_name,
                    **_metrics(artifact, evaluation),
                }
            )
            group_rows.extend(_group_metrics(model_name, artifact, evaluation, evaluation_name))
    test = pd.DataFrame(test_rows)
    test.to_csv(OUTPUT_DIR / "test_comparison.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUTPUT_DIR / "test_comparison_by_time.csv", index=False)

    control_validation = validation.set_index("model").loc["five_minute_control"]
    challenger_validation = validation.set_index("model").loc["hourly_challenger"]
    validation_eligible = bool(
        challenger_validation["interval_nll"] < control_validation["interval_nll"]
        and challenger_validation["mean_bucket_brier"] < control_validation["mean_bucket_brier"]
    )
    indexed_test = test.set_index(["evaluation_set", "model"])
    replicated_on_test = all(
        indexed_test.loc[(evaluation_set, "hourly_challenger"), metric]
        < indexed_test.loc[(evaluation_set, "five_minute_incumbent"), metric]
        for evaluation_set in ["hourly_test", "all_five_minute_test"]
        for metric in ["interval_nll", "mean_bucket_brier"]
    )
    decision = (
        "promote_hourly_challenger"
        if validation_eligible and replicated_on_test
        else "retain_five_minute_incumbent"
    )
    metadata = {
        "status": "validated_challenger_not_promoted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "NGBoost shifted Poisson",
        "feature_count": len(features),
        "training_frequency": "top-of-hour rows only",
        "scoring_compatibility": "may score any five-minute feature row",
        "training_rows": len(pretest_hourly),
        "test_hourly_rows": len(test_hourly),
        "test_all_five_minute_rows": len(test_5m),
        "splits": folds,
        "validation_comparison": validation.to_dict(orient="records"),
        "test_diagnostics_not_used_for_selection": test.to_dict(orient="records"),
        "decision": decision,
        "validation_eligible": validation_eligible,
        "replicated_on_untouched_test": replicated_on_test,
        "replacement_rule": "challenger must improve interval NLL and bucket Brier on validation and replicate both gains on hourly and all-five-minute test diagnostics",
        "dataset_sha256": _sha256(DATASET_PATH),
        "incumbent_sha256": _sha256(INCUMBENT_PATH),
        "challenger_sha256": _sha256(CHALLENGER_PATH),
        "features_sha256": _sha256(FEATURES_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = [
        "# Hourly Conditional NGBoost Challenger",
        "",
        f"Decision: **{decision}**.",
        f"Validation eligible: **{validation_eligible}**; replicated on untouched test: **{replicated_on_test}**.",
        f"The challenger used {len(pretest_hourly):,} top-of-hour positive rows versus {len(pretest_5m):,} five-minute positive rows.",
        "Both validation models were scored on the identical hourly rows.",
        "",
        "## Validation",
        "",
        validation.to_markdown(index=False),
        "",
        "## Untouched test diagnostics",
        "",
        test.to_markdown(index=False),
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(validation.to_string(index=False))
    print(test.to_string(index=False))
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
