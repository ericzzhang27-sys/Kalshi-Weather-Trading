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

from src.conditional_increase_model import (  # noqa: E402
    load_conditional_increase_model,
    positive_increase_rows,
    predict_conditional_distribution,
    train_shifted_poisson_ngboost,
)
from src.hurdle_model import expanding_window_splits, load_hurdle_predictor, materialize_fold  # noqa: E402
from src.ordinal_hazard_model import (  # noqa: E402
    choose_tail_start,
    evaluate_ordinal_probabilities,
    evaluate_threshold_probabilities,
    predict_continuation_probabilities,
    predict_ordinal_probabilities,
    reliability_table,
    threshold_sample_table,
    train_ordinal_hazard_model,
)


DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
EXCEEDANCE_BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
FEATURES_PATH = REPO_ROOT / "models" / "remaining_increase_features.json"
POISSON_PATH = REPO_ROOT / "models" / "remaining_increase_ngboost.pkl"
MODEL_PATH = REPO_ROOT / "models" / "remaining_increase_ordinal_hazard.pkl"
METADATA_PATH = REPO_ROOT / "models" / "remaining_increase_ordinal_hazard_metadata.json"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "remaining_increase" / "ordinal_hazard_challenger"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_exceedance_bundle():
    bundle = json.loads(EXCEEDANCE_BUNDLE_PATH.read_text(encoding="utf-8"))
    if bundle.get("status") not in {"frozen_validated", "frozen_validated_user_override"}:
        raise ValueError("Exceedance bundle is not frozen")
    for label, filename in bundle["paths"].items():
        path = EXCEEDANCE_BUNDLE_PATH.parent / filename
        if _sha256(path) != bundle["sha256"][label]:
            raise ValueError(f"Exceedance bundle {label} hash mismatch")
    if _sha256(DATASET_PATH) != bundle["sha256"]["dataset"]:
        raise ValueError("Conditional and exceedance stages must use the identical dataset")
    predictor = load_hurdle_predictor(
        EXCEEDANCE_BUNDLE_PATH.parent / bundle["paths"]["classifier"],
        EXCEEDANCE_BUNDLE_PATH.parent / bundle["paths"]["features"],
        EXCEEDANCE_BUNDLE_PATH.parent / bundle["paths"]["calibrator"],
        bundle["calibration"],
    )
    return bundle, predictor


def _poisson_probabilities(artifact: dict, frame: pd.DataFrame, tail_start: int) -> np.ndarray:
    distribution = predict_conditional_distribution(artifact, frame)
    exact = np.asarray(
        distribution.dist.pmf(np.arange(tail_start - 1, dtype=int)[:, None]), dtype=float
    ).T
    tail = np.asarray(distribution.dist.sf(tail_start - 2), dtype=float)
    result = np.column_stack([exact, tail])
    result = np.clip(result, 0.0, 1.0)
    result /= result.sum(axis=1, keepdims=True)
    return result


def _full_probability(p_increase: np.ndarray, conditional: np.ndarray) -> np.ndarray:
    p = np.asarray(p_increase, dtype=float).reshape(-1)
    result = np.column_stack([1.0 - p, p[:, None] * conditional])
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Full hurdle probabilities are invalid")
    return np.clip(result, 0.0, 1.0)


def _full_metrics(probability: np.ndarray, target: np.ndarray, tail_start: int) -> dict[str, float | int]:
    y = np.asarray(target, dtype=int)
    category = np.minimum(y, tail_start)
    p = np.asarray(probability, dtype=float)
    if p.shape != (len(y), tail_start + 1):
        raise ValueError("Full probability matrix has the wrong shape")
    realized = np.clip(p[np.arange(len(y)), category], 1e-15, 1.0)
    one_hot = np.eye(tail_start + 1)[category]
    cdf = np.cumsum(p[:, :-1], axis=1)
    actual_cdf = y[:, None] <= np.arange(tail_start)[None, :]
    return {
        "n": int(len(y)),
        "nll": float(-np.mean(np.log(realized))),
        "mean_bucket_brier": float(np.mean((p - one_hot) ** 2)),
        "crps": float(np.mean(np.sum((cdf - actual_cdf) ** 2, axis=1))),
        "cdf_calibration_error": float(np.mean(np.abs(cdf.mean(axis=0) - actual_cdf.mean(axis=0)))),
    }


def _cdf_table(model: str, probability: np.ndarray, target: np.ndarray, first_value: int) -> pd.DataFrame:
    cdf = np.cumsum(probability[:, :-1], axis=1)
    thresholds = np.arange(first_value, first_value + cdf.shape[1])
    actual = np.asarray(target)[:, None] <= thresholds[None, :]
    return pd.DataFrame(
        {
            "model": model,
            "threshold": thresholds,
            "mean_predicted_cdf": cdf.mean(axis=0),
            "empirical_cdf": actual.mean(axis=0),
            "calibration_gap": cdf.mean(axis=0) - actual.mean(axis=0),
        }
    )


def _individual_losses(probability: np.ndarray, target: np.ndarray, tail_start: int) -> tuple[np.ndarray, np.ndarray]:
    category = np.minimum(np.asarray(target, dtype=int), tail_start)
    realized = np.clip(probability[np.arange(len(category)), category], 1e-15, 1.0)
    one_hot = np.eye(tail_start + 1)[category]
    return -np.log(realized), np.mean((probability - one_hot) ** 2, axis=1)


def _plot_reliability(table: pd.DataFrame, threshold: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="perfect")
    ax.plot(table["mean_probability"], table["observed_frequency"], marker="o", label="NGBoost")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"q{threshold}: P(delta >= {threshold + 1} | delta >= {threshold})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"threshold_{threshold:02d}_reliability.png", dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare discrete NGBoost continuation models to shifted Poisson")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-train-at-risk", type=int, default=500)
    parser.add_argument("--min-train-per-class", type=int, default=100)
    parser.add_argument("--min-validation-at-risk", type=int, default=100)
    parser.add_argument("--min-validation-per-class", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exceedance_bundle, exceedance = _verify_exceedance_bundle()
    features = list(json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["features"])
    dataset = pd.read_csv(DATASET_PATH, low_memory=False)
    dataset["target_date"] = pd.to_datetime(dataset["target_date"], errors="raise").dt.normalize()
    dataset["prediction_time"] = pd.to_datetime(dataset["prediction_time"], errors="raise")
    positive = positive_increase_rows(dataset)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validation_start = pd.Timestamp(str(config["splits"]["validation_start"]))
    test_start = pd.Timestamp(str(config["splits"]["test_start"]))
    if validation_start >= test_start:
        raise ValueError("Configured validation must precede test")
    pretest = positive.loc[positive["target_date"] < test_start].copy()
    positive_test = positive.loc[positive["target_date"] >= test_start].copy()
    full_test = dataset.loc[dataset["target_date"] >= test_start].copy()
    fold_specs = expanding_window_splits(pretest, test_start=str(test_start.date()), minimum_training_years=2)
    materialized = [(fold["name"], *materialize_fold(pretest, fold)) for fold in fold_specs]
    if any(set(train["target_date"]) & set(validation["target_date"]) for _, train, validation in materialized):
        raise AssertionError("A date crossed a train/validation boundary")

    tail_start, cutoff_diagnostics = choose_tail_start(
        materialized,
        min_train_at_risk=args.min_train_at_risk,
        min_train_per_class=args.min_train_per_class,
        min_validation_at_risk=args.min_validation_at_risk,
        min_validation_per_class=args.min_validation_per_class,
    )
    cutoff_diagnostics.to_csv(OUTPUT_DIR / "tail_cutoff_diagnostics.csv", index=False)
    threshold_sample_table(pretest).to_csv(OUTPUT_DIR / "pretest_delta_threshold_counts.csv", index=False)
    positive["remaining_increase"].value_counts().sort_index().rename_axis("delta").reset_index(name="count").to_csv(
        OUTPUT_DIR / "delta_exact_counts.csv", index=False
    )
    print(f"Selected conditional support: 1..{tail_start - 1}, {tail_start}+")

    fold_metric_rows: list[dict] = []
    threshold_metric_rows: list[dict] = []
    oof_threshold: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        threshold: [] for threshold in range(1, tail_start)
    }
    oof_distribution: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "shifted_poisson": [], "discrete_hazard_ngboost": []
    }
    for fold_name, train, validation in materialized:
        print(f"Training {fold_name}: {len(train):,} positive rows, {tail_start - 1} classifiers")
        discrete = train_ordinal_hazard_model(
            train,
            features,
            tail_start=tail_start,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
        )
        poisson = train_shifted_poisson_ngboost(
            train, features, n_estimators=args.n_estimators, learning_rate=args.learning_rate
        )
        target = validation["remaining_increase"].to_numpy(dtype=int)
        discrete_probability = predict_ordinal_probabilities(discrete, validation)
        poisson_probability = _poisson_probabilities(poisson, validation, tail_start)
        for model_name, probability in (
            ("shifted_poisson", poisson_probability),
            ("discrete_hazard_ngboost", discrete_probability),
        ):
            metrics = evaluate_ordinal_probabilities(probability, target, tail_start)
            fold_metric_rows.append({"model": model_name, "fold": fold_name, **metrics})
            oof_distribution[model_name].append((probability, target))

        continuation = predict_continuation_probabilities(discrete, validation)
        for index, threshold_model in enumerate(discrete["models"]):
            threshold = int(threshold_model["threshold"])
            at_risk = target >= threshold
            outcome = target[at_risk] >= threshold + 1
            predicted = continuation[at_risk, index]
            metrics = evaluate_threshold_probabilities(outcome, predicted, threshold_model["prevalence"])
            threshold_metric_rows.append(
                {
                    "fold": fold_name,
                    "threshold": threshold,
                    "n_train": threshold_model["n_train"],
                    "train_prevalence": threshold_model["prevalence"],
                    **metrics,
                }
            )
            oof_threshold[threshold].append((outcome.astype(int), predicted))

    pd.DataFrame(fold_metric_rows).to_csv(OUTPUT_DIR / "validation_fold_metrics.csv", index=False)
    validation_rows = []
    for model_name, parts in oof_distribution.items():
        probability = np.vstack([part[0] for part in parts])
        target = np.concatenate([part[1] for part in parts])
        validation_rows.append(
            {"model": model_name, **evaluate_ordinal_probabilities(probability, target, tail_start)}
        )
    validation_comparison = pd.DataFrame(validation_rows)
    validation_comparison.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)

    threshold_fold_metrics = pd.DataFrame(threshold_metric_rows)
    threshold_fold_metrics.to_csv(OUTPUT_DIR / "threshold_validation_metrics_by_fold.csv", index=False)
    pooled_threshold_rows = []
    reliability_parts = []
    for threshold, parts in oof_threshold.items():
        outcome = np.concatenate([part[0] for part in parts])
        predicted = np.concatenate([part[1] for part in parts])
        fold_rows = threshold_fold_metrics.loc[threshold_fold_metrics["threshold"] == threshold]
        reference = float(np.average(fold_rows["train_prevalence"], weights=fold_rows["n"]))
        pooled_threshold_rows.append(
            {
                "threshold": threshold,
                "n_train_min_fold": int(fold_rows["n_train"].min()),
                "train_prevalence_weighted": reference,
                **evaluate_threshold_probabilities(outcome, predicted, reference),
            }
        )
        reliability = reliability_table(outcome, predicted)
        reliability["threshold"] = threshold
        reliability_parts.append(reliability)
        _plot_reliability(reliability, threshold, OUTPUT_DIR)
    threshold_summary = pd.DataFrame(pooled_threshold_rows)
    threshold_summary.to_csv(OUTPUT_DIR / "threshold_validation_metrics.csv", index=False)
    pd.concat(reliability_parts, ignore_index=True).to_csv(
        OUTPUT_DIR / "threshold_reliability.csv", index=False
    )

    print("Refitting discrete continuation model on all pre-test positive rows")
    final_discrete = train_ordinal_hazard_model(
        pretest,
        features,
        tail_start=tail_start,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
    )
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_discrete, handle)
    incumbent_poisson = load_conditional_increase_model(POISSON_PATH)

    conditional_probabilities = {
        "shifted_poisson": _poisson_probabilities(incumbent_poisson, positive_test, tail_start),
        "discrete_hazard_ngboost": predict_ordinal_probabilities(final_discrete, positive_test),
    }
    conditional_rows = []
    conditional_group_rows = []
    conditional_cdf_parts = []
    positive_target = positive_test["remaining_increase"].to_numpy(dtype=int)
    size_group = np.select(
        [positive_target == 1, positive_target == 2, positive_target == 3],
        ["delta=1", "delta=2", "delta=3"],
        default="delta>=4",
    )
    for model_name, probability in conditional_probabilities.items():
        conditional_rows.append(
            {"model": model_name, **evaluate_ordinal_probabilities(probability, positive_target, tail_start)}
        )
        conditional_cdf_parts.append(_cdf_table(model_name, probability, positive_target, 1))
        for group in ["delta=1", "delta=2", "delta=3", "delta>=4"]:
            mask = size_group == group
            conditional_group_rows.append(
                {
                    "model": model_name,
                    "realized_group": group,
                    **evaluate_ordinal_probabilities(probability[mask], positive_target[mask], tail_start),
                }
            )
    conditional_test = pd.DataFrame(conditional_rows)
    conditional_test.to_csv(OUTPUT_DIR / "conditional_test_diagnostics.csv", index=False)
    pd.DataFrame(conditional_group_rows).to_csv(OUTPUT_DIR / "conditional_test_by_realized_delta.csv", index=False)
    pd.concat(conditional_cdf_parts, ignore_index=True).to_csv(OUTPUT_DIR / "conditional_test_cdf_calibration.csv", index=False)

    final_continuation = predict_continuation_probabilities(final_discrete, positive_test)
    threshold_test_rows = []
    for index, threshold_model in enumerate(final_discrete["models"]):
        threshold = int(threshold_model["threshold"])
        at_risk = positive_target >= threshold
        outcome = positive_target[at_risk] >= threshold + 1
        threshold_test_rows.append(
            {
                "threshold": threshold,
                "n_train": threshold_model["n_train"],
                "train_prevalence": threshold_model["prevalence"],
                **evaluate_threshold_probabilities(
                    outcome, final_continuation[at_risk, index], threshold_model["prevalence"]
                ),
            }
        )
    pd.DataFrame(threshold_test_rows).to_csv(OUTPUT_DIR / "threshold_test_diagnostics.csv", index=False)

    p_increase = exceedance.predict_proba(full_test)
    full_probabilities = {
        "shifted_poisson_hurdle": _full_probability(
            p_increase, _poisson_probabilities(incumbent_poisson, full_test, tail_start)
        ),
        "discrete_hazard_ngboost_hurdle": _full_probability(
            p_increase, predict_ordinal_probabilities(final_discrete, full_test)
        ),
    }
    full_target = full_test["remaining_increase"].to_numpy(dtype=int)
    late = (full_test["prediction_time"].dt.hour >= 16).to_numpy()
    full_rows = []
    full_cdf_parts = []
    paired_parts = []
    for model_name, probability in full_probabilities.items():
        full_rows.append(
            {
                "model": model_name,
                **_full_metrics(probability, full_target, tail_start),
                "after_4pm_nll": _full_metrics(probability[late], full_target[late], tail_start)["nll"],
                "after_4pm_brier": _full_metrics(probability[late], full_target[late], tail_start)["mean_bucket_brier"],
            }
        )
        full_cdf_parts.append(_cdf_table(model_name, probability, full_target, 0))
        nll, brier = _individual_losses(probability, full_target, tail_start)
        paired_parts.append(
            pd.DataFrame(
                {"target_date": full_test["target_date"], "model": model_name, "nll": nll, "bucket_brier": brier}
            )
            .groupby(["target_date", "model"], as_index=False)
            .mean()
        )
    full_comparison = pd.DataFrame(full_rows)
    full_comparison.to_csv(OUTPUT_DIR / "full_hurdle_test_diagnostics.csv", index=False)
    pd.concat(full_cdf_parts, ignore_index=True).to_csv(OUTPUT_DIR / "full_hurdle_test_cdf_calibration.csv", index=False)
    paired = pd.concat(paired_parts).pivot(index="target_date", columns="model", values=["nll", "bucket_brier"])
    paired.columns = [f"{metric}_{model}" for metric, model in paired.columns]
    paired = paired.reset_index()
    for metric in ["nll", "bucket_brier"]:
        paired[f"{metric}_discrete_minus_poisson"] = (
            paired[f"{metric}_discrete_hazard_ngboost_hurdle"]
            - paired[f"{metric}_shifted_poisson_hurdle"]
        )
    paired.to_csv(OUTPUT_DIR / "paired_test_performance_by_day.csv", index=False)

    validation_index = validation_comparison.set_index("model")
    full_index = full_comparison.set_index("model")
    challenger = full_index.loc["discrete_hazard_ngboost_hurdle"]
    incumbent = full_index.loc["shifted_poisson_hurdle"]
    threshold_stable = bool(
        np.isfinite(threshold_summary[["brier", "brier_skill_score", "log_loss"]]).all().all()
        and (threshold_summary["brier_skill_score"] > -0.05).all()
        and (threshold_summary["calibration_gap"].abs() <= 0.10).all()
    )
    validation_win = bool(
        validation_index.loc["discrete_hazard_ngboost", "interval_nll"]
        < validation_index.loc["shifted_poisson", "interval_nll"]
        and validation_index.loc["discrete_hazard_ngboost", "mean_bucket_brier"]
        < validation_index.loc["shifted_poisson", "mean_bucket_brier"]
    )
    acceptance = {
        "validation_nll_and_brier_improve": validation_win,
        "full_test_nll_and_brier_improve": bool(
            challenger["nll"] < incumbent["nll"]
            and challenger["mean_bucket_brier"] < incumbent["mean_bucket_brier"]
        ),
        "calibration_preserved": bool(
            challenger["cdf_calibration_error"] <= incumbent["cdf_calibration_error"] + 0.005
        ),
        "late_day_preserved": bool(
            challenger["after_4pm_nll"] <= incumbent["after_4pm_nll"]
            and challenger["after_4pm_brier"] <= incumbent["after_4pm_brier"]
        ),
        "higher_thresholds_stable": threshold_stable,
    }
    accepted = all(acceptance.values())
    decision = "challenger_supported_not_promoted" if accepted else "retain_shifted_poisson"
    metadata = {
        "status": "validated_challenger_not_promoted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "promotion": "none",
        "model_type": "separate Bernoulli NGBoost continuation classifiers",
        "probability_definition": "q_k=P(delta>=k+1|delta>=k,X)",
        "tail_start": tail_start,
        "categories": [*range(1, tail_start), f">={tail_start}"],
        "tail_selection": vars(args),
        "splits": {"validation_start": str(validation_start.date()), "test_start": str(test_start.date()), "folds": fold_specs},
        "feature_count": len(features),
        "features": features,
        "acceptance_checks": acceptance,
        "test_period_previously_inspected": True,
        "exceedance_model_modified": False,
        "exceedance_winner": exceedance_bundle["winner"],
        "exceedance_bundle_sha256": _sha256(EXCEEDANCE_BUNDLE_PATH),
        "dataset_sha256": _sha256(DATASET_PATH),
        "model_sha256": _sha256(MODEL_PATH),
        "poisson_model_sha256": _sha256(POISSON_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = [
        "# Discrete Hazard / Ordinal Conditional Challenger",
        "",
        f"Decision: **{decision}**. Production and the frozen exceedance model were not changed.",
        f"The pre-test support rule selected exact outcomes 1 through {tail_start - 1} and a `{tail_start}+` tail.",
        "The 2025+ test period is diagnostic because it was inspected by earlier experiments.",
        "",
        "## Acceptance checks",
        "",
        *[f"- {name}: **{value}**" for name, value in acceptance.items()],
        "",
        "## Pre-test validation",
        "",
        validation_comparison.to_markdown(index=False),
        "",
        "## Positive-only held-out test",
        "",
        conditional_test.to_markdown(index=False),
        "",
        "## Full held-out hurdle test",
        "",
        full_comparison.to_markdown(index=False),
        "",
        "## Threshold validation summary",
        "",
        threshold_summary.to_markdown(index=False),
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(validation_comparison.to_string(index=False))
    print(conditional_test.to_string(index=False))
    print(full_comparison.to_string(index=False))
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
