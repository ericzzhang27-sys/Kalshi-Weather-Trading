from __future__ import annotations

import argparse
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

from src.conditional_increase_model import (  # noqa: E402
    conditional_dispersion,
    load_conditional_increase_model,
    positive_increase_rows,
    predict_conditional_distribution,
    train_shifted_negative_binomial_ngboost,
    train_shifted_poisson_ngboost,
)
from src.hurdle_comparison import time_of_day_bucket  # noqa: E402
from src.hurdle_model import expanding_window_splits, load_hurdle_predictor, materialize_fold  # noqa: E402


DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
EXCEEDANCE_BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
FEATURES_PATH = REPO_ROOT / "models" / "remaining_increase_features.json"
POISSON_PATH = REPO_ROOT / "models" / "remaining_increase_ngboost.pkl"
MODEL_PATH = REPO_ROOT / "models" / "remaining_increase_negative_binomial.pkl"
METADATA_PATH = REPO_ROOT / "models" / "remaining_increase_negative_binomial_metadata.json"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "remaining_increase" / "negative_binomial_challenger"
MAX_DELTA = 10
MATERIAL_DISPERSION = 1.25


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


def _state_groups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    gap = -pd.to_numeric(frame["current_temp_minus_max_so_far"], errors="coerce")
    minutes = pd.to_numeric(frame["minutes_since_max_temp_so_far"], errors="coerce")
    trajectory = pd.to_numeric(frame["temp_change_60m"], errors="coerce")
    forecast_gap = pd.to_numeric(frame["forecast_gap"], errors="coerce")
    return {
        "time_of_day": pd.Series(time_of_day_bucket(frame["prediction_time"]), index=frame.index),
        "temperature_gap": pd.cut(
            gap,
            [-np.inf, 0.0, 0.5, 1.0, 2.0, np.inf],
            labels=["at max", "0-0.5F below", "0.5-1F below", "1-2F below", ">2F below"],
        ),
        "minutes_since_max": pd.cut(
            minutes,
            [-np.inf, 10, 30, 60, 120, np.inf],
            labels=["0-10 min", "10-30 min", "30-60 min", "60-120 min", ">120 min"],
        ),
        "trajectory_60m": pd.cut(
            trajectory,
            [-np.inf, -2.0, -0.5, 0.5, 2.0, np.inf],
            labels=["strongly falling", "falling", "flat", "rising", "strongly rising"],
        ).astype(object).where(trajectory.notna(), "missing"),
        "forecast_gap": pd.cut(
            forecast_gap,
            [-np.inf, -1.0, 0.0, 1.0, 2.0, np.inf],
            labels=["<=-1F", "-1 to 0F", "0 to 1F", "1 to 2F", ">2F"],
        ),
    }


def _dispersion_report(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    y = pd.to_numeric(frame["remaining_increase"], errors="raise") - 1.0
    rows = [{"split": split, "dimension": "overall", "group": "all", **conditional_dispersion(y)}]
    for dimension, groups in _state_groups(frame).items():
        working = pd.DataFrame({"group": groups, "y": y}).dropna(subset=["group"])
        for group, values in working.groupby("group", observed=True)["y"]:
            rows.append(
                {"split": split, "dimension": dimension, "group": str(group), **conditional_dispersion(values)}
            )
    return pd.DataFrame(rows)


def _integer_probabilities(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    distribution = predict_conditional_distribution(artifact, frame)
    exact = np.asarray(distribution.dist.pmf(np.arange(MAX_DELTA, dtype=int)[:, None]), dtype=float).T
    tail = np.asarray(distribution.dist.sf(MAX_DELTA - 1), dtype=float)
    probability = np.column_stack([exact, tail])
    probability = np.clip(probability, 0.0, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def _conditional_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=int)
    category = np.where(y > MAX_DELTA, MAX_DELTA, y - 1)
    p = np.asarray(probability, dtype=float)
    realized = np.clip(p[np.arange(len(y)), category], 1e-15, 1.0)
    one_hot = np.eye(MAX_DELTA + 1)[category]
    cdf = np.cumsum(p[:, :-1], axis=1)
    actual_cdf = y[:, None] <= np.arange(1, MAX_DELTA + 1)[None, :]
    return {
        "n": int(len(y)),
        "interval_nll": float(-np.mean(np.log(realized))),
        "mean_bucket_brier": float(np.mean((p - one_hot) ** 2)),
        "crps": float(np.mean(np.sum((cdf - actual_cdf) ** 2, axis=1))),
        "cdf_calibration_error": float(np.mean(np.abs(cdf.mean(axis=0) - actual_cdf.mean(axis=0)))),
    }


def _full_probability(p_increase: np.ndarray, conditional: np.ndarray) -> np.ndarray:
    p = np.asarray(p_increase, dtype=float).reshape(-1)
    result = np.column_stack([1.0 - p, p[:, None] * conditional])
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Full hurdle probabilities are invalid")
    return np.clip(result, 0.0, 1.0)


def _full_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=int)
    category = np.where(y > MAX_DELTA, MAX_DELTA + 1, y)
    p = np.asarray(probability, dtype=float)
    realized = np.clip(p[np.arange(len(y)), category], 1e-15, 1.0)
    one_hot = np.eye(MAX_DELTA + 2)[category]
    cdf = np.cumsum(p[:, :-1], axis=1)
    actual_cdf = y[:, None] <= np.arange(MAX_DELTA + 1)[None, :]
    return {
        "n": int(len(y)),
        "nll": float(-np.mean(np.log(realized))),
        "mean_bucket_brier": float(np.mean((p - one_hot) ** 2)),
        "crps": float(np.mean(np.sum((cdf - actual_cdf) ** 2, axis=1))),
        "cdf_calibration_error": float(np.mean(np.abs(cdf.mean(axis=0) - actual_cdf.mean(axis=0)))),
    }


def _cdf_table(model: str, probability: np.ndarray, target: np.ndarray, first: int) -> pd.DataFrame:
    cdf = np.cumsum(probability[:, :-1], axis=1)
    thresholds = np.arange(first, first + cdf.shape[1])
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


def _individual_full_losses(probability: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=int)
    category = np.where(y > MAX_DELTA, MAX_DELTA + 1, y)
    realized = np.clip(probability[np.arange(len(y)), category], 1e-15, 1.0)
    one_hot = np.eye(MAX_DELTA + 2)[category]
    return -np.log(realized), np.mean((probability - one_hot) ** 2, axis=1)


def _alpha_summary(artifact: dict, frame: pd.DataFrame) -> pd.DataFrame:
    distribution = predict_conditional_distribution(artifact, frame)
    alpha = np.asarray(distribution.params["alpha"], dtype=float)
    mu = np.asarray(distribution.params["mu"], dtype=float)
    working = pd.DataFrame(
        {
            "alpha": alpha,
            "mu": mu,
            "implied_variance": mu + alpha * mu**2,
            "time_bucket": time_of_day_bucket(frame["prediction_time"]),
        }
    )
    rows = []
    for group, part in [("overall", working), *working.groupby("time_bucket", observed=True)]:
        rows.append(
            {
                "group": str(group),
                "n": len(part),
                "mean_mu": float(part["mu"].mean()),
                "mean_alpha": float(part["alpha"].mean()),
                "median_alpha": float(part["alpha"].median()),
                "p05_alpha": float(part["alpha"].quantile(0.05)),
                "p95_alpha": float(part["alpha"].quantile(0.95)),
                "mean_implied_variance": float(part["implied_variance"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare shifted NB2 against shifted Poisson")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
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
    test_start = pd.Timestamp(str(config["splits"]["test_start"]))
    pretest = positive.loc[positive["target_date"] < test_start].copy()
    positive_test = positive.loc[positive["target_date"] >= test_start].copy()
    full_test = dataset.loc[dataset["target_date"] >= test_start].copy()

    dispersion = pd.concat(
        [
            _dispersion_report(positive, "all_available_diagnostic"),
            _dispersion_report(pretest, "pretest"),
            _dispersion_report(positive_test, "heldout_test_diagnostic"),
        ],
        ignore_index=True,
    )
    dispersion.to_csv(OUTPUT_DIR / "conditional_dispersion.csv", index=False)
    pretest_dispersion = dispersion.loc[
        (dispersion["split"] == "pretest") & (dispersion["dimension"] == "overall"), "dispersion"
    ].iloc[0]
    stable_groups = dispersion.loc[
        (dispersion["split"] == "pretest")
        & (dispersion["dimension"] != "overall")
        & (dispersion["n"] >= 100)
    ]
    overdispersed_group_share = float((stable_groups["dispersion"] > 1.0).mean())
    materially_overdispersed = bool(
        pretest_dispersion >= MATERIAL_DISPERSION and overdispersed_group_share >= 0.75
    )
    dispersion_decision = {
        "pretest_dispersion": float(pretest_dispersion),
        "material_threshold": MATERIAL_DISPERSION,
        "eligible_state_time_groups": int(len(stable_groups)),
        "overdispersed_group_share": overdispersed_group_share,
        "fit_negative_binomial": materially_overdispersed,
    }
    (OUTPUT_DIR / "dispersion_decision.json").write_text(
        json.dumps(dispersion_decision, indent=2), encoding="utf-8"
    )
    if not materially_overdispersed:
        raise SystemExit("Y is not materially and consistently overdispersed; Negative Binomial was not fit")
    print(f"Pre-test dispersion={pretest_dispersion:.3f}; fitting shifted NB2")

    fold_specs = expanding_window_splits(pretest, test_start=str(test_start.date()), minimum_training_years=2)
    fold_rows = []
    oof: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "shifted_poisson": [], "shifted_negative_binomial": []
    }
    for fold in fold_specs:
        train, validation = materialize_fold(pretest, fold)
        print(f"Training {fold['name']}: {len(train):,} -> {len(validation):,}")
        poisson = train_shifted_poisson_ngboost(
            train, features, n_estimators=args.n_estimators, learning_rate=args.learning_rate
        )
        negative_binomial = train_shifted_negative_binomial_ngboost(
            train, features, n_estimators=args.n_estimators, learning_rate=args.learning_rate
        )
        target = validation["remaining_increase"].to_numpy(dtype=int)
        for model_name, artifact in (
            ("shifted_poisson", poisson),
            ("shifted_negative_binomial", negative_binomial),
        ):
            probability = _integer_probabilities(artifact, validation)
            fold_rows.append({"model": model_name, "fold": fold["name"], **_conditional_metrics(probability, target)})
            oof[model_name].append((probability, target))
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "validation_fold_metrics.csv", index=False)
    validation_rows = []
    for model_name, parts in oof.items():
        probability = np.vstack([part[0] for part in parts])
        target = np.concatenate([part[1] for part in parts])
        validation_rows.append({"model": model_name, **_conditional_metrics(probability, target)})
    validation_comparison = pd.DataFrame(validation_rows)
    validation_comparison.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)

    print("Refitting shifted NB2 on all pre-test positive rows")
    final_nb = train_shifted_negative_binomial_ngboost(
        pretest, features, n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(final_nb, handle)
    incumbent_poisson = load_conditional_increase_model(POISSON_PATH)
    positive_target = positive_test["remaining_increase"].to_numpy(dtype=int)
    conditional_probabilities = {
        "shifted_poisson": _integer_probabilities(incumbent_poisson, positive_test),
        "shifted_negative_binomial": _integer_probabilities(final_nb, positive_test),
    }
    conditional_rows = []
    conditional_cdf = []
    delta_group_rows = []
    delta_group = np.select(
        [positive_target == 1, positive_target == 2, positive_target == 3],
        ["delta=1", "delta=2", "delta=3"],
        default="delta>=4",
    )
    for model_name, probability in conditional_probabilities.items():
        conditional_rows.append({"model": model_name, **_conditional_metrics(probability, positive_target)})
        conditional_cdf.append(_cdf_table(model_name, probability, positive_target, 1))
        for group in ["delta=1", "delta=2", "delta=3", "delta>=4"]:
            mask = delta_group == group
            delta_group_rows.append(
                {"model": model_name, "realized_group": group, **_conditional_metrics(probability[mask], positive_target[mask])}
            )
    conditional_test = pd.DataFrame(conditional_rows)
    conditional_test.to_csv(OUTPUT_DIR / "conditional_test_diagnostics.csv", index=False)
    pd.concat(conditional_cdf, ignore_index=True).to_csv(OUTPUT_DIR / "conditional_test_cdf_calibration.csv", index=False)
    pd.DataFrame(delta_group_rows).to_csv(OUTPUT_DIR / "conditional_test_by_realized_delta.csv", index=False)
    _alpha_summary(final_nb, positive_test).to_csv(OUTPUT_DIR / "negative_binomial_alpha_test.csv", index=False)

    p_increase = exceedance.predict_proba(full_test)
    full_probabilities = {
        "shifted_poisson_hurdle": _full_probability(
            p_increase, _integer_probabilities(incumbent_poisson, full_test)
        ),
        "shifted_negative_binomial_hurdle": _full_probability(
            p_increase, _integer_probabilities(final_nb, full_test)
        ),
    }
    full_target = full_test["remaining_increase"].to_numpy(dtype=int)
    late = (full_test["prediction_time"].dt.hour >= 16).to_numpy()
    full_rows = []
    full_cdf = []
    paired_parts = []
    for model_name, probability in full_probabilities.items():
        late_metrics = _full_metrics(probability[late], full_target[late])
        full_rows.append(
            {
                "model": model_name,
                **_full_metrics(probability, full_target),
                "after_4pm_nll": late_metrics["nll"],
                "after_4pm_brier": late_metrics["mean_bucket_brier"],
                "after_4pm_crps": late_metrics["crps"],
                "after_4pm_cdf_calibration_error": late_metrics["cdf_calibration_error"],
            }
        )
        full_cdf.append(_cdf_table(model_name, probability, full_target, 0))
        nll, brier = _individual_full_losses(probability, full_target)
        paired_parts.append(
            pd.DataFrame(
                {"target_date": full_test["target_date"], "model": model_name, "nll": nll, "bucket_brier": brier}
            ).groupby(["target_date", "model"], as_index=False).mean()
        )
    full_comparison = pd.DataFrame(full_rows)
    full_comparison.to_csv(OUTPUT_DIR / "full_hurdle_test_diagnostics.csv", index=False)
    pd.concat(full_cdf, ignore_index=True).to_csv(OUTPUT_DIR / "full_hurdle_test_cdf_calibration.csv", index=False)
    paired = pd.concat(paired_parts).pivot(index="target_date", columns="model", values=["nll", "bucket_brier"])
    paired.columns = [f"{metric}_{model}" for metric, model in paired.columns]
    paired = paired.reset_index()
    for metric in ["nll", "bucket_brier"]:
        paired[f"{metric}_nb_minus_poisson"] = (
            paired[f"{metric}_shifted_negative_binomial_hurdle"]
            - paired[f"{metric}_shifted_poisson_hurdle"]
        )
    paired.to_csv(OUTPUT_DIR / "paired_test_performance_by_day.csv", index=False)

    validation_index = validation_comparison.set_index("model")
    conditional_index = conditional_test.set_index("model")
    full_index = full_comparison.set_index("model")
    validation_nb = validation_index.loc["shifted_negative_binomial"]
    validation_poisson = validation_index.loc["shifted_poisson"]
    test_nb = conditional_index.loc["shifted_negative_binomial"]
    test_poisson = conditional_index.loc["shifted_poisson"]
    full_nb = full_index.loc["shifted_negative_binomial_hurdle"]
    full_poisson = full_index.loc["shifted_poisson_hurdle"]
    acceptance = {
        "validation_nll_brier_crps_improve": bool(
            validation_nb["interval_nll"] < validation_poisson["interval_nll"]
            and validation_nb["mean_bucket_brier"] < validation_poisson["mean_bucket_brier"]
            and validation_nb["crps"] < validation_poisson["crps"]
        ),
        "positive_test_nll_and_brier_improve": bool(
            test_nb["interval_nll"] < test_poisson["interval_nll"]
            and test_nb["mean_bucket_brier"] < test_poisson["mean_bucket_brier"]
        ),
        "full_test_nll_and_brier_improve": bool(
            full_nb["nll"] < full_poisson["nll"]
            and full_nb["mean_bucket_brier"] < full_poisson["mean_bucket_brier"]
        ),
        "calibration_preserved": bool(
            full_nb["cdf_calibration_error"] <= full_poisson["cdf_calibration_error"] + 0.005
        ),
        "late_day_preserved": bool(
            full_nb["after_4pm_nll"] <= full_poisson["after_4pm_nll"]
            and full_nb["after_4pm_brier"] <= full_poisson["after_4pm_brier"]
        ),
    }
    accepted = all(acceptance.values())
    decision = "negative_binomial_supported_not_promoted" if accepted else "retain_shifted_poisson"
    metadata = {
        "status": "validated_challenger_not_promoted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "promotion": "none",
        "distribution": "shifted NB2 for Y=delta-1",
        "variance": "mu + alpha * mu^2",
        "ngboost_natural_gradient": False,
        "reason_natural_gradient_disabled": "custom NB2 uses analytic ordinary score gradients and identity metric",
        "dispersion_decision": dispersion_decision,
        "hyperparameters": vars(args),
        "max_exact_delta": MAX_DELTA,
        "features": features,
        "folds": fold_specs,
        "test_start": str(test_start.date()),
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
        "# Shifted Negative Binomial Conditional Challenger",
        "",
        f"Decision: **{decision}**. Production and the frozen exceedance model were not changed.",
        f"Pre-test Y dispersion was **{pretest_dispersion:.3f}**; {overdispersed_group_share:.1%} of eligible time/state groups had D>1.",
        "The 2025+ period is diagnostic because it was inspected by prior experiments.",
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
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(validation_comparison.to_string(index=False))
    print(conditional_test.to_string(index=False))
    print(full_comparison.to_string(index=False))
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
