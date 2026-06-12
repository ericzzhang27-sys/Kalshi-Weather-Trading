from __future__ import annotations

import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
    distribution_cdf,
    distribution_nll,
    predict_distribution_details,
    train_ngboost_distribution,
)
from src.evaluation import bucket_brier_scores, interval_log_loss  # noqa: E402
from src.features import load_feature_list, validate_feature_columns_exist  # noqa: E402
from src.splits import chronological_train_validation_test_split  # noqa: E402
from src.train_ngboost import build_imputed_feature_frames, load_modeling_table, validate_target_column  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"
FEATURE_LIST_PATH = REPO_ROOT / "outputs" / "final_feature_list.json"
EMPIRICAL_PATH = REPO_ROOT / "outputs" / "day9_empirical_baseline" / "empirical_baseline_predictions.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "robust_laplace_baseline"
MODEL_PATH = REPO_ROOT / "models" / "ngboost_laplace_robust_current36.pkl"

INTERVALS = [
    (None, -3.0),
    (-3.0, -1.0),
    (-1.0, 1.0),
    (1.0, 3.0),
    (3.0, None),
]
PROBABILITY_COLUMNS = [
    "prob_error_le_-3",
    "prob_error_-3_to_-1",
    "prob_error_-1_to_1",
    "prob_error_1_to_3",
    "prob_error_gt_3",
]
INTERVAL_LABELS = [
    "(-inf, -3]",
    "(-3, -1]",
    "(-1, 1]",
    "(1, 3]",
    "(3, inf)",
]

TRAINING_PARAMS = {
    "distribution": "laplace",
    "n_estimators": 120,
    "learning_rate": 0.05,
    "max_depth": 2,
    "min_samples_leaf": 50,
    "minibatch_frac": 1.0,
    "natural_gradient": True,
    "random_state": 11,
    "early_stopping_rounds": None,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_modeling_table(DATASET_PATH)
    validate_target_column(df)
    feature_columns = validate_feature_columns_exist(df, load_feature_list(FEATURE_LIST_PATH))

    split = chronological_train_validation_test_split(
        df,
        train_end_date="2023-12-31",
        validation_end_date="2024-12-31",
    )
    fit_df = pd.concat([split.train, split.validation], ignore_index=True)
    test_df = split.test.reset_index(drop=True)

    X_fit, X_test, _unused, imputer, preprocessing_notes = build_imputed_feature_frames(
        train_df=fit_df,
        validation_df=test_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )
    y_fit = fit_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model = train_ngboost_distribution(
        X_train=X_fit,
        y_train=y_fit,
        distribution=TRAINING_PARAMS["distribution"],
        n_estimators=TRAINING_PARAMS["n_estimators"],
        learning_rate=TRAINING_PARAMS["learning_rate"],
        max_depth=TRAINING_PARAMS["max_depth"],
        min_samples_leaf=TRAINING_PARAMS["min_samples_leaf"],
        minibatch_frac=TRAINING_PARAMS["minibatch_frac"],
        natural_gradient=TRAINING_PARAMS["natural_gradient"],
        random_state=TRAINING_PARAMS["random_state"],
        early_stopping_rounds=TRAINING_PARAMS["early_stopping_rounds"],
    )

    details = predict_distribution_details(model, X_test, distribution="laplace")
    mu = np.asarray(details["mu"], dtype=float)
    sigma = np.asarray(details["sigma"], dtype=float)
    robust_nll = distribution_nll(y_test, mu, sigma, distribution="laplace")

    train_mu = float(np.mean(y_fit))
    train_sigma = float(np.std(y_fit, ddof=1))
    if not math.isfinite(train_sigma) or train_sigma <= 0.0:
        raise ValueError("Train forecast_error standard deviation is not positive")
    normal_mu = np.full(len(test_df), train_mu, dtype=float)
    normal_sigma = np.full(len(test_df), train_sigma, dtype=float)
    normal_nll = distribution_nll(y_test, normal_mu, normal_sigma, distribution="normal")

    labels = pd.Series([_interval_label(value) for value in y_test], name="true_interval")
    robust_probs = _distribution_interval_probs(mu, sigma, "laplace")
    normal_probs = _distribution_interval_probs(normal_mu, normal_sigma, "normal")

    empirical = _load_aligned_empirical(test_df)
    empirical_probs = empirical[PROBABILITY_COLUMNS].reset_index(drop=True)
    empirical_probs.columns = INTERVAL_LABELS
    empirical_normal_probs = empirical[[f"normal_{column}" for column in PROBABILITY_COLUMNS]].copy()
    empirical_normal_probs.columns = INTERVAL_LABELS
    empirical_labels = empirical["true_interval"].reset_index(drop=True)
    if not labels.reset_index(drop=True).equals(empirical_labels):
        raise ValueError("Empirical baseline labels do not align with robust model test labels")

    params = _prediction_frame(test_df, mu, sigma, robust_nll)
    params.to_csv(OUTPUT_DIR / "robust_laplace_test_params.csv", index=False)

    comparison = pd.DataFrame(
        [
            _comparison_row(
                model_name="robust_laplace_ngboost_current36",
                probabilities=robust_probs,
                labels=labels,
                continuous_nll=float(np.mean(robust_nll)),
                extra={"mean_sigma": float(np.mean(sigma)), "mean_mu": float(np.mean(mu))},
            ),
            _comparison_row(
                model_name="constant_normal_train_baseline",
                probabilities=normal_probs,
                labels=labels,
                continuous_nll=float(np.mean(normal_nll)),
                extra={"mean_sigma": train_sigma, "mean_mu": train_mu},
            ),
            _comparison_row(
                model_name="empirical_baseline_day9",
                probabilities=empirical_probs,
                labels=labels,
                continuous_nll=math.nan,
                extra={
                    "mean_sigma": math.nan,
                    "mean_mu": math.nan,
                    "mean_sample_size": float(empirical["sample_size"].mean()),
                },
            ),
            _comparison_row(
                model_name="empirical_local_normal_baseline",
                probabilities=empirical_normal_probs,
                labels=labels,
                continuous_nll=math.nan,
                extra={
                    "mean_sigma": float(empirical["normal_sigma"].mean()),
                    "mean_mu": float(empirical["normal_mu"].mean()),
                    "mean_sample_size": float(empirical["sample_size"].mean()),
                },
            ),
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    _write_report(comparison, split.summary, feature_columns)

    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_columns": feature_columns,
        "target": TARGET_COLUMN,
        "model_name": "robust_laplace_ngboost_current36",
        "distribution_type": "laplace",
        "training_params": TRAINING_PARAMS,
        "split_summary": split.summary,
        "fit_rows": int(len(fit_df)),
        "test_rows": int(len(test_df)),
        "preprocessing_notes": preprocessing_notes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Single fixed robust model. No hyperparameter search, sigma scaling, "
            "or validation-based model selection was performed."
        ),
    }
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(artifact, handle)

    metadata = {key: value for key, value in artifact.items() if key not in {"model", "imputer"}}
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("Trained robust_laplace_ngboost_current36.")
    print(f"Fit rows: {len(fit_df):,}; test rows: {len(test_df):,}; features: {len(feature_columns)}")
    print(comparison.to_string(index=False))
    print(f"Model: {MODEL_PATH}")
    print(f"Comparison: {OUTPUT_DIR / 'comparison.csv'}")


def _distribution_interval_probs(mu: np.ndarray, sigma: np.ndarray, distribution: str) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {}
    for interval, column in zip(INTERVALS, INTERVAL_LABELS, strict=True):
        lower, upper = interval
        if lower is None:
            probs = distribution_cdf(upper, mu, sigma, distribution=distribution)
        elif upper is None:
            probs = 1.0 - distribution_cdf(lower, mu, sigma, distribution=distribution)
        else:
            probs = distribution_cdf(upper, mu, sigma, distribution=distribution) - distribution_cdf(
                lower,
                mu,
                sigma,
                distribution=distribution,
            )
        columns[column] = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    frame = pd.DataFrame(columns)
    row_sums = frame.sum(axis=1)
    frame = frame.div(row_sums, axis=0)
    return frame


def _interval_label(value: float) -> str:
    numeric = float(value)
    for lower, upper in INTERVALS:
        if lower is None and numeric <= float(upper):
            return "(-inf, -3]"
        if upper is None and numeric > float(lower):
            return "(3, inf)"
        if lower is not None and upper is not None and float(lower) < numeric <= float(upper):
            return f"({_format_boundary(lower)}, {_format_boundary(upper)}]"
    raise ValueError(f"No interval found for forecast_error={value!r}")


def _format_boundary(value: float) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def _comparison_row(
    *,
    model_name: str,
    probabilities: pd.DataFrame,
    labels: pd.Series,
    continuous_nll: float,
    extra: dict[str, float],
) -> dict[str, Any]:
    brier = bucket_brier_scores(probabilities, labels)
    true_prob = np.asarray(
        [
            probabilities.iloc[row_index][label]
            for row_index, label in enumerate(labels.reset_index(drop=True))
        ],
        dtype=float,
    )
    top_labels = probabilities.idxmax(axis=1)
    row: dict[str, Any] = {
        "model": model_name,
        "split": "test_2025_plus",
        "n_rows": int(len(probabilities)),
        "continuous_nll": continuous_nll,
        "interval_log_loss": interval_log_loss(probabilities, labels),
        "mean_bucket_brier": float(brier["brier_score"].mean()),
        "top_interval_accuracy": float((top_labels.reset_index(drop=True) == labels.reset_index(drop=True)).mean()),
        "mean_probability_true_interval": float(np.mean(true_prob)),
    }
    row.update(extra)
    return row


def _load_aligned_empirical(test_df: pd.DataFrame) -> pd.DataFrame:
    if not EMPIRICAL_PATH.exists():
        raise FileNotFoundError(
            f"Empirical baseline predictions are missing: {EMPIRICAL_PATH}. "
            "Run `python src/evaluate_empirical_baseline.py` first."
        )
    empirical = pd.read_csv(EMPIRICAL_PATH)
    required = {"date", "station_id", "prediction_hour", "true_interval", *PROBABILITY_COLUMNS}
    missing = sorted(required - set(empirical.columns))
    if missing:
        raise ValueError(f"Empirical baseline output is missing columns: {missing}")

    expected_keys = _key_frame(test_df)
    empirical_keyed = empirical.merge(expected_keys, on=["date", "station_id", "prediction_hour"], how="inner")
    if len(empirical_keyed) != len(expected_keys):
        raise ValueError(
            "Empirical baseline rows do not align with robust model test rows: "
            f"empirical_aligned={len(empirical_keyed)}, expected={len(expected_keys)}"
        )
    return empirical_keyed.sort_values(["date", "station_id", "prediction_hour"]).reset_index(drop=True)


def _key_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="raise").dt.date.astype(str),
            "station_id": df["location"].astype(str) if "location" in df.columns else df["station_id"].astype(str),
            "prediction_hour": pd.to_datetime(df["prediction_time"], errors="raise").dt.hour.astype(int),
        }
    )
    return result.sort_values(["date", "station_id", "prediction_hour"]).reset_index(drop=True)


def _prediction_frame(test_df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, nll: np.ndarray) -> pd.DataFrame:
    frame = test_df[
        [column for column in ["date", "target_date", "location", "prediction_time", "forecast_high", "actual_high", TARGET_COLUMN] if column in test_df.columns]
    ].copy()
    frame["mu"] = mu
    frame["sigma"] = sigma
    frame["nll"] = nll
    frame["distribution_type"] = "laplace"
    return frame


def _write_report(comparison: pd.DataFrame, split_summary: dict[str, Any], feature_columns: list[str]) -> None:
    lines = [
        "# Robust Laplace Baseline Comparison",
        "",
        "Single fixed robust model; no hyperparameter search, sigma scaling, or validation selection.",
        "",
        "## Setup",
        "",
        f"- Feature list: `{FEATURE_LIST_PATH.relative_to(REPO_ROOT)}` ({len(feature_columns)} features)",
        f"- Dataset: `{DATASET_PATH.relative_to(REPO_ROOT)}`",
        f"- Fit split: train + validation through 2024-12-31",
        f"- Test split: {split_summary['splits']['test']['date_min']} to {split_summary['splits']['test']['date_max']}",
        f"- Robust model params: `{json.dumps(TRAINING_PARAMS)}`",
        "",
        "## Metrics",
        "",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Notes",
        "",
        "- `continuous_nll` is only directly meaningful for parametric density models.",
        "- Interval metrics use fixed forecast-error buckets: `(-inf, -3]`, `(-3, -1]`, `(-1, 1]`, `(1, 3]`, `(3, inf)`.",
        "- Empirical baselines are from `outputs/day9_empirical_baseline/empirical_baseline_predictions.csv` regenerated on the current feature table.",
        "",
    ]
    (OUTPUT_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
