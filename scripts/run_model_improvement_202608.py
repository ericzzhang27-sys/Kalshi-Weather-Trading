from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_ngboost import (  # noqa: E402
    assign_error_interval_labels,
    error_interval_probabilities,
)
from scripts.search_ngboost_model_space import append_features, stable_run_id  # noqa: E402
from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
    distribution_nll,
    normalize_distribution_name,
    predict_distribution_details,
    train_ngboost_distribution,
)
from src.evaluation import bucket_brier_scores, interval_log_loss  # noqa: E402
from src.features import load_feature_list  # noqa: E402
from src.splits import chronological_train_validation_test_split  # noqa: E402
from src.train_ngboost import (  # noqa: E402
    DEFAULT_FINAL_FEATURE_LIST_PATH,
    DEFAULT_MODELING_TABLE_PATH,
    build_imputed_feature_frames,
    load_modeling_table,
    validate_target_column,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_FEATURE_LIST_PATH = REPO_ROOT / "outputs" / "day8_features" / "feature_columns.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_improvement_202608"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_improved_202608.pkl"
METADATA_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_improved_202608_metadata.json"

CALIB_SLICE_START = "2023-07-01"
GLOBAL_ALPHA_GRID = np.round(np.arange(0.60, 2.01, 0.05), 3)
PER_HORIZON_GRID = np.round(np.arange(0.60, 2.51, 0.05), 3)
HORIZON_BIN_EDGES = [-np.inf, 0.0, 2.0, 5.0, 9.0, np.inf]
HORIZON_BIN_LABELS = ["h<=0", "h1-2", "h3-5", "h6-9", "h10+"]

HYPERPARAMS = [
    {
        "name": "current_120_lr005_d2_leaf50",
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
        "name": "mid_300_lr002_d3_leaf50_sub08",
        "n_estimators": 300,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "minibatch_frac": 0.8,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 15,
    },
    {
        "name": "deep_500_lr001_d3_leaf30_sub08",
        "n_estimators": 500,
        "learning_rate": 0.01,
        "max_depth": 4,
        "min_samples_leaf": 30,
        "minibatch_frac": 0.8,
        "natural_gradient": False,
        "random_state": 42,
        "early_stopping_rounds": 15,
    },
]

DISTRIBUTIONS = ["normal", "laplace"]


def build_feature_sets() -> dict[str, list[str]]:
    final_features = load_feature_list(DEFAULT_FINAL_FEATURE_LIST_PATH)
    safe_features = load_feature_list(SAFE_FEATURE_LIST_PATH)
    return {
        "final": list(final_features),
        "curated_weather_plus": append_features(
            final_features,
            [
                "forecast_high",
                "nws_forecast_high_f",
                "ndfd_lead_hours",
                "ndfd_grid_distance_km",
                "day_of_year_cos",
                "temp_range_so_far",
                "min_temp_so_far",
                "recent_forecast_revision",
            ],
            safe_features,
        ),
    }


def horizon_bin_labels(hours: pd.Series) -> np.ndarray:
    return pd.cut(
        pd.to_numeric(hours, errors="coerce"),
        bins=HORIZON_BIN_EDGES,
        labels=HORIZON_BIN_LABELS,
        ordered=True,
    ).astype(object)


def fit_per_horizon_factors(
    hours_calib: pd.Series,
    mu: np.ndarray,
    sigma: np.ndarray,
    y_calib: np.ndarray,
    dist: str,
) -> dict[str, float]:
    bins = horizon_bin_labels(pd.Series(np.asarray(hours_calib)))
    factors: dict[str, float] = {}
    for label in HORIZON_BIN_LABELS:
        mask = (bins == label).to_numpy()
        if mask.sum() < 200:
            factors[label] = float("nan")
            continue
        mu_h = mu[mask]
        sig_h = sigma[mask]
        y_h = y_calib[mask]
        best_factor, best_nll = 1.0, np.inf
        for factor in PER_HORIZON_GRID:
            nll = distribution_nll(
                y_h, mu=mu_h, sigma=sig_h * factor, distribution=dist
            )
            mean_nll = float(np.mean(nll))
            if mean_nll < best_nll:
                best_nll, best_factor = mean_nll, float(factor)
        factors[label] = best_factor
    # Fill any sparse bins with the global median of fitted factors
    valid = [v for v in factors.values() if np.isfinite(v)]
    fallback = float(np.median(valid)) if valid else 1.0
    return {k: (v if np.isfinite(v) else fallback) for k, v in factors.items()}


def apply_per_horizon_factors(
    hours: pd.Series,
    sigma: np.ndarray,
    factors: dict[str, float],
) -> np.ndarray:
    bins = horizon_bin_labels(hours)
    scaled = sigma.copy()
    for label, factor in factors.items():
        mask = (bins == label).to_numpy()
        scaled[mask] = sigma[mask] * factor
    return scaled


def score_frame(
    *,
    frame: pd.DataFrame,
    details: dict[str, Any],
    dist: str,
    sigma: np.ndarray,
    split_name: str,
    method: str,
    hyperparams_name: str,
    feature_set: str,
    run_id: str,
) -> dict[str, Any]:
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=float)
    mu = np.asarray(details["mu"], dtype=float)
    nll = distribution_nll(y_true, mu=mu, sigma=sigma, distribution=dist)
    prediction_frame = frame.copy()
    prediction_frame["mu"] = mu
    prediction_frame["sigma"] = sigma
    prediction_frame["distribution_type"] = dist
    probs = error_interval_probabilities(prediction_frame, dist_type=dist)
    labels = assign_error_interval_labels(prediction_frame[TARGET_COLUMN])
    brier = bucket_brier_scores(probs, labels)
    coverage_80 = float(np.mean(np.abs(y_true - mu) <= 1.281552 * sigma))
    return {
        "run_id": run_id,
        "feature_set": feature_set,
        "distribution": dist,
        "hyperparams_name": hyperparams_name,
        "method": method,
        "split": split_name,
        "n_rows": int(len(frame)),
        "nll": float(np.mean(nll)),
        "bucket_brier": float(brier["brier_score"].mean()),
        "bucket_log_loss": float(interval_log_loss(probs, labels)),
        "coverage_80": coverage_80,
        "coverage_err_80": abs(coverage_80 - 0.80),
        "mean_sigma": float(np.mean(sigma)),
        "median_sigma": float(np.median(sigma)),
    }


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "improvement_metrics.csv"
    runs_path = OUTPUT_DIR / "improvement_runs.csv"

    df = load_modeling_table(DEFAULT_MODELING_TABLE_PATH)
    validate_target_column(df)
    split = chronological_train_validation_test_split(df)

    dates_train = pd.to_datetime(split.train["date"])
    calib_mask = (dates_train >= CALIB_SLICE_START).to_numpy()
    print(
        f"rows train={len(split.train)} (calib={int(calib_mask.sum())}) "
        f"val={len(split.validation)} test={len(split.test)}",
        flush=True,
    )

    feature_sets = build_feature_sets()
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    done_keys: set[tuple[str, str, str]] = set()
    if metrics_path.exists():
        prior = pd.read_csv(metrics_path)
        for _, row in prior[prior["split"] == "validation"].iterrows():
            if row["method"] == "raw":
                done_keys.add((row["feature_set"], row["distribution"], row["hyperparams_name"]))
        print(f"resuming: {len(done_keys)} fits already complete", flush=True)

    specs = [
        (fname, normalize_distribution_name(dist), hp)
        for fname in feature_sets
        for dist in DISTRIBUTIONS
        for hp in HYPERPARAMS
    ]

    for index, (fname, dist, hp) in enumerate(specs, start=1):
        key = (fname, dist, hp["name"])
        if key in done_keys:
            print(f"[{index}/{len(specs)}] skip completed {key}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            X_train, X_val, X_test, imputer, _notes = build_imputed_feature_frames(
                train_df=split.train,
                validation_df=split.validation,
                test_df=split.test,
                feature_columns=feature_sets[fname],
            )
            X_calib = X_train[calib_mask]
            y_train = split.train[TARGET_COLUMN].to_numpy(dtype=float)
            y_calib = y_train[calib_mask]
            model = train_ngboost_distribution(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=split.validation[TARGET_COLUMN].to_numpy(dtype=float),
                distribution=dist,
                n_estimators=int(hp["n_estimators"]),
                learning_rate=float(hp["learning_rate"]),
                max_depth=int(hp["max_depth"]),
                min_samples_leaf=int(hp["min_samples_leaf"]),
                minibatch_frac=float(hp["minibatch_frac"]),
                natural_gradient=bool(hp["natural_gradient"]),
                random_state=int(hp["random_state"]),
                early_stopping_rounds=hp.get("early_stopping_rounds"),
            )
            run_id = stable_run_id(fname, dist, hp["name"])
            details = {
                "calib": predict_distribution_details(model, X_calib, dist),
                "validation": predict_distribution_details(model, X_val, dist),
                "test": predict_distribution_details(model, X_test, dist),
            }
            frames = {
                "calib": split.train.loc[calib_mask].reset_index(drop=True),
                "validation": split.validation.reset_index(drop=True),
                "test": split.test.reset_index(drop=True),
            }
            hours = {
                name: f["forecast_horizon_hours"] for name, f in frames.items()
            }

            raw_sigma_val = np.asarray(details["validation"]["sigma"], dtype=float)
            raw_sigma_test = np.asarray(details["test"]["sigma"], dtype=float)

            # Method 1: raw
            metric_rows.append(
                score_frame(
                    frame=frames["validation"],
                    details=details["validation"],
                    dist=dist,
                    sigma=raw_sigma_val,
                    split_name="validation",
                    method="raw",
                    hyperparams_name=hp["name"],
                    feature_set=fname,
                    run_id=run_id,
                )
            )

            # Method 2: global sigma scale fit on train-calibration slice
            best_global, best_nll = 1.0, np.inf
            sig_c = np.asarray(details["calib"]["sigma"], dtype=float)
            mu_c = np.asarray(details["calib"]["mu"], dtype=float)
            for factor in GLOBAL_ALPHA_GRID:
                nll_c = distribution_nll(y_calib, mu=mu_c, sigma=sig_c * float(factor), distribution=dist)
                m = float(np.mean(nll_c))
                if m < best_nll:
                    best_nll, best_global = m, float(factor)
            metric_rows.append(
                score_frame(
                    frame=frames["validation"],
                    details=details["validation"],
                    dist=dist,
                    sigma=raw_sigma_val * best_global,
                    split_name="validation",
                    method="global_calib_fit",
                    hyperparams_name=hp["name"],
                    feature_set=fname,
                    run_id=run_id,
                )
            )

            # Method 3: per-horizon sigma scaling fit on train-calibration slice
            factors = fit_per_horizon_factors(
                hours["calib"], mu_c, sig_c, y_calib, dist
            )
            metric_rows.append(
                score_frame(
                    frame=frames["validation"],
                    details=details["validation"],
                    dist=dist,
                    sigma=apply_per_horizon_factors(hours["validation"], raw_sigma_val, factors),
                    split_name="validation",
                    method="per_horizon_calib_fit",
                    hyperparams_name=hp["name"],
                    feature_set=fname,
                    run_id=run_id,
                )
            )

            trained_cache[key] = {
                "model": model,
                "imputer": imputer,
                "dist": dist,
                "hp": hp,
                "feature_set": fname,
                "run_id": run_id,
                "best_global": best_global,
                "factors": factors,
                "frames": frames,
                "details": details,
                "raw_sigma_test": raw_sigma_test,
            }
            run_rows.append(
                {
                    "run_id": run_id,
                    "status": "success",
                    "feature_set": fname,
                    "distribution": dist,
                    "hyperparams_name": hp["name"],
                    "best_global_factor_calib": best_global,
                    "per_horizon_factors": json.dumps(factors),
                    "elapsed_seconds": time.perf_counter() - t0,
                }
            )
            pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
            pd.DataFrame(run_rows).to_csv(runs_path, index=False)
            print(
                f"[{index}/{len(specs)}] {fname}/{dist}/{hp['name']} ok "
                f"({time.perf_counter() - t0:.0f}s) global={best_global:.2f} "
                f"factors={factors}",
                flush=True,
            )
        except Exception as exc:
            run_rows.append(
                {
                    "run_id": stable_run_id(fname, dist, hp["name"]),
                    "status": "failed",
                    "feature_set": fname,
                    "distribution": dist,
                    "hyperparams_name": hp["name"],
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - t0,
                }
            )
            pd.DataFrame(run_rows).to_csv(runs_path, index=False)
            print(f"[{index}/{len(specs)}] FAILED {key}: {exc}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        print("no successful runs", flush=True)
        return

    val = metrics[metrics["split"] == "validation"].copy()
    for col in ["nll", "bucket_brier"]:
        val[f"{col}_rank"] = val.groupby("method")[col].rank()
    val["rank_sum"] = val["nll_rank"] + val["bucket_brier_rank"]
    val = val.sort_values(["rank_sum", "nll"])
    val.to_csv(OUTPUT_DIR / "validation_leaderboard.csv", index=False)

    winner_row = val.iloc[0]
    wkey = (winner_row["feature_set"], winner_row["distribution"], winner_row["hyperparams_name"])
    wmethod = winner_row["method"]
    cached = trained_cache[wkey]

    # Test-set scoring ONCE for the selected configuration/method only.
    test_rows = []
    if wmethod == "raw":
        test_sigma = cached["raw_sigma_test"]
    elif wmethod == "global_calib_fit":
        test_sigma = cached["raw_sigma_test"] * cached["best_global"]
    else:
        test_sigma = apply_per_horizon_factors(
            cached["frames"]["test"]["forecast_horizon_hours"],
            cached["raw_sigma_test"],
            cached["factors"],
        )
    test_rows.append(
        score_frame(
            frame=cached["frames"]["test"],
            details=cached["details"]["test"],
            dist=cached["dist"],
            sigma=test_sigma,
            split_name="test",
            method=wmethod,
            hyperparams_name=wkey[2],
            feature_set=wkey[0],
            run_id=cached["run_id"],
        )
    )
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUTPUT_DIR / "test_final_metrics.csv", index=False)

    with open(MODEL_OUTPUT_PATH, "wb") as fh:
        pickle.dump(
            {
                "model": cached["model"],
                "imputer": cached["imputer"],
                "distribution": cached["dist"],
                "hyperparams": cached["hp"],
                "feature_set_name": wkey[0],
                "sigma_method": wmethod,
                "global_factor": cached["best_global"],
                "per_horizon_factors": cached["factors"],
            },
            fh,
        )
    metadata = {
        "selected_on": "validation rank_sum(nll_rank+brier_rank)",
        "winner": {
            "feature_set": wkey[0],
            "distribution": wkey[1],
            "hyperparams": wkey[2],
            "sigma_method": wmethod,
        },
        "validation_metrics": {k: float(winner_row[k]) for k in ["nll", "bucket_brier", "bucket_log_loss", "coverage_80"]},
        "test_metrics": {k: float(test_df.iloc[0][k]) for k in ["nll", "bucket_brier", "bucket_log_loss", "coverage_80"]},
        "calibration_slice_start": CALIB_SLICE_START,
        "note": "Sigma calibration factors fit on 2023-07..2023-12 train slice only; test scored once.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(METADATA_OUTPUT_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("=== VALIDATION LEADERBOARD (top 10) ===", flush=True)
    print(
        val[
            ["feature_set", "distribution", "hyperparams_name", "method", "nll", "bucket_brier", "bucket_log_loss", "coverage_80", "rank_sum"]
        ].head(10).to_string(index=False),
        flush=True,
    )
    print("=== FINAL TEST (winner, scored once) ===", flush=True)
    print(test_df.to_string(index=False), flush=True)
    print(f"total elapsed {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
