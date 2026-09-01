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
from scripts.run_model_improvement_202608 import (  # noqa: E402
    apply_per_horizon_factors,
    build_feature_sets,
    fit_per_horizon_factors,
    score_frame,
)
from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
    distribution_nll,
    normalize_distribution_name,
    predict_distribution_details,
    train_ngboost_distribution,
)
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
OUTPUT_DIR = REPO_ROOT / "outputs" / "model_improvement_202608"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_improved_wave2.pkl"
METADATA_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_improved_wave2_metadata.json"
PARAMS_CSV = REPO_ROOT / "outputs" / "ngboost_distribution_params_v0.csv"

CALIB_SLICE_START = "2023-07-01"
GLOBAL_ALPHA_GRID = np.round(np.arange(0.60, 2.01, 0.05), 3)

NEW_FEATURES_SINCE_HEAD = [
    "hours_since_max",
    "is_post_peak_hour",
    "is_post_window_hour",
    "is_verified_peak",
    "temp_drop_since_max",
    "temp_range_so_far",
]

HYPERPARAMS = [
    {
        "name": "prod_replica_500_lr001_d3_ng",
        "n_estimators": 500,
        "learning_rate": 0.01,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "minibatch_frac": 0.8,
        "natural_gradient": True,
        "random_state": 42,
        "early_stopping_rounds": 5,
    },
    {
        "name": "prod_more_800_lr00075_d3_ng",
        "n_estimators": 800,
        "learning_rate": 0.0075,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "minibatch_frac": 0.8,
        "natural_gradient": True,
        "random_state": 42,
        "early_stopping_rounds": 10,
    },
    {
        "name": "prod_d2_500_lr001_ng",
        "n_estimators": 500,
        "learning_rate": 0.01,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "minibatch_frac": 0.8,
        "natural_gradient": True,
        "random_state": 42,
        "early_stopping_rounds": 5,
    },
]


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "wave2_metrics.csv"

    df = load_modeling_table(DEFAULT_MODELING_TABLE_PATH)
    validate_target_column(df)
    split = chronological_train_validation_test_split(df)
    dates_train = pd.to_datetime(split.train["date"])
    calib_mask = (dates_train >= CALIB_SLICE_START).to_numpy()

    final_features = load_feature_list(DEFAULT_FINAL_FEATURE_LIST_PATH)
    core36 = [f for f in final_features if f not in NEW_FEATURES_SINCE_HEAD]
    feature_sets = {"core36": core36, "final42": list(final_features)}

    metric_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str], dict[str, Any]] = {}

    # ---- Fair baseline: score the current production artifact under this harness
    print("scoring production artifact baseline...", flush=True)
    params = pd.read_csv(PARAMS_CSV)
    for split_name, frame in [("validation", split.validation), ("test", split.test)]:
        p = params[params["split"] == split_name].reset_index(drop=True)
        merged = frame.reset_index(drop=True)
        assert len(p) == len(merged), f"params/frame mismatch for {split_name}"
        details = {"mu": p["mu"].to_numpy(dtype=float), "sigma": p["sigma"].to_numpy(dtype=float)}
        metric_rows.append(
            score_frame(
                frame=merged,
                details=details,
                dist="normal",
                sigma=details["sigma"],
                split_name=split_name,
                method="raw",
                hyperparams_name="PRODUCTION_ARTIFACT",
                feature_set="final42",
                run_id="production_baseline",
            )
        )

    specs = [
        (fname, hp)
        for fname in feature_sets
        for hp in HYPERPARAMS
    ]
    dists = {"prod_replica_500_lr001_d3_ng": ["normal", "laplace"]}

    jobs: list[tuple[str, dict[str, Any], str]] = []
    for fname, hp in specs:
        if hp["name"] == "prod_replica_500_lr001_d3_ng":
            for d in ["normal", "laplace"]:
                jobs.append((fname, hp, d))
        else:
            jobs.append((fname, hp, "normal"))

    for index, (fname, hp, dist_label) in enumerate(jobs, start=1):
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
            dist = normalize_distribution_name(dist_label)
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
            sig_c = np.asarray(details["calib"]["sigma"], dtype=float)
            mu_c = np.asarray(details["calib"]["mu"], dtype=float)
            best_global, best_nll = 1.0, np.inf
            for factor in GLOBAL_ALPHA_GRID:
                m = float(np.mean(distribution_nll(y_calib, mu=mu_c, sigma=sig_c * float(factor), distribution=dist)))
                if m < best_nll:
                    best_nll, best_global = m, float(factor)

            raw_sigma_val = np.asarray(details["validation"]["sigma"], dtype=float)
            common = dict(
                dist=dist,
                hyperparams_name=hp["name"],
                feature_set=fname,
                run_id=f"{fname}__{dist}__{hp['name']}",
            )
            metric_rows.append(
                score_frame(frame=frames["validation"], details=details["validation"], sigma=raw_sigma_val,
                            split_name="validation", method="raw", **common)
            )
            metric_rows.append(
                score_frame(frame=frames["validation"], details=details["validation"], sigma=raw_sigma_val * best_global,
                            split_name="validation", method="global_calib_fit", **common)
            )
            trained_cache[(fname + "|" + dist, hp["name"])] = {
                "model": model, "imputer": imputer, "dist": dist, "hp": hp,
                "feature_set": fname, "best_global": best_global,
                "frames": frames, "details": details,
                "raw_sigma_test": np.asarray(details["test"]["sigma"], dtype=float),
            }
            run_rows.append({"job": common["run_id"], "status": "success",
                             "best_global_factor_calib": best_global,
                             "elapsed_seconds": time.perf_counter() - t0})
            pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
            print(f"[{index}/{len(jobs)}] {fname}/{dist}/{hp['name']} ok "
                  f"({time.perf_counter() - t0:.0f}s) global={best_global:.2f}", flush=True)
        except Exception as exc:
            run_rows.append({"job": f"{fname}|{dist_label}|{hp['name']}", "status": "failed",
                             "error_message": f"{type(exc).__name__}: {exc}",
                             "elapsed_seconds": time.perf_counter() - t0})
            print(f"[{index}/{len(jobs)}] FAILED {fname}/{dist_label}/{hp['name']}: {exc}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(metrics_path, index=False)
    val = metrics[(metrics["split"] == "validation") & (metrics["method"] == "raw")].copy()
    for col in ["nll", "bucket_brier"]:
        val[f"{col}_rank"] = val[col].rank()
    val["rank_sum"] = val["nll_rank"] + val["bucket_brier_rank"]
    val = val.sort_values(["rank_sum", "nll"])
    val.to_csv(OUTPUT_DIR / "wave2_leaderboard.csv", index=False)

    winner = val.iloc[0]
    wkey = (winner["feature_set"] + "|" + winner["distribution"], winner["hyperparams_name"])
    cached = trained_cache[wkey]

    test_sigma = cached["raw_sigma_test"]
    method = "raw"
    if winner["method"] != "raw":
        pass
    # also score global_calib_fit on test for the winner config (pre-registered rule:
    # choose whichever of raw/global won on validation for THIS config)
    if winner["method"] == "global_calib_fit":
        test_sigma = cached["raw_sigma_test"] * cached["best_global"]
        method = "global_calib_fit"
    test_df = pd.DataFrame([
        score_frame(
            frame=cached["frames"]["test"],
            details=cached["details"]["test"],
            dist=cached["dist"],
            sigma=test_sigma,
            split_name="test",
            method=method,
            hyperparams_name=wkey[1],
            feature_set=cached["feature_set"],
            run_id="wave2_winner",
        )
    ])
    test_df.to_csv(OUTPUT_DIR / "wave2_test_final_metrics.csv", index=False)

    with open(MODEL_OUTPUT_PATH, "wb") as fh:
        pickle.dump({
            "model": cached["model"],
            "imputer": cached["imputer"],
            "distribution": cached["dist"],
            "hyperparams": cached["hp"],
            "feature_set_name": cached["feature_set"],
            "sigma_method": method,
            "global_factor": cached["best_global"],
        }, fh)
    metadata = {
        "baseline_production_raw_validation": {
            k: float(metrics[(metrics['hyperparams_name'] == 'PRODUCTION_ARTIFACT') & (metrics['split'] == 'validation')].iloc[0][k])
            for k in ["nll", "bucket_brier", "bucket_log_loss", "coverage_80"]},
        "winner_validation": {k: float(winner[k]) for k in ["nll", "bucket_brier", "coverage_80"]} | {k: str(winner[k]) for k in ["feature_set", "distribution", "hyperparams_name"]},
        "winner_test_once": {k: float(test_df.iloc[0][k]) for k in ["nll", "bucket_brier", "bucket_log_loss", "coverage_80"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(METADATA_OUTPUT_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print("=== WAVE2 VALIDATION LEADERBOARD ===", flush=True)
    print(val[["feature_set", "distribution", "hyperparams_name", "method", "nll", "bucket_brier", "bucket_log_loss", "coverage_80", "rank_sum"]].to_string(index=False), flush=True)
    print("=== WINNER TEST (once) ===", flush=True)
    print(test_df.to_string(index=False), flush=True)
    print(f"total elapsed {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
