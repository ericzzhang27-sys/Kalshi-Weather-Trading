from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_model_improvement_wave2 import NEW_FEATURES_SINCE_HEAD  # noqa: E402
from scripts.run_model_improvement_202608 import score_frame  # noqa: E402
from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
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
ENSEMBLE_MODEL_PATH = REPO_ROOT / "models" / "ngboost_ensemble_202608.pkl"

HP = {
    "n_estimators": 800,
    "learning_rate": 0.0075,
    "max_depth": 3,
    "min_samples_leaf": 50,
    "minibatch_frac": 0.8,
    "natural_gradient": True,
    "early_stopping_rounds": 10,
}
SEEDS = [42, 7, 13, 2026, 99]


def main() -> None:
    started = time.perf_counter()
    df = load_modeling_table(DEFAULT_MODELING_TABLE_PATH)
    validate_target_column(df)
    split = chronological_train_validation_test_split(df)

    final_features = load_feature_list(DEFAULT_FINAL_FEATURE_LIST_PATH)
    core36 = [f for f in final_features if f not in NEW_FEATURES_SINCE_HEAD]

    X_train, X_val, X_test, imputer, _ = build_imputed_feature_frames(
        train_df=split.train,
        validation_df=split.validation,
        test_df=split.test,
        feature_columns=core36,
    )
    y_train = split.train[TARGET_COLUMN].to_numpy(dtype=float)
    frames = {
        "validation": split.validation.reset_index(drop=True),
        "test": split.test.reset_index(drop=True),
    }

    members = []
    rows = []
    for i, seed in enumerate(SEEDS, start=1):
        t0 = time.perf_counter()
        model = train_ngboost_distribution(
            X_train=X_train, y_train=y_train, X_val=X_val,
            y_val=split.validation[TARGET_COLUMN].to_numpy(dtype=float),
            distribution="normal", random_state=seed, **HP,
        )
        d_val = predict_distribution_details(model, X_val, "normal")
        single = score_frame(
            frame=frames["validation"], details=d_val, dist="normal",
            sigma=np.asarray(d_val["sigma"], dtype=float),
            split_name="validation", method=f"seed_{seed}",
            hyperparams_name="ensemble_member", feature_set="core36",
            run_id=f"seed_{seed}",
        )
        rows.append(single)
        members.append({"seed": seed, "model": model})
        print(f"[{i}/{len(SEEDS)}] seed {seed}: val nll={single['nll']:.4f} "
              f"brier={single['bucket_brier']:.5f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # Ensemble: average mu and sigma across members (validation + test)
    ens_rows = []
    for split_name, X in [("validation", X_val), ("test", X_test)]:
        mus, sigmas = [], []
        for m in members:
            d = predict_distribution_details(m["model"], X, "normal")
            mus.append(np.asarray(d["mu"], dtype=float))
            sigmas.append(np.asarray(d["sigma"], dtype=float))
        details = {"mu": np.mean(mus, axis=0), "sigma": np.sqrt(np.mean(np.array(sigmas) ** 2 + np.array(mus) ** 2, axis=0) - np.mean(mus, axis=0) ** 2)}
        row = score_frame(
            frame=frames[split_name], details=details, dist="normal",
            sigma=details["sigma"], split_name=split_name,
            method="mu_sigma_ensemble_5",
            hyperparams_name="ensemble_member", feature_set="core36",
            run_id="ensemble5",
        )
        ens_rows.append(row)
        print(f"ENSEMBLE {split_name}: nll={row['nll']:.4f} brier={row['bucket_brier']:.5f} "
              f"logloss={row['bucket_log_loss']:.4f} cov80={row['coverage_80']:.3f}", flush=True)

    pd.DataFrame(rows + ens_rows).to_csv(OUTPUT_DIR / "wave3_ensemble_metrics.csv", index=False)
    with open(ENSEMBLE_MODEL_PATH, "wb") as fh:
        pickle.dump({
            "members": [{"seed": m["seed"], "model": m["model"]} for m in members],
            "imputer": imputer,
            "distribution": "normal",
            "hyperparams": HP,
            "feature_columns": core36,
            "combine_rule": "mean_mu; sigma=mixture_std(mean_mu, mean_second_moment)",
        }, fh)
    print(f"saved {ENSEMBLE_MODEL_PATH}; elapsed {time.perf_counter()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
