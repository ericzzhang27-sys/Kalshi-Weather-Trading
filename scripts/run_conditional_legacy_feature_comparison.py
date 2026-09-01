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
from scipy.stats import norm
from sklearn.impute import SimpleImputer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conditional_increase_model import (  # noqa: E402
    evaluate_conditional_distribution,
    train_shifted_poisson_ngboost,
)
from src.distributional_model import (  # noqa: E402
    predict_distribution_details,
    train_ngboost_distribution,
    validate_no_leakage_feature_columns,
)
from src.hurdle_distribution import categorical_scores, integer_delta_probabilities  # noqa: E402
from src.hurdle_model import load_hurdle_predictor  # noqa: E402


HURDLE_DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
REGIONAL_DATASET_PATH = (
    REPO_ROOT / "data" / "processed" / "modeling_rows_v1_with_regional_features.csv"
)
BASE_FEATURES_PATH = REPO_ROOT / "models" / "remaining_increase_features.json"
REGIONAL_FEATURES_PATH = REPO_ROOT / "outputs" / "final_feature_list_with_regional.json"
LEGACY_BASE_FEATURES_PATH = REPO_ROOT / "outputs" / "final_feature_list.json"
EXCEEDANCE_BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "feature_count_model_comparison"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = list(payload.get("features", payload.get("feature_columns", [])))
    if not features:
        raise ValueError(f"No features found in {path}")
    return features


def _load_aligned_data(regional_features: list[str]) -> pd.DataFrame:
    hurdle = pd.read_csv(HURDLE_DATASET_PATH, low_memory=False)
    regional = pd.read_csv(REGIONAL_DATASET_PATH, low_memory=False)
    for frame in (hurdle, regional):
        frame["prediction_time"] = pd.to_datetime(frame["prediction_time"], errors="raise")
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise").dt.normalize()

    regional_columns = [
        "target_date",
        "prediction_time",
        "forecast_error",
        "forecast_high",
        "actual_high",
        *regional_features,
    ]
    regional = regional[regional_columns].drop_duplicates(
        ["target_date", "prediction_time"], keep="last"
    )
    regional = regional.rename(
        columns={
            "forecast_error": "legacy_forecast_error",
            "forecast_high": "legacy_forecast_high",
            "actual_high": "legacy_actual_high",
        }
    )
    aligned = hurdle.merge(
        regional,
        on=["target_date", "prediction_time"],
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        raise ValueError("No exact timestamps align between hurdle and regional datasets")
    if (pd.to_numeric(aligned["remaining_increase"], errors="raise") < 0).any():
        raise ValueError("Aligned remaining-increase target contains negative values")
    return aligned.sort_values(["target_date", "prediction_time"]).reset_index(drop=True)


def _load_exceedance_predictor():
    bundle = json.loads(EXCEEDANCE_BUNDLE_PATH.read_text(encoding="utf-8"))
    if bundle.get("status") not in {"frozen_validated", "frozen_validated_user_override"}:
        raise ValueError("Exceedance bundle is not frozen and validated")
    models_dir = EXCEEDANCE_BUNDLE_PATH.parent
    predictor = load_hurdle_predictor(
        models_dir / bundle["paths"]["classifier"],
        models_dir / bundle["paths"]["features"],
        models_dir / bundle["paths"]["calibrator"],
        bundle["calibration"],
    )
    return bundle, predictor


def _train_legacy(
    train: pd.DataFrame,
    features: list[str],
    training_config: dict,
) -> dict:
    X = train[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train["legacy_forecast_error"], errors="raise").to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    transformed = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)
    model = train_ngboost_distribution(
        transformed,
        y,
        distribution="normal",
        n_estimators=int(training_config["n_estimators"]),
        learning_rate=float(training_config["learning_rate"]),
        max_depth=int(training_config["max_depth"]),
        min_samples_leaf=int(training_config["min_samples_leaf"]),
        minibatch_frac=float(training_config["minibatch_frac"]),
        natural_gradient=bool(training_config["natural_gradient"]),
        random_state=int(training_config["random_state"]),
        early_stopping_rounds=None,
    )
    return {"model": model, "imputer": imputer, "features": features}


def _legacy_details(artifact: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    X = frame[artifact["features"]].apply(pd.to_numeric, errors="coerce")
    transformed = pd.DataFrame(
        artifact["imputer"].transform(X), columns=artifact["features"], index=X.index
    )
    details = predict_distribution_details(artifact["model"], transformed, "normal")
    return {"mu": np.asarray(details["mu"]), "sigma": np.asarray(details["sigma"])}


def _legacy_native_metrics(artifact: dict, frame: pd.DataFrame) -> dict[str, float | int]:
    details = _legacy_details(artifact, frame)
    target = frame["legacy_forecast_error"].to_numpy(dtype=float)
    mu, sigma = details["mu"], np.maximum(details["sigma"], 1e-9)
    nll = float(-np.mean(norm.logpdf(target, loc=mu, scale=sigma)))
    z = (target - mu) / sigma
    crps = sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return {
        "n": int(len(frame)),
        "mean_target": float(np.mean(target)),
        "mean_prediction": float(np.mean(mu)),
        "density_nll": nll,
        "interval_nll": nll,
        "mae": float(np.mean(np.abs(mu - target))),
        "rmse": float(np.sqrt(np.mean((mu - target) ** 2))),
        "coverage_80": float(np.mean(np.abs(target - mu) <= norm.ppf(0.90) * sigma)),
        "coverage_90": float(np.mean(np.abs(target - mu) <= norm.ppf(0.95) * sigma)),
        "crps": float(np.mean(crps)),
        "nll": nll,
    }


def _legacy_delta_probabilities(
    artifact: dict, frame: pd.DataFrame, max_delta: int = 10
) -> np.ndarray:
    details = _legacy_details(artifact, frame)
    mu, sigma = details["mu"], np.maximum(details["sigma"], 1e-9)
    forecast = frame["legacy_forecast_high"].to_numpy(dtype=float)
    current_max = frame["current_max_so_far"].to_numpy(dtype=float)

    def final_cdf(boundary: np.ndarray) -> np.ndarray:
        return norm.cdf(boundary - forecast, loc=mu, scale=sigma)

    columns = [final_cdf(current_max + 0.5)]
    for delta in range(1, max_delta + 1):
        columns.append(
            np.clip(
                final_cdf(current_max + delta + 0.5)
                - final_cdf(current_max + delta - 0.5),
                0.0,
                1.0,
            )
        )
    columns.append(1.0 - final_cdf(current_max + max_delta + 0.5))
    matrix = np.clip(np.column_stack(columns), 0.0, 1.0)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def _score_slices(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    max_delta: int = 10,
) -> pd.DataFrame:
    hour = frame["prediction_time"].dt.hour
    slices = {
        "overall": np.ones(len(frame), dtype=bool),
        "before_12": (hour < 12).to_numpy(),
        "12_to_4": ((hour >= 12) & (hour < 16)).to_numpy(),
        "after_4": (hour >= 16).to_numpy(),
    }
    target = frame["remaining_increase"].to_numpy(dtype=int)
    rows = []
    for slice_name, mask in slices.items():
        for model_name, probability in probabilities.items():
            rows.append(
                {
                    "slice": slice_name,
                    "model": model_name,
                    "n": int(mask.sum()),
                    **categorical_scores(probability[mask], target[mask], max_delta),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_features = _read_features(BASE_FEATURES_PATH)
    legacy_base_features = _read_features(LEGACY_BASE_FEATURES_PATH)
    regional_contract = _read_features(REGIONAL_FEATURES_PATH)
    regional_only = [feature for feature in regional_contract if feature not in legacy_base_features]
    expanded_features = [*base_features, *[f for f in regional_only if f not in base_features]]
    if len(base_features) != 40 or len(expanded_features) <= 100:
        raise ValueError(
            f"Unexpected feature counts: base={len(base_features)}, expanded={len(expanded_features)}"
        )
    validate_no_leakage_feature_columns(base_features)
    validate_no_leakage_feature_columns(expanded_features)

    aligned = _load_aligned_data(regional_only)
    missing = sorted(set(expanded_features).difference(aligned.columns))
    if missing:
        raise ValueError(f"Expanded features missing after alignment: {missing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    test_start = pd.Timestamp(str(config["splits"]["test_start"]))
    train = aligned.loc[aligned["target_date"] < test_start].copy()
    test = aligned.loc[aligned["target_date"] >= test_start].copy()
    positive_train = train.loc[train["remaining_increase"] > 0].copy()
    positive_test = test.loc[test["remaining_increase"] > 0].copy()
    if train.empty or test.empty or positive_train.empty or positive_test.empty:
        raise ValueError("Chronological comparison split contains an empty required subset")

    training_config = dict(config["ngboost"]["standard_training"])
    exceedance_bundle, exceedance = _load_exceedance_predictor()
    p_increase = exceedance.predict_proba(test)
    models: dict[str, dict] = {}
    native_rows: list[dict] = []
    probabilities: dict[str, np.ndarray] = {}

    for label, features in (("40_features", base_features), ("119_features", expanded_features)):
        print(f"Training conditional_{label}: {len(positive_train):,} rows x {len(features)} features")
        conditional = train_shifted_poisson_ngboost(
            positive_train,
            features,
            n_estimators=200,
            learning_rate=0.03,
            random_state=42,
            min_samples_leaf=50,
        )
        models[f"conditional_{label}"] = conditional
        native_rows.append(
            {
                "model": f"conditional_{label}",
                "target": "remaining_increase_given_positive",
                **evaluate_conditional_distribution(conditional, positive_test),
            }
        )
        probabilities[f"conditional_{label}"] = integer_delta_probabilities(
            p_increase, conditional, test, max_delta=10
        )

        print(f"Training legacy_{label}: {len(train):,} rows x {len(features)} features")
        legacy = _train_legacy(train, features, training_config)
        models[f"legacy_{label}"] = legacy
        native_rows.append(
            {
                "model": f"legacy_{label}",
                "target": "forecast_error",
                **_legacy_native_metrics(legacy, test),
            }
        )
        probabilities[f"legacy_{label}"] = _legacy_delta_probabilities(legacy, test)

    native = pd.DataFrame(native_rows)
    scores = _score_slices(test, probabilities)
    native.to_csv(OUTPUT_DIR / "native_test_metrics.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "common_bucket_test_metrics.csv", index=False)
    with (OUTPUT_DIR / "trained_models.pkl").open("wb") as handle:
        pickle.dump(models, handle)

    overall = scores.loc[scores["slice"].eq("overall")].set_index("model")
    comparison_rows = []
    for family in ("conditional", "legacy"):
        small = overall.loc[f"{family}_40_features"]
        large = overall.loc[f"{family}_119_features"]
        comparison_rows.append(
            {
                "comparison": f"{family}: 119 minus 40",
                "nll_change": float(large["multiclass_nll"] - small["multiclass_nll"]),
                "brier_change": float(
                    large["mean_bucket_brier"] - small["mean_bucket_brier"]
                ),
            }
        )
    best_conditional = overall.loc[[i for i in overall.index if i.startswith("conditional")]][
        "multiclass_nll"
    ].idxmin()
    best_legacy = overall.loc[[i for i in overall.index if i.startswith("legacy")]][
        "multiclass_nll"
    ].idxmin()
    comparison_rows.append(
        {
            "comparison": f"best conditional ({best_conditional}) minus best legacy ({best_legacy})",
            "nll_change": float(
                overall.loc[best_conditional, "multiclass_nll"]
                - overall.loc[best_legacy, "multiclass_nll"]
            ),
            "brier_change": float(
                overall.loc[best_conditional, "mean_bucket_brier"]
                - overall.loc[best_legacy, "mean_bucket_brier"]
            ),
        }
    )
    deltas = pd.DataFrame(comparison_rows)
    deltas.to_csv(OUTPUT_DIR / "comparison_deltas.csv", index=False)

    manifest = {
        "status": "completed_research_comparison_not_promoted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train),
        "positive_conditional_training_rows": len(positive_train),
        "test_rows": len(test),
        "positive_conditional_test_rows": len(positive_test),
        "training_dates": f"{train['target_date'].min().date()} to {train['target_date'].max().date()}",
        "test_dates": f"{test['target_date'].min().date()} to {test['target_date'].max().date()}",
        "base_feature_count": len(base_features),
        "expanded_feature_count": len(expanded_features),
        "expanded_definition": "existing conditional 40 plus the 79 regional ASOS features",
        "conditional_model": "NGBoost shifted Poisson, 200 estimators",
        "legacy_model": "NGBoost Normal using configured production hyperparameters",
        "legacy_training_config": training_config,
        "exceedance_winner": exceedance_bundle["winner"],
        "selection_or_promotion": "none; held-out test is diagnostic only",
        "data_sha256": {
            "hurdle": _sha256(HURDLE_DATASET_PATH),
            "regional": _sha256(REGIONAL_DATASET_PATH),
        },
        "features": {"base": base_features, "expanded": expanded_features},
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    report = [
        "# Conditional vs Legacy Feature-Count Training Run",
        "",
        f"Trained on {len(train):,} exact hourly-aligned pre-test rows; scored on "
        f"{len(test):,} untouched rows from {test_start.date()} onward.",
        f"The conditional fits used {len(positive_train):,} positive training rows. "
        "The frozen exceedance classifier supplied P(increase) for both conditional variants.",
        "",
        "The 119-feature contract is the existing conditional 40-feature contract plus all "
        "79 timestamp-safe regional ASOS features. No production artifact was replaced.",
        "",
        "## Common final-increase bucket scores",
        "",
        scores.to_markdown(index=False),
        "",
        "## Direct comparison deltas",
        "",
        "Negative changes favor the first/broader model named in each comparison.",
        "",
        deltas.to_markdown(index=False),
        "",
        "## Native target metrics",
        "",
        native.to_markdown(index=False),
        "",
        "## Limitation",
        "",
        "Regional features are hourly, so this run uses only exact hourly timestamps shared by "
        "the five-minute hurdle table and the regional modeling table. Results are not directly "
        "comparable to earlier five-minute conditional runs.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(scores.loc[scores["slice"].eq("overall")].to_string(index=False))
    print(deltas.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
