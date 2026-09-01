from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conditional_increase_model import load_conditional_increase_model
from src.distributional_model import distribution_cdf
from src.hurdle_distribution import categorical_scores, integer_delta_probabilities
from src.hurdle_model import load_hurdle_predictor
from src.predict_distribution import load_probability_engine


HURDLE_DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
LEGACY_DATASET_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"
EXCEEDANCE_BUNDLE_PATH = REPO_ROOT / "models" / "exceedance_model_bundle.json"
CONDITIONAL_BUNDLE_PATH = REPO_ROOT / "models" / "remaining_increase_bundle.json"
CONDITIONAL_MODEL_PATH = REPO_ROOT / "models" / "remaining_increase_ngboost.pkl"
CONFIG_PATH = REPO_ROOT / "config" / "model_config.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "remaining_increase" / "full_hurdle_backtest"


def _load_legacy_engine():
    bundle_path = REPO_ROOT / "models" / "production_model_bundle.json"
    try:
        return load_probability_engine(model_bundle_path=bundle_path), None
    except ValueError as exc:
        if "hash mismatch" not in str(exc):
            raise
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        model_path = (bundle_path.parent / bundle["model_path"]).resolve()
        feature_path = (bundle_path.parent / bundle["feature_list_path"]).resolve()
        engine = load_probability_engine(
            model_path=model_path,
            feature_list_path=feature_path,
            calibration_config_path=None,
            model_bundle_path=None,
        )
        return engine, str(exc)


def _load_frozen_components():
    exceedance = json.loads(EXCEEDANCE_BUNDLE_PATH.read_text(encoding="utf-8"))
    conditional = json.loads(CONDITIONAL_BUNDLE_PATH.read_text(encoding="utf-8"))
    if exceedance.get("status") not in {"frozen_validated", "frozen_validated_user_override"}:
        raise ValueError("Exceedance bundle is not frozen")
    if conditional.get("status") != "frozen_validated":
        raise ValueError("Conditional bundle is not frozen")
    models_dir = EXCEEDANCE_BUNDLE_PATH.parent
    predictor = load_hurdle_predictor(
        models_dir / exceedance["paths"]["classifier"],
        models_dir / exceedance["paths"]["features"],
        models_dir / exceedance["paths"]["calibrator"],
        exceedance["calibration"],
    )
    return exceedance, conditional, predictor, load_conditional_increase_model(CONDITIONAL_MODEL_PATH)


def _aligned_test_rows(test_start: str, legacy_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hurdle = pd.read_csv(HURDLE_DATASET_PATH)
    legacy = pd.read_csv(LEGACY_DATASET_PATH, low_memory=False)
    for frame in (hurdle, legacy):
        frame["prediction_time"] = pd.to_datetime(frame["prediction_time"], errors="raise")
    hurdle["target_date"] = pd.to_datetime(hurdle["target_date"], errors="raise").dt.normalize()
    legacy["target_date"] = pd.to_datetime(legacy["target_date"], errors="raise").dt.normalize()
    hurdle = hurdle.loc[hurdle["target_date"] >= pd.Timestamp(test_start)].copy()
    legacy = legacy.loc[legacy["target_date"] >= pd.Timestamp(test_start)].copy()
    legacy_columns = ["target_date", "prediction_time", "forecast_high", "actual_high", *legacy_features]
    legacy = legacy[legacy_columns].drop_duplicates(["target_date", "prediction_time"], keep="last")
    rename = {column: f"legacy__{column}" for column in legacy.columns if column not in {"target_date", "prediction_time"}}
    legacy = legacy.rename(columns=rename)
    aligned = hurdle.merge(legacy, on=["target_date", "prediction_time"], how="inner", validate="one_to_one")
    if aligned.empty:
        raise ValueError("No exact held-out timestamps align between hurdle and legacy tables")
    old_input = pd.DataFrame(index=aligned.index)
    for feature in legacy_features:
        old_input[feature] = aligned[f"legacy__{feature}"]
    old_input["date"] = aligned["target_date"]
    old_input["prediction_time"] = aligned["prediction_time"]
    old_input["forecast_high"] = aligned["legacy__forecast_high"]
    old_input["actual_high"] = aligned["legacy__actual_high"]
    return aligned, old_input


def _legacy_probability_matrix(
    params: pd.DataFrame,
    current_max: np.ndarray,
    forecast_high: np.ndarray,
    max_delta: int,
) -> np.ndarray:
    kwargs = {
        "mu": params["mu"].to_numpy(dtype=float),
        "sigma": params["sigma"].to_numpy(dtype=float),
        "distribution": str(params["distribution_type"].iloc[0]),
        "df": params["df"].to_numpy(dtype=float) if "df" in params and params["df"].notna().any() else None,
        "skew": params["skew"].to_numpy(dtype=float) if "skew" in params and params["skew"].notna().any() else None,
    }

    def final_cdf(boundary: np.ndarray) -> np.ndarray:
        return distribution_cdf(boundary - forecast_high, **kwargs)

    columns = [final_cdf(current_max + 0.5)]
    for delta in range(1, max_delta + 1):
        upper = final_cdf(current_max + delta + 0.5)
        lower = final_cdf(current_max + delta - 0.5)
        columns.append(np.clip(upper - lower, 0.0, 1.0))
    columns.append(1.0 - final_cdf(current_max + max_delta + 0.5))
    matrix = np.column_stack(columns)
    matrix = np.clip(matrix, 0.0, 1.0)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def _score_slices(
    aligned: pd.DataFrame,
    hurdle_probability: np.ndarray,
    legacy_probability: np.ndarray,
    max_delta: int,
) -> pd.DataFrame:
    hour = aligned["prediction_time"].dt.hour
    slices = {
        "overall": np.ones(len(aligned), dtype=bool),
        "before_12": (hour < 12).to_numpy(),
        "12_to_4": ((hour >= 12) & (hour < 16)).to_numpy(),
        "after_4": (hour >= 16).to_numpy(),
    }
    target = aligned["remaining_increase"].to_numpy(dtype=int)
    rows: list[dict] = []
    for slice_name, mask in slices.items():
        if not mask.any():
            continue
        for model_name, probability in {
            "full_hurdle": hurdle_probability,
            "legacy_single_ngboost": legacy_probability,
        }.items():
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
    exceedance_bundle, conditional_bundle, predictor, conditional_model = _load_frozen_components()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    test_start = str(config["splits"]["test_start"])
    legacy_engine, legacy_integrity_warning = _load_legacy_engine()
    aligned, old_input = _aligned_test_rows(test_start, legacy_engine.feature_columns)
    p_increase = predictor.predict_proba(aligned)
    max_delta = 10
    hurdle_probability = integer_delta_probabilities(
        p_increase, conditional_model, aligned, max_delta=max_delta
    )
    legacy_params = legacy_engine.predict_distribution_params(old_input)
    legacy_probability = _legacy_probability_matrix(
        legacy_params,
        aligned["current_max_so_far"].to_numpy(dtype=float),
        aligned["legacy__forecast_high"].to_numpy(dtype=float),
        max_delta,
    )
    scores = _score_slices(aligned, hurdle_probability, legacy_probability, max_delta)
    scores.to_csv(OUTPUT_DIR / "hurdle_vs_legacy_scores.csv", index=False)

    categories = ["delta_0", *[f"delta_{value}" for value in range(1, max_delta + 1)], f"delta_gt_{max_delta}"]
    export = aligned[["target_date", "prediction_time", "remaining_increase"]].copy()
    for index, category in enumerate(categories):
        export[f"hurdle_{category}"] = hurdle_probability[:, index]
        export[f"legacy_{category}"] = legacy_probability[:, index]
    export.to_csv(OUTPUT_DIR / "aligned_test_probabilities.csv", index=False)

    overall = scores.loc[scores["slice"].eq("overall")].set_index("model")
    late = scores.loc[scores["slice"].eq("after_4")].set_index("model")
    report = [
        "# Full Hurdle Distribution Backtest",
        "",
        f"Exact shared held-out timestamps: {len(aligned):,} (test begins {test_start}).",
        f"Exceedance winner: {exceedance_bundle['winner']}; conditional winner: {conditional_bundle['winner']}.",
        f"Legacy integrity warning: {legacy_integrity_warning or 'none'}.",
        "",
        "Both systems are scored on delta=0, each +1 deg F category through +10, and a >+10 tail.",
        "",
        scores.to_markdown(index=False),
        "",
        f"Overall NLL change (hurdle - legacy): {overall.loc['full_hurdle', 'multiclass_nll'] - overall.loc['legacy_single_ngboost', 'multiclass_nll']:.6f}.",
        f"After-4 PM Brier change (hurdle - legacy): {late.loc['full_hurdle', 'mean_bucket_brier'] - late.loc['legacy_single_ngboost', 'mean_bucket_brier']:.6f}.",
    ]
    (OUTPUT_DIR / "backtest_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(scores.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
