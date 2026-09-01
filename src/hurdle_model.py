from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def climatological_baseline_fit(
    df: pd.DataFrame,
    time_col: str = "prediction_time",
    target_col: str = "will_increase",
    *,
    prior_strength: float = 20.0,
) -> dict[str, Any]:
    """Fit smoothed P(increase | five-minute time-of-day bucket)."""
    minute = pd.to_datetime(df[time_col]).dt.hour * 60 + pd.to_datetime(df[time_col]).dt.minute
    target = pd.to_numeric(df[target_col], errors="raise").astype(int)
    global_rate = float(target.mean())
    grouped = pd.DataFrame({"minute": minute, "target": target}).groupby("minute")["target"].agg(["sum", "count"])
    rates = (grouped["sum"] + prior_strength * global_rate) / (grouped["count"] + prior_strength)
    return {
        "type": "climatology_minute_of_day",
        "minute_rates": {int(key): float(value) for key, value in rates.items()},
        "minute_counts": {int(key): int(value) for key, value in grouped["count"].items()},
        "global_rate": global_rate,
        "time_col": time_col,
        "prior_strength": float(prior_strength),
    }


def climatological_predict(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    timestamp = pd.to_datetime(df[model["time_col"]])
    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    probability = minute.map(model["minute_rates"]).fillna(model["global_rate"])
    return np.clip(probability.to_numpy(dtype=float), 0.0, 1.0)


def _numeric_frame(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    columns = list(feature_cols)
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Missing hurdle features: {missing}")
    return df[columns].apply(pd.to_numeric, errors="coerce")


def train_logistic_regression(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "will_increase",
    use_scaler: bool = True,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    # Probability calibration is the objective; class weighting is intentionally absent.
    steps.append(("clf", LogisticRegression(max_iter=2000, solver="lbfgs")))
    model = Pipeline(steps)
    model.fit(_numeric_frame(train_df, feature_cols), train_df[target_col].astype(int))
    return model


def train_boosted_classifier(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "will_increase",
    kind: str = "histgb",
    random_state: int = 42,
) -> Any:
    X = _numeric_frame(train_df, feature_cols)
    y = train_df[target_col].astype(int).to_numpy()
    if kind == "ngb":
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli

        imputer = SimpleImputer(strategy="median")
        transformed = imputer.fit_transform(X)
        model = NGBClassifier(
            Dist=Bernoulli,
            n_estimators=150,
            learning_rate=0.03,
            minibatch_frac=0.8,
            natural_gradient=True,
            random_state=random_state,
            verbose=False,
        )
        model.fit(transformed, y)
        return {"type": "ngb", "imputer": imputer, "model": model, "features": list(feature_cols)}
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=4,
                        min_child_samples=50,
                        subsample=0.8,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
        model.fit(X, y)
        return model
    if kind != "histgb":
        raise ValueError(f"Unknown boosted classifier kind: {kind}")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def predict_proba(model: Any, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    if isinstance(model, dict) and model.get("type") == "climatology_minute_of_day":
        return climatological_predict(model, df)
    if isinstance(model, dict) and model.get("type") == "ngb":
        X = _numeric_frame(df, model["features"])
        probability = np.asarray(model["model"].predict_proba(model["imputer"].transform(X)))
    else:
        if feature_cols is None:
            raise ValueError("feature_cols is required for sklearn hurdle models")
        probability = np.asarray(model.predict_proba(_numeric_frame(df, feature_cols)))
    if probability.ndim == 2:
        probability = probability[:, 1] if probability.shape[1] > 1 else probability[:, 0]
    return np.clip(probability.reshape(-1), 0.0, 1.0)


def evaluate_classifier(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    prediction = probability >= 0.5
    tp = int(np.sum(prediction & (y == 1)))
    fp = int(np.sum(prediction & (y == 0)))
    fn = int(np.sum(~prediction & (y == 1)))
    return {
        "brier": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
        "roc_auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, probability)) if np.unique(y).size == 2 else float("nan"),
        "accuracy": float(np.mean(prediction == y)),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "n": int(len(y)),
        "mean_pred": float(probability.mean()),
        "empirical_rate": float(y.mean()),
    }


def expanding_window_splits(
    df: pd.DataFrame,
    date_col: str = "target_date",
    folds: list[dict[str, Any]] | None = None,
    *,
    test_start: str = "2025-01-01",
    minimum_training_years: int = 2,
) -> list[dict[str, Any]]:
    """Return whole-day, yearly expanding folds strictly before final test."""
    if folds is not None:
        return folds
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    test_start_date = pd.Timestamp(test_start).normalize()
    pretest_years = sorted(dates.loc[dates < test_start_date].dt.year.unique().tolist())
    if len(pretest_years) <= minimum_training_years:
        raise ValueError("Not enough pre-test years for expanding-window validation")
    result: list[dict[str, Any]] = []
    for validation_year in pretest_years[minimum_training_years:]:
        result.append(
            {
                "name": f"validation_{validation_year}",
                "train_start": str(dates.min().date()),
                "train_end": f"{validation_year - 1}-12-31",
                "val_start": f"{validation_year}-01-01",
                "val_end": f"{validation_year}-12-31",
            }
        )
    return result


def materialize_fold(
    df: pd.DataFrame,
    fold: dict[str, Any],
    date_col: str = "target_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    train_end = pd.Timestamp(fold["train_end"]).normalize()
    validation_start = pd.Timestamp(fold["val_start"]).normalize()
    validation_end = pd.Timestamp(fold["val_end"]).normalize()
    train = df.loc[dates <= train_end].copy()
    validation = df.loc[(dates >= validation_start) & (dates <= validation_end)].copy()
    if train.empty or validation.empty:
        raise ValueError(f"Empty expanding fold: {fold}")
    if set(pd.to_datetime(train[date_col]).dt.date) & set(pd.to_datetime(validation[date_col]).dt.date):
        raise AssertionError("A weather day appears in both train and validation")
    return train, validation


@dataclass
class HurdlePredictor:
    """Live inference facade returning calibrated P(final high increases)."""

    classifier: Any
    feature_names: list[str]
    calibrator: Any = None
    calibrator_type: str = "none"

    def predict_raw(self, features: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        frame = pd.DataFrame([features]) if isinstance(features, dict) else features
        return predict_proba(self.classifier, frame, self.feature_names)

    def predict_proba(self, features: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        from src.hurdle_calibration import apply_calibrator

        raw = self.predict_raw(features)
        calibrated = apply_calibrator(self.calibrator_type, self.calibrator, raw)
        if not np.isfinite(calibrated).all() or ((calibrated < 0) | (calibrated > 1)).any():
            raise ValueError("Hurdle probabilities must be finite and inside [0, 1]")
        return calibrated


def load_hurdle_predictor(
    classifier_path: str | Path,
    features_path: str | Path,
    calibrator_path: str | Path | None = None,
    calibrator_type: str = "none",
) -> HurdlePredictor:
    import json

    with Path(classifier_path).open("rb") as handle:
        classifier = pickle.load(handle)
    feature_payload = json.loads(Path(features_path).read_text(encoding="utf-8"))
    calibrator = None
    if calibrator_path is not None and Path(calibrator_path).exists():
        with Path(calibrator_path).open("rb") as handle:
            calibrator = pickle.load(handle)
    return HurdlePredictor(classifier, list(feature_payload["features"]), calibrator, calibrator_type)
