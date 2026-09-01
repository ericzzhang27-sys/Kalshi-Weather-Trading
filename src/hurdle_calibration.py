from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    probability = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(probability / (1.0 - probability))


def fit_platt_scaler(val_probs: np.ndarray, val_true: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(_logit(val_probs).reshape(-1, 1), np.asarray(val_true, dtype=int))
    return model


def fit_isotonic(val_probs: np.ndarray, val_true: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(val_probs, dtype=float), np.asarray(val_true, dtype=float))
    return model


def apply_platt(scaler: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return scaler.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


def apply_isotonic(model: IsotonicRegression, probabilities: np.ndarray) -> np.ndarray:
    return np.clip(model.transform(np.asarray(probabilities, dtype=float)), 0.0, 1.0)


def apply_calibrator(kind: str, calibrator: Any, probabilities: np.ndarray) -> np.ndarray:
    if kind in {"none", "raw"}:
        return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if kind == "platt":
        return apply_platt(calibrator, probabilities)
    if kind == "isotonic":
        return apply_isotonic(calibrator, probabilities)
    raise ValueError(f"Unknown calibrator type: {kind}")


def fit_calibrator(kind: str, probabilities: np.ndarray, y_true: np.ndarray) -> Any:
    if kind in {"none", "raw"}:
        return None
    if kind == "platt":
        return fit_platt_scaler(probabilities, y_true)
    if kind == "isotonic":
        return fit_isotonic(probabilities, y_true)
    raise ValueError(f"Unknown calibrator type: {kind}")


def evaluate_calibration_comparison(
    raw: np.ndarray,
    platt: np.ndarray,
    iso: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    from src.hurdle_evaluation import brier_score, calibration_error_metrics, logloss_score

    rows = []
    for name, probabilities in (("raw", raw), ("platt", platt), ("isotonic", iso)):
        rows.append(
            {
                "method": name,
                "brier": brier_score(y_true, probabilities),
                "log_loss": logloss_score(y_true, probabilities),
                **calibration_error_metrics(y_true, probabilities),
            }
        )
    return pd.DataFrame(rows)
