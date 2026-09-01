from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor


ARTIFACT_TYPE = "discrete_continuation_ngboost"


def _positive_integer_target(frame: pd.DataFrame, target_col: str = "remaining_increase") -> np.ndarray:
    target = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    if len(target) == 0:
        raise ValueError("Discrete continuation model received no rows")
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Conditional target must be finite and strictly positive")
    if not np.allclose(target, np.round(target)):
        raise ValueError("Conditional target must be integer-valued")
    return target.astype(int)


def _numeric_features(frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    columns = list(features)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing discrete-continuation features: {missing}")
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def threshold_sample_table(
    frame: pd.DataFrame,
    *,
    target_col: str = "remaining_increase",
    date_col: str = "target_date",
    max_threshold: int | None = None,
) -> pd.DataFrame:
    """Describe each q_k risk set: P(delta >= k+1 | delta >= k)."""
    target = _positive_integer_target(frame, target_col)
    dates = pd.to_datetime(frame[date_col], errors="raise").dt.normalize()
    upper = int(target.max() - 1) if max_threshold is None else int(max_threshold)
    rows: list[dict[str, Any]] = []
    for threshold in range(1, max(0, upper) + 1):
        at_risk = target >= threshold
        outcome = target[at_risk] >= threshold + 1
        risk_dates = dates.loc[at_risk]
        rows.append(
            {
                "threshold": threshold,
                "target": f"delta>={threshold + 1}|delta>={threshold}",
                "n_at_risk": int(at_risk.sum()),
                "n_continue": int(outcome.sum()),
                "n_stop": int((~outcome).sum()),
                "prevalence": float(outcome.mean()) if len(outcome) else float("nan"),
                "n_days_at_risk": int(risk_dates.nunique()),
                "n_days_continue": int(risk_dates.loc[outcome].nunique()),
                "n_days_stop": int(risk_dates.loc[~outcome].nunique()),
            }
        )
    return pd.DataFrame(rows)


def choose_tail_start(
    folds: Iterable[tuple[str, pd.DataFrame, pd.DataFrame]],
    *,
    target_col: str = "remaining_increase",
    date_col: str = "target_date",
    min_train_at_risk: int = 500,
    min_train_per_class: int = 100,
    min_validation_at_risk: int = 100,
    min_validation_per_class: int = 30,
) -> tuple[int, pd.DataFrame]:
    """Select the first pooled tail value using pre-test fold support only."""
    fold_list = list(folds)
    if not fold_list:
        raise ValueError("At least one pre-test validation fold is required")
    maximum = min(
        int(_positive_integer_target(pd.concat([train, validation]), target_col).max() - 1)
        for _, train, validation in fold_list
    )
    rows: list[dict[str, Any]] = []
    last_eligible = 0
    stopped = False
    for threshold in range(1, maximum + 1):
        for fold_name, train, validation in fold_list:
            train_row = threshold_sample_table(
                train, target_col=target_col, date_col=date_col, max_threshold=threshold
            ).iloc[-1]
            validation_row = threshold_sample_table(
                validation, target_col=target_col, date_col=date_col, max_threshold=threshold
            ).iloc[-1]
            eligible = bool(
                train_row["n_at_risk"] >= min_train_at_risk
                and min(train_row["n_continue"], train_row["n_stop"]) >= min_train_per_class
                and validation_row["n_at_risk"] >= min_validation_at_risk
                and min(validation_row["n_continue"], validation_row["n_stop"])
                >= min_validation_per_class
            )
            rows.append(
                {
                    "fold": fold_name,
                    "threshold": threshold,
                    "train_at_risk": int(train_row["n_at_risk"]),
                    "train_continue": int(train_row["n_continue"]),
                    "train_stop": int(train_row["n_stop"]),
                    "validation_at_risk": int(validation_row["n_at_risk"]),
                    "validation_continue": int(validation_row["n_continue"]),
                    "validation_stop": int(validation_row["n_stop"]),
                    "eligible": eligible,
                }
            )
        threshold_rows = rows[-len(fold_list) :]
        if all(row["eligible"] for row in threshold_rows) and not stopped:
            last_eligible = threshold
        else:
            stopped = True
    if last_eligible < 1:
        raise ValueError("No conditional threshold has adequate training/validation support")
    diagnostics = pd.DataFrame(rows)
    diagnostics["selected"] = diagnostics["threshold"] <= last_eligible
    return last_eligible + 1, diagnostics


def _fit_threshold_classifier(
    risk_frame: pd.DataFrame,
    features: list[str],
    outcome: np.ndarray,
    *,
    n_estimators: int,
    learning_rate: float,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[NGBClassifier, SimpleImputer]:
    if np.unique(outcome).size != 2:
        raise ValueError("Every continuation threshold requires both outcome classes")
    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(_numeric_features(risk_frame, features))
    base = DecisionTreeRegressor(
        criterion="friedman_mse",
        max_depth=3,
        min_samples_leaf=min(min_samples_leaf, max(2, len(risk_frame) // 10)),
        random_state=random_state,
    )
    model = NGBClassifier(
        Dist=Bernoulli,
        Base=base,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        minibatch_frac=0.8,
        natural_gradient=True,
        random_state=random_state,
        verbose=False,
    )
    model.fit(transformed, outcome.astype(int))
    return model, imputer


def train_ordinal_hazard_model(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    target_col: str = "remaining_increase",
    date_col: str = "target_date",
    tail_start: int | None = None,
    max_delta: int | None = None,
    n_estimators: int = 150,
    learning_rate: float = 0.03,
    min_samples_leaf: int = 50,
    random_state: int = 42,
    calibrators: Any = None,
) -> dict[str, Any]:
    """Fit separate Bernoulli NGBoost q_k continuation classifiers.

    ``max_delta`` is a compatibility alias for ``tail_start``. The resulting
    support is 1, ..., tail_start-1, tail_start+.
    """
    if calibrators is not None:
        raise ValueError("Post-hoc calibrators are not part of the first NGBoost implementation")
    if tail_start is None:
        tail_start = max_delta
    if tail_start is None or int(tail_start) < 2:
        raise ValueError("tail_start must be at least 2")
    tail_start = int(tail_start)
    target = _positive_integer_target(train_df, target_col)
    threshold_models: list[dict[str, Any]] = []
    for threshold in range(1, tail_start):
        at_risk = target >= threshold
        risk_frame = train_df.loc[at_risk]
        outcome = target[at_risk] >= threshold + 1
        model, imputer = _fit_threshold_classifier(
            risk_frame,
            features,
            outcome,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + threshold,
        )
        threshold_models.append(
            {
                "threshold": threshold,
                "model": model,
                "imputer": imputer,
                "n_train": int(at_risk.sum()),
                "n_continue": int(outcome.sum()),
                "n_stop": int((~outcome).sum()),
                "prevalence": float(outcome.mean()),
            }
        )
    return {
        "type": ARTIFACT_TYPE,
        "models": threshold_models,
        "features": list(features),
        "target_col": target_col,
        "date_col": date_col,
        "tail_start": tail_start,
        "thresholds": list(range(1, tail_start)),
        "category_labels": [*map(str, range(1, tail_start)), f">={tail_start}"],
        "probability_definition": "q_k=P(delta>=k+1|delta>=k,X)",
    }


def predict_continuation_probabilities(
    artifact: dict[str, Any], frame: pd.DataFrame | dict[str, Any]
) -> np.ndarray:
    if artifact.get("type") != ARTIFACT_TYPE:
        raise ValueError("Unsupported discrete-continuation artifact")
    working = pd.DataFrame([frame]) if isinstance(frame, dict) else frame
    numeric = _numeric_features(working, artifact["features"])
    columns: list[np.ndarray] = []
    for threshold_model in artifact["models"]:
        transformed = threshold_model["imputer"].transform(numeric)
        probability = np.asarray(threshold_model["model"].predict_proba(transformed), dtype=float)
        probability = probability[:, 1] if probability.ndim == 2 else probability.reshape(-1)
        columns.append(np.clip(probability, 1e-8, 1.0 - 1e-8))
    return np.column_stack(columns)


def hazards_to_probabilities(continuation: np.ndarray) -> np.ndarray:
    """Convert q_k continuation probabilities to exact outcomes and a final tail."""
    q = np.asarray(continuation, dtype=float)
    if q.ndim != 2 or q.shape[1] < 1 or not np.isfinite(q).all():
        raise ValueError("Continuation probabilities must be a finite 2-D matrix")
    if ((q < 0) | (q > 1)).any():
        raise ValueError("Continuation probabilities must be inside [0, 1]")
    survival_before = np.column_stack([np.ones(len(q)), np.cumprod(q[:, :-1], axis=1)])
    exact = survival_before * (1.0 - q)
    tail = np.prod(q, axis=1)
    result = np.column_stack([exact, tail])
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Conditional probabilities do not sum to one")
    return np.clip(result, 0.0, 1.0)


def predict_ordinal_probabilities(
    artifact: dict[str, Any],
    frame: pd.DataFrame | dict[str, Any],
    *,
    calibrated: bool = True,
) -> np.ndarray:
    del calibrated
    return hazards_to_probabilities(predict_continuation_probabilities(artifact, frame))


def evaluate_threshold_probabilities(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    reference_prevalence: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1.0 - 1e-15)
    brier = float(np.mean((p - y) ** 2))
    reference_brier = float(np.mean((float(reference_prevalence) - y) ** 2))
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier": brier,
        "brier_skill_score": float(1.0 - brier / reference_brier)
        if reference_brier > 0
        else float("nan"),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "calibration_gap": float(p.mean() - y.mean()),
    }


def reliability_table(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        mask = (p >= edges[index]) & (p < edges[index + 1])
        if index == bins - 1:
            mask |= p == 1.0
        if mask.any():
            rows.append(
                {
                    "bin": index + 1,
                    "lower": edges[index],
                    "upper": edges[index + 1],
                    "n": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()),
                    "observed_frequency": float(y[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_ordinal_probabilities(
    probability: np.ndarray,
    target: np.ndarray | pd.Series,
    tail_start: int,
) -> dict[str, float | int]:
    y = np.asarray(target, dtype=int)
    category = np.minimum(y, tail_start) - 1
    p = np.asarray(probability, dtype=float)
    if p.shape != (len(y), tail_start):
        raise ValueError("Conditional probability matrix has the wrong shape")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Conditional probability rows must sum to one")
    realized = np.clip(p[np.arange(len(y)), category], 1e-15, 1.0)
    one_hot = np.eye(tail_start)[category]
    cdf = np.cumsum(p[:, :-1], axis=1)
    actual_cdf = y[:, None] <= np.arange(1, tail_start)[None, :]
    cdf_gap = cdf.mean(axis=0) - actual_cdf.mean(axis=0)
    return {
        "n": int(len(y)),
        "interval_nll": float(-np.mean(np.log(realized))),
        "mean_bucket_brier": float(np.mean((p - one_hot) ** 2)),
        "ordinal_crps": float(np.mean(np.sum((cdf - actual_cdf) ** 2, axis=1))),
        "cdf_calibration_error": float(np.mean(np.abs(cdf_gap))),
    }


def evaluate_ordinal_model(
    artifact: dict[str, Any], frame: pd.DataFrame, target_col: str = "remaining_increase"
) -> dict[str, float | int]:
    target = _positive_integer_target(frame, target_col)
    return evaluate_ordinal_probabilities(
        predict_ordinal_probabilities(artifact, frame), target, int(artifact["tail_start"])
    )


predict_raw_hazards = predict_continuation_probabilities


def load_ordinal_hazard_model(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    if artifact.get("type") != ARTIFACT_TYPE:
        raise ValueError(f"Invalid discrete-continuation model: {path}")
    return artifact
