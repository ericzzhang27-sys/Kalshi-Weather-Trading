from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ordinal_hazard_model import (
    choose_tail_start,
    evaluate_ordinal_model,
    evaluate_threshold_probabilities,
    hazards_to_probabilities,
    predict_continuation_probabilities,
    predict_ordinal_probabilities,
    reliability_table,
    threshold_sample_table,
    train_ordinal_hazard_model,
)


def _training_frame(days: int = 24) -> pd.DataFrame:
    rows = []
    for day_index in range(days):
        for row_index in range(10):
            x = float(row_index + day_index % 4)
            rows.append(
                {
                    "target_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_index),
                    "x": x,
                    "remaining_increase": 1 + int((row_index + day_index) % 6),
                }
            )
    return pd.DataFrame(rows)


def test_continuation_probabilities_construct_expected_distribution():
    continuation = np.array([[0.5, 0.25, 0.1], [0.1, 0.2, 0.3]])
    probability = hazards_to_probabilities(continuation)
    assert probability.shape == (2, 4)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.all(probability >= 0)
    assert np.allclose(probability[0], [0.5, 0.375, 0.1125, 0.0125])


def test_threshold_sample_table_uses_nested_risk_sets():
    frame = pd.DataFrame(
        {
            "target_date": pd.date_range("2024-01-01", periods=5),
            "remaining_increase": [1, 2, 2, 3, 4],
        }
    )
    table = threshold_sample_table(frame, max_threshold=3)
    assert table["n_at_risk"].tolist() == [5, 4, 2]
    assert table["n_continue"].tolist() == [4, 2, 1]
    assert table["n_stop"].tolist() == [1, 2, 1]


def test_tail_selection_stops_at_first_unstable_threshold():
    train = _training_frame(16)
    validation = _training_frame(8)
    tail_start, diagnostics = choose_tail_start(
        [("fold", train, validation)],
        min_train_at_risk=40,
        min_train_per_class=10,
        min_validation_at_risk=20,
        min_validation_per_class=5,
    )
    assert tail_start >= 2
    selected = diagnostics.loc[diagnostics["selected"], "threshold"]
    assert selected.tolist() == list(range(1, tail_start))
    if (diagnostics["threshold"] == tail_start).any():
        assert not diagnostics.loc[diagnostics["threshold"] == tail_start, "eligible"].all()


def test_separate_ngboost_models_train_predict_and_score():
    training = _training_frame()
    artifact = train_ordinal_hazard_model(
        training,
        ["x"],
        tail_start=5,
        n_estimators=5,
        learning_rate=0.1,
        min_samples_leaf=5,
    )
    continuation = predict_continuation_probabilities(artifact, training)
    probability = predict_ordinal_probabilities(artifact, training)
    assert artifact["type"] == "discrete_continuation_ngboost"
    assert [model["threshold"] for model in artifact["models"]] == [1, 2, 3, 4]
    assert continuation.shape == (len(training), 4)
    assert probability.shape == (len(training), 5)
    assert np.allclose(probability.sum(axis=1), 1.0)
    metrics = evaluate_ordinal_model(artifact, training)
    assert np.isfinite(metrics["interval_nll"])
    assert 0 <= metrics["mean_bucket_brier"] <= 1


def test_threshold_metrics_include_skill_and_reliability():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = evaluate_threshold_probabilities(y, probability, reference_prevalence=0.5)
    assert metrics["brier_skill_score"] > 0
    assert np.isfinite(metrics["log_loss"])
    table = reliability_table(y, probability, bins=5)
    assert table["n"].sum() == len(y)
