from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    bucket_brier_scores,
    compute_pit_values,
    interval_log_loss,
    negative_log_likelihood,
    prediction_interval_coverage,
    validate_bucket_probabilities,
)


def _simple_bucket_probs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cold": [0.8, 0.3, 0.1],
            "warm": [0.2, 0.7, 0.9],
        }
    )


def test_negative_log_likelihood_is_finite_for_simple_normal_inputs() -> None:
    nll = negative_log_likelihood(
        y_true=np.array([0.0, 1.0, -1.0]),
        mu=np.array([0.0, 0.5, -0.5]),
        sigma=np.array([1.0, 1.0, 1.0]),
    )

    assert np.isfinite(nll)


def test_prediction_interval_coverage_returns_one_row_per_level() -> None:
    coverage = prediction_interval_coverage(
        y_true=np.array([0.0, 0.5, -0.5]),
        mu=np.zeros(3),
        sigma=np.ones(3),
        levels=(0.5, 0.8, 0.9),
    )

    assert coverage["level"].tolist() == [0.5, 0.8, 0.9]
    assert len(coverage) == 3


def test_bucket_brier_scores_are_between_zero_and_one() -> None:
    scores = bucket_brier_scores(
        _simple_bucket_probs(),
        pd.Series(["cold", "warm", "warm"]),
    )

    assert ((scores["brier_score"] >= 0.0) & (scores["brier_score"] <= 1.0)).all()


def test_interval_log_loss_is_finite() -> None:
    loss = interval_log_loss(
        _simple_bucket_probs(),
        pd.Series(["cold", "warm", "warm"]),
    )

    assert np.isfinite(loss)


def test_pit_values_are_between_zero_and_one() -> None:
    pit = compute_pit_values(
        y_true=np.array([-1.0, 0.0, 1.0]),
        mu=np.zeros(3),
        sigma=np.ones(3),
    )

    assert ((pit >= 0.0) & (pit <= 1.0)).all()


def test_bucket_probability_validation_catches_bad_row_sums() -> None:
    bad_probs = pd.DataFrame({"cold": [0.2, 0.3], "warm": [0.4, 0.3]})

    with pytest.raises(ValueError, match="row sums"):
        validate_bucket_probabilities(bad_probs)


def test_realized_bucket_labels_missing_from_columns_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="Realized bucket labels"):
        interval_log_loss(
            _simple_bucket_probs(),
            pd.Series(["cold", "hot", "warm"]),
        )
