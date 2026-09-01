from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost.manifold import manifold
from ngboost.scores import LogScore


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conditional_increase_model import (
    conditional_cdf,
    conditional_dispersion,
    predict_conditional_distribution,
    train_shifted_negative_binomial_ngboost,
)
from src.negative_binomial import NegativeBinomial


def test_nb2_parameters_have_requested_mean_and_variance():
    distribution = NegativeBinomial(np.log(np.array([[2.0, 4.0], [0.5, 1.0]])))
    assert np.allclose(distribution.dist.mean(), [2.0, 4.0])
    assert np.allclose(distribution.dist.var(), [4.0, 20.0])


def test_nb2_analytic_score_matches_finite_difference():
    params = np.log(np.array([[2.3, 4.1, 1.2], [0.7, 0.4, 1.1]]))
    target = np.array([0, 3, 5])
    scored_distribution = manifold(LogScore, NegativeBinomial)
    distribution = scored_distribution(params)
    analytic = distribution.d_score(target)
    epsilon = 1e-6
    numeric = np.zeros_like(analytic)
    for parameter in range(2):
        plus = params.copy()
        minus = params.copy()
        plus[parameter] += epsilon
        minus[parameter] -= epsilon
        numeric[:, parameter] = (
            scored_distribution(plus).score(target) - scored_distribution(minus).score(target)
        ) / (2 * epsilon)
    assert np.allclose(analytic, numeric, rtol=2e-5, atol=2e-5)


def test_conditional_dispersion_detects_overdispersion():
    metrics = conditional_dispersion(np.array([0, 0, 0, 1, 1, 8]))
    assert metrics["variance_y"] > metrics["mean_y"]
    assert metrics["dispersion"] > 1.0


def test_shifted_negative_binomial_trains_and_returns_discrete_cdf():
    rng = np.random.default_rng(7)
    rows = 180
    x = np.linspace(0, 3, rows)
    mu = 0.5 + x
    alpha = 0.8
    size = 1.0 / alpha
    shifted = rng.negative_binomial(size, size / (size + mu))
    frame = pd.DataFrame({"x": x, "remaining_increase": shifted + 1})
    artifact = train_shifted_negative_binomial_ngboost(
        frame,
        ["x"],
        n_estimators=5,
        learning_rate=0.03,
        min_samples_leaf=10,
    )
    distribution = predict_conditional_distribution(artifact, frame.iloc[:8])
    cdf = conditional_cdf(artifact, frame.iloc[:8], np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    assert artifact["type"] == "shifted_negative_binomial_ngboost"
    assert np.isfinite(distribution.params["mu"]).all()
    assert (distribution.params["alpha"] > 0).all()
    assert np.all((cdf >= 0) & (cdf <= 1))
