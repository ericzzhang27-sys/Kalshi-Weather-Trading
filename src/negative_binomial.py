from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma
from scipy.stats import nbinom

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import LogScore


def _validate_count_target(target: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=float)
    if not np.isfinite(y).all() or (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ValueError("Negative Binomial targets must be finite non-negative integers")
    return y.astype(int)


def _negative_log_likelihood(log_params: np.ndarray, target: np.ndarray) -> float:
    mu = np.exp(np.clip(log_params[0], -20.0, 20.0))
    alpha = np.exp(np.clip(log_params[1], -12.0, 8.0))
    size = 1.0 / alpha
    probability = size / (size + mu)
    return float(-np.sum(nbinom.logpmf(target, size, probability)))


class NegativeBinomialLogScore(LogScore):
    """NB2 log score parameterized by log(mu) and log(alpha)."""

    def score(self, target):
        return -self.dist.logpmf(target)

    def d_score(self, target):
        y = np.asarray(target, dtype=float)
        size = self.size
        denominator = size + self.mu
        gradient = np.zeros((len(y), 2), dtype=float)
        gradient[:, 0] = size * (self.mu - y) / denominator
        gradient[:, 1] = size * (
            digamma(y + size)
            - digamma(size)
            + np.log(size / denominator)
            + 1.0
            - (size + y) / denominator
        )
        return gradient

    def metric(self):
        # The challenger deliberately uses ordinary (non-natural) gradients.
        count = np.asarray(self.mu).reshape(-1).shape[0]
        return np.repeat(np.eye(2, dtype=float)[None, :, :], count, axis=0)


class NegativeBinomial(RegressionDistn):
    """NB2 count distribution with Var(Y|X) = mu + alpha * mu**2."""

    n_params = 2
    scores = [NegativeBinomialLogScore]

    def __init__(self, params):
        self._params = params
        self.logmu = np.asarray(params[0], dtype=float)
        self.logalpha = np.asarray(params[1], dtype=float)
        self.mu = np.exp(np.clip(self.logmu, -20.0, 20.0))
        self.alpha = np.exp(np.clip(self.logalpha, -12.0, 8.0))
        self.size = 1.0 / self.alpha
        self.probability = self.size / (self.size + self.mu)
        self.dist = nbinom(n=self.size, p=self.probability)

    @staticmethod
    def fit(target):
        y = _validate_count_target(target)
        mean = max(float(y.mean()), 1e-6)
        variance = float(y.var())
        alpha = max((variance - mean) / (mean * mean), 1e-4)
        fitted = minimize(
            _negative_log_likelihood,
            x0=np.log([mean, alpha]),
            args=(y,),
            method="L-BFGS-B",
            bounds=[(-20.0, 20.0), (-12.0, 8.0)],
        )
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise RuntimeError(f"Negative Binomial initialization failed: {fitted.message}")
        return fitted.x

    def sample(self, count):
        return np.asarray([self.dist.rvs() for _ in range(count)])

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    @property
    def params(self):
        return {
            "mu": self.mu,
            "alpha": self.alpha,
            "size": self.size,
            "probability": self.probability,
        }
