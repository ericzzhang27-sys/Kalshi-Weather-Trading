from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from src.backtest.fees import kalshi_taker_fee
from .interfaces import TradeDecision


def project_simplex(values: Sequence[float]) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
        raise ValueError("simplex input must be a finite one-dimensional vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    positive = np.flatnonzero(ordered - cumulative / (np.arange(len(vector)) + 1) > 0)
    theta = cumulative[positive[-1]] / float(positive[-1] + 1)
    projected = np.maximum(vector - theta, 0.0)
    return projected / projected.sum()


def project_bounded_simplex(
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Project to sum-one probabilities while respecting executable quote bounds."""
    vector = np.asarray(values, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if vector.shape != lo.shape or vector.shape != hi.shape or vector.ndim != 1:
        raise ValueError("values and bounds must be same-length vectors")
    if not np.isfinite(vector).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("bounded simplex inputs must be finite")
    if (lo < 0).any() or (hi > 1).any() or (lo > hi).any():
        raise ValueError("invalid probability bounds")
    if lo.sum() > 1.0 + tolerance or hi.sum() < 1.0 - tolerance:
        raise ValueError("quote bounds do not admit a coherent event distribution")
    left = float(np.min(vector - hi) - 1.0)
    right = float(np.max(vector - lo) + 1.0)
    for _ in range(100):
        midpoint = (left + right) / 2.0
        candidate = np.clip(vector - midpoint, lo, hi)
        if candidate.sum() > 1.0:
            left = midpoint
        else:
            right = midpoint
    result = np.clip(vector - (left + right) / 2.0, lo, hi)
    residual = 1.0 - result.sum()
    if abs(residual) > tolerance:
        capacity = (hi - result) if residual > 0 else (result - lo)
        eligible = np.flatnonzero(capacity > tolerance)
        if not len(eligible):
            raise ValueError("bounded simplex projection could not resolve its sum")
        result[eligible[0]] += residual
    return result


def coherent_market_distribution(
    yes_bids: Sequence[float | None],
    yes_asks: Sequence[float | None],
) -> np.ndarray:
    """Reconstruct one coherent event distribution from contemporaneous quotes."""
    if len(yes_bids) != len(yes_asks) or len(yes_bids) < 2:
        raise ValueError("bid and ask vectors must have equal length >= 2")
    bid = np.array([np.nan if value is None else float(value) for value in yes_bids])
    ask = np.array([np.nan if value is None else float(value) for value in yes_asks])
    if np.any(np.isfinite(bid) & ((bid < 0) | (bid > 1))) or np.any(np.isfinite(ask) & ((ask < 0) | (ask > 1))):
        raise ValueError("market probabilities must be in [0, 1]")
    if np.any(np.isfinite(bid) & np.isfinite(ask) & (bid > ask)):
        raise ValueError("crossed bid/ask inputs cannot define fair value")
    midpoint = np.where(np.isfinite(bid) & np.isfinite(ask), (bid + ask) / 2.0, np.where(np.isfinite(bid), bid, ask))
    if not np.isfinite(midpoint).all():
        raise ValueError("every outcome needs at least one contemporaneous quote")
    lower = np.where(np.isfinite(bid), bid, 0.0)
    upper = np.where(np.isfinite(ask), ask, 1.0)
    try:
        return project_bounded_simplex(midpoint, lower, upper)
    except ValueError:
        return project_simplex(midpoint)


def _clip_probabilities(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


@dataclass
class LogOddsPool:
    """Regularized weather-plus-market pool, fit only on prior/cross-fitted rows."""

    regularization_c: float = 0.25
    random_state: int = 42
    _model: LogisticRegression | None = None

    def fit(
        self,
        p_weather: np.ndarray,
        p_market: np.ndarray,
        outcomes: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "LogOddsPool":
        weather = np.asarray(p_weather, dtype=float)
        market = np.asarray(p_market, dtype=float)
        truth = np.asarray(outcomes, dtype=int)
        if weather.shape != market.shape or weather.ndim != 2 or len(truth) != len(weather):
            raise ValueError("weather, market, and outcome shapes are incompatible")
        if not np.allclose(weather.sum(axis=1), 1.0, atol=1e-6) or not np.allclose(market.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("weather and market distributions must be coherent")
        one_hot = np.eye(weather.shape[1], dtype=int)[truth].ravel()
        features = np.column_stack((logit(_clip_probabilities(weather)).ravel(), logit(_clip_probabilities(market)).ravel()))
        self._model = LogisticRegression(C=self.regularization_c, random_state=self.random_state, max_iter=2000)
        weights = None
        if sample_weight is not None:
            event_weights = np.asarray(sample_weight, dtype=float)
            if event_weights.shape != (len(weather),) or not np.isfinite(event_weights).all() or (event_weights <= 0).any():
                raise ValueError("sample_weight must contain one positive finite weight per event")
            weights = np.repeat(event_weights / weather.shape[1], weather.shape[1])
        self._model.fit(features, one_hot, sample_weight=weights)
        return self

    def predict(self, p_weather: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("log-odds pool has not been fit")
        weather = np.asarray(p_weather, dtype=float)
        market = np.asarray(p_market, dtype=float)
        if weather.shape != market.shape or weather.ndim != 2:
            raise ValueError("weather and market arrays must have the same 2D shape")
        features = np.column_stack((logit(_clip_probabilities(weather)).ravel(), logit(_clip_probabilities(market)).ravel()))
        raw = self._model.predict_proba(features)[:, 1].reshape(weather.shape)
        return np.apply_along_axis(project_simplex, 1, raw)


def cross_fitted_log_odds_pool(
    p_weather: np.ndarray,
    p_market: np.ndarray,
    outcomes: np.ndarray,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    *,
    regularization_c: float = 0.25,
    random_state: int = 42,
) -> np.ndarray:
    result = np.full_like(np.asarray(p_weather, dtype=float), np.nan)
    for train_indices, validation_indices in folds:
        train = np.asarray(train_indices, dtype=int)
        validation = np.asarray(validation_indices, dtype=int)
        model = LogOddsPool(regularization_c=regularization_c, random_state=random_state)
        model.fit(p_weather[train], p_market[train], outcomes[train])
        result[validation] = model.predict(p_weather[validation], p_market[validation])
    return result


@dataclass
class BinaryLogOddsPool:
    """Log-odds pool for variable-size event schemas; normalize per event after prediction."""

    regularization_c: float = 0.25
    random_state: int = 42
    _model: LogisticRegression | None = None

    def fit(
        self,
        p_weather: Sequence[float],
        p_market: Sequence[float],
        outcomes: Sequence[int],
        *,
        sample_weight: Sequence[float] | None = None,
    ) -> "BinaryLogOddsPool":
        weather = _clip_probabilities(np.asarray(p_weather, dtype=float))
        market = _clip_probabilities(np.asarray(p_market, dtype=float))
        truth = np.asarray(outcomes, dtype=int)
        if weather.ndim != 1 or weather.shape != market.shape or truth.shape != weather.shape:
            raise ValueError("binary pool inputs must be same-length vectors")
        if set(np.unique(truth)) - {0, 1} or len(np.unique(truth)) < 2:
            raise ValueError("binary pool outcomes must contain both classes")
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights is not None and (weights.shape != weather.shape or not np.isfinite(weights).all() or (weights <= 0).any()):
            raise ValueError("sample weights must be positive, finite, and aligned")
        features = np.column_stack((logit(weather), logit(market)))
        self._model = LogisticRegression(C=self.regularization_c, random_state=self.random_state, max_iter=2000)
        self._model.fit(features, truth, sample_weight=weights)
        return self

    def predict_event(self, p_weather: Sequence[float], p_market: Sequence[float]) -> np.ndarray:
        if self._model is None:
            raise ValueError("binary log-odds pool has not been fit")
        weather = _clip_probabilities(np.asarray(p_weather, dtype=float))
        market = _clip_probabilities(np.asarray(p_market, dtype=float))
        if weather.ndim != 1 or weather.shape != market.shape:
            raise ValueError("event weather and market probabilities must be aligned vectors")
        raw = self._model.predict_proba(np.column_stack((logit(weather), logit(market))))[:, 1]
        return project_simplex(raw)


def conservative_trade_decision(
    *,
    side: str,
    fair_probability: float,
    lower_probability: float,
    upper_probability: float,
    executable_price: float,
    slippage_ticks: int = 1,
    tick_size: float = 0.01,
    fee_rate: float = 0.07,
    min_lower_edge: float = 0.0,
    contracts: int = 1,
    model_version: str,
    risk_state: str = "OK",
) -> TradeDecision:
    side_upper = side.upper()
    if side_upper not in {"YES", "NO"}:
        raise ValueError("side must be YES or NO")
    if contracts < 1:
        raise ValueError("contracts must be a positive whole number")
    price = float(executable_price)
    if not 0.0 < price < 1.0:
        raise ValueError("executable price must be strictly between zero and one")
    fee = kalshi_taker_fee(price, contracts=1, fee_rate=fee_rate)
    slippage = slippage_ticks * tick_size
    if side_upper == "YES":
        point = fair_probability - price - fee - slippage
        lower_edge = lower_probability - price - fee - slippage
    else:
        point = (1.0 - fair_probability) - price - fee - slippage
        lower_edge = (1.0 - upper_probability) - price - fee - slippage
    reasons: list[str] = []
    if risk_state != "OK":
        reasons.append("risk_blocked")
    if lower_edge <= min_lower_edge:
        reasons.append("lower_confidence_edge_not_positive")
    if reasons:
        return TradeDecision("NO_TRADE", None, None, 0, point, lower_edge, fee + slippage, tuple(reasons), model_version, risk_state)
    from decimal import Decimal

    return TradeDecision(
        "TRADE", side_upper, Decimal(str(round(price + slippage, 4))), contracts,
        point, lower_edge, fee + slippage, ("conservative_edge_pass",), model_version, risk_state,
    )
