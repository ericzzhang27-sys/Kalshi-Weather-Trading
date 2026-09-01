from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Sequence
import uuid

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.metrics import ledger_metrics
from src.backtest.sizing import SizingConfig
from .folds import ExpandingFold, event_day_folds, inner_rolling_folds
from .gates import evaluate_competence_gates
from .interfaces import ExperimentRecord, canonical_json_hash
from .probability import BinaryLogOddsPool, coherent_market_distribution
from .registry import ExperimentRegistry, sha256_file
from .statistics import (
    block_bootstrap_pnl,
    daily_returns,
    deflated_sharpe_confidence,
    probability_of_backtest_overfitting,
    proper_score_skill,
)


@dataclass(frozen=True)
class EventSnapshot:
    target_date: str
    event_ticker: str
    signal_timestamp: pd.Timestamp
    market_tickers: tuple[str, ...]
    p_weather: np.ndarray
    p_market: np.ndarray
    p_baseline: np.ndarray | None
    outcome_index: int
    coverage_80_hit: float | None = None
    coverage_90_hit: float | None = None
    weather_cdf_below_truth: float | None = None
    weather_cdf_at_truth: float | None = None


@dataclass
class RaggedConvexPool:
    """Coherent linear or logarithmic pool with validation-fit regime weights."""

    kind: str
    weights: dict[str, float]

    def predict_event(self, p_weather: Sequence[float], p_market: Sequence[float], *, regime: str = "all") -> np.ndarray:
        weight = float(self.weights.get(regime, self.weights.get("all", 0.0)))
        weather = np.clip(np.asarray(p_weather, dtype=float), 1e-12, 1.0)
        market = np.clip(np.asarray(p_market, dtype=float), 1e-12, 1.0)
        if self.kind in {"logarithmic", "regime_logarithmic"}:
            raw = np.exp(weight * np.log(weather) + (1.0 - weight) * np.log(market))
        elif self.kind == "linear":
            raw = weight * weather + (1.0 - weight) * market
        else:
            raise ValueError(f"unsupported convex pool kind: {self.kind}")
        return raw / raw.sum()


def audit_candle_granularity(aligned: pd.DataFrame, *, maximum_gap_minutes: int = 5) -> dict[str, Any]:
    gaps: list[float] = []
    for _, group in aligned.groupby("market_ticker", sort=False):
        timestamps = pd.to_datetime(group["timestamp"], errors="raise", utc=True).drop_duplicates().sort_values()
        if len(timestamps) > 1:
            gaps.extend(timestamps.diff().dropna().dt.total_seconds().div(60.0).tolist())
    if not gaps:
        return {"passed": False, "reason": "no_consecutive_market_candles", "n_gaps": 0}
    values = np.asarray(gaps, dtype=float)
    fraction = float(np.mean(values <= maximum_gap_minutes + 1e-9))
    return {
        "passed": bool(fraction >= 0.95),
        "n_gaps": int(len(values)),
        "median_gap_minutes": float(np.median(values)),
        "p95_gap_minutes": float(np.quantile(values, 0.95)),
        "fraction_within_execution_window": fraction,
        "maximum_execution_gap_minutes": int(maximum_gap_minutes),
        "reason": "ok" if fraction >= 0.95 else "stored_candles_are_too_coarse_for_causal_proxy_fills",
    }


def event_snapshots_from_aligned(aligned: pd.DataFrame) -> list[EventSnapshot]:
    required = {
        "target_date", "event_ticker", "market_ticker", "signal_timestamp",
        "model_probability", "yes_bid_close", "yes_ask_close", "settlement",
    }
    missing = sorted(required - set(aligned.columns))
    if missing:
        raise ValueError(f"aligned data missing event-distribution columns: {missing}")
    if "city" in aligned and set(aligned["city"].dropna().astype(str).str.upper()) != {"NYC"}:
        raise ValueError("research wave accepts NYC rows only")
    working = aligned.copy()
    working["signal_timestamp"] = pd.to_datetime(working["signal_timestamp"], errors="raise", utc=True)
    snapshots: list[EventSnapshot] = []
    keys = ["target_date", "event_ticker", "signal_timestamp"]
    for (target_date, event_ticker, signal_timestamp), group in working.groupby(keys, sort=True):
        ordered = group.assign(
            _lower=pd.to_numeric(group.get("bucket_lower"), errors="coerce").fillna(float("-inf")),
            _upper=pd.to_numeric(group.get("bucket_upper"), errors="coerce").fillna(float("inf")),
        ).sort_values(["_lower", "_upper", "market_ticker"], kind="stable")
        if ordered["market_ticker"].duplicated().any():
            continue
        truth = pd.to_numeric(ordered["settlement"], errors="coerce").to_numpy(float)
        if not np.isfinite(truth).all() or not np.isclose(truth.sum(), 1.0):
            continue
        weather = pd.to_numeric(ordered["model_probability"], errors="coerce").to_numpy(float)
        if not np.isfinite(weather).all() or (weather < 0).any() or weather.sum() <= 0:
            continue
        weather = weather / weather.sum()
        baseline: np.ndarray | None = None
        if "baseline_probability" in ordered:
            candidate = pd.to_numeric(ordered["baseline_probability"], errors="coerce").to_numpy(float)
            if np.isfinite(candidate).all() and (candidate >= 0).all() and candidate.sum() > 0:
                baseline = candidate / candidate.sum()
        bids = pd.to_numeric(ordered["yes_bid_close"], errors="coerce").to_numpy(float)
        asks = pd.to_numeric(ordered["yes_ask_close"], errors="coerce").to_numpy(float)
        try:
            market = coherent_market_distribution(bids, asks)
        except ValueError:
            continue
        coverage_80 = _constant_optional_value(ordered, "coverage_80_hit")
        coverage_90 = _constant_optional_value(ordered, "coverage_90_hit")
        cdf_below = _constant_optional_value(ordered, "weather_cdf_below_truth")
        cdf_at = _constant_optional_value(ordered, "weather_cdf_at_truth")
        snapshots.append(
            EventSnapshot(
                str(pd.Timestamp(target_date).date()), str(event_ticker), pd.Timestamp(signal_timestamp),
                tuple(ordered["market_ticker"].astype(str)), weather, market, baseline,
                int(np.argmax(truth)), coverage_80, coverage_90, cdf_below, cdf_at,
            )
        )
    if not snapshots:
        raise ValueError("no complete coherent event snapshots could be built")
    return snapshots


def _constant_optional_value(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1 or not 0.0 <= float(values[0]) <= 1.0:
        return None
    return float(values[0])


def _day_weights(snapshots: Sequence[EventSnapshot]) -> np.ndarray:
    counts = pd.Series([item.target_date for item in snapshots]).value_counts()
    return np.array([1.0 / counts[item.target_date] for item in snapshots], dtype=float)


def ragged_probability_metrics(
    snapshots: Sequence[EventSnapshot],
    probabilities: Sequence[np.ndarray],
    *,
    use_snapshot_coverage: bool = False,
    coverage_80_override: Sequence[float] | None = None,
    coverage_90_override: Sequence[float] | None = None,
) -> dict[str, float | int]:
    if len(snapshots) != len(probabilities) or not snapshots:
        raise ValueError("snapshots and ragged probabilities must be nonempty and aligned")
    weights = _day_weights(snapshots)
    weights = weights / weights.sum()
    log_losses, rps_values, briers, confidences, correct = [], [], [], [], []
    coverage80, coverage90 = [], []
    if coverage_80_override is not None and len(coverage_80_override) != len(snapshots):
        raise ValueError("80% coverage override is not aligned")
    if coverage_90_override is not None and len(coverage_90_override) != len(snapshots):
        raise ValueError("90% coverage override is not aligned")
    for position, (snapshot, raw) in enumerate(zip(snapshots, probabilities)):
        probs = np.asarray(raw, dtype=float)
        if probs.shape != snapshot.p_weather.shape or not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1.0):
            raise ValueError("invalid ragged event probability distribution")
        truth = snapshot.outcome_index
        one_hot = np.eye(len(probs))[truth]
        log_losses.append(-math.log(max(1e-12, probs[truth])))
        rps_values.append(float(np.sum((np.cumsum(probs)[:-1] - np.cumsum(one_hot)[:-1]) ** 2) / max(1, len(probs) - 1)))
        briers.append(float(np.sum((probs - one_hot) ** 2)))
        confidences.append(float(probs.max()))
        correct.append(float(np.argmax(probs) == truth))
        coverage80.append(
            float(coverage_80_override[position])
            if coverage_80_override is not None
            else snapshot.coverage_80_hit
            if use_snapshot_coverage and snapshot.coverage_80_hit is not None
            else _central_interval_contains(probs, truth, 0.80)
        )
        coverage90.append(
            float(coverage_90_override[position])
            if coverage_90_override is not None
            else snapshot.coverage_90_hit
            if use_snapshot_coverage and snapshot.coverage_90_hit is not None
            else _central_interval_contains(probs, truth, 0.90)
        )
    confidence_array = np.asarray(confidences)
    correct_array = np.asarray(correct)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence_array >= lower) & (confidence_array < upper if upper < 1 else confidence_array <= upper)
        if mask.any():
            bin_weight = float(weights[mask].sum())
            normalized = weights[mask] / bin_weight
            ece += bin_weight * abs(float(np.sum(normalized * correct_array[mask])) - float(np.sum(normalized * confidence_array[mask])))
    return {
        "n_snapshots": len(snapshots),
        "n_event_days": len({item.target_date for item in snapshots}),
        "log_loss": float(np.sum(weights * np.asarray(log_losses))),
        "ranked_probability_score": float(np.sum(weights * np.asarray(rps_values))),
        "multiclass_brier": float(np.sum(weights * np.asarray(briers))),
        "ece": float(ece),
        "coverage_80": float(np.sum(weights * np.asarray(coverage80))),
        "coverage_90": float(np.sum(weights * np.asarray(coverage90))),
        "coverage_error_80": float(np.sum(weights * np.asarray(coverage80)) - 0.80),
        "coverage_error_90": float(np.sum(weights * np.asarray(coverage90)) - 0.90),
    }


def _central_interval_contains(probabilities: np.ndarray, truth: int, level: float) -> float:
    tail = (1.0 - level) / 2.0
    cdf = np.cumsum(probabilities)
    lower = int(np.searchsorted(cdf, tail, side="left"))
    upper = int(np.searchsorted(cdf, 1.0 - tail, side="left"))
    return float(lower <= truth <= min(upper, len(probabilities) - 1))


def _time_regime(snapshot: EventSnapshot) -> str:
    hour = snapshot.signal_timestamp.tz_convert("America/New_York").hour
    if hour < 12:
        return "before_12"
    if hour < 16:
        return "12_to_16"
    return "after_16"


def _fit_convex_pool(train: Sequence[EventSnapshot], *, kind: str) -> RaggedConvexPool:
    day_weights = _day_weights(train)
    day_weights /= day_weights.sum()

    def fit_group(indices: np.ndarray) -> float:
        if not len(indices):
            return 0.0
        local_weights = day_weights[indices]
        local_weights /= local_weights.sum()

        def objective(weight: float) -> float:
            pool = RaggedConvexPool(kind, {"all": float(weight)})
            losses = []
            for index in indices:
                snapshot = train[int(index)]
                probability = pool.predict_event(snapshot.p_weather, snapshot.p_market)
                losses.append(-math.log(max(1e-12, probability[snapshot.outcome_index])))
            return float(np.sum(local_weights * np.asarray(losses)))

        result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
        return float(result.x) if result.success else 0.0

    if kind == "regime_logarithmic":
        regimes = np.asarray([_time_regime(item) for item in train])
        weights = {
            regime: fit_group(np.flatnonzero(regimes == regime))
            for regime in ("before_12", "12_to_16", "after_16")
        }
        weights["all"] = fit_group(np.arange(len(train)))
        return RaggedConvexPool(kind, weights)
    return RaggedConvexPool(kind, {"all": fit_group(np.arange(len(train)))})


def _fit_weather_champion_pool(
    train: Sequence[EventSnapshot],
    *,
    kind: str,
) -> RaggedConvexPool:
    if kind not in {"linear", "logarithmic"}:
        raise ValueError("weather ensemble must be linear or logarithmic")
    if any(item.p_baseline is None for item in train):
        raise ValueError("weather ensemble requires a frozen champion distribution")
    day_weights = _day_weights(train)
    day_weights /= day_weights.sum()

    def objective(weight: float) -> float:
        pool = RaggedConvexPool(kind, {"all": float(weight)})
        losses = []
        for item in train:
            probability = pool.predict_event(item.p_weather, np.asarray(item.p_baseline, dtype=float))
            losses.append(-math.log(max(1e-12, probability[item.outcome_index])))
        return float(np.sum(day_weights * np.asarray(losses)))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
    weight = float(result.x) if result.success else 0.0
    return RaggedConvexPool(kind, {"all": weight})


def _weather_ensemble_predict(
    pool: RaggedConvexPool,
    snapshot: EventSnapshot,
) -> np.ndarray:
    if snapshot.p_baseline is None:
        raise ValueError("weather ensemble prediction is missing its champion baseline")
    return pool.predict_event(snapshot.p_weather, snapshot.p_baseline)


def _cross_fitted_weather_ensemble(
    train: Sequence[EventSnapshot],
    validation: Sequence[EventSnapshot],
    *,
    kind: str | None,
) -> tuple[list[EventSnapshot], list[EventSnapshot], dict[str, float] | None]:
    if kind is None:
        return list(train), list(validation), None
    by_day: dict[str, list[EventSnapshot]] = {}
    for item in train:
        by_day.setdefault(item.target_date, []).append(item)
    transformed_train: list[EventSnapshot] = []
    for fold in inner_rolling_folds([item.target_date for item in train], n_folds=3, purge_days=1):
        inner_train = [item for day in fold.train_dates for item in by_day.get(day.isoformat(), [])]
        inner_validation = [item for day in fold.validation_dates for item in by_day.get(day.isoformat(), [])]
        if not inner_train or not inner_validation:
            continue
        pool = _fit_weather_champion_pool(inner_train, kind=kind)
        transformed_train.extend(
            replace(item, p_weather=_weather_ensemble_predict(pool, item))
            for item in inner_validation
        )
    if not transformed_train:
        raise ValueError("weather ensemble could not produce temporally cross-fitted training rows")
    final_pool = _fit_weather_champion_pool(train, kind=kind)
    transformed_validation = [
        replace(item, p_weather=_weather_ensemble_predict(final_pool, item))
        for item in validation
    ]
    return transformed_train, transformed_validation, dict(final_pool.weights)


def _fit_pool(
    train: Sequence[EventSnapshot],
    *,
    regularization_c: float,
    seed: int,
    pool_kind: str = "binary_log_odds",
) -> BinaryLogOddsPool | RaggedConvexPool:
    if pool_kind != "binary_log_odds":
        return _fit_convex_pool(train, kind=pool_kind)
    day_counts = pd.Series([item.target_date for item in train]).value_counts()
    weather: list[float] = []
    market: list[float] = []
    truth: list[int] = []
    weights: list[float] = []
    for item in train:
        size = len(item.p_weather)
        event_weight = 1.0 / day_counts[item.target_date]
        weather.extend(item.p_weather.tolist())
        market.extend(item.p_market.tolist())
        truth.extend((np.arange(size) == item.outcome_index).astype(int).tolist())
        weights.extend([event_weight / size] * size)
    return BinaryLogOddsPool(regularization_c, seed).fit(weather, market, truth, sample_weight=weights)


def _pool_predict(
    pool: BinaryLogOddsPool | RaggedConvexPool,
    snapshot: EventSnapshot,
) -> np.ndarray:
    if isinstance(pool, RaggedConvexPool):
        return pool.predict_event(
            snapshot.p_weather,
            snapshot.p_market,
            regime=_time_regime(snapshot),
        )
    return pool.predict_event(snapshot.p_weather, snapshot.p_market)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    scaled = np.exp(np.log(values) / float(temperature))
    return scaled / scaled.sum()


def _fit_pool_temperature(
    train: Sequence[EventSnapshot],
    *,
    regularization_c: float,
    seed: int,
    pool_kind: str,
) -> float:
    """Fit a scalar calibrator on temporally cross-fitted training predictions."""
    folds = inner_rolling_folds(
        [item.target_date for item in train], n_folds=3, purge_days=1
    )
    if not folds:
        return 1.0
    by_day: dict[str, list[EventSnapshot]] = {}
    for item in train:
        by_day.setdefault(item.target_date, []).append(item)
    calibration_snapshots: list[EventSnapshot] = []
    raw_predictions: list[np.ndarray] = []
    for fold_number, fold in enumerate(folds):
        inner_train = [item for day in fold.train_dates for item in by_day.get(day.isoformat(), [])]
        inner_validation = [item for day in fold.validation_dates for item in by_day.get(day.isoformat(), [])]
        if not inner_train or not inner_validation:
            continue
        pool = _fit_pool(
            inner_train,
            regularization_c=regularization_c,
            seed=seed + fold_number,
            pool_kind=pool_kind,
        )
        calibration_snapshots.extend(inner_validation)
        raw_predictions.extend(
            _pool_predict(pool, item) for item in inner_validation
        )
    if not calibration_snapshots:
        return 1.0
    weights = _day_weights(calibration_snapshots)
    weights /= weights.sum()

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        losses = [
            -math.log(max(1e-12, _temperature_scale(probability, temperature)[snapshot.outcome_index]))
            for snapshot, probability in zip(calibration_snapshots, raw_predictions)
        ]
        return float(np.sum(weights * np.asarray(losses)))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.35), math.log(4.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(math.exp(result.x)) if result.success and math.isfinite(result.fun) else 1.0


def _fit_coverage_tail(
    train: Sequence[EventSnapshot],
    target_coverage: float,
) -> float | None:
    """Calibrate a central discrete interval using prior whole-day outcomes."""
    usable = [
        item for item in train
        if item.weather_cdf_below_truth is not None and item.weather_cdf_at_truth is not None
    ]
    if len(usable) != len(train) or not usable:
        return None
    weights = _day_weights(usable)
    weights /= weights.sum()
    below = np.asarray([item.weather_cdf_below_truth for item in usable], dtype=float)
    at = np.asarray([item.weather_cdf_at_truth for item in usable], dtype=float)
    nominal_tail = (1.0 - float(target_coverage)) / 2.0
    best: tuple[float, float, float] | None = None
    for tail in np.linspace(0.0, 0.5, 2001):
        hits = (at >= tail) & (below <= 1.0 - tail)
        coverage = float(np.sum(weights * hits))
        candidate = (abs(coverage - target_coverage), abs(tail - nominal_tail), float(tail))
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[2]


def _coverage_hit(snapshot: EventSnapshot, tail: float | None, fallback_level: float) -> float:
    if (
        tail is not None
        and snapshot.weather_cdf_below_truth is not None
        and snapshot.weather_cdf_at_truth is not None
    ):
        return float(
            snapshot.weather_cdf_at_truth >= tail
            and snapshot.weather_cdf_below_truth <= 1.0 - tail
        )
    fallback = snapshot.coverage_80_hit if math.isclose(fallback_level, 0.80) else snapshot.coverage_90_hit
    return float(fallback) if fallback is not None else float("nan")


def _paired_baseline_skill(
    snapshots: Sequence[EventSnapshot],
    candidates: Sequence[np.ndarray],
    *,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, float | int | str]:
    """Paired whole-day block bootstrap skill against the frozen champion."""
    if len(snapshots) != len(candidates) or not snapshots:
        raise ValueError("paired skill inputs must be nonempty and aligned")
    if any(item.p_baseline is None for item in snapshots):
        return {
            "baseline": "missing",
            "n_event_days": 0,
            "log_loss_skill": float("nan"),
            "rps_skill": float("nan"),
            "log_loss_skill_ci_2_5": float("nan"),
            "rps_skill_ci_2_5": float("nan"),
        }
    rows: list[dict[str, float | str]] = []
    for snapshot, raw_candidate in zip(snapshots, candidates):
        candidate = np.asarray(raw_candidate, dtype=float)
        baseline = np.asarray(snapshot.p_baseline, dtype=float)
        truth = snapshot.outcome_index
        one_hot = np.eye(len(candidate))[truth]
        rows.append({
            "target_date": snapshot.target_date,
            "candidate_nll": -math.log(max(1e-12, candidate[truth])),
            "baseline_nll": -math.log(max(1e-12, baseline[truth])),
            "candidate_rps": float(np.sum((np.cumsum(candidate)[:-1] - np.cumsum(one_hot)[:-1]) ** 2) / max(1, len(candidate) - 1)),
            "baseline_rps": float(np.sum((np.cumsum(baseline)[:-1] - np.cumsum(one_hot)[:-1]) ** 2) / max(1, len(candidate) - 1)),
        })
    daily = pd.DataFrame(rows).groupby("target_date", sort=True).mean(numeric_only=True)
    candidate_nll = daily["candidate_nll"].to_numpy(float)
    baseline_nll = daily["baseline_nll"].to_numpy(float)
    candidate_rps = daily["candidate_rps"].to_numpy(float)
    baseline_rps = daily["baseline_rps"].to_numpy(float)

    def skill(candidate: np.ndarray, baseline: np.ndarray) -> float:
        denominator = float(np.mean(baseline))
        return float(1.0 - np.mean(candidate) / denominator) if denominator > 0 else float("nan")

    n_days = len(daily)
    rng = np.random.default_rng(seed)
    block_length = max(2, int(round(math.sqrt(n_days))))
    sampled_nll = np.empty(n_resamples, dtype=float)
    sampled_rps = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        selected: list[int] = []
        while len(selected) < n_days:
            start = int(rng.integers(0, n_days))
            selected.extend((start + offset) % n_days for offset in range(block_length))
        chosen = np.asarray(selected[:n_days], dtype=int)
        sampled_nll[index] = skill(candidate_nll[chosen], baseline_nll[chosen])
        sampled_rps[index] = skill(candidate_rps[chosen], baseline_rps[chosen])
    return {
        "baseline": "frozen_production_ngboost",
        "n_event_days": n_days,
        "block_length_days": block_length,
        "log_loss_skill": skill(candidate_nll, baseline_nll),
        "rps_skill": skill(candidate_rps, baseline_rps),
        "log_loss_skill_ci_2_5": float(np.quantile(sampled_nll, 0.025)),
        "log_loss_skill_ci_97_5": float(np.quantile(sampled_nll, 0.975)),
        "rps_skill_ci_2_5": float(np.quantile(sampled_rps, 0.025)),
        "rps_skill_ci_97_5": float(np.quantile(sampled_rps, 0.975)),
    }


def _prediction_mapping(
    snapshots: Sequence[EventSnapshot],
    predictions: Sequence[np.ndarray],
    *,
    effective_training_days: int,
    uncertainty_method: str = "wilson_plus_disagreement",
) -> dict[tuple[str, str, pd.Timestamp, str], tuple[float, float, float]]:
    result: dict[tuple[str, str, pd.Timestamp, str], tuple[float, float, float]] = {}
    n_days = max(1, effective_training_days)
    for snapshot, fair in zip(snapshots, predictions):
        z = 1.959963984540054
        denominator = 1.0 + z * z / n_days
        center = (fair + z * z / (2.0 * n_days)) / denominator
        half_width = (
            z
            * np.sqrt(np.maximum(fair * (1.0 - fair), 1e-12) / n_days + z * z / (4.0 * n_days * n_days))
            / denominator
        )
        lower = np.maximum(0.0, center - half_width)
        upper = np.minimum(1.0, center + half_width)
        if uncertainty_method == "wilson_plus_disagreement":
            disagreement = 0.5 * np.abs(snapshot.p_weather - snapshot.p_market)
            lower = np.maximum(0.0, lower - disagreement)
            upper = np.minimum(1.0, upper + disagreement)
        elif uncertainty_method != "effective_day_wilson":
            raise ValueError("unsupported probability uncertainty method")
        for ticker, value, low, high in zip(snapshot.market_tickers, fair, lower, upper):
            result[(snapshot.target_date, snapshot.event_ticker, snapshot.signal_timestamp, ticker)] = (
                float(value), float(low), float(high),
            )
    return result


def _apply_predictions(rows: pd.DataFrame, mapping: dict[tuple[str, str, pd.Timestamp, str], tuple[float, float, float]]) -> pd.DataFrame:
    result = rows.copy()
    timestamps = pd.to_datetime(result["signal_timestamp"], errors="raise", utc=True)
    values = []
    for target, event, timestamp, ticker in zip(result["target_date"], result["event_ticker"], timestamps, result["market_ticker"]):
        values.append(mapping.get((str(pd.Timestamp(target).date()), str(event), pd.Timestamp(timestamp), str(ticker))))
    result["model_probability"] = [value[0] if value else np.nan for value in values]
    result["probability_lower"] = [value[1] if value else np.nan for value in values]
    result["probability_upper"] = [value[2] if value else np.nan for value in values]
    return result.dropna(subset=["model_probability", "probability_lower", "probability_upper"])


def _vectorized_fee(prices: np.ndarray, fee_rate: float) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    raw = np.ceil((fee_rate * values * (1.0 - values) - 1e-12) / 0.0001) * 0.0001
    total = np.ceil((values + raw - 1e-12) / 0.01) * 0.01
    return total - values


def _minimal_execution_rows(
    signal_rows: pd.DataFrame,
    full_market_rows: pd.DataFrame,
    *,
    threshold: float,
    adverse_slippage_ticks: int,
    fee_rate: float,
    maximum_gap_minutes: int = 5,
) -> pd.DataFrame:
    """Keep earliest eligible signal plus its one-minute execution window per market."""
    if signal_rows.empty:
        return signal_rows.copy()
    signals = signal_rows.copy()
    ask = pd.to_numeric(signals["yes_ask_close"], errors="coerce").to_numpy(float)
    bid = pd.to_numeric(signals["yes_bid_close"], errors="coerce").to_numpy(float)
    lower = pd.to_numeric(signals["probability_lower"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(signals["probability_upper"], errors="coerce").to_numpy(float)
    no_price = 1.0 - bid
    slippage = adverse_slippage_ticks * 0.01
    yes_edge = lower - ask - _vectorized_fee(ask, fee_rate) - slippage
    no_edge = (1.0 - upper) - no_price - _vectorized_fee(no_price, fee_rate) - slippage
    edge = np.maximum(yes_edge, no_edge)
    eligible = np.isfinite(edge) & (
        edge > float(threshold) if float(threshold) == 0.0 else edge >= float(threshold)
    )
    candidates = signals.loc[eligible].sort_values(["signal_timestamp", "market_ticker"], kind="stable")
    if candidates.empty:
        return signals.iloc[0:0].copy()
    candidates = candidates.drop_duplicates("market_ticker", keep="first")
    full = full_market_rows.copy()
    full["timestamp"] = pd.to_datetime(full["timestamp"], errors="raise", utc=True)
    full_groups = {str(key): group.sort_values("timestamp", kind="stable") for key, group in full.groupby("market_ticker", sort=False)}
    pieces: list[pd.DataFrame] = []
    for _, signal in candidates.iterrows():
        ticker = str(signal["market_ticker"])
        group = full_groups.get(ticker)
        if group is None:
            continue
        signal_time = pd.Timestamp(signal["signal_timestamp"])
        window = group[(group["timestamp"] >= signal_time) & (group["timestamp"] <= signal_time + pd.Timedelta(minutes=maximum_gap_minutes))].copy()
        if window.empty:
            continue
        for column in ("model_probability", "probability_lower", "probability_upper", "model_name"):
            if column in signal:
                window[column] = signal[column]
        window["signal_timestamp"] = window["timestamp"]
        pieces.append(window)
    if not pieces:
        return signals.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True).sort_values(["signal_timestamp", "market_ticker"], kind="stable")


def _combine_ledgers(ledgers: Iterable[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [item for item in ledgers if not item.empty]
    if not nonempty:
        return pd.DataFrame()
    result = pd.concat(nonempty, ignore_index=True)
    result["trade_id"] = np.arange(1, len(result) + 1)
    return result.sort_values(["settlement_timestamp", "trade_id"], kind="stable").reset_index(drop=True)


def run_baseline_research_wave(
    aligned_path: str | Path,
    *,
    output_root: str | Path,
    registry_path: str | Path,
    manifest_dir: str | Path,
    model_path: str | Path,
    requirements_path: str | Path,
    hypothesis: str,
    seed: int = 42,
    regularization_c: float = 0.25,
    pool_kind: str = "binary_log_odds",
    weather_ensemble_kind: str | None = None,
    uncertainty_method: str = "wilson_plus_disagreement",
    thresholds: Sequence[float] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15),
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    aligned_file = Path(aligned_path)
    if pool_kind not in {"binary_log_odds", "linear", "logarithmic", "regime_logarithmic"}:
        raise ValueError("unsupported weather/market pool kind")
    if weather_ensemble_kind not in {None, "linear", "logarithmic"}:
        raise ValueError("unsupported weather/champion ensemble kind")
    if uncertainty_method not in {"effective_day_wilson", "wilson_plus_disagreement"}:
        raise ValueError("unsupported probability uncertainty method")
    aligned = pd.read_parquet(aligned_file) if aligned_file.suffix.lower() == ".parquet" else pd.read_csv(aligned_file)
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], errors="raise", utc=True)
    # The weather/intraday model operates on a five-minute decision grid.  Full
    # one-minute rows remain available solely for the later execution lookup.
    decision_aligned = aligned[aligned["timestamp"].dt.minute.mod(5).eq(0)].copy()
    snapshots = event_snapshots_from_aligned(decision_aligned)
    folds = event_day_folds([item.target_date for item in snapshots], warmup_days=90, validation_days=90, purge_days=1)
    if not folds:
        raise ValueError("at least 181 distinct Kalshi event-days are required for a research wave")
    granularity = audit_candle_granularity(aligned)
    index_by_day: dict[str, list[int]] = {}
    for index, snapshot in enumerate(snapshots):
        index_by_day.setdefault(snapshot.target_date, []).append(index)

    scored_snapshots: list[EventSnapshot] = []
    weather_predictions: list[np.ndarray] = []
    market_predictions: list[np.ndarray] = []
    fair_predictions: list[np.ndarray] = []
    base_ledgers: list[pd.DataFrame] = []
    doubled_fee_ledgers: list[pd.DataFrame] = []
    two_tick_ledgers: list[pd.DataFrame] = []
    fold_results: list[dict[str, Any]] = []
    calibration_temperatures: list[float] = []
    coverage_tails: list[dict[str, float | None]] = []
    cross_fitted_coverage_80: list[float] = []
    cross_fitted_coverage_90: list[float] = []
    threshold_ledgers: dict[float, list[pd.DataFrame]] = {float(value): [] for value in thresholds}
    prior_threshold_pnl = {float(value): 0.0 for value in thresholds}
    sizing = SizingConfig(
        method="fixed_contracts", fixed_contracts=1, max_contracts_per_order=1,
        max_contracts_per_market=1, max_dollars_per_order=5.0,
        max_dollars_per_market=5.0, max_dollars_per_event=10.0,
        max_daily_exposure=10.0, max_total_exposure=50.0, bankroll=1000.0,
    )
    for fold in folds:
        train_indices = [index for day in fold.train_dates for index in index_by_day.get(day.isoformat(), [])]
        validation_indices = [index for day in fold.validation_dates for index in index_by_day.get(day.isoformat(), [])]
        if not train_indices or not validation_indices:
            continue
        raw_train = [snapshots[index] for index in train_indices]
        raw_validation = [snapshots[index] for index in validation_indices]
        train, validation, weather_ensemble_weights = _cross_fitted_weather_ensemble(
            raw_train,
            raw_validation,
            kind=weather_ensemble_kind,
        )
        temperature = _fit_pool_temperature(
            train,
            regularization_c=regularization_c,
            seed=seed + len(fold_results),
            pool_kind=pool_kind,
        )
        tail_80 = _fit_coverage_tail(train, 0.80)
        tail_90 = _fit_coverage_tail(train, 0.90)
        pool = _fit_pool(
            train,
            regularization_c=regularization_c,
            seed=seed,
            pool_kind=pool_kind,
        )
        fair = [
            _temperature_scale(
                _pool_predict(pool, item), temperature
            )
            for item in validation
        ]
        calibration_temperatures.append(temperature)
        coverage_tails.append({"tail_80": tail_80, "tail_90": tail_90})
        cross_fitted_coverage_80.extend(
            _coverage_hit(item, tail_80, 0.80) for item in validation
        )
        cross_fitted_coverage_90.extend(
            _coverage_hit(item, tail_90, 0.90) for item in validation
        )
        scored_snapshots.extend(validation)
        weather_predictions.extend([item.p_weather for item in validation])
        market_predictions.extend([item.p_market for item in validation])
        fair_predictions.extend(fair)
        mapping = _prediction_mapping(
            validation,
            fair,
            effective_training_days=len(set(item.target_date for item in train)),
            uncertainty_method=uncertainty_method,
        )
        validation_days = {item.target_date for item in validation}
        full_validation_rows = aligned[pd.to_datetime(aligned["target_date"]).dt.date.astype(str).isin(validation_days)]
        validation_rows = decision_aligned[pd.to_datetime(decision_aligned["target_date"]).dt.date.astype(str).isin(validation_days)]
        validation_rows = _apply_predictions(validation_rows, mapping)
        selected_threshold = sorted(prior_threshold_pnl, key=lambda value: (-prior_threshold_pnl[value], -value))[0] if any(prior_threshold_pnl.values()) else 0.0
        fold_threshold_pnl: dict[str, float] = {}
        for threshold in thresholds:
            config = BacktestConfig(
                threshold=float(threshold), adverse_slippage_ticks=1,
                execution_mode="next_candle_open", sizing=sizing,
            )
            engine_rows = _minimal_execution_rows(
                validation_rows, full_validation_rows, threshold=float(threshold),
                adverse_slippage_ticks=1, fee_rate=0.07,
            )
            ledger = run_backtest(engine_rows, config)
            threshold_ledgers[float(threshold)].append(ledger)
            pnl = float(ledger["net_pnl"].sum()) if not ledger.empty else 0.0
            fold_threshold_pnl[str(threshold)] = pnl
            prior_threshold_pnl[float(threshold)] += pnl
            if math.isclose(float(threshold), selected_threshold):
                base_ledgers.append(ledger)
        double_rows = _minimal_execution_rows(validation_rows, full_validation_rows, threshold=selected_threshold, adverse_slippage_ticks=1, fee_rate=0.14)
        two_tick_rows = _minimal_execution_rows(validation_rows, full_validation_rows, threshold=selected_threshold, adverse_slippage_ticks=2, fee_rate=0.07)
        doubled_fee_ledgers.append(run_backtest(double_rows, BacktestConfig(threshold=selected_threshold, adverse_slippage_ticks=1, fee_rate=0.14, sizing=sizing)))
        two_tick_ledgers.append(run_backtest(two_tick_rows, BacktestConfig(threshold=selected_threshold, adverse_slippage_ticks=2, sizing=sizing)))
        chosen = threshold_ledgers[selected_threshold][-1]
        fold_results.append({
            **fold.as_dict(),
            "cross_fitted_calibration_temperature": temperature,
            "pool_kind": pool_kind,
            "pool_weights": pool.weights if isinstance(pool, RaggedConvexPool) else None,
            "weather_ensemble_kind": weather_ensemble_kind,
            "weather_ensemble_weights": weather_ensemble_weights,
            "cross_fitted_coverage_tail_80": tail_80,
            "cross_fitted_coverage_tail_90": tail_90,
            "selected_threshold_from_prior_folds": selected_threshold,
            "n_trades": int(len(chosen)),
            "net_pnl": float(chosen["net_pnl"].sum()) if not chosen.empty else 0.0,
            "candidate_threshold_pnl": fold_threshold_pnl,
        })

    weather_metrics = ragged_probability_metrics(
        scored_snapshots,
        weather_predictions,
        use_snapshot_coverage=True,
        coverage_80_override=(
            cross_fitted_coverage_80
            if len(cross_fitted_coverage_80) == len(scored_snapshots)
            and np.isfinite(cross_fitted_coverage_80).all()
            else None
        ),
        coverage_90_override=(
            cross_fitted_coverage_90
            if len(cross_fitted_coverage_90) == len(scored_snapshots)
            and np.isfinite(cross_fitted_coverage_90).all()
            else None
        ),
    )
    market_metrics = ragged_probability_metrics(scored_snapshots, market_predictions)
    fair_metrics = ragged_probability_metrics(scored_snapshots, fair_predictions)
    baseline_predictions = [item.p_baseline for item in scored_snapshots]
    baseline_metrics = (
        ragged_probability_metrics(
            scored_snapshots,
            [np.asarray(value, dtype=float) for value in baseline_predictions],
        )
        if baseline_predictions and all(value is not None for value in baseline_predictions)
        else None
    )
    baseline_skill = _paired_baseline_skill(
        scored_snapshots, weather_predictions, seed=seed
    )
    probability = {
        "weather": weather_metrics,
        "frozen_champion": baseline_metrics,
        "coherent_market": market_metrics,
        "hybrid": fair_metrics,
        "log_loss_skill": baseline_skill["log_loss_skill"],
        "rps_skill": baseline_skill["rps_skill"],
        "log_loss_skill_ci_2_5": baseline_skill["log_loss_skill_ci_2_5"],
        "rps_skill_ci_2_5": baseline_skill["rps_skill_ci_2_5"],
        "frozen_champion_comparison": baseline_skill,
        "weather_log_loss_skill_vs_market": proper_score_skill(float(weather_metrics["log_loss"]), float(market_metrics["log_loss"])),
        "weather_rps_skill_vs_market": proper_score_skill(float(weather_metrics["ranked_probability_score"]), float(market_metrics["ranked_probability_score"])),
        "hybrid_log_loss_skill": proper_score_skill(float(fair_metrics["log_loss"]), float(market_metrics["log_loss"])),
        "hybrid_rps_skill": proper_score_skill(float(fair_metrics["ranked_probability_score"]), float(market_metrics["ranked_probability_score"])),
        "ece": fair_metrics["ece"],
        "coverage_error_80": weather_metrics["coverage_error_80"],
        "coverage_error_90": weather_metrics["coverage_error_90"],
        "coverage_source": (
            "integer_temperature_hurdle_distribution"
            if all(item.coverage_80_hit is not None and item.coverage_90_hit is not None for item in scored_snapshots)
            else "kalshi_bucket_central_interval_fallback"
        ),
        "calibration_temperatures": calibration_temperatures,
        "coverage_calibration_tails": coverage_tails,
        "frozen_champion_reference": "exact shared timestamp and Kalshi bucket frozen production NGBoost",
    }
    ledger = _combine_ledgers(base_ledgers)
    stress_double = _combine_ledgers(doubled_fee_ledgers)
    stress_ticks = _combine_ledgers(two_tick_ledgers)
    unique_days = sorted({item.target_date for item in scored_snapshots})
    base_metrics = ledger_metrics(ledger, evaluation_dates=unique_days)
    bootstrap = block_bootstrap_pnl(ledger, seed=seed, evaluation_dates=unique_days)
    dsr = deflated_sharpe_confidence(
        daily_returns(ledger, evaluation_dates=unique_days),
        trials=len(thresholds),
        observed_sharpe=base_metrics.get("sharpe_ratio"),
    )
    trial_matrix = np.zeros((len(unique_days), len(thresholds)), dtype=float)
    for column, threshold in enumerate(thresholds):
        candidate_ledger = _combine_ledgers(threshold_ledgers[float(threshold)])
        if not candidate_ledger.empty:
            daily = candidate_ledger.groupby(candidate_ledger["target_date"].astype(str))["net_pnl"].sum()
            trial_matrix[:, column] = [float(daily.get(day, 0.0)) for day in unique_days]
    pbo = probability_of_backtest_overfitting(trial_matrix)
    trading = {
        **base_metrics,
        "sharpe": base_metrics.get("sharpe_ratio"),
        "sortino": base_metrics.get("sortino_ratio"),
        "calmar": base_metrics.get("calmar_ratio"),
        "max_drawdown_fraction": base_metrics.get("max_drawdown_pct"),
        "n_event_days": base_metrics.get("n_event_days", 0),
        "net_pnl_ci_2_5": bootstrap.get("net_pnl_ci_2_5"),
        "net_pnl_ci_97_5": bootstrap.get("net_pnl_ci_97_5"),
        "dsr_confidence": dsr.get("confidence"),
        "deflated_sharpe": dsr,
        "pbo": pbo.get("pbo"),
        "pbo_details": pbo,
    }
    robustness = {
        "positive_fold_fraction": float(np.mean([row["net_pnl"] > 0 for row in fold_results])) if fold_results else 0.0,
        "doubled_cost_net_pnl": float(stress_double["net_pnl"].sum()) if not stress_double.empty else 0.0,
        "two_tick_net_pnl": float(stress_ticks["net_pnl"].sum()) if not stress_ticks.empty else 0.0,
        "candle_granularity": granularity,
    }
    gates = evaluate_competence_gates(probability, trading, robustness)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    ledger.to_csv(output_dir / "ledger.csv", index=False)
    pd.DataFrame(fold_results).to_json(output_dir / "fold_results.json", orient="records", indent=2)
    report = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": hypothesis,
        "evidence_label": "historical_proxy_validated" if granularity["passed"] else "blocked_hourly_candles_no_proxy_fills",
        "probability_metrics": probability,
        "trading_metrics": trading,
        "robustness": robustness,
        "competence_gates": gates,
        "folds": fold_results,
        "assumptions": {
            "scope": "NYC KXHIGHNY only",
            "execution": "first later executable one-minute quote within five minutes; one contract",
            "historical_depth": "unavailable; no executable-depth claim",
            "selection": "whole event-day expanding folds with one-day purge",
            "live_trading": False,
            "weather_market_pool": pool_kind,
            "weather_champion_ensemble": weather_ensemble_kind,
            "probability_uncertainty": uncertainty_method,
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    elapsed = time.perf_counter() - started
    config_payload = {
        "regularization_c": regularization_c,
        "pool_kind": pool_kind,
        "weather_ensemble_kind": weather_ensemble_kind,
        "uncertainty_method": uncertainty_method,
        "thresholds": list(thresholds),
        "seed": seed,
    }
    record = ExperimentRecord(
        experiment_id=run_id,
        created_at_utc=datetime.now(timezone.utc),
        hypothesis=hypothesis,
        family=f"frozen_weather_{weather_ensemble_kind or 'standalone'}_plus_{pool_kind}_market_pool",
        seed=seed,
        data_hash=sha256_file(aligned_file),
        model_hash=sha256_file(model_path),
        config_hash=canonical_json_hash(config_payload),
        package_hash=sha256_file(requirements_path),
        folds=tuple(fold.as_dict() for fold in folds),
        trial_count=len(thresholds),
        elapsed_seconds=elapsed,
        probability_metrics=probability,
        trading_metrics=trading,
        robustness_tests=robustness,
        promotion_decision=gates["promotion_decision"],
        evidence_label=report["evidence_label"],
    )
    registry = ExperimentRegistry(registry_path, manifest_dir)
    registry.register(record)
    for number, threshold in enumerate(thresholds):
        candidate = _combine_ledgers(threshold_ledgers[float(threshold)])
        registry.register_trial(
            run_id, number, seed=seed, state="succeeded", elapsed_seconds=0.0,
            params={"threshold": threshold},
            metrics=ledger_metrics(candidate, evaluation_dates=unique_days),
        )
    return output_dir, report


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _markdown_report(report: dict[str, Any]) -> str:
    probability = report["probability_metrics"]
    trading = report["trading_metrics"]
    gates = report["competence_gates"]
    return "\n".join([
        "# Closed-loop research wave",
        "",
        f"Evidence label: `{report['evidence_label']}`",
        f"Promotion: `{gates['promotion_decision']}`",
        "",
        "## Probability evidence",
        "",
        f"- Weather NLL: {probability['weather']['log_loss']:.6f}; RPS: {probability['weather']['ranked_probability_score']:.6f}; Brier: {probability['weather']['multiclass_brier']:.6f}",
        f"- Market NLL: {probability['coherent_market']['log_loss']:.6f}; RPS: {probability['coherent_market']['ranked_probability_score']:.6f}",
        f"- Hybrid NLL: {probability['hybrid']['log_loss']:.6f}; RPS: {probability['hybrid']['ranked_probability_score']:.6f}; ECE: {probability['hybrid']['ece']:.6f}",
        f"- Hybrid log-loss skill versus market: {probability['hybrid_log_loss_skill']:.4%}",
        "",
        "## Trading evidence",
        "",
        f"- Trades/event-days: {trading.get('n_trades', 0)}/{trading.get('n_event_days', 0)}",
        f"- Net P&L: {trading.get('net_pnl', 0):.2f}; profit factor: {trading.get('profit_factor')}",
        f"- Sharpe/Sortino/Calmar: {trading.get('sharpe')}/{trading.get('sortino')}/{trading.get('calmar')}",
        f"- DSR confidence/PBO: {trading.get('dsr_confidence')}/{trading.get('pbo')}",
        "",
        "## Gate result",
        "",
        f"Passed: **{gates['passed']}**",
        f"Failed gates: {', '.join(gates['failed_gates']) or 'none'}",
        "",
        "This is paper research, not a profitability guarantee or executable-depth proof.",
    ])
