from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, skew, kurtosis


def probability_metrics(probabilities: np.ndarray, outcomes: np.ndarray, *, n_bins: int = 10) -> dict[str, float | int]:
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(outcomes, dtype=int)
    if probs.ndim != 2 or len(truth) != len(probs) or probs.shape[1] < 2:
        raise ValueError("probabilities must be [events, buckets] and match outcomes")
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("every probability distribution must sum to one")
    if (truth < 0).any() or (truth >= probs.shape[1]).any():
        raise ValueError("outcomes contain an invalid bucket index")
    one_hot = np.eye(probs.shape[1])[truth]
    realized = probs[np.arange(len(truth)), truth]
    cdf_error = np.cumsum(probs, axis=1) - np.cumsum(one_hot, axis=1)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == truth).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "n_events": int(len(truth)),
        "log_loss": float(-np.mean(np.log(np.clip(realized, 1e-12, 1.0)))),
        "ranked_probability_score": float(np.mean(np.sum(cdf_error[:, :-1] ** 2, axis=1) / (probs.shape[1] - 1))),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece": ece,
        "accuracy": float(correct.mean()),
    }


def proper_score_skill(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("proper scores must be finite and the baseline positive")
    return float(1.0 - candidate / baseline)


def daily_returns(
    ledger: pd.DataFrame,
    *,
    starting_cash: float = 1000.0,
    evaluation_dates: list[object] | tuple[object, ...] | None = None,
) -> pd.Series:
    if ledger.empty and evaluation_dates is None:
        return pd.Series(dtype=float, name="return")
    if ledger.empty:
        daily = pd.Series(dtype=float)
    else:
        dates = pd.to_datetime(ledger["target_date"], errors="raise").dt.date
        daily = ledger.assign(_date=dates).groupby("_date", sort=True)["net_pnl"].sum()
    if evaluation_dates is not None:
        calendar = pd.Index(
            sorted(set(pd.to_datetime(pd.Series(evaluation_dates), errors="raise").dt.date))
        )
        daily = daily.reindex(calendar, fill_value=0.0)
    return (daily / float(starting_cash)).rename("return")


def block_bootstrap_pnl(
    ledger: pd.DataFrame,
    *,
    block_days: int = 7,
    n_resamples: int = 2000,
    seed: int = 42,
    evaluation_dates: list[object] | tuple[object, ...] | None = None,
) -> dict[str, float | int | str]:
    if ledger.empty and evaluation_dates is None:
        return {"method": "moving_date_block_bootstrap", "n_resamples": 0, "n_days": 0}
    if ledger.empty:
        daily_series = pd.Series(dtype=float)
    else:
        dates = pd.to_datetime(ledger["target_date"], errors="raise").dt.date
        daily_series = ledger.assign(_date=dates).groupby("_date", sort=True)["net_pnl"].sum()
    if evaluation_dates is not None:
        calendar = pd.Index(
            sorted(set(pd.to_datetime(pd.Series(evaluation_dates), errors="raise").dt.date))
        )
        daily_series = daily_series.reindex(calendar, fill_value=0.0)
    daily = daily_series.to_numpy(float)
    size = len(daily)
    block = min(max(1, int(block_days)), size)
    starts = np.arange(max(1, size - block + 1))
    rng = np.random.default_rng(seed)
    totals = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample: list[float] = []
        while len(sample) < size:
            start = int(rng.choice(starts))
            sample.extend(daily[start : start + block].tolist())
        totals[index] = float(np.sum(sample[:size]))
    return {
        "method": "moving_date_block_bootstrap",
        "n_resamples": int(n_resamples),
        "n_days": int(size),
        "block_days": int(block),
        "net_pnl_mean": float(totals.mean()),
        "net_pnl_ci_2_5": float(np.quantile(totals, 0.025)),
        "net_pnl_ci_97_5": float(np.quantile(totals, 0.975)),
    }


def deflated_sharpe_confidence(
    returns: np.ndarray | pd.Series,
    *,
    trials: int,
    observed_sharpe: float | None = None,
) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return {"n_observations": int(len(values)), "trials": int(trials), "confidence": 0.0, "deflated_sharpe": float("nan")}
    annual_sharpe = float(observed_sharpe) if observed_sharpe is not None else float(np.sqrt(252.0) * values.mean() / values.std(ddof=1))
    daily_sharpe = annual_sharpe / np.sqrt(252.0)
    n_trials = max(1, int(trials))
    euler_gamma = 0.5772156649015329
    expected_max = 0.0
    if n_trials > 1:
        expected_max = (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / n_trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        ) / math.sqrt(max(1, len(values) - 1))
    variance = (
        1.0 - float(skew(values, bias=False)) * daily_sharpe
        + ((float(kurtosis(values, fisher=False, bias=False)) - 1.0) / 4.0) * daily_sharpe**2
    ) / max(1, len(values) - 1)
    standard_error = math.sqrt(max(variance, 1e-15))
    confidence = float(norm.cdf((daily_sharpe - expected_max) / standard_error))
    return {
        "n_observations": int(len(values)),
        "trials": n_trials,
        "observed_sharpe": annual_sharpe,
        "expected_max_sharpe": float(expected_max * np.sqrt(252.0)),
        "deflated_sharpe": float((daily_sharpe - expected_max) / standard_error),
        "confidence": confidence,
    }


def probability_of_backtest_overfitting(
    trial_returns: np.ndarray,
    *,
    slices: int = 8,
    max_combinations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Combinatorially symmetric cross-validation estimate of PBO."""
    matrix = np.asarray(trial_returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 * slices or matrix.shape[1] < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "reason": "insufficient_trials_or_days"}
    if slices < 4 or slices % 2:
        raise ValueError("slices must be an even integer >= 4")
    blocks = np.array_split(np.arange(matrix.shape[0]), slices)
    candidates = list(combinations(range(slices), slices // 2))
    rng = np.random.default_rng(seed)
    if len(candidates) > max_combinations:
        chosen = rng.choice(len(candidates), size=max_combinations, replace=False)
        candidates = [candidates[index] for index in sorted(chosen)]
    logits: list[float] = []
    all_blocks = set(range(slices))
    for train_blocks in candidates:
        test_blocks = sorted(all_blocks - set(train_blocks))
        train_rows = np.concatenate([blocks[index] for index in train_blocks])
        test_rows = np.concatenate([blocks[index] for index in test_blocks])
        train_mean = np.nanmean(matrix[train_rows], axis=0)
        winner = int(np.nanargmax(train_mean))
        test_mean = np.nanmean(matrix[test_rows], axis=0)
        ranks = rankdata(test_mean, method="average")
        percentile = float(ranks[winner] / (matrix.shape[1] + 1.0))
        percentile = float(np.clip(percentile, 1e-9, 1.0 - 1e-9))
        logits.append(float(math.log(percentile / (1.0 - percentile))))
    return {
        "pbo": float(np.mean(np.asarray(logits) <= 0.0)),
        "n_combinations": int(len(logits)),
        "median_logit": float(np.median(logits)),
    }
