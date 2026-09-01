from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.research.feature_store import PointInTimeFeatureStore
from src.research.folds import event_day_folds
from src.research.gates import evaluate_competence_gates
from src.research.interfaces import ForecastRequest, TemperatureBucket
from src.research.loop import assess_research_loop
from src.research.probability import (
    coherent_market_distribution,
    conservative_trade_decision,
    project_bounded_simplex,
)
from src.research.statistics import block_bootstrap_pnl, probability_metrics
from src.research.wave import EventSnapshot, _paired_baseline_skill
from src.kalshi.download_weather_history import _infer_target_date


def _buckets() -> tuple[TemperatureBucket, ...]:
    return (
        TemperatureBucket("79 or below", None, 79.5),
        TemperatureBucket("80 to 81", 79.5, 81.5),
        TemperatureBucket("82 or above", 81.5, None),
    )


def test_forecast_request_requires_aware_utc_and_exhaustive_nyc_schema() -> None:
    request = ForecastRequest(date(2026, 9, 1), datetime(2026, 9, 1, 12, tzinfo=timezone.utc), "NYC", _buckets())
    assert request.buckets[1].contains(81.0)
    with pytest.raises(ValueError, match="timezone-aware"):
        ForecastRequest(date(2026, 9, 1), datetime(2026, 9, 1, 12), "NYC", _buckets())


def test_point_in_time_store_blocks_future_issue_and_revisions(tmp_path) -> None:
    store = PointInTimeFeatureStore(tmp_path / "features.sqlite")
    key = {
        "target_date": "2026-09-01",
        "as_of_utc": "2026-09-01T12:00:00Z",
        "location": "NYC",
        "forecast_issue_utc": "2026-09-01T11:00:00Z",
        "valid_time_utc": "2026-09-01T18:00:00Z",
        "station_or_grid": "KNYC",
        "source": "NBM",
        "source_version": "v1",
    }
    store.insert(key, {"temperature_f": 80.0})
    assert store.as_of(target_date="2026-09-01", as_of_utc="2026-09-01T12:00:00Z")[0]["features"]["temperature_f"] == 80.0
    assert store.as_of(target_date="2026-09-01", as_of_utc="2026-09-01T11:30:00Z") == []
    with pytest.raises(ValueError, match="overwrite"):
        store.insert(key, {"temperature_f": 99.0})
    with pytest.raises(ValueError, match="later"):
        store.insert({**key, "as_of_utc": "2026-09-01T10:00:00Z"}, {"temperature_f": 80.0})


def test_event_day_folds_never_split_same_day_and_apply_purge() -> None:
    days = pd.date_range("2025-01-01", periods=300, freq="D")
    repeated = np.repeat(days, 4)
    folds = event_day_folds(repeated, warmup_days=90, validation_days=90, purge_days=1)
    assert len(folds) == 3
    for fold in folds:
        assert not set(fold.train_dates) & set(fold.validation_dates)
        assert not set(fold.purge_dates) & set(fold.train_dates)
        assert max(fold.train_dates) < min(fold.validation_dates)


def test_coherent_market_projection_respects_sum_and_bounds() -> None:
    result = coherent_market_distribution([0.10, 0.20, 0.30], [0.20, 0.30, 0.50])
    assert result.sum() == pytest.approx(1.0)
    assert np.all(result >= np.array([0.10, 0.20, 0.30]))
    assert np.all(result <= np.array([0.20, 0.30, 0.50]))
    bounded = project_bounded_simplex([0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [0.5, 0.5, 0.5])
    assert bounded.sum() == pytest.approx(1.0)


def test_conservative_trade_uses_uncertainty_fee_and_slippage() -> None:
    blocked = conservative_trade_decision(
        side="YES", fair_probability=0.60, lower_probability=0.52, upper_probability=0.68,
        executable_price=0.50, slippage_ticks=1, model_version="test",
    )
    assert blocked.action == "NO_TRADE"
    passed = conservative_trade_decision(
        side="YES", fair_probability=0.75, lower_probability=0.70, upper_probability=0.80,
        executable_price=0.50, slippage_ticks=1, model_version="test",
    )
    assert passed.action == "TRADE" and passed.mode == "shadow"


def test_probability_metrics_and_bootstrap_are_deterministic() -> None:
    metrics = probability_metrics(np.array([[0.8, 0.2], [0.1, 0.9]]), np.array([0, 1]))
    assert metrics["log_loss"] < 0.2
    ledger = pd.DataFrame({"target_date": pd.date_range("2025-01-01", periods=20), "net_pnl": np.arange(20) - 5})
    assert block_bootstrap_pnl(ledger, n_resamples=100, seed=3) == block_bootstrap_pnl(ledger, n_resamples=100, seed=3)


def test_competence_gates_fail_closed_when_metrics_are_missing() -> None:
    result = evaluate_competence_gates({}, {}, {})
    assert result["passed"] is False
    assert "sharpe" in result["failed_gates"]


def test_weather_market_target_date_comes_from_event_not_late_expiration() -> None:
    market = {
        "event_ticker": "HIGHNY-24JUN05",
        "close_time": "2024-06-06T03:59:00Z",
        "expiration_time": "2024-06-12T14:00:00Z",
    }
    assert _infer_target_date(market) == "2024-06-05"


def test_paired_weather_skill_uses_whole_days_and_fails_without_baseline() -> None:
    snapshots = []
    candidates = []
    for index, day in enumerate(("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04")):
        snapshots.append(EventSnapshot(
            day, f"event-{index}", pd.Timestamp(f"{day}T15:00:00Z"),
            ("low", "high"), np.array([0.8, 0.2]), np.array([0.5, 0.5]),
            np.array([0.6, 0.4]), 0, 1.0, 1.0,
        ))
        candidates.append(np.array([0.8, 0.2]))
    result = _paired_baseline_skill(snapshots, candidates, seed=7, n_resamples=100)
    assert result["n_event_days"] == 4
    assert result["log_loss_skill"] > 0
    assert result["rps_skill"] > 0
    missing = _paired_baseline_skill(
        [EventSnapshot(
            "2025-01-01", "event", pd.Timestamp("2025-01-01T15:00:00Z"),
            ("low", "high"), np.array([0.8, 0.2]), np.array([0.5, 0.5]),
            None, 0,
        )],
        [np.array([0.8, 0.2])],
        seed=7,
        n_resamples=10,
    )
    assert np.isnan(missing["log_loss_skill"])


def test_loop_never_stops_on_plateau_before_competence_floors_pass() -> None:
    report = {
        "run_id": "failed-wave",
        "generated_at_utc": "2026-09-01T00:00:00Z",
        "competence_gates": {"passed": False, "failed_gates": ["hybrid_skill_positive"]},
        "probability_metrics": {"hybrid": {"log_loss": 0.5}, "ece": 0.01},
        "trading_metrics": {"net_pnl": 10.0, "calmar": 2.0},
        "robustness": {"candle_granularity": {"passed": True}},
    }
    assessment = assess_research_loop([report, {**report, "run_id": "failed-wave-2"}])
    assert assessment["continue_required"] is True
    assert assessment["status"] == "continue_research_floors_not_met"
    assert assessment["dominant_failure"]["component"] == "market_combination"
