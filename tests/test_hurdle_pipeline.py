from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hurdle_calibration import apply_calibrator, fit_platt_scaler
from src.hurdle_comparison import brier_skill_score, choose_exceedance_winner, time_of_day_bucket
from src.conditional_increase_model import (
    conditional_cdf,
    evaluate_conditional_distribution,
    positive_increase_rows,
    predict_conditional_mean,
    train_positive_ngboost,
    train_shifted_poisson_ngboost,
)
from src.hurdle_distribution import (
    categorical_scores,
    hurdle_cdf,
    integer_delta_probabilities,
)
from src.hurdle_dataset import HurdleDatasetConfig, build_hurdle_dataset, settlement_round_f, validate_hurdle_dataset
from src.hurdle_features import add_hurdle_core_features
from src.hurdle_model import (
    HurdlePredictor,
    expanding_window_splits,
    materialize_fold,
    predict_proba,
    train_boosted_classifier,
    train_logistic_regression,
)


def _write_fixture_files(tmp_path):
    observations = []
    for date, temperatures in {
        "2021-01-01": [50.0, 51.0, 52.0, 52.0, 53.0],
        "2022-01-01": [60.0, 61.0, 62.0, 63.0, 64.0],
        "2023-01-01": [40.0, 41.0, 42.0, 43.0, 44.0],
        "2024-01-01": [45.0, 46.0, 47.0, 48.0, 49.0],
    }.items():
        local = pd.date_range(f"{date} 09:00", periods=5, freq="5min", tz="America/New_York")
        for timestamp, temperature in zip(local, temperatures):
            observations.append(
                {
                    "station": "KNYC",
                    "timestamp_utc": timestamp.tz_convert("UTC").isoformat(),
                    "temp_f": temperature,
                    "dewpoint_f": temperature - 10,
                }
            )
    observation_path = tmp_path / "observations.csv"
    pd.DataFrame(observations).to_csv(observation_path, index=False)

    daily_path = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
            "actual_high": [55, 62, 46, 51],  # 2022 is deliberately inconsistent.
            "actual_source": ["noaa_nws_daily_tmax"] * 4,
            "source_station": ["USW00094728"] * 4,
            "source_station_name": ["Central Park"] * 4,
            "source_file": ["fixture"] * 4,
        }
    ).to_csv(daily_path, index=False)

    forecast_rows = []
    for date, high in zip(
        ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"], [55, 65, 46, 52]
    ):
        forecast_rows.extend(
            [
                {
                    "date": date,
                    "forecast_high": high,
                    "forecast_source": "nws_ndfd_historical_forecast",
                    "forecast_issue_time": (
                        pd.Timestamp(f"{date} 08:00", tz="America/New_York").tz_convert("UTC").isoformat()
                    ),
                },
                {
                    "date": date,
                    "forecast_high": high + 10,
                    "forecast_source": "nws_ndfd_historical_forecast",
                    "forecast_issue_time": (
                        pd.Timestamp(f"{date} 09:30", tz="America/New_York").tz_convert("UTC").isoformat()
                    ),
                },
            ]
        )
    forecast_path = tmp_path / "forecasts.csv"
    pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)

    hourly_path = tmp_path / "hourly.csv"
    pd.DataFrame(
        {
            "timestamp": [f"{date} 08:55:00" for date in ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]],
            "nws_relative_humidity": [50.0] * 4,
            "nws_cloud_cover_pct": [20.0] * 4,
            "nws_wind_speed_kt": [5.0] * 4,
            "nws_wind_dir": [180.0] * 4,
            "nws_precip_1h": [0.0] * 4,
            "nws_mslp": [1010.0] * 4,
        }
    ).to_csv(hourly_path, index=False)
    return observation_path, daily_path, forecast_path, hourly_path


def test_settlement_rounding_is_explicit_half_up():
    result = settlement_round_f(pd.Series([49.49, 49.50, 50.49, 50.50]))
    assert result.tolist() == [49.0, 50.0, 50.0, 51.0]


def _fixture_config() -> HurdleDatasetConfig:
    return HurdleDatasetConfig(
        start_time="09:00",
        end_time="09:20",
        require_forecast=True,
        minimum_daily_observations=5,
        latest_first_observation_minute=540,
        earliest_last_observation_minute=560,
    )


def test_builder_uses_same_feed_target_and_audits_official_disagreement(tmp_path):
    paths = _write_fixture_files(tmp_path)
    dataset, summary = build_hurdle_dataset(*paths, config=_fixture_config())
    assert summary["violating_dates"] == []
    assert "2022-01-01" in set(dataset["date"])
    assert not dataset.empty
    assert (dataset["remaining_increase"] >= 0).all()
    assert (dataset["final_daily_high"] >= dataset["current_max_so_far"]).all()
    assert dataset.attrs["invariant_violations"].empty
    disagreement = dataset.attrs["official_settlement_disagreements"]
    assert set(disagreement["date"]) == {"2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"}
    day = dataset.loc[dataset["date"].eq("2022-01-01")]
    assert day["final_daily_high"].eq(64).all()
    assert day["official_final_daily_high"].eq(62).all()
    validate_hurdle_dataset(dataset)


def test_builder_never_attaches_future_forecast(tmp_path):
    paths = _write_fixture_files(tmp_path)
    dataset, _ = build_hurdle_dataset(*paths, config=_fixture_config())
    assert (dataset["forecast_issue_time_utc"] <= dataset["prediction_time_utc"]).all()
    # The 09:30 revision must not appear on the 09:00-09:20 rows.
    expected = {"2021-01-01": 55, "2022-01-01": 65, "2023-01-01": 46, "2024-01-01": 52}
    for date, high in expected.items():
        assert dataset.loc[dataset["date"].eq(date), "forecast_high"].eq(high).all()


def test_real_short_window_momentum_is_computed(tmp_path):
    paths = _write_fixture_files(tmp_path)
    dataset, _ = build_hurdle_dataset(*paths, config=_fixture_config())
    day = dataset.loc[dataset["date"].eq("2021-01-01")].sort_values("prediction_time")
    assert np.isnan(day.iloc[0]["temp_change_5m"])
    assert day.iloc[1]["temp_change_5m"] == pytest.approx(1.0)
    assert day.iloc[3]["temp_change_15m"] == pytest.approx(2.0)
    assert day.iloc[4]["max_change_last_30m"] != 0.0 or np.isnan(day.iloc[4]["max_change_last_30m"])


def test_expanding_folds_use_disjoint_whole_days():
    rows = []
    for year in range(2020, 2025):
        for minute in (0, 5):
            rows.append({"target_date": f"{year}-06-01", "prediction_time": f"{year}-06-01 12:{minute:02d}"})
    frame = pd.DataFrame(rows)
    folds = expanding_window_splits(frame, test_start="2025-01-01", minimum_training_years=2)
    assert [fold["val_start"] for fold in folds] == ["2022-01-01", "2023-01-01", "2024-01-01"]
    for fold in folds:
        train, validation = materialize_fold(frame, fold)
        assert set(train["target_date"]).isdisjoint(set(validation["target_date"]))


def test_hurdle_predictor_returns_bounded_one_dimensional_probability():
    training = pd.DataFrame({"x": [-2, -1, 1, 2], "will_increase": [0, 0, 1, 1]})
    classifier = train_logistic_regression(training, ["x"])
    raw = classifier.predict_proba(pd.DataFrame({"x": [-1, 1]}))[:, 1]
    calibrator = fit_platt_scaler(raw, np.array([0, 1]))
    predictor = HurdlePredictor(classifier, ["x"], calibrator, "platt")
    probability = predictor.predict_proba({"x": 0.25})
    assert probability.shape == (1,)
    assert 0.0 <= probability[0] <= 1.0
    assert np.array_equal(apply_calibrator("raw", None, np.array([-1.0, 2.0])), np.array([0.0, 1.0]))


def test_brier_skill_score_uses_reference_brier():
    y = np.array([0, 0, 1, 1])
    model = np.array([0.1, 0.2, 0.8, 0.9])
    reference = np.full(4, 0.5)
    assert brier_skill_score(y, model, reference) == pytest.approx(0.9)


def test_replacement_rule_ignores_small_bss_gain_but_accepts_meaningful_gain():
    base = pd.DataFrame(
        [
            {
                "model": "ngboost_bernoulli",
                "brier_skill_score": 0.36,
                "log_loss": 0.33,
                "late_day_calibration_gap": 0.04,
                "late_day_brier": 0.14,
            },
            {
                "model": "lightgbm_classifier",
                "brier_skill_score": 0.37,
                "log_loss": 0.329,
                "late_day_calibration_gap": 0.035,
                "late_day_brier": 0.139,
            },
        ]
    )
    winner, decision = choose_exceedance_winner(base)
    assert winner == "ngboost_bernoulli"
    assert decision["decision"] == "retain_incumbent"
    improved = base.copy()
    improved.loc[improved["model"].eq("lightgbm_classifier"), "brier_skill_score"] = 0.385
    winner, decision = choose_exceedance_winner(improved)
    assert winner == "lightgbm_classifier"
    assert decision["decision"] == "replace_incumbent"


def test_time_buckets_match_late_day_boundaries():
    timestamp = pd.Series(pd.to_datetime(["2025-01-01 11:55", "2025-01-01 16:00", "2025-01-01 19:00"]))
    assert list(time_of_day_bucket(timestamp).astype(str)) == ["before 12 PM", "4–5 PM", "after 7 PM"]


def test_lightgbm_challenger_returns_probabilities():
    training = pd.DataFrame(
        {
            "x": np.tile(np.arange(20, dtype=float), 4),
            "will_increase": np.tile(np.array([0] * 10 + [1] * 10), 4),
        }
    )
    model = train_boosted_classifier(training, ["x"], kind="lightgbm")
    probability = predict_proba(model, pd.DataFrame({"x": [2.0, 18.0]}), ["x"])
    assert probability.shape == (2,)
    assert ((probability >= 0) & (probability <= 1)).all()


def test_conditional_stage_uses_only_positive_rows_and_positive_integer_support():
    frame = pd.DataFrame(
        {
            "x": np.tile(np.arange(20, dtype=float), 4),
            "remaining_increase": np.tile(np.array([0] * 5 + [1, 2, 3, 4, 5] * 3), 4)[:80],
        }
    )
    positive = positive_increase_rows(frame)
    assert (positive["remaining_increase"] > 0).all()
    artifact = train_shifted_poisson_ngboost(positive, ["x"], n_estimators=5)
    sample = pd.DataFrame({"x": [2.0, 18.0]})
    mean = predict_conditional_mean(artifact, sample)
    assert (mean >= 1.0).all()
    assert np.array_equal(conditional_cdf(artifact, sample, 0.99), np.zeros(2))


@pytest.mark.parametrize("distribution", ["weibull", "halfnormal", "lognormal", "exponential"])
def test_positive_continuous_candidates_produce_discretized_probabilities(distribution):
    training = pd.DataFrame(
        {
            "x": np.tile(np.arange(20, dtype=float), 5),
            "remaining_increase": np.tile(np.array([1, 2, 3, 4, 5], dtype=float), 20),
        }
    )
    artifact = train_positive_ngboost(training, ["x"], distribution, n_estimators=3, min_samples_leaf=5)
    sample = training.iloc[:8].copy()
    assert np.array_equal(conditional_cdf(artifact, sample, 0.99), np.zeros(len(sample)))
    assert ((conditional_cdf(artifact, sample, 3) >= 0) & (conditional_cdf(artifact, sample, 3) <= 1)).all()
    metrics = evaluate_conditional_distribution(artifact, sample, compute_crps=False)
    assert np.isfinite(metrics["interval_nll"])
    assert 0 <= metrics["coverage_80"] <= 1


def test_full_hurdle_has_zero_mass_and_rows_sum_to_one():
    training = pd.DataFrame(
        {
            "x": np.tile(np.arange(20, dtype=float), 5),
            "remaining_increase": np.tile(np.array([1, 2, 3, 4, 5], dtype=float), 20),
        }
    )
    artifact = train_positive_ngboost(training, ["x"], "weibull", n_estimators=3, min_samples_leaf=5)
    sample = pd.DataFrame({"x": [2.0, 18.0]})
    p_increase = np.array([0.25, 0.8])
    assert np.allclose(hurdle_cdf(p_increase, artifact, sample, 0), 1 - p_increase)
    matrix = integer_delta_probabilities(p_increase, artifact, sample, max_delta=5)
    assert np.allclose(matrix[:, 0], 1 - p_increase)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    scores = categorical_scores(matrix, np.array([0, 3]), max_delta=5)
    assert np.isfinite(scores["multiclass_nll"])
    assert 0 <= scores["mean_bucket_brier"] <= 1
