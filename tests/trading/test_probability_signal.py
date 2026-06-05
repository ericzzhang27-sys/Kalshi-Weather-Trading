from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict_distribution import EngineDiagnostics, PredictionResult
from src.trading.contract_mapping import map_event_contracts
from src.trading.probability_signal import score_live_probabilities


def test_score_live_probabilities_against_saved_day2_row() -> None:
    feature_rows_path = ROOT / "outputs/live_trading/live_feature_rows.csv"
    mapping_path = ROOT / "outputs/live_trading/contract_bucket_mapping.csv"

    if not feature_rows_path.exists() or not mapping_path.exists():
        pytest.skip("Day 2 live feature artifacts are not available")

    feature_rows = pd.read_csv(feature_rows_path)
    mapping = pd.read_csv(mapping_path)

    result = score_live_probabilities(feature_rows, mapping)

    assert not result.distribution_params.empty
    assert not result.bucket_probabilities.empty
    assert result.bucket_probabilities["probability"].sum() == pytest.approx(1.0, abs=1e-6)
    assert result.bucket_probabilities["ticker"].notna().all()
    assert set(result.bucket_probabilities["probability_signal_status"]) == {
        _expected_signal_status(feature_rows)
    }


def test_score_live_probabilities_accepts_contract_mapping_result() -> None:
    feature_rows_path = ROOT / "outputs/live_trading/live_feature_rows.csv"
    markets_path = ROOT / "outputs/live_trading/market_discovery_snapshot.csv"

    if not feature_rows_path.exists() or not markets_path.exists():
        pytest.skip("Day 2 live feature artifacts are not available")

    feature_rows = pd.read_csv(feature_rows_path)
    markets = pd.read_csv(markets_path)
    event_ticker = str(feature_rows.iloc[0]["event_ticker"])
    mapping = map_event_contracts(markets, event_ticker)

    result = score_live_probabilities(feature_rows, mapping)

    assert len(result.bucket_probabilities) == len(mapping.buckets)
    assert result.diagnostics.probability_row_count == len(mapping.buckets)
    assert set(result.bucket_probabilities["probability_signal_status"]) == {
        _expected_signal_status(feature_rows)
    }


def test_score_live_probabilities_propagates_feature_no_trade(monkeypatch) -> None:
    feature_rows = pd.DataFrame(
        [
            {
                "row_id": "live:test",
                "live_feature_status": "NO_TRADE",
                "no_trade_reason": "unverified_observed_high_window",
                "max_temp_so_far": 81.2,
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "mapping_status": "MAPPED",
                "ticker": "YESLOW",
                "event_ticker": "EVENT",
                "bucket_name": "80 or below",
                "bucket_lower_temp": None,
                "bucket_upper_temp": 80.5,
            },
            {
                "mapping_status": "MAPPED",
                "ticker": "YESHIGH",
                "event_ticker": "EVENT",
                "bucket_name": "81 or above",
                "bucket_lower_temp": 80.5,
                "bucket_upper_temp": None,
            },
        ]
    )

    class FakeEngine:
        def predict(self, rows, buckets):
            return PredictionResult(
                distribution_params=pd.DataFrame([{"row_id": "live:test"}]),
                bucket_probabilities=pd.DataFrame(
                    [
                        {
                            "row_id": "live:test",
                            "bucket_index": 0,
                            "bucket_name": "80 or below",
                            "bucket_lower_temp": None,
                            "bucket_upper_temp": 80.5,
                            "probability": 0.25,
                            "mu": 0.0,
                            "sigma": 1.0,
                        },
                        {
                            "row_id": "live:test",
                            "bucket_index": 1,
                            "bucket_name": "81 or above",
                            "bucket_lower_temp": 80.5,
                            "bucket_upper_temp": None,
                            "probability": 0.75,
                            "mu": 0.0,
                            "sigma": 1.0,
                        },
                    ]
                ),
                diagnostics=EngineDiagnostics(
                    model_path="fake.pkl",
                    model_name="fake",
                    distribution_type="normal",
                    feature_count=0,
                    model_sigma_scale=1.0,
                    calibration_alpha=1.0,
                    calibration_method="none",
                    prediction_row_count=1,
                    probability_row_count=2,
                    bucket_count_per_prediction=2,
                    max_abs_row_probability_sum_deviation=0.0,
                    total_feature_values_imputed_or_replaced=0,
                ),
            )

    monkeypatch.setattr(
        "src.trading.probability_signal.load_probability_engine",
        lambda **kwargs: FakeEngine(),
    )

    result = score_live_probabilities(feature_rows, mapping)

    assert set(result.bucket_probabilities["probability_signal_status"]) == {"NO_TRADE"}
    assert set(result.bucket_probabilities["probability_signal_reason"]) == {
        "unverified_observed_high_window"
    }
    by_bucket = result.bucket_probabilities.set_index("bucket_name")["probability"]
    assert by_bucket["80 or below"] == pytest.approx(0.0)
    assert by_bucket["81 or above"] == pytest.approx(1.0)
    assert set(result.bucket_probabilities["probability_constraint"]) == {"observed_high_floor"}


def test_score_live_probabilities_applies_observed_high_floor(monkeypatch) -> None:
    feature_rows = pd.DataFrame(
        [
            {
                "row_id": "live:test",
                "live_feature_status": "SCOREABLE_SHADOW",
                "max_temp_so_far": 82.1,
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "mapping_status": "MAPPED",
                "ticker": "LOW",
                "event_ticker": "EVENT",
                "bucket_name": "80 or below",
                "bucket_lower_temp": None,
                "bucket_upper_temp": 80.5,
            },
            {
                "mapping_status": "MAPPED",
                "ticker": "MID",
                "event_ticker": "EVENT",
                "bucket_name": "81 to 82",
                "bucket_lower_temp": 80.5,
                "bucket_upper_temp": 82.5,
            },
            {
                "mapping_status": "MAPPED",
                "ticker": "HIGH",
                "event_ticker": "EVENT",
                "bucket_name": "83 or above",
                "bucket_lower_temp": 82.5,
                "bucket_upper_temp": None,
            },
        ]
    )

    class FakeEngine:
        def predict(self, rows, buckets):
            return PredictionResult(
                distribution_params=pd.DataFrame([{"row_id": "live:test"}]),
                bucket_probabilities=pd.DataFrame(
                    [
                        {
                            "row_id": "live:test",
                            "bucket_index": 0,
                            "bucket_name": "80 or below",
                            "bucket_lower_temp": None,
                            "bucket_upper_temp": 80.5,
                            "probability": 0.4,
                            "mu": 0.0,
                            "sigma": 1.0,
                        },
                        {
                            "row_id": "live:test",
                            "bucket_index": 1,
                            "bucket_name": "81 to 82",
                            "bucket_lower_temp": 80.5,
                            "bucket_upper_temp": 82.5,
                            "probability": 0.4,
                            "mu": 0.0,
                            "sigma": 1.0,
                        },
                        {
                            "row_id": "live:test",
                            "bucket_index": 2,
                            "bucket_name": "83 or above",
                            "bucket_lower_temp": 82.5,
                            "bucket_upper_temp": None,
                            "probability": 0.2,
                            "mu": 0.0,
                            "sigma": 1.0,
                        },
                    ]
                ),
                diagnostics=EngineDiagnostics(
                    model_path="fake.pkl",
                    model_name="fake",
                    distribution_type="normal",
                    feature_count=0,
                    model_sigma_scale=1.0,
                    calibration_alpha=1.0,
                    calibration_method="none",
                    prediction_row_count=1,
                    probability_row_count=3,
                    bucket_count_per_prediction=3,
                    max_abs_row_probability_sum_deviation=0.0,
                    total_feature_values_imputed_or_replaced=0,
                ),
            )

    monkeypatch.setattr(
        "src.trading.probability_signal.load_probability_engine",
        lambda **kwargs: FakeEngine(),
    )

    result = score_live_probabilities(feature_rows, mapping)
    by_bucket = result.bucket_probabilities.set_index("bucket_name")

    assert by_bucket.loc["80 or below", "probability"] == pytest.approx(0.0)
    assert by_bucket.loc["81 to 82", "probability"] == pytest.approx(2 / 3)
    assert by_bucket.loc["83 or above", "probability"] == pytest.approx(1 / 3)
    assert by_bucket.loc["80 or below", "unconstrained_probability"] == pytest.approx(0.4)
    assert set(result.bucket_probabilities["probability_constraint"]) == {"observed_high_floor"}


def _expected_signal_status(feature_rows: pd.DataFrame) -> str:
    if "live_feature_status" not in feature_rows.columns:
        return "OK"
    statuses = feature_rows["live_feature_status"].dropna().astype(str)
    return "NO_TRADE" if (statuses == "NO_TRADE").any() else "OK"
