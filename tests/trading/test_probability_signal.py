from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.trading.contract_mapping import map_event_contracts
from src.trading.probability_signal import score_live_probabilities


ROOT = Path(__file__).resolve().parents[2]


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
    assert set(result.bucket_probabilities["probability_signal_status"]) == {"OK"}


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
    assert set(result.bucket_probabilities["probability_signal_status"]) == {"OK"}
