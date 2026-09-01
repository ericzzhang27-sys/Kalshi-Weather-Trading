from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.align_probabilities import align_probabilities_with_markets
from src.backtest.engine import BacktestConfig, run_backtest
from src.backtest.fees import kalshi_taker_fee
from src.backtest.sizing import SizingConfig, cap_contracts_for_portfolio


def _probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_time": ["2026-06-01 09:00", "2026-06-01 10:00"],
            "target_date": ["2026-06-01", "2026-06-01"],
            "city": ["NYC", "NYC"],
            "bucket_lower": [80.5, 80.5],
            "bucket_upper": [82.5, 82.5],
            "model_probability": [0.40, 0.99],
            "model_name": ["safe", "future"],
        }
    )


def _markets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-06-01T13:30:00Z", "2026-06-01T13:31:00Z"],
            "settlement_timestamp": ["2026-06-02T01:00:00Z"] * 2,
            "target_date": ["2026-06-01"] * 2,
            "city": ["NYC"] * 2,
            "market_ticker": ["M1"] * 2,
            "event_ticker": ["E1"] * 2,
            "bucket_lower": [80.5] * 2,
            "bucket_upper": [82.5] * 2,
            "result": ["yes"] * 2,
            "yes_bid_open": [0.20, 0.21],
            "yes_ask_open": [0.25, 0.26],
            "yes_bid_close": [0.20, 0.21],
            "yes_ask_close": [0.25, 0.26],
        }
    )


def test_fee_uses_general_formula_and_rounds_up_to_cent() -> None:
    assert kalshi_taker_fee(0.50, 1) == pytest.approx(0.02)
    assert kalshi_taker_fee(0.50, 100) == pytest.approx(1.75)


def test_alignment_localizes_naive_eastern_and_never_uses_future_prediction() -> None:
    aligned = align_probabilities_with_markets(_probabilities(), _markets())
    assert aligned["model_probability"].tolist() == [0.40, 0.40]
    assert aligned["probability_age_minutes"].tolist() == [30.0, 31.0]
    assert set(aligned["settlement_source"]) == {"kalshi_result"}


def test_dst_nonexistent_and_ambiguous_predictions_are_blocked() -> None:
    probabilities = pd.DataFrame({
        "prediction_time": ["2024-03-10 02:00", "2024-11-03 01:00"],
        "target_date": ["2024-03-10", "2024-11-03"],
        "city": ["NYC", "NYC"],
        "bucket_lower": [70.5, 70.5],
        "bucket_upper": [72.5, 72.5],
        "model_probability": [0.5, 0.5],
    })
    from src.backtest.align_probabilities import standardize_model_probabilities
    standardized = standardize_model_probabilities(probabilities)
    assert standardized.empty
    assert standardized.attrs["blocked_dst_prediction_rows"] == 2


def test_alignment_refuses_missing_kalshi_result_even_if_weather_actual_exists() -> None:
    markets = _markets()
    markets["result"] = None
    markets["actual_high"] = 82.0
    with pytest.raises(ValueError, match="unresolved Kalshi results"):
        align_probabilities_with_markets(_probabilities(), markets)


def test_engine_executes_strictly_on_next_open_and_uses_whole_contracts() -> None:
    aligned = align_probabilities_with_markets(_probabilities(), _markets())
    ledger = run_backtest(
        aligned,
        BacktestConfig(
            threshold=0.10,
            allow_buy_no=False,
            sizing=SizingConfig(
                fixed_contracts=5,
                max_contracts_per_order=5,
                max_contracts_per_market=5,
                max_dollars_per_order=5.0,
                max_dollars_per_market=5.0,
                max_dollars_per_event=5.0,
                max_daily_exposure=5.0,
                max_total_exposure=5.0,
            ),
        ),
    )
    assert len(ledger) == 1
    trade = ledger.iloc[0]
    assert trade["execution_timestamp"] > trade["signal_timestamp"]
    assert trade["quote_field"] == "yes_ask_open"
    assert trade["entry_price"] == pytest.approx(0.26)
    # No historical order-book depth: multi-contract requests cap to one.
    assert trade["contracts"] == 1
    assert trade["settlement_source"] == "kalshi_result"
    assert trade["supports_profitability_claim"] == False  # noqa: E712


def test_engine_refuses_non_kalshi_settlement_source() -> None:
    aligned = align_probabilities_with_markets(_probabilities(), _markets())
    aligned["settlement_source"] = "weather_actual"
    with pytest.raises(ValueError, match="exclusively from Kalshi"):
        run_backtest(aligned)


def test_portfolio_caps_enforce_cash_market_event_day_and_total() -> None:
    config = SizingConfig(
        max_contracts_per_order=100,
        max_contracts_per_market=100,
        max_dollars_per_order=100,
        max_dollars_per_market=3,
        max_dollars_per_event=2,
        max_daily_exposure=4,
        max_total_exposure=5,
    )
    contracts = cap_contracts_for_portfolio(
        100,
        0.50,
        available_cash=10,
        market_exposure=0,
        event_exposure=1.25,
        daily_exposure=0,
        total_exposure=0,
        config=config,
    )
    assert contracts == 1
