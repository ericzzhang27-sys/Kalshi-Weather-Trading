from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.trading.config import SettlementSettings
from src.trading.settlement_state import (
    PEAK_WINDOW_CAUTION,
    POST_PEAK_NO_TRADE,
    PRE_PEAK_FORECAST,
    VERIFIED_SETTLEMENT_ONLY,
    apply_settlement_state_to_probabilities,
    evaluate_settlement_state,
)


def test_pre_peak_forecast_allows_model_probabilities() -> None:
    state = evaluate_settlement_state(
        weather=pd.DataFrame(),
        feature_rows=_feature_rows(prediction_time="2026-06-05T12:00:00"),
        settings=SettlementSettings(),
    )

    assert state.settlement_status == PRE_PEAK_FORECAST
    assert state.settlement_trading_allowed is True
    assert state.probability_mode == "ngboost_forecast"


def test_peak_window_stays_tradable_when_peak_is_not_established() -> None:
    state = evaluate_settlement_state(
        weather=pd.DataFrame(),
        feature_rows=_feature_rows(
            prediction_time="2026-06-05T16:00:00",
            current_temp=84.8,
            max_temp_so_far=85.0,
            minutes_since_high=20,
        ),
        settings=SettlementSettings(),
    )

    assert state.settlement_status == PEAK_WINDOW_CAUTION
    assert state.settlement_trading_allowed is True


def test_post_peak_drop_blocks_model_driven_trades() -> None:
    state = evaluate_settlement_state(
        weather=pd.DataFrame(),
        feature_rows=_feature_rows(
            prediction_time="2026-06-05T19:30:00",
            current_temp=82.0,
            max_temp_so_far=85.0,
            minutes_since_high=120,
        ),
        settings=SettlementSettings(),
    )

    assert state.settlement_status == POST_PEAK_NO_TRADE
    assert state.settlement_trading_allowed is False
    assert state.probability_mode == "diagnostic_no_trade"


def test_unverified_observed_high_window_blocks_trades() -> None:
    weather = pd.DataFrame(
        [
            {
                "source_role": "weather_diagnostics",
                "status": "NO_TRADE",
                "no_trade_reason": "unverified_observed_high_window",
            }
        ]
    )

    state = evaluate_settlement_state(
        weather=weather,
        feature_rows=_feature_rows(
            prediction_time="2026-06-05T18:30:00",
            current_temp=82.0,
            max_temp_so_far=85.0,
            minutes_since_high=45,
        ),
        settings=SettlementSettings(),
    )

    assert state.settlement_status == POST_PEAK_NO_TRADE
    assert state.settlement_trading_allowed is False
    assert state.settlement_reason == "unverified_observed_high_window"


def test_verified_nws_daily_high_overrides_to_settlement_distribution() -> None:
    weather = pd.DataFrame(
        [
            {
                "source_role": "hourly_observations",
                "timestamp": "2026-06-05T19:00:00",
                "date": "2026-06-05",
                "temperature_2m": 82.0,
                "nws_24h_max_temp": 86.7,
            }
        ]
    )
    settings = SettlementSettings(settlement_tail_probability=0.02)
    state = evaluate_settlement_state(
        weather=weather,
        feature_rows=_feature_rows(
            prediction_time="2026-06-05T19:30:00",
            current_temp=82.0,
            max_temp_so_far=86.7,
            minutes_since_high=120,
        ),
        settings=settings,
    )
    probabilities = apply_settlement_state_to_probabilities(
        _probabilities(),
        _mapping(),
        state,
        settings,
    )

    assert state.settlement_status == VERIFIED_SETTLEMENT_ONLY
    assert state.settlement_trading_allowed is True
    winner = probabilities[probabilities["bucket_name"] == "87 to 88"].iloc[0]
    losers = probabilities[probabilities["bucket_name"] != "87 to 88"]
    assert winner["probability"] == 0.98
    assert losers["probability"].sum() == 0.02
    assert set(probabilities["probability_signal_status"]) == {"OK"}
    assert set(probabilities["probability_constraint"]) == {"verified_settlement_state"}


def test_no_trade_state_marks_probabilities_diagnostic_only() -> None:
    state = evaluate_settlement_state(
        weather=pd.DataFrame(),
        feature_rows=_feature_rows(
            prediction_time="2026-06-05T19:30:00",
            current_temp=82.0,
            max_temp_so_far=85.0,
            minutes_since_high=120,
        ),
        settings=SettlementSettings(),
    )
    probabilities = apply_settlement_state_to_probabilities(
        _probabilities(),
        _mapping(),
        state,
        SettlementSettings(),
    )

    assert set(probabilities["probability_signal_status"]) == {"NO_TRADE"}
    assert probabilities["probability_signal_reason"].str.contains(
        "post_peak_temperature_path_no_verified_settlement"
    ).all()
    assert set(probabilities["settlement_status"]) == {POST_PEAK_NO_TRADE}


def _feature_rows(
    *,
    prediction_time: str,
    current_temp: float = 78.0,
    max_temp_so_far: float = 78.0,
    minutes_since_high: float = 0.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "live:KXHIGHNY-26JUN05:demo",
                "event_ticker": "KXHIGHNY-26JUN05",
                "target_date": "2026-06-05",
                "prediction_time": datetime.fromisoformat(prediction_time),
                "current_temp": current_temp,
                "current_temp_source_time": prediction_time,
                "max_temp_so_far": max_temp_so_far,
                "max_temp_so_far_source_time": "2026-06-05T15:30:00",
                "minutes_since_max_temp_so_far": minutes_since_high,
                "forecast_high": max_temp_so_far,
            }
        ]
    )


def _probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": "live:KXHIGHNY-26JUN05:demo",
                "ticker": "KXHIGHNY-26JUN05-T84",
                "bucket_name": "84 or below",
                "bucket_lower_temp": None,
                "bucket_upper_temp": 84.5,
                "probability": 0.10,
                "probability_signal_status": "OK",
                "probability_signal_reason": "",
            },
            {
                "row_id": "live:KXHIGHNY-26JUN05:demo",
                "ticker": "KXHIGHNY-26JUN05-B85.5",
                "bucket_name": "85 to 86",
                "bucket_lower_temp": 84.5,
                "bucket_upper_temp": 86.5,
                "probability": 0.45,
                "probability_signal_status": "OK",
                "probability_signal_reason": "",
            },
            {
                "row_id": "live:KXHIGHNY-26JUN05:demo",
                "ticker": "KXHIGHNY-26JUN05-B87.5",
                "bucket_name": "87 to 88",
                "bucket_lower_temp": 86.5,
                "bucket_upper_temp": 88.5,
                "probability": 0.45,
                "probability_signal_status": "OK",
                "probability_signal_reason": "",
            },
        ]
    )


def _mapping() -> pd.DataFrame:
    return _probabilities()[
        ["ticker", "bucket_name", "bucket_lower_temp", "bucket_upper_temp"]
    ].copy()
