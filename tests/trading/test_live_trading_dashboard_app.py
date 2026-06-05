from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_dashboard_module_imports_without_starting_app() -> None:
    module = _load_dashboard_module()

    assert callable(module.main)
    assert len(module.FEATURE_LABELS) >= 36


def test_day4_artifact_paths_fall_back_for_legacy_output_settings(tmp_path: Path) -> None:
    module = _load_dashboard_module()
    config = SimpleNamespace(
        outputs=SimpleNamespace(
            live_trading_dir=tmp_path,
        )
    )

    paths = module._day4_artifact_paths(config)

    assert paths["portfolio_snapshot.csv"] == tmp_path / "portfolio_snapshot.csv"
    assert paths["risk_decisions.csv"] == tmp_path / "risk_decisions.csv"


def test_probability_context_summary_describes_prediction_inputs() -> None:
    module = _load_dashboard_module()
    state = SimpleNamespace(
        status={
            "data_source": "live_refresh",
            "refreshed_at": "2026-06-05T18:30:00+00:00",
            "event_ticker": "KXHIGHNY-26JUN05",
            "target_date": "2026-06-05",
            "model_name": "fallback_model",
        },
        bucket_probabilities=pd.DataFrame(
            [
                {
                    "prediction_time": "2026-06-05T18:29:00",
                    "model_name": "ngboost_laplace_current36_default",
                    "distribution_type": "laplace",
                    "calibration_method": "global_sigma_scaling",
                }
            ]
        ),
        live_feature_rows=pd.DataFrame(
            [
                {
                    "prediction_time": "2026-06-05T18:29:00",
                    "forecast_high": 86.4,
                    "current_temp": 82.1,
                    "max_temp_so_far": 84.9,
                    "weather_status": "NO_TRADE",
                    "live_feature_status": "NO_TRADE",
                    "no_trade_reason": "unverified_observed_high_window",
                }
            ]
        ),
        feature_freshness=pd.DataFrame(
            [
                {
                    "feature": "max_temp_so_far",
                    "source_time": "2026-06-05T18:00:00",
                }
            ]
        ),
        live_weather=pd.DataFrame(
            [
                {
                    "source_role": "hourly_observations",
                    "timestamp": "2026-06-05T18:00:00",
                    "forecast_source": "nws_station_observations",
                    "observed_high_so_far_source_time": "2026-06-05T17:51:00",
                },
                {
                    "source_role": "hourly_forecasts",
                    "timestamp": "2026-06-05T23:00:00",
                    "forecast_source": "open_meteo_live_forecast",
                },
                {
                    "source_role": "daily_forecast",
                    "date": "2026-06-05",
                    "forecast_high": 86.4,
                    "forecast_source": "open_meteo_live_forecast",
                },
            ]
        ),
    )

    summary = module._probability_context_summary(state)
    records = module._probability_context_records(summary)

    assert summary["prediction_time"] == "2026-06-05 18:29:00"
    assert summary["observations_through"] == "2026-06-05 18:00:00"
    assert summary["forecast_through"] == "2026-06-05 23:00:00"
    assert summary["model_name"] == "ngboost_laplace_current36_default"
    assert summary["forecast_high"] == "86.40"
    assert any(row["Input"] == "Observed weather" for row in records)


def test_bucket_board_hides_blocked_edges() -> None:
    module = _load_dashboard_module()

    assert module._format_edge("BUY_YES", 0.25, "CANDIDATE") == "BUY_YES (25.0c)"
    assert module._format_edge("BUY_YES", 0.25, "NO_TRADE") == "--"


def _load_dashboard_module():
    app_path = ROOT / "apps/live_trading_dashboard.py"
    spec = importlib.util.spec_from_file_location("live_trading_dashboard_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
