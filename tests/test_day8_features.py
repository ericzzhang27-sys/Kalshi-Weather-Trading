from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    add_observed_weather_features,
    add_time_features,
    build_feature_matrix,
    write_feature_columns,
)
from src.leakage_checks import run_leakage_checks, write_leakage_report  # noqa: E402


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2026-05-20", "2026-05-20"]),
            "location": ["NYC", "NYC"],
            "prediction_time": pd.to_datetime(
                ["2026-05-20 09:00", "2026-05-20 11:00"]
            ),
            "actual_high": [110.0, 110.0],
            "forecast_high": [100.0, 100.0],
            "forecast_error": [10.0, 10.0],
        }
    )


def _sample_hourly() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-05-20 06:00",
            "2026-05-20 07:00",
            "2026-05-20 08:00",
            "2026-05-20 09:00",
            "2026-05-20 10:00",
            "2026-05-20 11:00",
            "2026-05-20 15:00",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "target_date": timestamps.normalize(),
            "location": ["NYC"] * len(timestamps),
            "temperature_2m": [60.0, 65.0, 70.0, 75.0, 90.0, 100.0, 110.0],
            "dew_point_2m": [49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
            "cloud_cover": [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "wind_speed_10m": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "wind_direction_10m": [315.0, 0.0, 90.0, 180.0, 270.0, 360.0, 45.0],
            "precipitation": [0.0] * len(timestamps),
        }
    )


def test_time_features_use_prediction_time_and_3pm_peak() -> None:
    featured = add_time_features(_sample_rows().iloc[[0]].copy())

    assert featured.loc[0, "forecast_horizon_hours"] == 6.0
    assert featured.loc[0, "month"] == 5
    assert featured.loc[0, "season"] == 1


def test_max_temp_so_far_uses_only_observations_at_or_before_prediction_time() -> None:
    featured = add_observed_weather_features(_sample_rows(), _sample_hourly())

    assert featured["current_temp"].tolist() == [75.0, 100.0]
    assert featured["max_temp_so_far"].tolist() == [75.0, 100.0]
    assert featured.loc[0, "temp_change_60m"] == 5.0
    assert featured.loc[0, "temp_change_120m"] == 10.0
    assert featured.loc[0, "temp_change_180m"] == 15.0
    assert featured.loc[0, "temp_acceleration_60m"] == 0.0
    assert featured.loc[0, "temp_change_60m_minus_3h_avg_rate"] == 0.0
    assert featured.loc[1, "temp_change_180m"] == 30.0
    assert featured.loc[1, "temp_change_240m"] == 35.0
    assert featured.loc[1, "temp_change_300m"] == 40.0
    assert featured.loc[1, "temp_acceleration_60m"] == -5.0
    assert featured.loc[1, "temp_change_60m_minus_3h_avg_rate"] == 0.0
    assert featured["temp_change_30m"].isna().all()
    assert (
        pd.to_datetime(featured["max_temp_so_far_source_time"])
        <= pd.to_datetime(featured["prediction_time"])
    ).all()


def test_feature_columns_exclude_target_and_actual_high(tmp_path: Path) -> None:
    df = add_observed_weather_features(add_time_features(_sample_rows()), _sample_hourly())
    spec = write_feature_columns(df, tmp_path / "feature_columns.json")

    assert "forecast_error" not in spec["feature_columns"]
    assert "actual_high" not in spec["feature_columns"]
    assert "forecast_high" in spec["feature_columns"]
    assert "temp_change_30m" not in spec["feature_columns"]

    saved = json.loads((tmp_path / "feature_columns.json").read_text(encoding="utf-8"))
    assert saved["target"] == "forecast_error"


def test_leakage_checks_fail_future_source_timestamps() -> None:
    df = pd.DataFrame(
        {
            "prediction_time": pd.to_datetime(["2026-05-20 09:00"]),
            "target_date": pd.to_datetime(["2026-05-20"]),
            "location": ["NYC"],
            "actual_high": [80.0],
            "forecast_high": [78.0],
            "forecast_error": [2.0],
            "current_temp": [70.0],
            "max_temp_so_far": [70.0],
            "current_temp_source_time": pd.to_datetime(["2026-05-20 10:00"]),
        }
    )

    checks = run_leakage_checks(df, ["forecast_high", "current_temp", "max_temp_so_far"])

    future_check = next(
        item for item in checks["checks"] if item["name"] == "Future timestamp check"
    )
    assert future_check["status"] == "FAIL"
    assert "current_temp_source_time" in future_check["affected_columns"]


def test_feature_matrix_and_reports_can_be_written(tmp_path: Path) -> None:
    inputs = {
        "rows": _sample_rows(),
        "hourly": _sample_hourly(),
        "hourly_forecasts": pd.DataFrame(),
        "forecasts": pd.DataFrame(),
        "notes": [],
    }
    missingness_path = tmp_path / "feature_missingness_report.csv"
    rows = build_feature_matrix(inputs, missingness_output_path=missingness_path)
    modeling_path = tmp_path / "modeling_rows_v1.csv"
    feature_path = tmp_path / "feature_columns.json"
    leakage_path = tmp_path / "leakage_check_report.md"

    rows.to_csv(modeling_path, index=False)
    spec = write_feature_columns(rows, feature_path)
    checks = run_leakage_checks(rows, spec["feature_columns"])
    write_leakage_report(checks, leakage_path)

    assert modeling_path.exists()
    assert missingness_path.exists()
    assert feature_path.exists()
    assert leakage_path.exists()
    assert len(rows) == 2
