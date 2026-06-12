from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.supervised_table import (  # noqa: E402
    AUDIT_ONLY_COLUMNS,
    BASELINE_FEATURE_COLUMNS,
    PREDICTION_TIMES,
    TARGET_COLUMN,
    expand_targets_to_prediction_times,
    validate_supervised_rows,
)
from src.forecast_data import (  # noqa: E402
    NDFD_FORECAST_SOURCE,
    _prediction_timestamp_utc,
    build_prediction_time_forecasts,
)
from src.target_builder import (  # noqa: E402
    build_daily_forecast_error_targets,
    build_prediction_forecast_error_rows,
    validate_daily_targets,
)


def _sample_daily_actuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-05-20", "2026-05-21"],
            "location": ["NYC", "NYC"],
            "actual_high": [80.0, 77.5],
        }
    )


def _sample_daily_forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-05-20", "2026-05-21"],
            "location": ["NYC", "NYC"],
            "forecast_high": [78.0, 79.0],
            "forecast_source": ["test_forecast", "test_forecast"],
        }
    )


def test_daily_forecast_error_targets_are_one_row_per_date_location() -> None:
    targets = build_daily_forecast_error_targets(
        _sample_daily_actuals(),
        _sample_daily_forecasts(),
    )

    validate_daily_targets(targets)

    assert len(targets) == 2
    assert targets.duplicated(subset=["date", "location"]).sum() == 0
    assert targets["forecast_error"].tolist() == [2.0, -1.5]


def test_daily_target_validation_rejects_duplicate_keys() -> None:
    targets = build_daily_forecast_error_targets(
        _sample_daily_actuals(),
        _sample_daily_forecasts(),
    )
    duplicated = pd.concat([targets, targets.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate date/location"):
        validate_daily_targets(duplicated)


def test_supervised_rows_expand_each_daily_target_to_hourly_prediction_times() -> None:
    targets = build_daily_forecast_error_targets(
        _sample_daily_actuals(),
        _sample_daily_forecasts(),
    )
    supervised = expand_targets_to_prediction_times(targets)

    validate_supervised_rows(supervised)

    assert len(supervised) == 2 * len(PREDICTION_TIMES)
    assert supervised.groupby(["date", "location"]).size().tolist() == [
        len(PREDICTION_TIMES),
        len(PREDICTION_TIMES),
    ]
    assert sorted(supervised["prediction_time"].unique()) == sorted(PREDICTION_TIMES)
    assert supervised["forecast_error"].isna().sum() == 0
    assert supervised.loc[0, "prediction_timestamp"] == pd.Timestamp("2026-05-20 00:00")
    assert supervised.loc[len(PREDICTION_TIMES) - 1, "prediction_timestamp"] == pd.Timestamp(
        "2026-05-20 23:00"
    )


def test_supervised_validation_rejects_duplicate_prediction_rows() -> None:
    targets = build_daily_forecast_error_targets(
        _sample_daily_actuals(),
        _sample_daily_forecasts(),
    )
    supervised = expand_targets_to_prediction_times(targets)
    duplicated = pd.concat([supervised, supervised.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate date/location/prediction_time"):
        validate_supervised_rows(duplicated)


def test_target_and_audit_columns_are_not_marked_as_baseline_features() -> None:
    assert TARGET_COLUMN == "forecast_error"
    assert "actual_high" in AUDIT_ONLY_COLUMNS
    assert TARGET_COLUMN not in BASELINE_FEATURE_COLUMNS
    assert "actual_high" not in BASELINE_FEATURE_COLUMNS
    assert "forecast_high" in BASELINE_FEATURE_COLUMNS


def test_prediction_time_targets_use_latest_ndfd_issue_with_openmeteo_fallback() -> None:
    openmeteo = pd.DataFrame(
        {
            "date": ["2026-05-20"],
            "location": ["NYC"],
            "forecast_high": [78.0],
            "forecast_source": ["open_meteo_historical_forecast"],
        }
    )
    ndfd = pd.DataFrame(
        {
            "date": ["2026-05-20"],
            "location": ["NYC"],
            "forecast_high": [79.0],
            "forecast_source": [NDFD_FORECAST_SOURCE],
            "forecast_issue_time": ["2026-05-20T10:00:00+00:00"],
            "nws_forecast_high_f": [79.0],
            "ndfd_valid_time_utc": ["2026-05-21T00:00:00+00:00"],
        }
    )

    forecasts = build_prediction_time_forecasts(openmeteo, ndfd, PREDICTION_TIMES)
    rows = build_prediction_forecast_error_rows(_sample_daily_actuals().head(1), forecasts)

    validate_supervised_rows(rows)

    before_issue = rows.loc[rows["prediction_time"] == "05:00"].iloc[0]
    at_issue = rows.loc[rows["prediction_time"] == "06:00"].iloc[0]

    assert before_issue["forecast_source"] == "open_meteo_historical_forecast"
    assert before_issue["forecast_high"] == pytest.approx(78.0)
    assert before_issue["forecast_error"] == pytest.approx(2.0)
    assert at_issue["forecast_source"] == NDFD_FORECAST_SOURCE
    assert at_issue["forecast_high"] == pytest.approx(79.0)
    assert at_issue["forecast_error"] == pytest.approx(1.0)
    assert at_issue["openmeteo_forecast_high_f"] == pytest.approx(78.0)
    assert at_issue["nws_forecast_high_f"] == pytest.approx(79.0)


def test_prediction_timestamp_utc_handles_dst_edges() -> None:
    converted = _prediction_timestamp_utc(
        pd.Series(
            [
                "2022-03-13 02:00",
                "2022-11-06 01:00",
            ]
        ),
        "America/New_York",
    )

    assert converted.tolist() == [
        pd.Timestamp("2022-03-13 07:00:00+00:00"),
        pd.Timestamp("2022-11-06 05:00:00+00:00"),
    ]
