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
from src.target_builder import (  # noqa: E402
    build_daily_forecast_error_targets,
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
