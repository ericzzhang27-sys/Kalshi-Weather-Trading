from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.distributional_model import get_feature_columns, normal_nll
from src.splits import chronological_train_validation_test_split


def test_chronological_split_uses_repo_calendar_boundaries() -> None:
    rows = pd.DataFrame(
        {
            "date": [
                "2022-01-01",
                "2023-12-31",
                "2024-01-01",
                "2024-12-31",
                "2025-01-01",
            ],
            "forecast_error": [0.0, 1.0, -1.0, 2.0, -2.0],
        }
    )

    result = chronological_train_validation_test_split(rows)

    assert result.summary["train_end_date"] == "2023-12-31"
    assert result.summary["validation_end_date"] == "2024-12-31"
    assert result.train["date"].max() < result.validation["date"].min()
    assert result.validation["date"].max() < result.test["date"].min()
    assert len(result.train) == 2
    assert len(result.validation) == 2
    assert len(result.test) == 1


def test_get_feature_columns_uses_spec_and_rejects_leakage(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "forecast_high": [70.0, 71.0],
            "current_temp": [66.0, 67.0],
            "actual_high": [72.0, 73.0],
            "forecast_error": [2.0, 2.0],
        }
    )
    feature_spec = tmp_path / "feature_columns.json"
    feature_spec.write_text(
        json.dumps({"feature_columns": ["forecast_high", "current_temp"]}),
        encoding="utf-8",
    )

    assert get_feature_columns(rows, feature_spec) == ["forecast_high", "current_temp"]

    feature_spec.write_text(
        json.dumps({"feature_columns": ["forecast_high", "actual_high"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leakage"):
        get_feature_columns(rows, feature_spec)


def test_normal_nll_is_finite_and_prefers_correct_mean() -> None:
    y = np.array([0.0, 1.0, -1.0])
    good = normal_nll(y, mu=y, sigma=np.ones_like(y))
    bad = normal_nll(y, mu=y + 10.0, sigma=np.ones_like(y))

    assert np.isfinite(good).all()
    assert float(good.mean()) < float(bad.mean())
