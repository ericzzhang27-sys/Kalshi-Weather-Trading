from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import (
    apply_sigma_scaling,
    cdf_reliability_table,
    fit_global_sigma_scale,
    make_calibration_table,
)


def test_calibration_table_has_expected_columns() -> None:
    table = make_calibration_table(
        pred_probs=pd.Series([0.05, 0.25, 0.75, 0.95]),
        actual_indicators=pd.Series([0, 0, 1, 1]),
        n_bins=4,
    )

    assert table.columns.tolist() == [
        "bin_lower",
        "bin_upper",
        "count",
        "mean_predicted_probability",
        "empirical_frequency",
        "calibration_gap",
    ]
    assert len(table) == 4


def test_apply_sigma_scaling_requires_positive_alpha_and_sigma() -> None:
    scaled = apply_sigma_scaling(pd.Series([1.0, 2.0]), alpha=1.25)

    assert np.allclose(scaled, [1.25, 2.5])

    with pytest.raises(ValueError, match="alpha"):
        apply_sigma_scaling(pd.Series([1.0, 2.0]), alpha=0.0)

    with pytest.raises(ValueError, match="sigma"):
        apply_sigma_scaling(pd.Series([1.0, -2.0]), alpha=1.0)


def test_fit_global_sigma_scale_uses_supplied_grid() -> None:
    selected, search = fit_global_sigma_scale(
        y_true=pd.Series([-0.2, 0.0, 0.2, 0.4]),
        mu=pd.Series([0.0, 0.0, 0.0, 0.0]),
        sigma=pd.Series([1.0, 1.0, 1.0, 1.0]),
        alpha_grid=(0.8, 1.0, 1.2),
    )

    assert selected in {0.8, 1.0, 1.2}
    assert search["alpha"].tolist() == [0.8, 1.0, 1.2]
    assert {"coverage_50", "coverage_gap_80", "selection_score"}.issubset(search.columns)


def test_cdf_reliability_table_has_threshold_bins() -> None:
    table = cdf_reliability_table(
        y_true=pd.Series([-1.0, 0.0, 1.0, 2.0]),
        mu=pd.Series([0.0, 0.0, 0.0, 0.0]),
        sigma=pd.Series([1.0, 1.0, 1.0, 1.0]),
        thresholds=(-1.0, 1.0),
        n_bins=5,
        split="validation",
        method="raw_ngboost",
    )

    assert set(table["threshold"]) == {-1.0, 1.0}
    assert set(table["split"]) == {"validation"}
    assert set(table["method"]) == {"raw_ngboost"}
    assert len(table) == 10
