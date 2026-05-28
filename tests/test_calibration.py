from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration import make_calibration_table


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
