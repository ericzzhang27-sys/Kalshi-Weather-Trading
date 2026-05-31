from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_ngboost import choose_selected_alpha


def test_alpha_selection_ignores_test_metrics() -> None:
    search = pd.DataFrame(
        {
            "alpha": [0.95, 1.0],
            "validation_nll": [1.01, 1.0],
            "coverage_penalty": [0.02, 0.10],
            "bucket_log_loss": [0.9, 0.8],
            "distance_from_raw_alpha": [0.05, 0.0],
            "selection_score": [1.015, 1.025],
            "test_nll": [99.0, 0.1],
        }
    )

    selected_alpha, details = choose_selected_alpha(search, nll_tolerance=0.02)

    assert selected_alpha == 0.95
    assert details["selection_rule"].startswith("validation-only alpha search")
