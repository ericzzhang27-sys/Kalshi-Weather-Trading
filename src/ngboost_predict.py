from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .distributional_model import predict_distribution_params as _predict_param_arrays
except ImportError:
    from distributional_model import predict_distribution_params as _predict_param_arrays


METADATA_COLUMNS = [
    "row_id",
    "date",
    "prediction_time",
    "prediction_timestamp",
    "timestamp",
    "station",
    "station_id",
    "location",
    "forecast_high",
    "actual_high",
    "official_high",
    "forecast_error",
    "model_version",
    "forecast_horizon_hours",
    "split",
]


def predict_distribution_params(
    model: Any,
    X: pd.DataFrame,
    metadata: pd.DataFrame | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Return one standardized Normal distribution-parameter row per prediction state.

    The project-level model helper in distributional_model intentionally returns
    raw arrays for training/evaluation code. This wrapper is the CSV/export
    shape used by downstream bucket pricing.
    """
    mu, sigma = _predict_param_arrays(model, X)
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    if not np.isfinite(mu_array).all():
        raise ValueError("Predicted mu contains non-finite values")
    if not np.isfinite(sigma_array).all() or (sigma_array <= 0.0).any():
        raise ValueError("Predicted sigma must be finite and greater than 0")

    if metadata is None:
        metadata_df = pd.DataFrame(index=range(len(X)))
        if "forecast_high" in X.columns:
            metadata_df["forecast_high"] = X["forecast_high"].to_numpy()
    else:
        metadata_df = pd.DataFrame(metadata).reset_index(drop=True)

    if len(metadata_df) != len(X):
        raise ValueError(
            "metadata length must match prediction rows: "
            f"metadata={len(metadata_df)}, X={len(X)}"
        )

    result = metadata_df.copy()
    if "row_id" not in result.columns:
        result.insert(0, "row_id", np.arange(len(result), dtype=int))
    result["mu"] = mu_array
    result["sigma"] = sigma_array

    ordered = [column for column in METADATA_COLUMNS if column in result.columns]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining]
