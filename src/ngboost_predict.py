from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .calibration import apply_sigma_scaling
    from .distributional_model import predict_distribution_details
except ImportError:
    from calibration import apply_sigma_scaling
    from distributional_model import predict_distribution_details


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
    "distribution_type",
    "df",
    "forecast_horizon_hours",
    "split",
]


def predict_distribution_params(
    model: Any,
    X: pd.DataFrame,
    metadata: pd.DataFrame | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Return one standardized distribution-parameter row per prediction state.

    The project-level model helper in distributional_model intentionally returns
    raw arrays and distribution metadata for training/evaluation code. This
    wrapper is the CSV/export shape used by downstream bucket pricing.
    """
    details = predict_distribution_details(model, X)
    mu = details["mu"]
    sigma = details["sigma"]
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
    result["distribution_type"] = str(details["distribution_type"])
    if details["df"] is not None:
        result["df"] = np.asarray(details["df"], dtype=float)

    ordered = [column for column in METADATA_COLUMNS if column in result.columns]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining]


def apply_sigma_scaling_to_predictions(
    predictions: pd.DataFrame,
    alpha: float,
    sigma_col: str = "sigma",
    output_sigma_col: str = "sigma",
    raw_sigma_col: str = "raw_sigma",
) -> pd.DataFrame:
    """
    Return distribution parameters with post-hoc sigma scaling applied.

    This keeps calibrated bucket generation transparent: mu is unchanged, sigma
    is multiplied by a validation-fit alpha, and the original sigma is retained
    when the calibrated values replace the standard sigma column.
    """
    if sigma_col not in predictions.columns:
        raise ValueError(f"Prediction frame is missing sigma column {sigma_col!r}")
    result = predictions.copy()
    if output_sigma_col == sigma_col and raw_sigma_col not in result.columns:
        result[raw_sigma_col] = pd.to_numeric(result[sigma_col], errors="raise")
    result[output_sigma_col] = apply_sigma_scaling(result[sigma_col], alpha)
    result["sigma_scaling_alpha"] = float(alpha)
    return result
