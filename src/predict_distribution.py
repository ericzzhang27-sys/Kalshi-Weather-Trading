from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

try:
    from .bucket_schema import Bucket, TemperatureBucket
    from .distribution_pricing import (
        load_bucket_schema,
        price_buckets_for_dataframe,
        validate_bucket_probabilities,
    )
    from .distributional_model import normalize_distribution_name
    from .ngboost_predict import METADATA_COLUMNS, predict_distribution_params
except ImportError:
    from bucket_schema import Bucket, TemperatureBucket
    from distribution_pricing import (
        load_bucket_schema,
        price_buckets_for_dataframe,
        validate_bucket_probabilities,
    )
    from distributional_model import normalize_distribution_name
    from ngboost_predict import METADATA_COLUMNS, predict_distribution_params


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "ngboost_laplace_current36_default.pkl"
DEFAULT_FEATURE_LIST_PATH = REPO_ROOT / "outputs" / "final_feature_list.json"
DEFAULT_CALIBRATION_CONFIG_PATH = REPO_ROOT / "models" / "calibration_config.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "final_bucket_probability_predictions.csv"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "outputs" / "prediction_schema.md"

ENGINE_DIAGNOSTIC_COLUMNS = [
    "model_name",
    "model_raw_sigma",
    "raw_sigma",
    "model_sigma_scale",
    "sigma_scaling_alpha",
    "calibration_method",
    "alpha",
    "feature_missing_values",
    "feature_infinite_values",
    "feature_values_imputed_or_replaced",
]

FINAL_OUTPUT_COLUMN_ORDER = [
    "row_id",
    "date",
    "prediction_time",
    "prediction_timestamp",
    "timestamp",
    "location",
    "station",
    "station_id",
    "split",
    "forecast_high",
    "actual_high",
    "official_high",
    "forecast_error",
    "forecast_horizon_hours",
    "bucket_index",
    "bucket_name",
    "bucket_lower_temp",
    "bucket_upper_temp",
    "error_lower",
    "error_upper",
    "probability",
    "mu",
    "sigma",
    "raw_sigma",
    "model_raw_sigma",
    "model_sigma_scale",
    "sigma_scaling_alpha",
    "alpha",
    "distribution_type",
    "df",
    "model_name",
    "calibration_method",
    "feature_missing_values",
    "feature_infinite_values",
    "feature_values_imputed_or_replaced",
]


BucketSchema = list[Bucket | TemperatureBucket | dict[str, Any]]


@dataclass(frozen=True)
class EngineDiagnostics:
    model_path: str
    model_name: str
    distribution_type: str
    feature_count: int
    model_sigma_scale: float
    calibration_alpha: float
    calibration_method: str
    prediction_row_count: int
    probability_row_count: int
    bucket_count_per_prediction: int
    max_abs_row_probability_sum_deviation: float
    total_feature_values_imputed_or_replaced: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "distribution_type": self.distribution_type,
            "feature_count": self.feature_count,
            "model_sigma_scale": self.model_sigma_scale,
            "calibration_alpha": self.calibration_alpha,
            "calibration_method": self.calibration_method,
            "prediction_row_count": self.prediction_row_count,
            "probability_row_count": self.probability_row_count,
            "bucket_count_per_prediction": self.bucket_count_per_prediction,
            "max_abs_row_probability_sum_deviation": (
                self.max_abs_row_probability_sum_deviation
            ),
            "total_feature_values_imputed_or_replaced": (
                self.total_feature_values_imputed_or_replaced
            ),
        }


@dataclass(frozen=True)
class PredictionResult:
    distribution_params: pd.DataFrame
    bucket_probabilities: pd.DataFrame
    diagnostics: EngineDiagnostics


@dataclass
class ProbabilityEngine:
    model: Any
    imputer: Any
    feature_columns: list[str]
    distribution_type: str
    model_name: str
    model_path: Path
    model_sigma_scale: float = 1.0
    calibration_alpha: float = 1.0
    calibration_method: str = "none"

    def predict_distribution_params(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Predict calibrated forecast-error distribution parameters for feature rows.
        """
        working = _validate_prediction_rows(rows)
        metadata = _metadata_frame(working)
        feature_frame, feature_diagnostics = _prepare_feature_frame(
            working,
            self.feature_columns,
            self.imputer,
        )
        predictions = predict_distribution_params(
            self.model,
            feature_frame,
            metadata=metadata,
            distribution=self.distribution_type,
        )
        predictions = _apply_engine_sigma_adjustments(
            predictions,
            model_sigma_scale=self.model_sigma_scale,
            calibration_alpha=self.calibration_alpha,
            calibration_method=self.calibration_method,
        )
        predictions["model_name"] = self.model_name
        predictions = predictions.merge(feature_diagnostics, on="row_id", how="left")
        _validate_distribution_params(predictions)
        return predictions

    def predict(
        self,
        rows: pd.DataFrame,
        buckets: BucketSchema | None = None,
        forecast_rounding: str = "nearest",
    ) -> PredictionResult:
        """
        Return distribution parameters, bucket probabilities, and diagnostics.
        """
        params = self.predict_distribution_params(rows)
        priced = price_buckets_for_dataframe(
            params,
            buckets=buckets,
            dist_type=self.distribution_type,
            forecast_rounding=forecast_rounding,
        )
        priced = _attach_engine_diagnostics(priced, params)
        priced = order_prediction_columns(priced)
        validation_summary = validate_bucket_probabilities(priced)
        diagnostics = EngineDiagnostics(
            model_path=str(self.model_path),
            model_name=self.model_name,
            distribution_type=self.distribution_type,
            feature_count=len(self.feature_columns),
            model_sigma_scale=float(self.model_sigma_scale),
            calibration_alpha=float(self.calibration_alpha),
            calibration_method=self.calibration_method,
            prediction_row_count=int(validation_summary["prediction_row_count"]),
            probability_row_count=int(validation_summary["probability_row_count"]),
            bucket_count_per_prediction=int(
                validation_summary["bucket_count_per_prediction"]
            ),
            max_abs_row_probability_sum_deviation=float(
                validation_summary["max_abs_row_probability_sum_deviation"]
            ),
            total_feature_values_imputed_or_replaced=int(
                params["feature_values_imputed_or_replaced"].sum()
            ),
        )
        return PredictionResult(
            distribution_params=params,
            bucket_probabilities=priced,
            diagnostics=diagnostics,
        )


def load_probability_engine(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_list_path: str | Path = DEFAULT_FEATURE_LIST_PATH,
    calibration_config_path: str | Path | None = DEFAULT_CALIBRATION_CONFIG_PATH,
) -> ProbabilityEngine:
    """
    Load the saved model artifact, feature list, and calibration adjustment.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    with model_file.open("rb") as handle:
        artifact = pickle.load(handle)

    model, artifact_metadata = _unwrap_model_artifact(artifact)
    feature_columns = _load_and_validate_feature_columns(
        feature_list_path,
        artifact_metadata.get("feature_columns"),
    )
    calibration = _load_calibration_config(calibration_config_path)
    distribution = normalize_distribution_name(
        str(artifact_metadata.get("distribution_type", calibration["distribution_type"]))
    )
    model_name = str(artifact_metadata.get("model_name", model_file.stem))
    artifact_sigma_scale = artifact_metadata.get("sigma_scale", 1.0)
    model_sigma_scale = _positive_float(
        1.0 if artifact_sigma_scale is None else artifact_sigma_scale,
        name="model sigma scale",
    )

    return ProbabilityEngine(
        model=model,
        imputer=_repair_legacy_simple_imputer(artifact_metadata.get("imputer")),
        feature_columns=feature_columns,
        distribution_type=distribution,
        model_name=model_name,
        model_path=model_file,
        model_sigma_scale=model_sigma_scale,
        calibration_alpha=calibration["alpha"],
        calibration_method=calibration["method"],
    )


def predict_bucket_probabilities(
    rows: pd.DataFrame,
    buckets: BucketSchema | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_list_path: str | Path = DEFAULT_FEATURE_LIST_PATH,
    calibration_config_path: str | Path | None = DEFAULT_CALIBRATION_CONFIG_PATH,
    forecast_rounding: str = "nearest",
) -> PredictionResult:
    """
    One-call probability engine entry point for notebooks and scripts.
    """
    engine = load_probability_engine(
        model_path=model_path,
        feature_list_path=feature_list_path,
        calibration_config_path=calibration_config_path,
    )
    return engine.predict(rows, buckets=buckets, forecast_rounding=forecast_rounding)


def load_bucket_schema_optional(path: str | Path | None) -> BucketSchema | None:
    if path is None:
        return None
    return load_bucket_schema(path)


def order_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [column for column in FINAL_OUTPUT_COLUMN_ORDER if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining]


def save_prediction_outputs(
    result: PredictionResult,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.bucket_probabilities.to_csv(output, index=False)
    write_prediction_schema(
        schema_path,
        diagnostics=result.diagnostics,
        output_path=output,
    )


def write_prediction_schema(
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    diagnostics: EngineDiagnostics | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    path = Path(schema_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_lines: list[str] = []
    if diagnostics is not None:
        diagnostics_lines = [
            "",
            "## Run Diagnostics",
            "",
            *[
                f"- `{key}`: {value}"
                for key, value in diagnostics.to_dict().items()
            ],
        ]

    lines = [
        "# Prediction Schema",
        "",
        f"- Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Default output path: `{Path(output_path)}`",
        "",
        "## Inputs",
        "",
        "- Prediction rows must contain every selected model feature in `outputs/final_feature_list.json`.",
        "- Prediction rows must contain `forecast_high`; it is metadata for bucket conversion, not a model feature.",
        "- Optional metadata columns are preserved when present: `row_id`, `date`, `prediction_time`, `location`, `actual_high`, `forecast_error`, `forecast_horizon_hours`, and split/station fields.",
        "- Bucket definitions use lower-open, upper-closed final-temperature intervals: `lower_temp < actual_high <= upper_temp`.",
        "- If no bucket schema is provided, Kalshi-style six-bucket schemas are built around each row's `forecast_high`.",
        "",
        "## Output Columns",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `row_id` | Stable prediction-row identifier. Generated if missing. |",
        "| `bucket_index`, `bucket_name` | Bucket position and display label. |",
        "| `bucket_lower_temp`, `bucket_upper_temp` | Final-temperature interval bounds. Blank means open-ended. |",
        "| `error_lower`, `error_upper` | Forecast-error interval after subtracting `forecast_high`. Blank means open-ended. |",
        "| `probability` | `P(error_lower < forecast_error <= error_upper) = F(error_upper) - F(error_lower)`. |",
        "| `mu` | Predicted forecast-error location. |",
        "| `sigma` | Final calibrated forecast-error scale used for bucket pricing. |",
        "| `raw_sigma` | Scale after model-level sigma scaling and before post-hoc calibration alpha. |",
        "| `model_raw_sigma` | Raw scale emitted by the NGBoost artifact before engine adjustments. |",
        "| `model_sigma_scale` | Model-level scale multiplier stored in the artifact. |",
        "| `sigma_scaling_alpha`, `alpha` | Post-hoc calibration multiplier from `models/calibration_config.json`. |",
        "| `distribution_type` | Distribution family used by the CDF, such as `laplace` or `normal`. |",
        "| `model_name`, `calibration_method` | Artifact and calibration provenance. |",
        "| `feature_*` diagnostics | Per-row missing/infinite feature values imputed or replaced before prediction. |",
        "",
        "## Validation",
        "",
        "- Required feature columns are checked before prediction and ordered to the model artifact's feature list.",
        "- Feature values are coerced to numeric, infinities are treated as missing, and the saved train-only imputer fills missing values.",
        "- Distribution parameters must be finite, and final `sigma` must be greater than zero.",
        "- Bucket probabilities must be finite, nonnegative, no greater than one, and sum to one per `row_id`.",
        *diagnostics_lines,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _unwrap_model_artifact(artifact: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], dict(artifact)
    return artifact, {}


def _load_feature_columns(path: str | Path) -> list[str]:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature list file not found: {feature_path}")
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    raw_features = payload.get("features", payload.get("feature_columns"))
    if not isinstance(raw_features, list) or not raw_features:
        raise ValueError(f"Feature list file does not contain features: {feature_path}")
    features = [str(column) for column in raw_features]
    if len(set(features)) != len(features):
        raise ValueError("Feature list contains duplicate columns")
    return features


def _load_and_validate_feature_columns(
    feature_list_path: str | Path,
    artifact_features: Any,
) -> list[str]:
    feature_columns = _load_feature_columns(feature_list_path)
    if artifact_features is not None:
        artifact_columns = [str(column) for column in artifact_features]
        if artifact_columns != feature_columns:
            raise ValueError(
                "Model artifact feature columns do not match the configured final feature list. "
                f"artifact_count={len(artifact_columns)}, feature_list_count={len(feature_columns)}"
            )
    return feature_columns


def _load_calibration_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"alpha": 1.0, "method": "none", "distribution_type": "normal"}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Calibration config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "alpha": _positive_float(payload.get("alpha", 1.0), name="calibration alpha"),
        "method": str(payload.get("calibration_method", "none")),
        "distribution_type": str(payload.get("base_distribution_type", "normal")),
    }


def _positive_float(value: Any, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and greater than 0, got {value!r}")
    return numeric


def _validate_prediction_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    if rows.empty:
        raise ValueError("rows cannot be empty")
    if "forecast_high" not in rows.columns:
        raise ValueError("Prediction rows must include forecast_high for bucket pricing")
    working = rows.reset_index(drop=True).copy()
    if "row_id" in working.columns:
        if working["row_id"].isna().any():
            raise ValueError("row_id cannot contain missing values")
        if working["row_id"].duplicated().any():
            raise ValueError("row_id values must be unique")
    else:
        working.insert(0, "row_id", np.arange(len(working), dtype=int))
    working["forecast_high"] = pd.to_numeric(working["forecast_high"], errors="raise")
    if not np.isfinite(working["forecast_high"].to_numpy(dtype=float)).all():
        raise ValueError("forecast_high must be finite for every prediction row")
    return working


def _metadata_frame(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in METADATA_COLUMNS if column in rows.columns]
    if "forecast_high" not in columns:
        columns.append("forecast_high")
    if "row_id" not in columns:
        columns.insert(0, "row_id")
    return rows[columns].copy()


def _prepare_feature_frame(
    rows: pd.DataFrame,
    feature_columns: list[str],
    imputer: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [column for column in feature_columns if column not in rows.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    frame = rows[feature_columns].copy()
    coerced = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        bad_values = frame[column].notna() & values.isna()
        if bad_values.any():
            raise ValueError(f"Feature column {column!r} contains non-numeric values")
        coerced[column] = values.astype(float)

    raw_values = coerced.to_numpy(dtype=float)
    missing_counts = np.isnan(raw_values).sum(axis=1)
    infinite_counts = np.isinf(raw_values).sum(axis=1)
    cleaned = coerced.replace([np.inf, -np.inf], np.nan)

    if imputer is None:
        values = cleaned.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Feature rows contain missing or infinite values and no imputer is available")
    else:
        values = _transform_with_imputer(imputer, cleaned)

    values = np.asarray(values, dtype=float)
    if values.shape != (len(rows), len(feature_columns)):
        raise ValueError(
            "Imputer output shape does not match feature matrix: "
            f"{values.shape} vs {(len(rows), len(feature_columns))}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite feature values remain after imputation")

    feature_frame = pd.DataFrame(values, columns=feature_columns, index=rows.index)
    diagnostics = pd.DataFrame(
        {
            "row_id": rows["row_id"].to_numpy(),
            "feature_missing_values": missing_counts.astype(int),
            "feature_infinite_values": infinite_counts.astype(int),
            "feature_values_imputed_or_replaced": (
                missing_counts + infinite_counts
            ).astype(int),
        }
    )
    return feature_frame, diagnostics


def _repair_legacy_simple_imputer(imputer: Any) -> Any:
    """
    Patch sklearn SimpleImputer artifacts pickled before newer private attrs existed.

    Streamlit Cloud can install a newer scikit-learn than the one that created the
    model artifact. Some newer SimpleImputer.transform paths expect _fill_dtype,
    while older pickles only contain _fit_dtype. The imputer is numeric-only here,
    so using the fit dtype preserves the trained imputation behavior.
    """
    if imputer is None or hasattr(imputer, "_fill_dtype"):
        return imputer
    fill_dtype = getattr(imputer, "_fit_dtype", None)
    if fill_dtype is None and hasattr(imputer, "statistics_"):
        try:
            fill_dtype = np.asarray(imputer.statistics_).dtype
        except Exception:
            fill_dtype = None
    if fill_dtype is not None:
        try:
            setattr(imputer, "_fill_dtype", fill_dtype)
        except Exception:
            pass
    return imputer


def _transform_with_imputer(imputer: Any, cleaned: pd.DataFrame) -> Any:
    _repair_legacy_simple_imputer(imputer)
    try:
        return imputer.transform(cleaned)
    except AttributeError as exc:
        if "_fill_dtype" not in str(exc):
            raise
        _repair_legacy_simple_imputer(imputer)
        return _manual_simple_imputer_transform(imputer, cleaned)


def _manual_simple_imputer_transform(imputer: Any, cleaned: pd.DataFrame) -> np.ndarray:
    if not hasattr(imputer, "statistics_"):
        raise AttributeError("Imputer is missing statistics_")
    if bool(getattr(imputer, "add_indicator", False)):
        raise AttributeError("Manual SimpleImputer fallback does not support add_indicator=True")

    statistics = np.asarray(imputer.statistics_, dtype=float)
    values = cleaned.to_numpy(dtype=float)
    if statistics.shape[0] != values.shape[1]:
        raise ValueError(
            "Imputer statistics shape does not match feature matrix: "
            f"{statistics.shape[0]} vs {values.shape[1]}"
        )
    if np.isnan(statistics).any():
        raise ValueError("Manual SimpleImputer fallback cannot drop all-missing feature columns")
    missing = np.isnan(values)
    if missing.any():
        values = values.copy()
        values[missing] = np.take(statistics, np.where(missing)[1])
    return values


def _apply_engine_sigma_adjustments(
    predictions: pd.DataFrame,
    *,
    model_sigma_scale: float,
    calibration_alpha: float,
    calibration_method: str,
) -> pd.DataFrame:
    result = predictions.copy()
    model_scale = _positive_float(model_sigma_scale, name="model sigma scale")
    alpha = _positive_float(calibration_alpha, name="calibration alpha")
    model_raw_sigma = pd.to_numeric(result["sigma"], errors="raise")
    result["model_raw_sigma"] = model_raw_sigma
    result["model_sigma_scale"] = model_scale
    result["raw_sigma"] = model_raw_sigma * model_scale
    result["sigma"] = result["raw_sigma"] * alpha
    result["sigma_scaling_alpha"] = alpha
    result["alpha"] = alpha
    result["calibration_method"] = calibration_method if alpha != 1.0 else "none"
    return result


def _validate_distribution_params(predictions: pd.DataFrame) -> None:
    for column in ["mu", "sigma", "raw_sigma", "model_raw_sigma"]:
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
    if (pd.to_numeric(predictions["sigma"], errors="raise") <= 0.0).any():
        raise ValueError("sigma must be greater than zero after calibration")


def _attach_engine_diagnostics(
    priced: pd.DataFrame,
    params: pd.DataFrame,
) -> pd.DataFrame:
    diagnostic_columns = [
        "row_id",
        *[column for column in ENGINE_DIAGNOSTIC_COLUMNS if column in params.columns],
    ]
    diagnostics = params[diagnostic_columns].drop_duplicates("row_id")
    result = priced.merge(diagnostics, on="row_id", how="left", validate="many_to_one")
    return result
