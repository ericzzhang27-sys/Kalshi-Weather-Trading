from __future__ import annotations

from math import isclose
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bucket_schema import make_integer_temperature_buckets
from src.predict_distribution import ProbabilityEngine, load_probability_engine, save_prediction_outputs


class Normal:
    pass


class _FakePredictedDistribution:
    def __init__(self, loc: np.ndarray, scale: np.ndarray) -> None:
        self.loc = loc
        self.scale = scale


class _FakeModel:
    Dist = Normal

    def pred_dist(self, X: pd.DataFrame) -> _FakePredictedDistribution:
        if X.isna().any().any():
            raise ValueError("Fake model received missing feature values")
        return _FakePredictedDistribution(
            loc=np.zeros(len(X), dtype=float),
            scale=np.ones(len(X), dtype=float),
        )


class _LegacySimpleImputerLike:
    add_indicator = False

    def __init__(self) -> None:
        self._fit_dtype = np.dtype("float64")
        self.statistics_ = np.array([1.5, 3.5], dtype=float)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "_fill_dtype"):
            raise AttributeError("'SimpleImputer' object has no attribute '_fill_dtype'")
        values = X.to_numpy(dtype=float)
        mask = np.isnan(values)
        if mask.any():
            values = values.copy()
            values[mask] = np.take(self.statistics_, np.where(mask)[1])
        return values


def _fake_engine() -> ProbabilityEngine:
    feature_columns = ["feature_a", "feature_b"]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]}))
    return ProbabilityEngine(
        model=_FakeModel(),
        imputer=imputer,
        feature_columns=feature_columns,
        distribution_type="normal",
        model_name="fake_ngboost",
        model_path=Path("fake.pkl"),
        model_sigma_scale=2.0,
        calibration_alpha=0.5,
        calibration_method="global_sigma_scaling",
    )


def test_probability_engine_returns_calibrated_distribution_and_bucket_probs() -> None:
    rows = pd.DataFrame(
        {
            "forecast_high": [72.5],
            "date": ["2025-07-01"],
            "location": ["NYC"],
            "feature_a": [1.0],
            "feature_b": [np.nan],
        }
    )
    buckets = make_integer_temperature_buckets(72, 74)

    result = _fake_engine().predict(rows, buckets=buckets)

    params = result.distribution_params.iloc[0]
    assert params["model_raw_sigma"] == 1.0
    assert params["raw_sigma"] == 2.0
    assert params["sigma"] == 1.0
    assert params["sigma_scaling_alpha"] == 0.5
    assert params["feature_missing_values"] == 1
    assert result.bucket_probabilities["bucket_name"].tolist() == [
        "72 or lower",
        "73",
        "74 or higher",
    ]
    assert isclose(
        float(result.bucket_probabilities["probability"].sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result.diagnostics.prediction_row_count == 1
    assert result.diagnostics.total_feature_values_imputed_or_replaced == 1


def test_probability_engine_validates_required_feature_columns() -> None:
    rows = pd.DataFrame({"forecast_high": [72.5], "feature_a": [1.0]})

    with pytest.raises(ValueError, match="feature_b"):
        _fake_engine().predict_distribution_params(rows)


def test_save_prediction_outputs_writes_csv_and_schema(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "forecast_high": [72.5],
            "feature_a": [1.0],
            "feature_b": [2.0],
        }
    )
    result = _fake_engine().predict(rows, buckets=make_integer_temperature_buckets(72, 74))
    output_path = tmp_path / "predictions.csv"
    schema_path = tmp_path / "schema.md"

    save_prediction_outputs(result, output_path=output_path, schema_path=schema_path)

    assert output_path.exists()
    assert schema_path.exists()
    assert "Prediction Schema" in schema_path.read_text(encoding="utf-8")


def test_load_probability_engine_repairs_legacy_simple_imputer_artifact() -> None:
    engine = load_probability_engine()

    assert engine.imputer is not None
    assert hasattr(engine.imputer, "_fill_dtype")


def test_probability_engine_repairs_missing_simple_imputer_fill_dtype() -> None:
    imputer = _LegacySimpleImputerLike()
    engine = ProbabilityEngine(
        model=_FakeModel(),
        imputer=imputer,
        feature_columns=["feature_a", "feature_b"],
        distribution_type="normal",
        model_name="fake_ngboost",
        model_path=Path("fake.pkl"),
    )
    rows = pd.DataFrame(
        {
            "forecast_high": [72.5],
            "feature_a": [np.nan],
            "feature_b": [2.0],
        }
    )

    result = engine.predict(rows, buckets=make_integer_temperature_buckets(72, 74))

    assert hasattr(imputer, "_fill_dtype")
    assert result.diagnostics.total_feature_values_imputed_or_replaced == 1
    assert result.bucket_probabilities["probability"].sum() == pytest.approx(1.0)
