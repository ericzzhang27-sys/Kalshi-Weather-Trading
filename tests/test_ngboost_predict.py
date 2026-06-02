from __future__ import annotations

from math import isclose

import numpy as np
import pandas as pd

from src.bucket_schema import make_integer_temperature_buckets
from src.ngboost_predict import predict_bucket_probabilities


class Normal:
    pass


class _FakePredictedDistribution:
    def __init__(self, loc: np.ndarray, scale: np.ndarray) -> None:
        self.loc = loc
        self.scale = scale


class _FakeNgboostModel:
    Dist = Normal

    def pred_dist(self, X: pd.DataFrame) -> _FakePredictedDistribution:
        return _FakePredictedDistribution(
            loc=np.zeros(len(X), dtype=float),
            scale=np.ones(len(X), dtype=float),
        )


def test_predict_bucket_probabilities_prices_model_distribution_params() -> None:
    X = pd.DataFrame({"forecast_high": [72.5], "feature": [1.0]})
    buckets = make_integer_temperature_buckets(72, 74)

    priced = predict_bucket_probabilities(_FakeNgboostModel(), X, buckets=buckets)

    assert len(priced) == 3
    assert priced["bucket_name"].tolist() == ["72 or lower", "73", "74 or higher"]
    assert priced["error_lower"].isna().tolist() == [True, False, False]
    assert priced["error_upper"].isna().tolist() == [False, False, True]
    assert isclose(float(priced["probability"].sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
