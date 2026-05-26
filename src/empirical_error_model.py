from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interval_probs import (  # noqa: E402
    Interval,
    cdf_to_interval_probs,
    normalize_probs,
    validate_interval_probs,
)


REQUIRED_COLUMNS = [
    "date",
    "station_id",
    "prediction_hour",
    "forecast_horizon_hours",
    "day_of_year",
    "month",
    "season",
    "forecast_error",
]

CRITICAL_MISSING_FRACTION_LIMIT = 0.05
FORECAST_HORIZON_ABS_SANITY_LIMIT_HOURS = 72.0


def circular_doy_distance(a: int, b: int) -> int:
    raw = abs(int(a) - int(b))
    return min(raw, 365 - raw)


class EmpiricalErrorModel:
    def __init__(
        self,
        min_samples: int = 30,
        doy_window: int = 30,
        horizon_window_hours: float = 6.0,
        smoothing_alpha: float = 1.0,
    ):
        if int(min_samples) < 1:
            raise ValueError("min_samples must be at least 1")
        if int(doy_window) < 0:
            raise ValueError("doy_window must be nonnegative")
        if float(horizon_window_hours) < 0.0:
            raise ValueError("horizon_window_hours must be nonnegative")
        if float(smoothing_alpha) < 0.0:
            raise ValueError("smoothing_alpha must be nonnegative")

        self.min_samples = int(min_samples)
        self.doy_window = int(doy_window)
        self.horizon_window_hours = float(horizon_window_hours)
        self.smoothing_alpha = float(smoothing_alpha)

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Empirical error input is missing required columns: {missing_columns}")
        if df.empty:
            raise ValueError("Empirical error input is empty")

        dates = pd.to_datetime(df["date"], errors="coerce")
        self._check_missing_fraction(dates, "date")
        if dates.isna().any() and df["date"].notna().any():
            bad_count = int((dates.isna() & df["date"].notna()).sum())
            if bad_count:
                raise ValueError(f"date contains {bad_count} unparseable values")

        for column in ["station_id", "season"]:
            self._check_missing_fraction(df[column], column)

        numeric_specs = {
            "forecast_error": (None, None),
            "prediction_hour": (0.0, 23.0),
            "forecast_horizon_hours": (None, None),
            "day_of_year": (1.0, 366.0),
            "month": (1.0, 12.0),
        }
        for column, (minimum, maximum) in numeric_specs.items():
            raw = df[column]
            numeric = pd.to_numeric(raw, errors="coerce")
            unparseable = raw.notna() & numeric.isna()
            if unparseable.any():
                raise ValueError(f"{column} contains {int(unparseable.sum())} non-numeric values")
            self._check_missing_fraction(numeric, column)

            present = numeric.dropna()
            if not np.isfinite(present).all():
                raise ValueError(f"{column} contains non-finite values")
            if minimum is not None and (present < minimum).any():
                raise ValueError(f"{column} contains values below {minimum:g}")
            if maximum is not None and (present > maximum).any():
                raise ValueError(f"{column} contains values above {maximum:g}")
            if (
                column == "forecast_horizon_hours"
                and (present.abs() > FORECAST_HORIZON_ABS_SANITY_LIMIT_HOURS).any()
            ):
                raise ValueError(
                    "forecast_horizon_hours contains values with absolute magnitude above "
                    f"{FORECAST_HORIZON_ABS_SANITY_LIMIT_HOURS:g} hours"
                )
            if column in {"prediction_hour", "day_of_year", "month"}:
                integer_like = np.isclose(present, np.round(present), rtol=0.0, atol=1e-9)
                if not bool(np.all(integer_like)):
                    raise ValueError(f"{column} must contain integer-like values")

    @staticmethod
    def _check_missing_fraction(series: pd.Series, column: str) -> None:
        missing_count = int(series.isna().sum())
        if missing_count == 0:
            return

        total = len(series)
        missing_fraction = missing_count / total if total else 1.0
        if missing_fraction > CRITICAL_MISSING_FRACTION_LIMIT:
            raise ValueError(
                f"{column} has excessive missing/unusable values: "
                f"{missing_count}/{total} ({missing_fraction:.1%})"
            )

    def fit(self, train_df: pd.DataFrame) -> "EmpiricalErrorModel":
        self._validate_input_df(train_df)

        cleaned = train_df.copy()
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="raise").dt.normalize()
        for column in [
            "forecast_error",
            "prediction_hour",
            "forecast_horizon_hours",
            "day_of_year",
            "month",
        ]:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

        cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS).copy()
        if cleaned.empty:
            raise ValueError("No usable empirical error rows remain after dropping missing critical fields")

        cleaned["station_id"] = cleaned["station_id"].astype(str)
        cleaned["prediction_hour"] = cleaned["prediction_hour"].astype(int)
        cleaned["day_of_year"] = cleaned["day_of_year"].astype(int)
        cleaned["month"] = cleaned["month"].astype(int)
        cleaned["forecast_error"] = cleaned["forecast_error"].astype(float)
        cleaned["forecast_horizon_hours"] = cleaned["forecast_horizon_hours"].astype(float)

        self.error_library_ = cleaned.sort_values("date").reset_index(drop=True)
        self._history_dates_ = self.error_library_["date"].to_numpy(dtype="datetime64[ns]")
        self._station_ids_ = self.error_library_["station_id"].to_numpy()
        self._prediction_hours_ = self.error_library_["prediction_hour"].to_numpy(dtype=int)
        self._horizons_ = self.error_library_["forecast_horizon_hours"].to_numpy(dtype=float)
        self._doys_ = self.error_library_["day_of_year"].to_numpy(dtype=int)
        self._months_ = self.error_library_["month"].to_numpy(dtype=int)
        self._seasons_ = self.error_library_["season"].to_numpy()
        self._errors_ = self.error_library_["forecast_error"].to_numpy(dtype=float)
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "error_library_"):
            raise ValueError("EmpiricalErrorModel must be fitted before prediction")

    def _candidate_errors(self, row: pd.Series) -> tuple[np.ndarray, str, int]:
        self._require_fitted()

        row_date = pd.to_datetime(row["date"], errors="raise").normalize()
        cutoff = np.datetime64(row_date.to_datetime64())
        past_end = int(np.searchsorted(self._history_dates_, cutoff, side="left"))
        if past_end <= 0:
            raise ValueError(f"No historical data before prediction row date {row_date.date()}")

        station_id = str(row["station_id"])
        prediction_hour = int(row["prediction_hour"])
        horizon = float(row["forecast_horizon_hours"])
        day_of_year = int(row["day_of_year"])
        month = int(row["month"])
        season = row["season"]

        dates = self._history_dates_[:past_end]
        station_ids = self._station_ids_[:past_end]
        prediction_hours = self._prediction_hours_[:past_end]
        horizons = self._horizons_[:past_end]
        doys = self._doys_[:past_end]
        months = self._months_[:past_end]
        seasons = self._seasons_[:past_end]
        errors = self._errors_[:past_end]

        raw_doy_distances = np.abs(doys - day_of_year)
        doy_distances = np.minimum(raw_doy_distances, 365 - raw_doy_distances)

        level_1_mask = (
            (station_ids == station_id)
            & (doy_distances <= self.doy_window)
            & (prediction_hours == prediction_hour)
            & (np.abs(horizons - horizon) <= self.horizon_window_hours)
        )
        selected_errors = self._errors_if_enough(
            errors,
            level_1_mask,
            "same_station_doy_hour_horizon",
        )

        if selected_errors is None:
            level_2_mask = (
                (seasons == season)
                & (prediction_hours == prediction_hour)
                & (np.abs(horizons - horizon) <= self.horizon_window_hours)
            )
            selected_errors = self._errors_if_enough(
                errors,
                level_2_mask,
                "same_season_hour_horizon",
            )

        if selected_errors is None:
            level_3_mask = (station_ids == station_id) & (months == month)
            selected_errors = self._errors_if_enough(errors, level_3_mask, "same_station_month")

        if selected_errors is None:
            selected_errors = (errors, "all_past")

        if (dates >= cutoff).any():
            raise AssertionError("Candidate history contains same-date or future rows")

        candidate_errors, fallback_level = selected_errors
        if len(candidate_errors) <= 0:
            raise ValueError(f"No historical data before prediction row date {row_date.date()}")
        return candidate_errors, fallback_level, int(len(candidate_errors))

    def _errors_if_enough(
        self,
        errors: np.ndarray,
        mask: np.ndarray,
        fallback_level: str,
    ) -> tuple[np.ndarray, str] | None:
        if int(mask.sum()) >= self.min_samples:
            return errors[mask], fallback_level
        return None

    def _cdf_from_errors(
        self,
        errors: np.ndarray,
        boundaries: list[float],
    ) -> dict[float, float]:
        finite_boundaries = sorted({float(boundary) for boundary in boundaries})
        for boundary in finite_boundaries:
            if not np.isfinite(boundary):
                raise ValueError(f"CDF boundary must be finite, got {boundary!r}")

        n = int(len(errors))
        if n <= 0:
            raise ValueError("Cannot estimate empirical CDF with no historical errors")

        alpha = self.smoothing_alpha
        denominator = n + 2.0 * alpha
        if denominator <= 0.0:
            raise ValueError("Invalid empirical CDF denominator")

        cdf_values: dict[float, float] = {}
        running_max = 0.0
        for boundary in finite_boundaries:
            count = int(np.count_nonzero(errors <= boundary))
            value = (count + alpha) / denominator
            value = max(running_max, min(1.0, max(0.0, float(value))))
            cdf_values[boundary] = value
            running_max = value

        return cdf_values

    def predict_cdf(
        self,
        row: pd.Series,
        boundaries: list[float],
    ) -> dict[float, float]:
        errors, _, _ = self._candidate_errors(row)
        return self._cdf_from_errors(errors, boundaries)

    def predict_interval_probs(
        self,
        row: pd.Series,
        intervals: list[Interval],
    ) -> dict[str, Any]:
        finite_boundaries = sorted(
            {
                float(boundary)
                for interval in intervals
                for boundary in interval
                if boundary is not None
            }
        )
        errors, fallback_level, sample_size = self._candidate_errors(row)
        cdf_values = self._cdf_from_errors(errors, finite_boundaries)
        probs = cdf_to_interval_probs(cdf_values, intervals)
        probs = normalize_probs(probs)
        validate_interval_probs(probs)

        return {
            "probs": probs,
            "cdf_values": cdf_values,
            "fallback_level": fallback_level,
            "sample_size": sample_size,
        }
