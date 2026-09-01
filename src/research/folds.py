from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpandingFold:
    fold_id: str
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]
    purge_dates: tuple[date, ...]

    def indices(self, frame: pd.DataFrame, date_col: str = "target_date") -> tuple[np.ndarray, np.ndarray]:
        dates = pd.to_datetime(frame[date_col], errors="raise").dt.date
        train = np.flatnonzero(dates.isin(self.train_dates).to_numpy())
        validation = np.flatnonzero(dates.isin(self.validation_dates).to_numpy())
        if set(train) & set(validation):
            raise AssertionError("train and validation indices overlap")
        return train, validation

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_start": min(self.train_dates).isoformat(),
            "train_end": max(self.train_dates).isoformat(),
            "validation_start": min(self.validation_dates).isoformat(),
            "validation_end": max(self.validation_dates).isoformat(),
            "n_train_days": len(self.train_dates),
            "n_validation_days": len(self.validation_dates),
            "purge_dates": [value.isoformat() for value in self.purge_dates],
        }


def _unique_dates(values: Iterable[object]) -> list[date]:
    parsed = pd.to_datetime(pd.Series(list(values)), errors="raise").dt.date
    return sorted(set(parsed.tolist()))


def event_day_folds(
    dates: Iterable[object],
    *,
    warmup_days: int = 90,
    validation_days: int = 90,
    purge_days: int = 1,
) -> list[ExpandingFold]:
    """Sequential trading folds grouped by whole event-day."""
    unique = _unique_dates(dates)
    if warmup_days < 1 or validation_days < 1 or purge_days < 0:
        raise ValueError("fold sizes must be positive and purge_days nonnegative")
    folds: list[ExpandingFold] = []
    cursor = warmup_days + purge_days
    while cursor < len(unique):
        validation = unique[cursor : cursor + validation_days]
        if not validation:
            break
        purge = unique[max(0, cursor - purge_days) : cursor]
        train = unique[: max(0, cursor - purge_days)]
        if train:
            folds.append(ExpandingFold(f"trading_outer_{len(folds):02d}", tuple(train), tuple(validation), tuple(purge)))
        cursor += validation_days
    return folds


def calendar_month_folds(
    dates: Iterable[object],
    *,
    minimum_training_days: int = 730,
    validation_months: int = 6,
    purge_days: int = 1,
) -> list[ExpandingFold]:
    """Expanding whole-day weather folds with calendar-month outer windows."""
    unique = _unique_dates(dates)
    if len(unique) <= minimum_training_days + purge_days:
        return []
    first_validation = pd.Timestamp(unique[minimum_training_days + purge_days]).normalize()
    end = pd.Timestamp(unique[-1]).normalize()
    folds: list[ExpandingFold] = []
    cursor = first_validation
    all_dates = pd.DatetimeIndex(unique)
    while cursor <= end:
        validation_end = cursor + pd.DateOffset(months=validation_months)
        train_end = cursor - pd.Timedelta(days=purge_days)
        train = tuple(value.date() for value in all_dates[all_dates < train_end])
        purge = tuple(value.date() for value in all_dates[(all_dates >= train_end) & (all_dates < cursor)])
        validation = tuple(value.date() for value in all_dates[(all_dates >= cursor) & (all_dates < validation_end)])
        if train and validation:
            folds.append(ExpandingFold(f"weather_outer_{len(folds):02d}", train, validation, purge))
        cursor = validation_end
    return folds


def inner_rolling_folds(train_dates: Iterable[object], *, n_folds: int = 3, purge_days: int = 1) -> list[ExpandingFold]:
    """Create rolling inner folds wholly inside an outer training period."""
    unique = _unique_dates(train_dates)
    if n_folds < 1 or len(unique) < (n_folds + 1) * 2:
        return []
    block = max(1, len(unique) // (n_folds + 1))
    folds: list[ExpandingFold] = []
    for index in range(n_folds):
        start = block * (index + 1)
        stop = len(unique) if index == n_folds - 1 else min(len(unique), start + block)
        purge = unique[max(0, start - purge_days) : start]
        train = unique[: max(0, start - purge_days)]
        validation = unique[start:stop]
        if train and validation:
            folds.append(ExpandingFold(f"inner_{index:02d}", tuple(train), tuple(validation), tuple(purge)))
    return folds
