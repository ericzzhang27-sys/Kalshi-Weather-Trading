from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


DATE_COLUMN_CANDIDATES = ("date", "target_date")


@dataclass(frozen=True)
class ChronologicalSplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    summary: dict[str, Any]


def get_preferred_date_column(
    df: pd.DataFrame,
    date_column: str | None = None,
) -> str:
    if date_column is not None:
        if date_column not in df.columns:
            raise ValueError(f"Requested date column {date_column!r} is not present")
        return date_column

    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "No chronological split date column found. Expected one of: "
        + ", ".join(DATE_COLUMN_CANDIDATES)
    )


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    train_end_date: str | pd.Timestamp | None = None,
    validation_end_date: str | pd.Timestamp | None = None,
    validation_start_date: str | pd.Timestamp | None = None,
    test_start_date: str | pd.Timestamp | None = None,
    date_column: str | None = None,
) -> ChronologicalSplitResult:
    """Split rows by date into train, validation, and test sets.

    The split is always chronological and date based. Random splitting is
    intentionally not supported for this weather/market workflow.
    """

    if df.empty:
        raise ValueError("Cannot split an empty dataframe")

    split_date_column = get_preferred_date_column(df, date_column)
    working = df.copy()
    parsed_dates = pd.to_datetime(working[split_date_column], errors="coerce").dt.normalize()
    if parsed_dates.isna().any():
        bad_count = int(parsed_dates.isna().sum())
        raise ValueError(
            f"{split_date_column!r} has {bad_count} unparsable values; cannot create chronological split"
        )

    working[split_date_column] = parsed_dates
    sort_columns = [
        column
        for column in [split_date_column, "location", "station_id", "prediction_time"]
        if column in working.columns
    ]
    working = working.sort_values(sort_columns).reset_index(drop=True)

    using_start_dates = validation_start_date is not None or test_start_date is not None
    if using_start_dates:
        if train_end_date is not None or validation_end_date is not None:
            raise ValueError(
                "Use either train_end_date/validation_end_date or "
                "validation_start_date/test_start_date, not both"
            )
        if validation_start_date is None or test_start_date is None:
            raise ValueError(
                "Both validation_start_date and test_start_date are required for start-date splits"
            )
        validation_start = pd.to_datetime(validation_start_date, errors="raise").normalize()
        test_start = pd.to_datetime(test_start_date, errors="raise").normalize()
        if validation_start >= test_start:
            raise ValueError(
                "validation_start_date must be earlier than test_start_date; "
                f"got {validation_start.date()} and {test_start.date()}"
            )

        split_dates = working[split_date_column]
        train = working[split_dates < validation_start].copy()
        validation = working[
            (split_dates >= validation_start) & (split_dates < test_start)
        ].copy()
        test = working[split_dates >= test_start].copy()

        summary = build_split_summary(
            train=train,
            validation=validation,
            test=test,
            date_column=split_date_column,
            train_end=_max_date(train, split_date_column) if not train.empty else validation_start,
            validation_end=_max_date(validation, split_date_column)
            if not validation.empty
            else test_start,
            validation_start=validation_start,
            test_start=test_start,
            strategy="chronological",
        )
        validate_chronological_splits(train, validation, test, split_date_column)
        return ChronologicalSplitResult(
            train=train,
            validation=validation,
            test=test,
            summary=summary,
        )

    if train_end_date is None or validation_end_date is None:
        default_train_end, default_validation_end = choose_default_split_dates(parsed_dates)
        if train_end_date is None:
            train_end_date = default_train_end
        if validation_end_date is None:
            validation_end_date = default_validation_end

    train_end = pd.to_datetime(train_end_date, errors="raise").normalize()
    validation_end = pd.to_datetime(validation_end_date, errors="raise").normalize()
    if train_end >= validation_end:
        raise ValueError(
            "train_end_date must be earlier than validation_end_date; "
            f"got {train_end.date()} and {validation_end.date()}"
        )

    split_dates = working[split_date_column]
    train = working[split_dates <= train_end].copy()
    validation = working[(split_dates > train_end) & (split_dates <= validation_end)].copy()
    test = working[split_dates > validation_end].copy()

    summary = build_split_summary(
        train=train,
        validation=validation,
        test=test,
        date_column=split_date_column,
        train_end=train_end,
        validation_end=validation_end,
        validation_start=_min_date(validation, split_date_column),
        test_start=_min_date(test, split_date_column),
        strategy="chronological",
    )
    validate_chronological_splits(train, validation, test, split_date_column)
    return ChronologicalSplitResult(
        train=train,
        validation=validation,
        test=test,
        summary=summary,
    )


def choose_default_split_dates(
    dates: pd.Series,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    parsed = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    unique_dates = pd.Index(parsed.unique()).sort_values()
    if len(unique_dates) < 3:
        raise ValueError("Need at least three unique dates for train/validation/test splitting")

    repo_train_end = pd.Timestamp("2023-12-31")
    repo_validation_end = pd.Timestamp("2024-12-31")
    has_repo_train = (unique_dates <= repo_train_end).any()
    has_repo_validation = ((unique_dates > repo_train_end) & (unique_dates <= repo_validation_end)).any()
    has_repo_test = (unique_dates > repo_validation_end).any()
    if has_repo_train and has_repo_validation and has_repo_test:
        return repo_train_end, repo_validation_end

    train_index = max(0, int(len(unique_dates) * 0.6) - 1)
    validation_index = max(train_index + 1, int(len(unique_dates) * 0.8) - 1)
    validation_index = min(validation_index, len(unique_dates) - 2)
    return pd.Timestamp(unique_dates[train_index]), pd.Timestamp(unique_dates[validation_index])


def validate_chronological_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    date_column: str,
) -> None:
    split_map = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    empty = [name for name, split_df in split_map.items() if split_df.empty]
    if empty:
        raise ValueError(f"Chronological split produced empty split(s): {', '.join(empty)}")

    train_max = _max_date(train, date_column)
    validation_min = _min_date(validation, date_column)
    validation_max = _max_date(validation, date_column)
    test_min = _min_date(test, date_column)

    if train_max >= validation_min:
        raise AssertionError(
            "Chronological split overlap: train max date must be before validation min date "
            f"({train_max.date()} >= {validation_min.date()})"
        )
    if validation_max >= test_min:
        raise AssertionError(
            "Chronological split overlap: validation max date must be before test min date "
            f"({validation_max.date()} >= {test_min.date()})"
        )

    if len(train.index.intersection(validation.index)) > 0:
        raise AssertionError("Chronological split overlap: train and validation share row indices")
    if len(train.index.intersection(test.index)) > 0:
        raise AssertionError("Chronological split overlap: train and test share row indices")
    if len(validation.index.intersection(test.index)) > 0:
        raise AssertionError("Chronological split overlap: validation and test share row indices")


def build_split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    date_column: str,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    validation_start: pd.Timestamp | None = None,
    test_start: pd.Timestamp | None = None,
    strategy: str = "chronological",
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "date_column": date_column,
        "train_end_date": train_end.date().isoformat(),
        "validation_start_date": None
        if validation_start is None
        else validation_start.date().isoformat(),
        "validation_end_date": validation_end.date().isoformat(),
        "test_start_date": None if test_start is None else test_start.date().isoformat(),
        "splits": {
            "train": _split_summary(train, date_column),
            "validation": _split_summary(validation, date_column),
            "test": _split_summary(test, date_column),
        },
    }


def _split_summary(df: pd.DataFrame, date_column: str) -> dict[str, Any]:
    if df.empty:
        return {
            "row_count": 0,
            "date_min": None,
            "date_max": None,
        }
    return {
        "row_count": int(len(df)),
        "date_min": _min_date(df, date_column).date().isoformat(),
        "date_max": _max_date(df, date_column).date().isoformat(),
    }


def _min_date(df: pd.DataFrame, date_column: str) -> pd.Timestamp:
    return pd.to_datetime(df[date_column], errors="raise").dt.normalize().min()


def _max_date(df: pd.DataFrame, date_column: str) -> pd.Timestamp:
    return pd.to_datetime(df[date_column], errors="raise").dt.normalize().max()
