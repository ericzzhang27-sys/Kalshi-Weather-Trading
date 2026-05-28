from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distributional_model import (  # noqa: E402
    TARGET_COLUMN,
    get_feature_columns,
    normal_nll,
    predict_distribution_params,
    train_ngboost_normal,
    validate_no_leakage_feature_columns,
)
from src.splits import chronological_train_validation_test_split  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELING_TABLE_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"
DEFAULT_FEATURE_COLUMNS_PATH = REPO_ROOT / "outputs" / "day8_features" / "feature_columns.json"
MODEL_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_normal_v0.pkl"
FEATURE_OUTPUT_PATH = REPO_ROOT / "models" / "ngboost_normal_v0_features.json"
PARAMS_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_distribution_params_v0.csv"
METRICS_OUTPUT_PATH = REPO_ROOT / "outputs" / "ngboost_nll_v0.json"
EMPIRICAL_BASELINE_PATH = (
    REPO_ROOT / "outputs" / "day9_empirical_baseline" / "empirical_baseline_predictions.csv"
)

DATETIME_COLUMNS = [
    "date",
    "target_date",
    "prediction_time",
    "prediction_timestamp",
    "current_temp_source_time",
    "max_temp_so_far_source_time",
    "forecast_temp_source_valid_time",
    "forecast_max_so_far_source_valid_time",
]

METADATA_COLUMNS = [
    "date",
    "timestamp",
    "prediction_time",
    "prediction_timestamp",
    "location",
    "station",
    "station_id",
    "forecast_high",
    "actual_high",
    "official_high",
    "forecast_horizon_hours",
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_path = _resolve_path(args.dataset_path)
    feature_columns_path = _resolve_path(args.feature_columns_path)

    df = load_modeling_table(dataset_path)
    validate_target_column(df)

    split_result = chronological_train_validation_test_split(
        df,
        train_end_date=args.train_end_date,
        validation_end_date=args.validation_end_date,
    )
    train_df = split_result.train
    validation_df = split_result.validation
    test_df = split_result.test
    split_summary = split_result.summary

    validate_target_column(train_df, split_name="train")
    validate_target_column(validation_df, split_name="validation")
    validate_target_column(test_df, split_name="test")

    feature_columns = get_feature_columns(df, feature_columns_path)
    validate_no_leakage_feature_columns(feature_columns)

    X_train, X_validation, X_test, imputer, preprocessing_notes = build_imputed_feature_frames(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
    )
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = validation_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model = train_ngboost_normal(
        X_train=X_train,
        y_train=y_train,
        X_val=X_validation,
        y_val=y_validation,
    )

    validation_mu, validation_sigma = predict_distribution_params(model, X_validation)
    test_mu, test_sigma = predict_distribution_params(model, X_test)

    validation_nll = normal_nll(y_validation, validation_mu, validation_sigma)
    test_nll = normal_nll(y_test, test_mu, test_sigma)
    validate_distribution_outputs(
        split_name="validation",
        expected_count=len(validation_df),
        mu=validation_mu,
        sigma=validation_sigma,
        nll=validation_nll,
    )
    validate_distribution_outputs(
        split_name="test",
        expected_count=len(test_df),
        mu=test_mu,
        sigma=test_sigma,
        nll=test_nll,
    )

    baseline_mu = float(np.mean(y_train))
    baseline_sigma = float(np.std(y_train, ddof=1))
    if not np.isfinite(baseline_sigma) or baseline_sigma <= 0:
        raise ValueError("Train target standard deviation is not positive; cannot score baseline")
    validation_baseline_nll = normal_nll(
        y_validation,
        np.full_like(y_validation, baseline_mu, dtype=float),
        np.full_like(y_validation, baseline_sigma, dtype=float),
    )
    test_baseline_nll = normal_nll(
        y_test,
        np.full_like(y_test, baseline_mu, dtype=float),
        np.full_like(y_test, baseline_sigma, dtype=float),
    )

    predictions = pd.concat(
        [
            build_prediction_frame(
                split_name="validation",
                split_df=validation_df,
                mu=validation_mu,
                sigma=validation_sigma,
                nll=validation_nll,
            ),
            build_prediction_frame(
                split_name="test",
                split_df=test_df,
                mu=test_mu,
                sigma=test_sigma,
                nll=test_nll,
            ),
        ],
        ignore_index=True,
    )
    predictions.insert(0, "row_id", np.arange(len(predictions), dtype=int))
    if len(predictions) != len(validation_df) + len(test_df):
        raise AssertionError("Distribution prediction count does not match validation/test rows")

    sigma_summary = {
        "validation": summarize_sigma(validation_sigma),
        "test": summarize_sigma(test_sigma),
    }
    metrics = build_metrics(
        split_summary=split_summary,
        feature_columns=feature_columns,
        validation_nll=validation_nll,
        test_nll=test_nll,
        validation_baseline_nll=validation_baseline_nll,
        test_baseline_nll=test_baseline_nll,
        baseline_mu=baseline_mu,
        baseline_sigma=baseline_sigma,
        sigma_summary=sigma_summary,
        preprocessing_notes=preprocessing_notes,
        empirical_baseline_note=build_empirical_baseline_note(EMPIRICAL_BASELINE_PATH),
    )

    write_outputs(
        model=model,
        imputer=imputer,
        feature_columns=feature_columns,
        split_summary=split_summary,
        preprocessing_notes=preprocessing_notes,
        predictions=predictions,
        metrics=metrics,
        feature_columns_path=feature_columns_path,
    )
    print_report(
        dataset_path=dataset_path,
        total_rows=len(df),
        split_summary=split_summary,
        feature_columns=feature_columns,
        validation_nll=validation_nll,
        test_nll=test_nll,
        validation_baseline_nll=validation_baseline_nll,
        test_baseline_nll=test_baseline_nll,
        sigma_summary=sigma_summary,
        preprocessing_notes=preprocessing_notes,
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Day 11 NGBoost Normal forecast-error model.")
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_MODELING_TABLE_PATH),
        help="Path to the modeling rows CSV.",
    )
    parser.add_argument(
        "--feature-columns-path",
        default=str(DEFAULT_FEATURE_COLUMNS_PATH),
        help="Path to the leakage-safe Day 8 feature columns JSON.",
    )
    parser.add_argument(
        "--train-end-date",
        default=None,
        help="Inclusive train end date. Defaults to repo-aware chronological split.",
    )
    parser.add_argument(
        "--validation-end-date",
        default=None,
        help="Inclusive validation end date. Defaults to repo-aware chronological split.",
    )
    return parser.parse_args(argv)


def load_modeling_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Modeling table not found: {path}")
    df = pd.read_csv(path)
    for column in DATETIME_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
            if df[column].isna().any():
                bad_count = int(df[column].isna().sum())
                raise ValueError(f"{column!r} has {bad_count} unparsable timestamp values")
    return df


def validate_target_column(df: pd.DataFrame, split_name: str | None = None) -> None:
    label = f" in {split_name}" if split_name else ""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    missing = int(target.isna().sum())
    if missing:
        raise ValueError(f"{TARGET_COLUMN!r} has {missing} missing/non-numeric values{label}")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(f"{TARGET_COLUMN!r} has non-finite values{label}")
    df[TARGET_COLUMN] = target.astype(float)


def build_imputed_feature_frames(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer, dict[str, Any]]:
    raw_frames = {
        "train": _numeric_feature_frame(train_df, feature_columns),
        "validation": _numeric_feature_frame(validation_df, feature_columns),
        "test": _numeric_feature_frame(test_df, feature_columns),
    }
    counts = {
        split: _missing_and_infinite_counts(frame)
        for split, frame in raw_frames.items()
    }
    cleaned_frames = {
        split: frame.replace([np.inf, -np.inf], np.nan)
        for split, frame in raw_frames.items()
    }

    all_missing_train = [
        column
        for column in feature_columns
        if cleaned_frames["train"][column].isna().all()
    ]
    if all_missing_train:
        raise ValueError(f"Train feature columns are entirely missing: {all_missing_train}")

    imputer = SimpleImputer(strategy="median")
    imputer.fit(cleaned_frames["train"])

    transformed: dict[str, pd.DataFrame] = {}
    for split, frame in cleaned_frames.items():
        values = imputer.transform(frame)
        if values.shape[1] != len(feature_columns):
            raise AssertionError(
                f"Imputer changed feature width for {split}: {values.shape[1]} vs {len(feature_columns)}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values remain after imputation for {split}")
        transformed[split] = pd.DataFrame(values, columns=feature_columns, index=frame.index)

    notes = {
        "rows_dropped": 0,
        "imputation_strategy": "median fit on train only",
        "missing_and_infinite_values_by_split": counts,
        "train_imputer_medians": {
            column: float(value)
            for column, value in zip(feature_columns, imputer.statistics_, strict=True)
        },
    }
    return (
        transformed["train"],
        transformed["validation"],
        transformed["test"],
        imputer,
        notes,
    )


def _numeric_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns: {missing}")
    frame = df[feature_columns].copy()
    for column in feature_columns:
        if not (
            pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        ):
            raise ValueError(f"Selected feature {column!r} is not numeric")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _missing_and_infinite_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.to_numpy(dtype=float)
    missing = int(np.isnan(values).sum())
    infinite = int(np.isinf(values).sum())
    return {
        "missing_values": missing,
        "infinite_values": infinite,
        "total_values_imputed_or_replaced": missing + infinite,
    }


def validate_distribution_outputs(
    split_name: str,
    expected_count: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    nll: np.ndarray,
) -> None:
    if len(mu) != expected_count or len(sigma) != expected_count or len(nll) != expected_count:
        raise AssertionError(
            f"{split_name} prediction lengths do not match rows: "
            f"mu={len(mu)}, sigma={len(sigma)}, nll={len(nll)}, rows={expected_count}"
        )
    if not np.isfinite(mu).all():
        raise ValueError(f"{split_name} mu contains non-finite values")
    if not np.isfinite(sigma).all():
        raise ValueError(f"{split_name} sigma contains non-finite values")
    if (sigma <= 0).any():
        raise ValueError(f"{split_name} sigma contains non-positive values")
    near_zero_share = float(np.mean(sigma < 1e-4))
    if near_zero_share > 0.5:
        raise ValueError(
            f"{split_name} sigma is nearly zero for {near_zero_share:.1%} of rows"
        )
    if not np.isfinite(nll).all():
        raise ValueError(f"{split_name} NLL contains non-finite values")


def build_prediction_frame(
    split_name: str,
    split_df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    nll: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "split": split_name,
            "forecast_error": split_df[TARGET_COLUMN].to_numpy(dtype=float),
            "mu": mu,
            "sigma": sigma,
            "nll": nll,
        }
    )

    metadata = pd.DataFrame(index=split_df.index)
    for column in METADATA_COLUMNS:
        if column == "timestamp":
            source = _first_existing_column(split_df, ["timestamp", "prediction_timestamp"])
            if source is not None:
                metadata[column] = split_df[source].values
            continue
        if column in split_df.columns:
            metadata[column] = split_df[column].values

    if "date" not in metadata.columns and "target_date" in split_df.columns:
        metadata["date"] = split_df["target_date"].values
    if "timestamp" not in metadata.columns and "prediction_time" in split_df.columns:
        metadata["timestamp"] = split_df["prediction_time"].values

    metadata = metadata.reset_index(drop=True)
    result = pd.concat([metadata, frame], axis=1)
    ordered = [
        column
        for column in [
            "split",
            "date",
            "timestamp",
            "prediction_time",
            "prediction_timestamp",
            "location",
            "station",
            "station_id",
            "forecast_high",
            "actual_high",
            "official_high",
            "forecast_horizon_hours",
            "forecast_error",
            "mu",
            "sigma",
            "nll",
        ]
        if column in result.columns
    ]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining]


def build_metrics(
    split_summary: dict[str, Any],
    feature_columns: list[str],
    validation_nll: np.ndarray,
    test_nll: np.ndarray,
    validation_baseline_nll: np.ndarray,
    test_baseline_nll: np.ndarray,
    baseline_mu: float,
    baseline_sigma: float,
    sigma_summary: dict[str, dict[str, float]],
    preprocessing_notes: dict[str, Any],
    empirical_baseline_note: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": "ngboost_normal_v0",
        "target_name": TARGET_COLUMN,
        "distribution_type": "Normal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_summary": split_summary,
        "train_date_range": split_summary["splits"]["train"],
        "validation_date_range": split_summary["splits"]["validation"],
        "test_date_range": split_summary["splits"]["test"],
        "train_row_count": split_summary["splits"]["train"]["row_count"],
        "validation_row_count": split_summary["splits"]["validation"]["row_count"],
        "test_row_count": split_summary["splits"]["test"]["row_count"],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "validation_ngboost_nll": float(np.mean(validation_nll)),
        "test_ngboost_nll": float(np.mean(test_nll)),
        "validation_constant_normal_baseline_nll": float(np.mean(validation_baseline_nll)),
        "test_constant_normal_baseline_nll": float(np.mean(test_baseline_nll)),
        "constant_normal_baseline": {
            "mu_train": baseline_mu,
            "sigma_train": baseline_sigma,
        },
        "sigma_summary": sigma_summary,
        "notes": {
            "preprocessing": preprocessing_notes,
            "empirical_baseline_comparability": empirical_baseline_note,
            "scope": "Probability-signal layer only; no trading logic or bucket conversion was run.",
        },
    }


def summarize_sigma(sigma: np.ndarray) -> dict[str, float]:
    sigma_array = np.asarray(sigma, dtype=float)
    return {
        "min": float(np.min(sigma_array)),
        "median": float(np.median(sigma_array)),
        "mean": float(np.mean(sigma_array)),
        "max": float(np.max(sigma_array)),
        "near_zero_share_lt_1e-4": float(np.mean(sigma_array < 1e-4)),
    }


def build_empirical_baseline_note(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "not_available",
            "note": "No empirical baseline prediction file was found.",
        }
    columns = list(pd.read_csv(path, nrows=1).columns)
    continuous_nll_columns = [
        column
        for column in columns
        if column in {"continuous_nll", "density_nll", "normal_density_nll"}
    ]
    if continuous_nll_columns:
        return {
            "status": "available_not_loaded",
            "note": (
                "An empirical baseline file has a possible continuous NLL column, "
                "but Day 11 metrics keep the comparison to the train-only constant Normal baseline."
            ),
            "candidate_columns": continuous_nll_columns,
        }
    return {
        "status": "not_directly_comparable",
        "note": (
            "Existing empirical baseline outputs bucket/interval log scores. "
            "Those are not directly comparable to continuous Normal density NLL."
        ),
        "available_columns": columns,
    }


def write_outputs(
    model: Any,
    imputer: SimpleImputer,
    feature_columns: list[str],
    split_summary: dict[str, Any],
    preprocessing_notes: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    feature_columns_path: Path,
) -> None:
    for path in [MODEL_OUTPUT_PATH, FEATURE_OUTPUT_PATH, PARAMS_OUTPUT_PATH, METRICS_OUTPUT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": model,
        "imputer": imputer,
        "feature_columns": feature_columns,
        "target": TARGET_COLUMN,
        "distribution_type": "Normal",
        "split_summary": split_summary,
        "preprocessing_notes": preprocessing_notes,
    }
    with MODEL_OUTPUT_PATH.open("wb") as file:
        pickle.dump(model_artifact, file)

    feature_payload = {
        "model_name": "ngboost_normal_v0",
        "target": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "source_feature_spec": str(feature_columns_path),
        "imputation_strategy": preprocessing_notes["imputation_strategy"],
    }
    FEATURE_OUTPUT_PATH.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")
    predictions.to_csv(PARAMS_OUTPUT_PATH, index=False)
    METRICS_OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for path in [MODEL_OUTPUT_PATH, FEATURE_OUTPUT_PATH, PARAMS_OUTPUT_PATH, METRICS_OUTPUT_PATH]:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Expected output was not written: {path}")


def print_report(
    dataset_path: Path,
    total_rows: int,
    split_summary: dict[str, Any],
    feature_columns: list[str],
    validation_nll: np.ndarray,
    test_nll: np.ndarray,
    validation_baseline_nll: np.ndarray,
    test_baseline_nll: np.ndarray,
    sigma_summary: dict[str, dict[str, float]],
    preprocessing_notes: dict[str, Any],
) -> None:
    print("Day 11 NGBoost Normal training complete.")
    print(f"Loaded dataset: {dataset_path}")
    print(f"Total rows: {total_rows:,}")
    for split in ["train", "validation", "test"]:
        item = split_summary["splits"][split]
        print(
            f"{split.title()}: {item['row_count']:,} rows, "
            f"{item['date_min']} to {item['date_max']}"
        )
    print(f"Selected feature count: {len(feature_columns):,}")
    print(f"First 10 selected features: {feature_columns[:10]}")
    print(
        "Imputed/replaced values: "
        f"{preprocessing_notes['missing_and_infinite_values_by_split']}"
    )
    print(f"Validation NGBoost NLL: {float(np.mean(validation_nll)):.6f}")
    print(f"Validation constant Normal NLL: {float(np.mean(validation_baseline_nll)):.6f}")
    print(f"Test NGBoost NLL: {float(np.mean(test_nll)):.6f}")
    print(f"Test constant Normal NLL: {float(np.mean(test_baseline_nll)):.6f}")
    for split in ["validation", "test"]:
        summary = sigma_summary[split]
        print(
            f"{split.title()} sigma min/median/mean/max: "
            f"{summary['min']:.6f}, {summary['median']:.6f}, "
            f"{summary['mean']:.6f}, {summary['max']:.6f}"
        )
    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved feature list: {FEATURE_OUTPUT_PATH}")
    print(f"Saved distribution params: {PARAMS_OUTPUT_PATH}")
    print(f"Saved metrics: {METRICS_OUTPUT_PATH}")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


if __name__ == "__main__":
    main()
