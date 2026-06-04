from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.bucket_schema import TemperatureBucket
from src.predict_distribution import (
    DEFAULT_CALIBRATION_CONFIG_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_MODEL_PATH,
    EngineDiagnostics,
    load_probability_engine,
)
from src.trading.contract_mapping import ContractMappingResult


@dataclass(frozen=True)
class ProbabilitySignalResult:
    distribution_params: pd.DataFrame
    bucket_probabilities: pd.DataFrame
    diagnostics: EngineDiagnostics


def score_live_probabilities(
    feature_rows: pd.DataFrame,
    mapping: ContractMappingResult | pd.DataFrame,
    model_path: str | Path | None = None,
    feature_list_path: str | Path = DEFAULT_FEATURE_LIST_PATH,
    calibration_config_path: str | Path | None = DEFAULT_CALIBRATION_CONFIG_PATH,
) -> ProbabilitySignalResult:
    """
    Score live feature rows against mapped Kalshi temperature buckets.
    """
    if not isinstance(feature_rows, pd.DataFrame) or feature_rows.empty:
        raise ValueError("feature_rows must be a non-empty DataFrame")

    mapping_frame = _mapping_frame(mapping)
    buckets = _buckets_from_mapping(mapping)
    if not buckets:
        raise ValueError("Contract mapping contains no mapped buckets")

    engine = load_probability_engine(
        model_path=DEFAULT_MODEL_PATH if model_path is None else model_path,
        feature_list_path=feature_list_path,
        calibration_config_path=calibration_config_path,
    )
    prediction = engine.predict(feature_rows, buckets=buckets)
    probabilities = prediction.bucket_probabilities.copy()
    probabilities = _attach_tickers(probabilities, mapping_frame)
    probabilities["probability_signal_status"] = "OK"
    probabilities["probability_signal_reason"] = ""

    return ProbabilitySignalResult(
        distribution_params=prediction.distribution_params,
        bucket_probabilities=probabilities,
        diagnostics=prediction.diagnostics,
    )


def save_probability_signal_outputs(
    result: ProbabilitySignalResult,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.bucket_probabilities.to_csv(path, index=False)


def _mapping_frame(mapping: ContractMappingResult | pd.DataFrame) -> pd.DataFrame:
    if isinstance(mapping, ContractMappingResult):
        return mapping.mapping.copy()
    if isinstance(mapping, pd.DataFrame):
        return mapping.copy()
    raise TypeError("mapping must be ContractMappingResult or DataFrame")


def _buckets_from_mapping(mapping: ContractMappingResult | pd.DataFrame) -> list[TemperatureBucket]:
    if isinstance(mapping, ContractMappingResult) and mapping.buckets:
        return list(mapping.buckets)

    frame = _mapping_frame(mapping)
    if "mapping_status" in frame.columns:
        frame = frame[frame["mapping_status"] == "MAPPED"].copy()
    if frame.empty:
        return []

    frame = _sort_mapping_for_buckets(frame)
    buckets: list[TemperatureBucket] = []
    for _, row in frame.iterrows():
        buckets.append(
            TemperatureBucket(
                label=str(row["bucket_name"]),
                lower_temp=_optional_float(row.get("bucket_lower_temp")),
                upper_temp=_optional_float(row.get("bucket_upper_temp")),
            )
        )
    return buckets


def _attach_tickers(probabilities: pd.DataFrame, mapping_frame: pd.DataFrame) -> pd.DataFrame:
    join_columns = [
        column
        for column in [
            "ticker",
            "event_ticker",
            "bucket_name",
            "bucket_lower_temp",
            "bucket_upper_temp",
            "mapping_status",
        ]
        if column in mapping_frame.columns
    ]
    mapped = mapping_frame[join_columns].copy()
    if "mapping_status" in mapped.columns:
        mapped = mapped[mapped["mapping_status"] == "MAPPED"].copy()
    mapped = mapped.drop_duplicates(subset=["bucket_name"], keep="first")
    result = probabilities.merge(
        mapped,
        on="bucket_name",
        how="left",
        suffixes=("", "_mapped"),
        validate="many_to_one",
    )
    for column in ["bucket_lower_temp", "bucket_upper_temp"]:
        mapped_column = f"{column}_mapped"
        if mapped_column in result.columns:
            result[column] = result[column].combine_first(result[mapped_column])
            result = result.drop(columns=[mapped_column])
    ordered = [
        column
        for column in [
            "row_id",
            "event_ticker",
            "ticker",
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
            "distribution_type",
            "model_name",
            "calibration_method",
            "probability_signal_status",
            "probability_signal_reason",
        ]
        if column in result.columns
    ]
    remaining = [column for column in result.columns if column not in ordered]
    return result[ordered + remaining]


def _sort_mapping_for_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["_sort_lower"] = working["bucket_lower_temp"].map(
        lambda value: float("-inf") if pd.isna(value) else float(value)
    )
    working["_sort_upper"] = working["bucket_upper_temp"].map(
        lambda value: float("inf") if pd.isna(value) else float(value)
    )
    return (
        working.sort_values(["_sort_lower", "_sort_upper"], kind="stable")
        .drop(columns=["_sort_lower", "_sort_upper"])
        .reset_index(drop=True)
    )


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
