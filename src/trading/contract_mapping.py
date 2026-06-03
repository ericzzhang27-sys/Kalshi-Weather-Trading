from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from src.bucket_schema import TemperatureBucket, validate_temperature_buckets


ACTIVE_MARKET_STATUSES = {"active", "open"}

CONTRACT_MAPPING_COLUMNS = [
    "event_ticker",
    "ticker",
    "status",
    "eligible",
    "title",
    "subtitle",
    "rules_primary",
    "strike_type",
    "floor_strike",
    "cap_strike",
    "bucket_name",
    "bucket_lower_temp",
    "bucket_upper_temp",
    "mapping_status",
    "no_trade_reason",
    "parse_source",
    "event_mapping_status",
    "event_no_trade_reason",
]


@dataclass(frozen=True)
class MappingValidation:
    valid: bool
    status: str
    no_trade_reason: str
    bucket_count: int


@dataclass(frozen=True)
class ContractMappingResult:
    event_ticker: str
    mapping: pd.DataFrame
    buckets: list[TemperatureBucket]
    validation: MappingValidation


@dataclass(frozen=True)
class ParsedContractBucket:
    bucket: TemperatureBucket
    strike_type: str
    parse_source: str


class ContractMappingError(ValueError):
    """Raised when a market cannot be safely mapped to a temperature bucket."""


def parse_contract_bucket(market: Mapping[str, Any]) -> TemperatureBucket:
    """
    Parse one Kalshi temperature contract into a lower-open, upper-closed bucket.
    """
    parsed = _parse_contract_bucket_with_source(market)
    return parsed.bucket


def validate_contract_mapping(buckets: list[TemperatureBucket]) -> MappingValidation:
    if len(buckets) == 0:
        return MappingValidation(
            valid=False,
            status="NO_TRADE",
            no_trade_reason="no_parsed_buckets",
            bucket_count=0,
        )

    ordered = _sort_buckets(buckets)
    interval_keys = [
        (_bound_key(bucket.lower_temp), _bound_key(bucket.upper_temp))
        for bucket in ordered
    ]
    if len(interval_keys) != len(set(interval_keys)):
        return MappingValidation(
            valid=False,
            status="NO_TRADE",
            no_trade_reason="duplicate_bucket_interval",
            bucket_count=len(ordered),
        )

    try:
        validate_temperature_buckets(ordered)
    except (TypeError, ValueError) as exc:
        return MappingValidation(
            valid=False,
            status="NO_TRADE",
            no_trade_reason=f"invalid_bucket_schema:{exc}",
            bucket_count=len(ordered),
        )

    return MappingValidation(
        valid=True,
        status="MAPPED",
        no_trade_reason="",
        bucket_count=len(ordered),
    )


def map_event_contracts(markets: pd.DataFrame, event_ticker: str) -> ContractMappingResult:
    if not isinstance(markets, pd.DataFrame):
        raise TypeError("markets must be a pandas DataFrame")
    if not event_ticker:
        raise ValueError("event_ticker must be provided")

    if "event_ticker" not in markets.columns:
        event_rows = pd.DataFrame(columns=markets.columns)
    else:
        event_rows = markets[markets["event_ticker"].astype(str) == str(event_ticker)].copy()
    if event_rows.empty:
        empty = pd.DataFrame(columns=CONTRACT_MAPPING_COLUMNS)
        validation = MappingValidation(
            valid=False,
            status="NO_TRADE",
            no_trade_reason="event_ticker_not_found",
            bucket_count=0,
        )
        return ContractMappingResult(str(event_ticker), empty, [], validation)

    records: list[dict[str, Any]] = []
    parsed_buckets: list[TemperatureBucket] = []
    parsed_row_indexes: list[int] = []

    for row_index, row in event_rows.reset_index(drop=True).iterrows():
        market = _combined_market_payload(row.to_dict())
        record = _base_mapping_record(market)
        if not _is_eligible_market(market):
            record["mapping_status"] = "NO_TRADE"
            record["no_trade_reason"] = _market_rejection_reason(market)
            records.append(record)
            continue

        try:
            parsed = _parse_contract_bucket_with_source(market)
        except ContractMappingError as exc:
            record["mapping_status"] = "NO_TRADE"
            record["no_trade_reason"] = str(exc)
            records.append(record)
            continue

        bucket = parsed.bucket
        record.update(
            {
                "strike_type": parsed.strike_type,
                "bucket_name": bucket.label,
                "bucket_lower_temp": bucket.lower_temp,
                "bucket_upper_temp": bucket.upper_temp,
                "mapping_status": "PARSED",
                "no_trade_reason": "",
                "parse_source": parsed.parse_source,
            }
        )
        parsed_buckets.append(bucket)
        parsed_row_indexes.append(len(records))
        records.append(record)

    duplicate_tickers = _duplicate_values(
        [
            str(record["ticker"])
            for record in records
            if record.get("mapping_status") == "PARSED"
        ]
    )
    if duplicate_tickers:
        for record in records:
            if record.get("ticker") in duplicate_tickers:
                record["mapping_status"] = "NO_TRADE"
                record["no_trade_reason"] = "duplicate_ticker"

    validation = validate_contract_mapping(
        [
            bucket
            for row_number, bucket in enumerate(parsed_buckets)
            if records[parsed_row_indexes[row_number]].get("mapping_status") == "PARSED"
        ]
    )
    event_status = validation.status
    event_reason = validation.no_trade_reason

    for record in records:
        if validation.valid and record.get("mapping_status") == "PARSED":
            record["mapping_status"] = "MAPPED"
        elif record.get("mapping_status") == "PARSED":
            record["mapping_status"] = "NO_TRADE"
            record["no_trade_reason"] = event_reason
        record["event_mapping_status"] = event_status
        record["event_no_trade_reason"] = event_reason

    frame = pd.DataFrame.from_records(records)
    frame = _sort_mapping_frame(frame).reindex(columns=CONTRACT_MAPPING_COLUMNS)
    buckets = _sort_buckets(parsed_buckets) if validation.valid else []
    return ContractMappingResult(str(event_ticker), frame, buckets, validation)


def save_contract_mapping_result(result: ContractMappingResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.mapping.reindex(columns=CONTRACT_MAPPING_COLUMNS).to_csv(path, index=False)


def _parse_contract_bucket_with_source(market: Mapping[str, Any]) -> ParsedContractBucket:
    combined = _combined_market_payload(market)
    strike_type = _infer_strike_type(combined)
    floor_strike = _number_field(combined, "floor_strike")
    cap_strike = _number_field(combined, "cap_strike")
    source = "strike_fields"

    lower: float | None
    upper: float | None
    if strike_type == "less":
        if cap_strike is None:
            cap_strike = _fallback_less_temperature(combined)
            source = "text_fallback"
        if cap_strike is None:
            raise ContractMappingError("missing_cap_strike_for_less_contract")
        lower = None
        upper = float(cap_strike) - 0.5
    elif strike_type == "greater":
        if floor_strike is None:
            floor_strike = _fallback_greater_temperature(combined)
            source = "text_fallback"
        if floor_strike is None:
            raise ContractMappingError("missing_floor_strike_for_greater_contract")
        lower = float(floor_strike) + 0.5
        upper = None
    elif strike_type == "between":
        if floor_strike is None or cap_strike is None:
            text_bounds = _fallback_between_temperatures(combined)
            if text_bounds is not None:
                floor_strike, cap_strike = text_bounds
                source = "text_fallback"
        if floor_strike is None or cap_strike is None:
            raise ContractMappingError("missing_strikes_for_between_contract")
        lower = float(floor_strike) - 0.5
        upper = float(cap_strike) + 0.5
    else:
        raise ContractMappingError("ambiguous_or_unsupported_strike_type")

    label = _bucket_label(combined, strike_type, lower=lower, upper=upper)
    bucket = TemperatureBucket(label=label, lower_temp=lower, upper_temp=upper)
    return ParsedContractBucket(bucket=bucket, strike_type=strike_type, parse_source=source)


def _combined_market_payload(market: Mapping[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    raw_json = market.get("raw_market_json")
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            raw = json.loads(raw_json)
        except json.JSONDecodeError:
            raw = {}
        if isinstance(raw, dict):
            combined.update(raw)
    for key, value in dict(market).items():
        if _is_present(value):
            combined[key] = value
    return combined


def _base_mapping_record(market: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_ticker": _string_field(market, "event_ticker"),
        "ticker": _string_field(market, "ticker"),
        "status": _string_field(market, "status"),
        "eligible": _bool_field(market, "eligible"),
        "title": _string_field(market, "title"),
        "subtitle": _string_field(market, "subtitle"),
        "rules_primary": _string_field(market, "rules_primary"),
        "strike_type": _string_field(market, "strike_type"),
        "floor_strike": _number_field(market, "floor_strike"),
        "cap_strike": _number_field(market, "cap_strike"),
        "bucket_name": "",
        "bucket_lower_temp": None,
        "bucket_upper_temp": None,
        "mapping_status": "",
        "no_trade_reason": "",
        "parse_source": "",
        "event_mapping_status": "",
        "event_no_trade_reason": "",
    }


def _is_eligible_market(market: Mapping[str, Any]) -> bool:
    status = _string_field(market, "status").lower()
    if status not in ACTIVE_MARKET_STATUSES:
        return False
    if "eligible" in market and not _bool_field(market, "eligible"):
        return False
    return True


def _market_rejection_reason(market: Mapping[str, Any]) -> str:
    status = _string_field(market, "status").lower()
    if status not in ACTIVE_MARKET_STATUSES:
        return f"status_is_{status or 'missing'}"
    if "eligible" in market and not _bool_field(market, "eligible"):
        reason = _string_field(market, "rejection_reason")
        return reason or "market_not_eligible"
    return "market_not_eligible"


def _infer_strike_type(market: Mapping[str, Any]) -> str:
    raw_type = _string_field(market, "strike_type").lower()
    if raw_type in {"less", "between", "greater"}:
        return raw_type

    text = _contract_text(market).lower()
    if re.search(r"(<|less than|below|or below)", text):
        return "less"
    if re.search(r"(>|greater than|above|or above)", text):
        return "greater"
    if re.search(r"(between|\bto\b|\d+\s*-\s*\d+)", text):
        return "between"
    return ""


def _fallback_less_temperature(market: Mapping[str, Any]) -> float | None:
    text = _contract_text(market)
    for pattern, offset in [
        (r"<\s*(-?\d+(?:\.\d+)?)", 0.0),
        (r"less than\s*(-?\d+(?:\.\d+)?)", 0.0),
        (r"(-?\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)?\s*or below", 1.0),
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) + offset
    return None


def _fallback_greater_temperature(market: Mapping[str, Any]) -> float | None:
    text = _contract_text(market)
    for pattern, offset in [
        (r">\s*(-?\d+(?:\.\d+)?)", 0.0),
        (r"greater than\s*(-?\d+(?:\.\d+)?)", 0.0),
        (r"(-?\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)?\s*or above", -1.0),
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) + offset
    return None


def _fallback_between_temperatures(market: Mapping[str, Any]) -> tuple[float, float] | None:
    text = _contract_text(market)
    patterns = [
        r"between\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)",
        r"(-?\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)?\s*to\s*(-?\d+(?:\.\d+)?)",
        r"\b(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            lower = float(match.group(1))
            upper = float(match.group(2))
            if upper > lower:
                return lower, upper
    return None


def _bucket_label(
    market: Mapping[str, Any],
    strike_type: str,
    *,
    lower: float | None,
    upper: float | None,
) -> str:
    for field in ["subtitle", "yes_sub_title", "title"]:
        value = _clean_label(_string_field(market, field))
        if value:
            return value
    if strike_type == "less" and upper is not None:
        return f"{int(round(float(upper) - 0.5))} or lower"
    if strike_type == "greater" and lower is not None:
        return f"{int(round(float(lower) + 0.5))} or higher"
    if lower is not None and upper is not None:
        return f"{int(round(float(lower) + 0.5))} to {int(round(float(upper) - 0.5))}"
    return "unknown"


def _clean_label(value: str) -> str:
    cleaned = re.sub(r"\*\*", "", value)
    cleaned = cleaned.replace("Â°", "deg").replace("°", "deg")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _contract_text(market: Mapping[str, Any]) -> str:
    return " ".join(
        _string_field(market, field)
        for field in [
            "strike_type",
            "ticker",
            "title",
            "subtitle",
            "yes_sub_title",
            "no_sub_title",
            "rules_primary",
            "functional_strike",
        ]
    )


def _sort_buckets(buckets: list[TemperatureBucket]) -> list[TemperatureBucket]:
    return sorted(
        buckets,
        key=lambda bucket: (
            -math.inf if bucket.lower_temp is None else float(bucket.lower_temp),
            math.inf if bucket.upper_temp is None else float(bucket.upper_temp),
        ),
    )


def _sort_mapping_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "bucket_lower_temp" not in frame.columns:
        return frame
    sortable = frame.copy()
    sortable["_sort_lower"] = sortable["bucket_lower_temp"].map(
        lambda value: -math.inf if not _is_present(value) else float(value)
    )
    sortable["_sort_upper"] = sortable["bucket_upper_temp"].map(
        lambda value: math.inf if not _is_present(value) else float(value)
    )
    sortable["_sort_status"] = sortable["mapping_status"].map(
        lambda value: 0 if str(value) == "MAPPED" else 1
    )
    sortable = sortable.sort_values(
        ["_sort_status", "_sort_lower", "_sort_upper", "ticker"],
        kind="stable",
    )
    return sortable.drop(columns=["_sort_status", "_sort_lower", "_sort_upper"]).reset_index(drop=True)


def _bound_key(value: float | None) -> str:
    if value is None:
        return "open"
    return f"{float(value):.8f}"


def _duplicate_values(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _number_field(market: Mapping[str, Any], field: str) -> float | None:
    value = market.get(field)
    if not _is_present(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _string_field(market: Mapping[str, Any], field: str) -> str:
    value = market.get(field, "")
    if not _is_present(value):
        return ""
    return str(value)


def _bool_field(market: Mapping[str, Any], field: str) -> bool:
    value = market.get(field)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return bool(value)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""
