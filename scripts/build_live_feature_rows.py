from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import re
import sys

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import load_feature_list  # noqa: E402
from src.predict_distribution import DEFAULT_FEATURE_LIST_PATH, load_probability_engine  # noqa: E402
from src.trading.config import DEFAULT_TRADING_CONFIG_PATH, load_trading_config  # noqa: E402
from src.trading.contract_mapping import (  # noqa: E402
    ContractMappingResult,
    map_event_contracts,
    save_contract_mapping_result,
)
from src.trading.live_features import (  # noqa: E402
    build_live_feature_rows,
    save_live_feature_outputs,
)
from src.trading.live_weather import (  # noqa: E402
    fetch_live_weather,
    save_live_weather_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELING_FIXTURE_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Day 2 read-only Kalshi contract mapping and live feature rows."
    )
    parser.add_argument("--config-path", default=str(DEFAULT_TRADING_CONFIG_PATH))
    parser.add_argument("--market-discovery-path", default=None)
    parser.add_argument("--event-ticker", default=None)
    parser.add_argument("--target-date", default=None, help="YYYY-MM-DD target date.")
    parser.add_argument(
        "--prediction-time",
        default=None,
        help="Local prediction timestamp. Defaults to current local time.",
    )
    parser.add_argument("--feature-list-path", default=str(DEFAULT_FEATURE_LIST_PATH))
    parser.add_argument("--mapping-output-path", default=None)
    parser.add_argument("--weather-output-path", default=None)
    parser.add_argument("--feature-output-path", default=None)
    parser.add_argument("--freshness-output-path", default=None)
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Only parse and save contract mapping; do not fetch weather.",
    )
    parser.add_argument(
        "--dry-run-fixture",
        action="store_true",
        help="Do not call Open-Meteo; emit a scoreable feature row from processed modeling data.",
    )
    parser.add_argument(
        "--fixture-row-index",
        type=int,
        default=0,
        help="Processed modeling row index to use with --dry-run-fixture.",
    )
    parser.add_argument(
        "--skip-scoreability-check",
        action="store_true",
        help="Skip loading the probability engine to validate the feature row.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_trading_config(args.config_path)
    discovery_path = Path(args.market_discovery_path) if args.market_discovery_path else config.outputs.market_discovery_snapshot_path
    markets = pd.read_csv(discovery_path)
    event_ticker = args.event_ticker or _select_event_ticker(markets, args.target_date)
    mapping = map_event_contracts(markets, event_ticker)
    mapping_output_path = Path(args.mapping_output_path) if args.mapping_output_path else config.outputs.contract_bucket_mapping_path
    save_contract_mapping_result(mapping, mapping_output_path)

    if not mapping.validation.valid:
        print(
            "Saved contract mapping with NO_TRADE status: "
            f"{mapping.validation.no_trade_reason}"
        )
        print(f"Mapping CSV: {mapping_output_path}")
        return

    if args.mapping_only:
        print(
            "Saved contract mapping: "
            f"{mapping.validation.bucket_count} buckets for {event_ticker}."
        )
        print(f"Mapping CSV: {mapping_output_path}")
        return

    feature_list_path = Path(args.feature_list_path)
    if args.dry_run_fixture:
        feature_rows = _build_fixture_feature_rows(
            mapping=mapping,
            feature_list_path=feature_list_path,
            fixture_row_index=args.fixture_row_index,
        )
        _write_dry_run_weather_snapshot(
            Path(args.weather_output_path) if args.weather_output_path else config.outputs.live_weather_snapshot_path,
            mapping=mapping,
        )
    else:
        target_date = _target_date_for_args(args, event_ticker)
        prediction_time = _prediction_time_for_args(args)
        weather = fetch_live_weather(
            location=config.markets.default_location,
            target_date=target_date,
            prediction_time=prediction_time,
            config=config,
        )
        weather_output_path = Path(args.weather_output_path) if args.weather_output_path else config.outputs.live_weather_snapshot_path
        save_live_weather_snapshot(weather, weather_output_path)
        feature_rows = build_live_feature_rows(
            weather=weather,
            mapping=mapping,
            feature_list_path=feature_list_path,
        )

    if not args.skip_scoreability_check:
        engine = load_probability_engine(feature_list_path=feature_list_path)
        engine.predict_distribution_params(feature_rows)

    feature_output_path = Path(args.feature_output_path) if args.feature_output_path else config.outputs.live_feature_rows_path
    freshness_output_path = Path(args.freshness_output_path) if args.freshness_output_path else config.outputs.live_feature_freshness_path
    save_live_feature_outputs(feature_rows, feature_output_path, freshness_output_path)

    print(
        "Saved Day 2 live feature rows: "
        f"{len(feature_rows)} row(s), {mapping.validation.bucket_count} mapped buckets."
    )
    print(f"Mapping CSV: {mapping_output_path}")
    print(f"Feature rows CSV: {feature_output_path}")
    print(f"Freshness CSV: {freshness_output_path}")


def _select_event_ticker(markets: pd.DataFrame, target_date: str | None) -> str:
    if "event_ticker" not in markets.columns or markets.empty:
        raise ValueError("Market discovery snapshot has no event_ticker values")
    candidates = markets.copy()
    if target_date:
        requested = date.fromisoformat(target_date)
        candidates = candidates[
            candidates["event_ticker"].astype(str).map(_event_date_from_ticker) == requested
        ]
        if candidates.empty:
            raise ValueError(f"No discovered event matched target date {target_date}")
    if "close_time" in candidates.columns:
        candidates["_close_time"] = pd.to_datetime(candidates["close_time"], errors="coerce")
        candidates = candidates.sort_values("_close_time", kind="stable")
    return str(candidates["event_ticker"].dropna().iloc[-1])


def _target_date_for_args(args: argparse.Namespace, event_ticker: str) -> date:
    if args.target_date:
        return date.fromisoformat(args.target_date)
    event_date = _event_date_from_ticker(event_ticker)
    if event_date is None:
        raise ValueError("Target date must be provided when event ticker date cannot be parsed")
    return event_date


def _prediction_time_for_args(args: argparse.Namespace) -> datetime:
    if args.prediction_time:
        return datetime.fromisoformat(args.prediction_time)
    return datetime.now()


def _event_date_from_ticker(event_ticker: str) -> date | None:
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", str(event_ticker))
    if not match:
        return None
    year = 2000 + int(match.group(1))
    month_lookup = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    month = month_lookup.get(match.group(2))
    if month is None:
        return None
    return date(year, month, int(match.group(3)))


def _build_fixture_feature_rows(
    *,
    mapping: ContractMappingResult,
    feature_list_path: Path,
    fixture_row_index: int,
) -> pd.DataFrame:
    feature_columns = load_feature_list(feature_list_path)
    modeling_rows = pd.read_csv(DEFAULT_MODELING_FIXTURE_PATH)
    if fixture_row_index < 0 or fixture_row_index >= len(modeling_rows):
        raise ValueError(f"fixture_row_index out of range: {fixture_row_index}")
    row = modeling_rows.iloc[[fixture_row_index]].copy()
    missing = [column for column in feature_columns if column not in row.columns]
    if missing:
        raise ValueError(f"Fixture modeling row is missing features: {missing}")
    row["row_id"] = f"dry_run:{mapping.event_ticker}:{fixture_row_index}"
    row["event_ticker"] = mapping.event_ticker
    row["bucket_count"] = mapping.validation.bucket_count
    row["mapping_status"] = mapping.validation.status
    row["weather_status"] = "DRY_RUN_FIXTURE"
    row["live_feature_status"] = "SCOREABLE_SHADOW"
    row["no_trade_reason"] = "dry_run_fixture_not_live_weather"
    metadata = [
        "row_id",
        "date",
        "target_date",
        "prediction_time",
        "prediction_timestamp",
        "location",
        "event_ticker",
        "forecast_high",
        "bucket_count",
        "mapping_status",
        "weather_status",
        "live_feature_status",
        "no_trade_reason",
    ]
    output = row[[column for column in metadata if column in row.columns] + feature_columns].copy()
    output.attrs["freshness"] = pd.DataFrame(
        {
            "row_id": output["row_id"].iloc[0],
            "feature": feature_columns,
            "source_role": "processed_modeling_fixture",
            "source_time": output["prediction_timestamp"].iloc[0],
            "prediction_time": output["prediction_timestamp"].iloc[0],
            "age_minutes": 0.0,
            "is_missing": [bool(pd.isna(output[column].iloc[0])) for column in feature_columns],
            "is_infinite": [False for _ in feature_columns],
        }
    )
    return output


def _write_dry_run_weather_snapshot(path: Path, *, mapping: ContractMappingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "source_role": "dry_run_fixture",
                "event_ticker": mapping.event_ticker,
                "detail": "No Open-Meteo request was made; feature row came from processed modeling fixture.",
            }
        ]
    ).to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
