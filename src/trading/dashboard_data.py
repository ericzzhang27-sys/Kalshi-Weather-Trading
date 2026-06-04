from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.predict_distribution import DEFAULT_FEATURE_LIST_PATH, DEFAULT_MODEL_PATH
from src.trading.config import TradingConfig
from src.trading.contract_mapping import (
    ContractMappingResult,
    map_event_contracts,
    save_contract_mapping_result,
)
from src.trading.kalshi_client import KalshiClient
from src.trading.live_features import build_live_feature_rows, save_live_feature_outputs
from src.trading.live_weather import (
    LiveWeatherSnapshot,
    diagnostics_for_combined_frame,
    fetch_live_weather,
    save_live_weather_snapshot,
)
from src.trading.market_discovery import (
    discover_weather_markets,
    settings_from_config,
)
from src.trading.orderbook import (
    OrderbookSnapshot,
    fetch_orderbooks,
    save_orderbook_snapshot,
    summarize_orderbook,
)
from src.trading.probability_signal import (
    ProbabilitySignalResult,
    save_probability_signal_outputs,
    score_live_probabilities,
)


@dataclass(frozen=True)
class DashboardState:
    status: dict[str, Any]
    market_discovery: pd.DataFrame
    mapping: pd.DataFrame
    live_weather: pd.DataFrame
    live_feature_rows: pd.DataFrame
    feature_freshness: pd.DataFrame
    bucket_probabilities: pd.DataFrame
    distribution_params: pd.DataFrame
    orderbook: pd.DataFrame
    orderbook_summary: pd.DataFrame
    bucket_board: pd.DataFrame


def load_dashboard_state(
    config: TradingConfig,
    event_ticker: str | None = None,
    target_date: date | str | None = None,
    depth: int = 20,
    *,
    kalshi_client: KalshiClient | None = None,
    weather_client: Any | None = None,
    prediction_time: datetime | None = None,
    auth_market_data: bool = False,
    auth_orderbooks: bool = False,
    write_outputs: bool = True,
) -> DashboardState:
    """
    Run one read-only dashboard refresh cycle.
    """
    refresh_time = datetime.now().astimezone()
    client = kalshi_client or KalshiClient(
        base_url=config.kalshi.base_url,
        timeout_seconds=config.kalshi.request_timeout_seconds,
        max_retries=config.kalshi.max_retries,
        retry_backoff_seconds=config.kalshi.retry_backoff_seconds,
    )
    settings = settings_from_config(config)
    discovered = discover_weather_markets(
        client,
        settings,
        fetched_at=refresh_time,
        auth=auth_market_data,
    )
    market_discovery = pd.DataFrame([asdict(market) for market in discovered])
    selected_event = event_ticker or _select_event_ticker(market_discovery, target_date)
    selected_target_date = _coerce_target_date(target_date) or _event_date_from_ticker(selected_event)
    if selected_target_date is None:
        raise ValueError("target_date is required when the event ticker date cannot be parsed")

    mapping_result = map_event_contracts(market_discovery, selected_event)
    if not mapping_result.validation.valid:
        status = _status_dict(
            config=config,
            data_source="live_refresh",
            refresh_time=refresh_time,
            event_ticker=selected_event,
            target_date=selected_target_date,
            warnings=[mapping_result.validation.no_trade_reason],
        )
        state = DashboardState(
            status=status,
            market_discovery=market_discovery,
            mapping=mapping_result.mapping,
            live_weather=pd.DataFrame(),
            live_feature_rows=pd.DataFrame(),
            feature_freshness=pd.DataFrame(),
            bucket_probabilities=pd.DataFrame(),
            distribution_params=pd.DataFrame(),
            orderbook=pd.DataFrame(),
            orderbook_summary=pd.DataFrame(),
            bucket_board=build_bucket_board(mapping_result.mapping, pd.DataFrame(), pd.DataFrame()),
        )
        if write_outputs:
            _write_outputs(config, state, mapping_result=mapping_result)
        return state

    tickers = mapping_result.mapping.loc[
        mapping_result.mapping["mapping_status"] == "MAPPED",
        "ticker",
    ].dropna().astype(str).tolist()
    orderbook_snapshot = fetch_orderbooks(
        client,
        tickers=tickers,
        depth=depth,
        auth=auth_orderbooks,
        fetched_at=refresh_time,
    )

    weather: LiveWeatherSnapshot | None = None
    feature_rows = pd.DataFrame()
    feature_freshness = pd.DataFrame()
    probability_result: ProbabilitySignalResult | None = None
    probability_warning = ""
    try:
        weather = fetch_live_weather(
            location=config.markets.default_location,
            target_date=selected_target_date,
            prediction_time=prediction_time or datetime.now(),
            config=config,
            client=weather_client,
            fetched_at=refresh_time,
        )
        feature_rows = build_live_feature_rows(
            weather=weather,
            mapping=mapping_result,
            feature_list_path=DEFAULT_FEATURE_LIST_PATH,
        )
        freshness = feature_rows.attrs.get("freshness")
        feature_freshness = freshness if isinstance(freshness, pd.DataFrame) else pd.DataFrame()
        probability_result = score_live_probabilities(feature_rows, mapping_result)
    except Exception as exc:
        probability_warning = f"live_feature_scoring_unavailable:{exc}"

    bucket_probabilities = (
        probability_result.bucket_probabilities
        if probability_result is not None
        else pd.DataFrame()
    )
    distribution_params = (
        probability_result.distribution_params
        if probability_result is not None
        else pd.DataFrame()
    )
    warnings = _state_warnings(
        mapping_result,
        weather,
        feature_rows,
        orderbook_snapshot,
        probability_warning=probability_warning,
    )
    status = _status_dict(
        config=config,
        data_source="live_refresh" if probability_result is not None else "live_partial",
        refresh_time=refresh_time,
        event_ticker=selected_event,
        target_date=selected_target_date,
        warnings=warnings,
        model_name=probability_result.diagnostics.model_name if probability_result is not None else "",
        model_path=probability_result.diagnostics.model_path if probability_result is not None else "",
        probability_rows=len(bucket_probabilities),
    )
    bucket_board = build_bucket_board(
        mapping_result.mapping,
        bucket_probabilities,
        orderbook_snapshot.summary,
    )
    state = DashboardState(
        status=status,
        market_discovery=market_discovery,
        mapping=mapping_result.mapping,
        live_weather=_weather_frame_for_state(weather) if weather is not None else pd.DataFrame(),
        live_feature_rows=feature_rows,
        feature_freshness=feature_freshness,
        bucket_probabilities=bucket_probabilities,
        distribution_params=distribution_params,
        orderbook=orderbook_snapshot.orderbook,
        orderbook_summary=orderbook_snapshot.summary,
        bucket_board=bucket_board,
    )
    if write_outputs:
        _write_outputs(
            config,
            state,
            mapping_result=mapping_result,
            weather=weather,
            probability_result=probability_result,
            orderbook_snapshot=orderbook_snapshot,
        )
    return state


def load_dashboard_state_from_artifacts(config: TradingConfig) -> DashboardState:
    mapping = _read_csv(config.outputs.contract_bucket_mapping_path)
    features = _read_csv(config.outputs.live_feature_rows_path)
    freshness = _read_csv(config.outputs.live_feature_freshness_path)
    probabilities = _read_csv(config.outputs.live_bucket_probabilities_path)
    distribution_params = pd.DataFrame()
    model_name = ""
    model_path = ""
    if not features.empty and probabilities.empty and not mapping.empty:
        try:
            probability_result = score_live_probabilities(features, mapping)
            probabilities = probability_result.bucket_probabilities
            distribution_params = probability_result.distribution_params
            model_name = probability_result.diagnostics.model_name
            model_path = probability_result.diagnostics.model_path
        except Exception:
            probabilities = pd.DataFrame()
    if not probabilities.empty:
        model_name = model_name or _first_nonempty(probabilities, "model_name") or ""
        model_path = model_path or str(DEFAULT_MODEL_PATH)
    market_discovery = _read_csv(config.outputs.market_discovery_snapshot_path)
    weather = _read_csv(config.outputs.live_weather_snapshot_path)
    orderbook = _read_csv(config.outputs.orderbook_snapshot_path)
    orderbook_summary = _summaries_from_orderbook(orderbook)
    bucket_board = build_bucket_board(mapping, probabilities, orderbook_summary)
    event = _first_nonempty(mapping, "event_ticker") or _first_nonempty(features, "event_ticker")
    status = _status_dict(
        config=config,
        data_source="saved_artifacts",
        refresh_time=datetime.now().astimezone(),
        event_ticker=event,
        target_date=_event_date_from_ticker(event or ""),
        warnings=_artifact_warnings(features, probabilities, orderbook),
        model_name=model_name,
        model_path=model_path,
        probability_rows=len(probabilities),
    )
    return DashboardState(
        status=status,
        market_discovery=market_discovery,
        mapping=mapping,
        live_weather=weather,
        live_feature_rows=features,
        feature_freshness=freshness,
        bucket_probabilities=probabilities,
        distribution_params=distribution_params,
        orderbook=orderbook,
        orderbook_summary=orderbook_summary,
        bucket_board=bucket_board,
    )


def build_bucket_board(
    mapping: pd.DataFrame,
    probabilities: pd.DataFrame,
    orderbook_summary: pd.DataFrame,
) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame()
    board = mapping.copy()
    if "mapping_status" in board.columns:
        board = board[board["mapping_status"] == "MAPPED"].copy()
    probability_cols = [
        column
        for column in [
            "ticker",
            "probability",
            "mu",
            "sigma",
            "model_name",
            "distribution_type",
            "calibration_method",
        ]
        if column in probabilities.columns
    ]
    if probability_cols and "ticker" in probability_cols:
        board = board.merge(
            probabilities[probability_cols].drop_duplicates("ticker"),
            on="ticker",
            how="left",
            validate="one_to_one",
        )
    if not orderbook_summary.empty and "ticker" in orderbook_summary.columns:
        board = board.merge(
            orderbook_summary,
            on="ticker",
            how="left",
            validate="one_to_one",
        )
    return board.reset_index(drop=True)


def save_dashboard_status(status: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_outputs(
    config: TradingConfig,
    state: DashboardState,
    *,
    mapping_result: ContractMappingResult | None = None,
    weather: LiveWeatherSnapshot | None = None,
    probability_result: ProbabilitySignalResult | None = None,
    orderbook_snapshot: OrderbookSnapshot | None = None,
) -> None:
    if not state.market_discovery.empty:
        state.market_discovery.to_csv(config.outputs.market_discovery_snapshot_path, index=False)
    if mapping_result is not None:
        save_contract_mapping_result(mapping_result, config.outputs.contract_bucket_mapping_path)
    elif not state.mapping.empty:
        state.mapping.to_csv(config.outputs.contract_bucket_mapping_path, index=False)
    if weather is not None:
        save_live_weather_snapshot(weather, config.outputs.live_weather_snapshot_path)
    elif not state.live_weather.empty:
        state.live_weather.to_csv(config.outputs.live_weather_snapshot_path, index=False)
    if not state.live_feature_rows.empty:
        save_live_feature_outputs(
            state.live_feature_rows,
            config.outputs.live_feature_rows_path,
            config.outputs.live_feature_freshness_path,
        )
    if probability_result is not None:
        save_probability_signal_outputs(probability_result, config.outputs.live_bucket_probabilities_path)
    elif not state.bucket_probabilities.empty:
        state.bucket_probabilities.to_csv(config.outputs.live_bucket_probabilities_path, index=False)
    if orderbook_snapshot is not None:
        save_orderbook_snapshot(orderbook_snapshot, config.outputs.orderbook_snapshot_path)
    elif not state.orderbook.empty:
        state.orderbook.to_csv(config.outputs.orderbook_snapshot_path, index=False)
    save_dashboard_status(state.status, config.outputs.dashboard_status_path)


def _state_warnings(
    mapping: ContractMappingResult,
    weather: LiveWeatherSnapshot | None,
    feature_rows: pd.DataFrame,
    orderbook: OrderbookSnapshot,
    probability_warning: str = "",
) -> list[str]:
    warnings: list[str] = []
    if not mapping.validation.valid:
        warnings.append(mapping.validation.no_trade_reason)
    if probability_warning:
        warnings.append(probability_warning)
    if weather is not None and not weather.diagnostics.empty:
        for _, row in weather.diagnostics[weather.diagnostics["status"].isin(["WARN", "NO_TRADE"])].iterrows():
            reason = str(row.get("no_trade_reason", "") or row.get("diagnostic_name", ""))
            if reason:
                warnings.append(reason)
    if not feature_rows.empty and "no_trade_reason" in feature_rows.columns:
        warnings.extend(
            str(value)
            for value in feature_rows["no_trade_reason"].dropna().unique()
            if str(value)
        )
    if not orderbook.summary.empty:
        warnings.extend(
            str(value)
            for value in orderbook.summary["orderbook_reason"].dropna().unique()
            if str(value)
        )
    return sorted(set(warnings))


def _artifact_warnings(
    feature_rows: pd.DataFrame,
    probabilities: pd.DataFrame,
    orderbook: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []
    if feature_rows.empty:
        warnings.append("missing_live_feature_rows")
    if probabilities.empty:
        warnings.append("missing_live_bucket_probabilities")
    if orderbook.empty:
        warnings.append("missing_orderbook_snapshot")
    if not feature_rows.empty and "no_trade_reason" in feature_rows.columns:
        warnings.extend(
            str(value)
            for value in feature_rows["no_trade_reason"].dropna().unique()
            if str(value)
        )
    return sorted(set(warnings))


def _status_dict(
    *,
    config: TradingConfig,
    data_source: str,
    refresh_time: datetime,
    event_ticker: str | None,
    target_date: date | None,
    warnings: list[str],
    model_name: str | None = None,
    model_path: str | None = None,
    probability_rows: int = 0,
) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "kalshi_env": config.kalshi.env,
        "trading_enabled": config.trading_enabled,
        "live_auto_enabled": config.live_auto_enabled,
        "data_source": data_source,
        "refreshed_at": refresh_time.isoformat(),
        "event_ticker": event_ticker or "",
        "target_date": "" if target_date is None else target_date.isoformat(),
        "model_name": model_name or "",
        "model_path": model_path or "",
        "probability_rows": int(probability_rows),
        "warning_count": len(warnings),
        "warnings": warnings,
        "dashboard_status": "WARN" if warnings else "OK",
        "read_only": True,
    }


def _select_event_ticker(markets: pd.DataFrame, target_date: date | str | None) -> str:
    if markets.empty or "event_ticker" not in markets.columns:
        raise ValueError("No discovered markets are available")
    candidates = markets.copy()
    selected_date = _coerce_target_date(target_date)
    if selected_date is not None:
        candidates = candidates[
            candidates["event_ticker"].astype(str).map(_event_date_from_ticker) == selected_date
        ]
        if candidates.empty:
            raise ValueError(f"No discovered event matched target date {selected_date.isoformat()}")
    if "eligible" in candidates.columns:
        eligible = candidates[candidates["eligible"].astype(bool)]
        if not eligible.empty:
            candidates = eligible
    if "close_time" in candidates.columns:
        candidates["_close_time"] = pd.to_datetime(candidates["close_time"], errors="coerce")
        candidates = candidates.sort_values("_close_time", kind="stable")
    return str(candidates["event_ticker"].dropna().iloc[0])


def _coerce_target_date(value: date | str | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _event_date_from_ticker(event_ticker: str) -> date | None:
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", str(event_ticker))
    if not match:
        return None
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
    return date(2000 + int(match.group(1)), month, int(match.group(3)))


def _weather_frame_for_state(weather: LiveWeatherSnapshot) -> pd.DataFrame:
    frames = [
        weather.hourly_observations,
        weather.hourly_forecasts,
        weather.daily_forecast,
        diagnostics_for_combined_frame(weather.diagnostics),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def _summaries_from_orderbook(orderbook: pd.DataFrame) -> pd.DataFrame:
    if orderbook.empty or "ticker" not in orderbook.columns:
        return pd.DataFrame()
    summaries = []
    for ticker, group in orderbook.groupby("ticker", sort=False):
        fetched_at = pd.to_datetime(group["fetched_at"].iloc[0]).to_pydatetime()
        summaries.append(summarize_orderbook(group, ticker=str(ticker), fetched_at=fetched_at))
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def _read_csv(path: str | Path) -> pd.DataFrame:
    candidate = Path(path)
    if not candidate.exists():
        return pd.DataFrame()
    return pd.read_csv(candidate)


def _first_nonempty(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str)
    values = values[values != ""]
    if values.empty:
        return None
    return str(values.iloc[0])
