from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.trading.edge import EdgeSettings


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRADING_CONFIG_PATH = REPO_ROOT / "config" / "trading_config.yaml"
ALLOWED_MODES = {"shadow", "paper", "live_manual_approve", "live_auto"}
ALLOWED_KALSHI_ENVS = {"demo", "production"}


class TradingConfigError(ValueError):
    """Raised when live-trading configuration is invalid."""


@dataclass(frozen=True)
class KalshiSettings:
    env: str = "demo"
    demo_base_url: str = "https://external-api.demo.kalshi.co/trade-api/v2"
    production_base_url: str = "https://external-api.kalshi.com/trade-api/v2"
    request_timeout_seconds: float = 15.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    allow_public_market_data_without_auth: bool = True

    @property
    def base_url(self) -> str:
        if self.env == "demo":
            return self.demo_base_url
        if self.env == "production":
            return self.production_base_url
        raise TradingConfigError(f"Unsupported Kalshi environment: {self.env}")


@dataclass(frozen=True)
class MarketSettings:
    default_location: str = "NYC"
    supported_locations: tuple[str, ...] = ("NYC",)
    target_market_type: str = "daily_high_temperature"
    supported_market_types: tuple[str, ...] = ("daily_high_temperature",)
    status: str = "open"
    tradable_statuses: tuple[str, ...] = ("open", "active")
    min_minutes_to_close: int = 30
    page_limit: int = 1000
    max_pages: int = 5
    series_tickers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    location_terms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    weather_terms: tuple[str, ...] = (
        "temperature",
        "high temperature",
        "high temp",
        "daily high",
        "weather",
    )


@dataclass(frozen=True)
class OutputSettings:
    live_trading_dir: Path
    market_discovery_snapshot_path: Path
    market_discovery_raw_path: Path
    contract_bucket_mapping_path: Path
    live_weather_snapshot_path: Path
    live_feature_rows_path: Path
    live_feature_freshness_path: Path
    live_bucket_probabilities_path: Path
    settlement_state_path: Path
    orderbook_snapshot_path: Path
    orderbook_summary_path: Path
    edge_table_path: Path
    portfolio_snapshot_path: Path
    risk_decisions_path: Path
    order_intents_path: Path
    paper_orders_path: Path
    paper_positions_path: Path
    paper_pnl_path: Path
    trading_cycle_log_path: Path
    dashboard_status_path: Path


@dataclass(frozen=True)
class RiskSettings:
    kill_switch_path: Path
    max_contracts_per_order: int = 1
    max_contracts_per_market: int = 5
    max_dollars_per_order: float = 5.0
    max_dollars_per_market: float = 20.0
    max_dollars_per_event: float = 30.0
    max_correlated_event_exposure_dollars: float = 30.0
    max_total_exposure: float = 50.0
    max_open_orders: int = 10
    max_daily_loss_dollars: float = 25.0
    min_cash_reserve_dollars: float = 0.0
    denylist_tickers: tuple[str, ...] = ()
    denylist_event_tickers: tuple[str, ...] = ()
    denylist_target_dates: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperSettings:
    starting_cash_dollars: float = 100.0
    fill_mode: str = "immediate"


@dataclass(frozen=True)
class WeatherGridSettings:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class NwsStationSettings:
    station_id: str = "KNYC"
    station_name: str = "NY City Central Park"
    ghcn_station_id: str = "GHCND:USW00094728"
    latitude: float = 40.77898
    longitude: float = -73.96925
    base_url: str = "https://api.weather.gov"
    user_agent: str = "KalshiWeatherTrading/0.1 (local research; contact user)"


@dataclass(frozen=True)
class LiveWeatherSettings:
    provider: str = "open_meteo"
    observations_provider: str = "nws_station"
    forecast_base_url: str = "https://api.open-meteo.com/v1/forecast"
    timezone: str = "America/New_York"
    temperature_unit: str = "fahrenheit"
    wind_speed_unit: str = "mph"
    precipitation_unit: str = "inch"
    observation_grid: WeatherGridSettings = WeatherGridSettings(
        latitude=40.808434,
        longitude=-74.0199,
    )
    forecast_grid: WeatherGridSettings = WeatherGridSettings(
        latitude=40.78858,
        longitude=-73.9661,
    )
    nws_station: NwsStationSettings = NwsStationSettings()
    observed_past_hours: int = 12
    forecast_past_hours: int = 12
    forecast_days: int = 2
    max_observation_age_minutes: int = 90
    max_forecast_age_minutes: int = 180
    max_unverified_observed_high_minutes: int = 20
    require_forecast_issue_time: bool = False


@dataclass(frozen=True)
class SettlementSettings:
    typical_peak_hour: int = 15
    peak_window_end_hour: int = 18
    verified_settlement_min_hour: int = 18
    post_peak_temp_drop_f: float = 2.0
    min_minutes_since_high: int = 60
    forecast_remaining_margin_f: float = 0.5
    settlement_tail_probability: float = 0.01
    block_unverified_observed_high: bool = True


@dataclass(frozen=True)
class TradingConfig:
    mode: str
    trading_enabled: bool
    live_auto_enabled: bool
    kalshi: KalshiSettings
    markets: MarketSettings
    weather: LiveWeatherSettings
    settlement: SettlementSettings
    edge: EdgeSettings
    paper: PaperSettings
    outputs: OutputSettings
    risk: RiskSettings


def load_trading_config(path: str | Path = DEFAULT_TRADING_CONFIG_PATH) -> TradingConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Trading config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TradingConfigError("Trading config must be a YAML mapping")
    return parse_trading_config(raw)


def parse_trading_config(raw: dict[str, Any]) -> TradingConfig:
    mode = str(raw.get("mode", "shadow")).strip()
    trading_enabled = bool(raw.get("trading_enabled", False))
    live_auto_enabled = bool(raw.get("live_auto_enabled", False))

    kalshi = _parse_kalshi_settings(raw.get("kalshi", {}))
    markets = _parse_market_settings(raw.get("markets", {}))
    weather = _parse_live_weather_settings(raw.get("weather", {}))
    settlement = _parse_settlement_settings(raw.get("settlement", {}))
    edge = _parse_edge_settings(raw.get("edge", {}))
    paper = _parse_paper_settings(raw.get("paper", {}))
    outputs = _parse_output_settings(raw.get("outputs", {}))
    risk = _parse_risk_settings(raw.get("risk", {}))

    config = TradingConfig(
        mode=mode,
        trading_enabled=trading_enabled,
        live_auto_enabled=live_auto_enabled,
        kalshi=kalshi,
        markets=markets,
        weather=weather,
        settlement=settlement,
        edge=edge,
        paper=paper,
        outputs=outputs,
        risk=risk,
    )
    validate_trading_config(config)
    return config


def validate_trading_config(config: TradingConfig) -> None:
    if config.mode not in ALLOWED_MODES:
        raise TradingConfigError(
            f"Unsupported trading mode {config.mode!r}; expected one of {sorted(ALLOWED_MODES)}"
        )
    if config.kalshi.env not in ALLOWED_KALSHI_ENVS:
        raise TradingConfigError(
            f"Unsupported Kalshi env {config.kalshi.env!r}; expected demo or production"
        )
    if config.mode == "live_auto" and not config.live_auto_enabled:
        raise TradingConfigError("live_auto mode requires live_auto_enabled: true")
    if config.mode.startswith("live") and not config.trading_enabled:
        raise TradingConfigError("Live modes require trading_enabled: true")
    if config.markets.default_location not in config.markets.supported_locations:
        raise TradingConfigError(
            "markets.default_location must be listed in markets.supported_locations"
        )
    if config.markets.target_market_type not in config.markets.supported_market_types:
        raise TradingConfigError(
            "markets.target_market_type must be listed in markets.supported_market_types"
        )
    if config.markets.page_limit < 1 or config.markets.page_limit > 1000:
        raise TradingConfigError("markets.page_limit must be between 1 and 1000")
    if config.markets.max_pages < 1:
        raise TradingConfigError("markets.max_pages must be positive")
    if config.weather.provider != "open_meteo":
        raise TradingConfigError("weather.provider must be open_meteo")
    if config.weather.observations_provider not in {"open_meteo", "nws_station"}:
        raise TradingConfigError(
            "weather.observations_provider must be open_meteo or nws_station"
        )
    if config.weather.observed_past_hours < 1:
        raise TradingConfigError("weather.observed_past_hours must be positive")
    if config.weather.forecast_past_hours < 1:
        raise TradingConfigError("weather.forecast_past_hours must be positive")
    if config.weather.forecast_days < 1:
        raise TradingConfigError("weather.forecast_days must be positive")
    if config.weather.max_unverified_observed_high_minutes < 0:
        raise TradingConfigError(
            "weather.max_unverified_observed_high_minutes must be nonnegative"
        )
    if not 0 <= config.settlement.typical_peak_hour <= 23:
        raise TradingConfigError("settlement.typical_peak_hour must be between 0 and 23")
    if not 0 <= config.settlement.peak_window_end_hour <= 23:
        raise TradingConfigError("settlement.peak_window_end_hour must be between 0 and 23")
    if config.settlement.peak_window_end_hour < config.settlement.typical_peak_hour:
        raise TradingConfigError(
            "settlement.peak_window_end_hour must be at or after typical_peak_hour"
        )
    if not 0 <= config.settlement.verified_settlement_min_hour <= 23:
        raise TradingConfigError(
            "settlement.verified_settlement_min_hour must be between 0 and 23"
        )
    if config.settlement.post_peak_temp_drop_f < 0.0:
        raise TradingConfigError("settlement.post_peak_temp_drop_f must be nonnegative")
    if config.settlement.min_minutes_since_high < 0:
        raise TradingConfigError("settlement.min_minutes_since_high must be nonnegative")
    if config.settlement.forecast_remaining_margin_f < 0.0:
        raise TradingConfigError("settlement.forecast_remaining_margin_f must be nonnegative")
    if not 0.0 <= config.settlement.settlement_tail_probability <= 0.5:
        raise TradingConfigError(
            "settlement.settlement_tail_probability must be between 0 and 0.5"
        )
    if config.risk.max_contracts_per_order < 1:
        raise TradingConfigError("risk.max_contracts_per_order must be positive")
    if config.risk.max_contracts_per_market < 1:
        raise TradingConfigError("risk.max_contracts_per_market must be positive")
    for name, value in {
        "risk.max_dollars_per_order": config.risk.max_dollars_per_order,
        "risk.max_dollars_per_market": config.risk.max_dollars_per_market,
        "risk.max_dollars_per_event": config.risk.max_dollars_per_event,
        "risk.max_correlated_event_exposure_dollars": (
            config.risk.max_correlated_event_exposure_dollars
        ),
        "risk.max_total_exposure": config.risk.max_total_exposure,
        "risk.max_daily_loss_dollars": config.risk.max_daily_loss_dollars,
        "paper.starting_cash_dollars": config.paper.starting_cash_dollars,
    }.items():
        if value <= 0.0:
            raise TradingConfigError(f"{name} must be positive")
    if config.risk.max_open_orders < 0:
        raise TradingConfigError("risk.max_open_orders must be nonnegative")
    if config.risk.min_cash_reserve_dollars < 0.0:
        raise TradingConfigError("risk.min_cash_reserve_dollars must be nonnegative")
    if config.paper.fill_mode not in {"immediate", "none"}:
        raise TradingConfigError("paper.fill_mode must be immediate or none")


def _parse_kalshi_settings(raw: Any) -> KalshiSettings:
    data = _mapping(raw, "kalshi")
    env = _normalize_env(str(data.get("env", "demo")))
    return KalshiSettings(
        env=env,
        demo_base_url=str(
            data.get("demo_base_url", "https://external-api.demo.kalshi.co/trade-api/v2")
        ).rstrip("/"),
        production_base_url=str(
            data.get("production_base_url", "https://external-api.kalshi.com/trade-api/v2")
        ).rstrip("/"),
        request_timeout_seconds=float(data.get("request_timeout_seconds", 15.0)),
        max_retries=int(data.get("max_retries", 2)),
        retry_backoff_seconds=float(data.get("retry_backoff_seconds", 0.5)),
        allow_public_market_data_without_auth=bool(
            data.get("allow_public_market_data_without_auth", True)
        ),
    )


def _parse_market_settings(raw: Any) -> MarketSettings:
    data = _mapping(raw, "markets")
    supported_locations = _tuple_of_strings(data.get("supported_locations", ["NYC"]))
    default_location = str(data.get("default_location", supported_locations[0])).strip()
    supported_market_types = _tuple_of_strings(
        data.get("supported_market_types", ["daily_high_temperature"])
    )
    target_market_type = str(
        data.get("target_market_type", supported_market_types[0])
    ).strip()
    return MarketSettings(
        default_location=default_location,
        supported_locations=supported_locations,
        target_market_type=target_market_type,
        supported_market_types=supported_market_types,
        status=str(data.get("status", "open")).strip(),
        tradable_statuses=_tuple_of_strings(data.get("tradable_statuses", ["open", "active"])),
        min_minutes_to_close=int(data.get("min_minutes_to_close", 30)),
        page_limit=int(data.get("page_limit", 1000)),
        max_pages=int(data.get("max_pages", 5)),
        series_tickers=_parse_string_tuple_mapping(data.get("series_tickers", {})),
        location_terms=_parse_string_tuple_mapping(data.get("location_terms", {})),
        weather_terms=_tuple_of_strings(data.get("weather_terms", [])),
    )


def _parse_output_settings(raw: Any) -> OutputSettings:
    data = _mapping(raw, "outputs")
    return OutputSettings(
        live_trading_dir=_repo_path(data.get("live_trading_dir", "outputs/live_trading")),
        market_discovery_snapshot_path=_repo_path(
            data.get(
                "market_discovery_snapshot_path",
                "outputs/live_trading/market_discovery_snapshot.csv",
            )
        ),
        market_discovery_raw_path=_repo_path(
            data.get(
                "market_discovery_raw_path",
                "outputs/live_trading/market_discovery_raw.json",
            )
        ),
        contract_bucket_mapping_path=_repo_path(
            data.get(
                "contract_bucket_mapping_path",
                "outputs/live_trading/contract_bucket_mapping.csv",
            )
        ),
        live_weather_snapshot_path=_repo_path(
            data.get(
                "live_weather_snapshot_path",
                "outputs/live_trading/live_weather_snapshot.csv",
            )
        ),
        live_feature_rows_path=_repo_path(
            data.get(
                "live_feature_rows_path",
                "outputs/live_trading/live_feature_rows.csv",
            )
        ),
        live_feature_freshness_path=_repo_path(
            data.get(
                "live_feature_freshness_path",
                "outputs/live_trading/live_feature_freshness.csv",
            )
        ),
        live_bucket_probabilities_path=_repo_path(
            data.get(
                "live_bucket_probabilities_path",
                "outputs/live_trading/live_bucket_probabilities.csv",
            )
        ),
        settlement_state_path=_repo_path(
            data.get(
                "settlement_state_path",
                "outputs/live_trading/settlement_state.csv",
            )
        ),
        orderbook_snapshot_path=_repo_path(
            data.get(
                "orderbook_snapshot_path",
                "outputs/live_trading/orderbook_snapshot.csv",
            )
        ),
        orderbook_summary_path=_repo_path(
            data.get(
                "orderbook_summary_path",
                "outputs/live_trading/orderbook_summary.csv",
            )
        ),
        edge_table_path=_repo_path(
            data.get(
                "edge_table_path",
                "outputs/live_trading/edge_table.csv",
            )
        ),
        portfolio_snapshot_path=_repo_path(
            data.get(
                "portfolio_snapshot_path",
                "outputs/live_trading/portfolio_snapshot.csv",
            )
        ),
        risk_decisions_path=_repo_path(
            data.get(
                "risk_decisions_path",
                "outputs/live_trading/risk_decisions.csv",
            )
        ),
        order_intents_path=_repo_path(
            data.get(
                "order_intents_path",
                "outputs/live_trading/order_intents.csv",
            )
        ),
        paper_orders_path=_repo_path(
            data.get(
                "paper_orders_path",
                "outputs/live_trading/paper_orders.csv",
            )
        ),
        paper_positions_path=_repo_path(
            data.get(
                "paper_positions_path",
                "outputs/live_trading/paper_positions.csv",
            )
        ),
        paper_pnl_path=_repo_path(
            data.get(
                "paper_pnl_path",
                "outputs/live_trading/paper_pnl.csv",
            )
        ),
        trading_cycle_log_path=_repo_path(
            data.get(
                "trading_cycle_log_path",
                "outputs/live_trading/trading_cycle_log.csv",
            )
        ),
        dashboard_status_path=_repo_path(
            data.get(
                "dashboard_status_path",
                "outputs/live_trading/dashboard_status.json",
            )
        ),
    )


def _parse_edge_settings(raw: Any) -> EdgeSettings:
    data = _mapping(raw, "edge")
    return EdgeSettings(
        min_edge_dollars=float(data.get("min_edge_dollars", 0.02)),
        min_edge_percent=float(data.get("min_edge_percent", 0.0)),
        slippage_buffer_dollars=float(data.get("slippage_buffer_dollars", 0.005)),
        min_liquidity_contracts=float(data.get("min_liquidity_contracts", 1.0)),
        max_spread_dollars=float(data.get("max_spread_dollars", 0.25)),
        max_staleness_seconds=float(data.get("max_staleness_seconds", 300.0)),
        fee_rate=float(data.get("fee_rate", 0.07)),
        evaluation_contracts=float(data.get("evaluation_contracts", 1.0)),
        trade_fee_rounding_increment=float(
            data.get("trade_fee_rounding_increment", 0.0001)
        ),
        balance_rounding_increment=float(
            data.get("balance_rounding_increment", 0.01)
        ),
    )


def _parse_settlement_settings(raw: Any) -> SettlementSettings:
    data = _mapping(raw, "settlement")
    return SettlementSettings(
        typical_peak_hour=int(data.get("typical_peak_hour", 15)),
        peak_window_end_hour=int(data.get("peak_window_end_hour", 18)),
        verified_settlement_min_hour=int(data.get("verified_settlement_min_hour", 18)),
        post_peak_temp_drop_f=float(data.get("post_peak_temp_drop_f", 2.0)),
        min_minutes_since_high=int(data.get("min_minutes_since_high", 60)),
        forecast_remaining_margin_f=float(data.get("forecast_remaining_margin_f", 0.5)),
        settlement_tail_probability=float(data.get("settlement_tail_probability", 0.01)),
        block_unverified_observed_high=bool(data.get("block_unverified_observed_high", True)),
    )


def _parse_paper_settings(raw: Any) -> PaperSettings:
    data = _mapping(raw, "paper")
    return PaperSettings(
        starting_cash_dollars=float(data.get("starting_cash_dollars", 100.0)),
        fill_mode=str(data.get("fill_mode", "immediate")).strip(),
    )


def _parse_risk_settings(raw: Any) -> RiskSettings:
    data = _mapping(raw, "risk")
    return RiskSettings(
        kill_switch_path=_repo_path(data.get("kill_switch_path", "runtime/KILL_SWITCH_TRADING")),
        max_contracts_per_order=int(data.get("max_contracts_per_order", 1)),
        max_contracts_per_market=int(data.get("max_contracts_per_market", 5)),
        max_dollars_per_order=float(data.get("max_dollars_per_order", 5.0)),
        max_dollars_per_market=float(data.get("max_dollars_per_market", 20.0)),
        max_dollars_per_event=float(data.get("max_dollars_per_event", 30.0)),
        max_correlated_event_exposure_dollars=float(
            data.get("max_correlated_event_exposure_dollars", 30.0)
        ),
        max_total_exposure=float(data.get("max_total_exposure", 50.0)),
        max_open_orders=int(data.get("max_open_orders", 10)),
        max_daily_loss_dollars=float(data.get("max_daily_loss_dollars", 25.0)),
        min_cash_reserve_dollars=float(data.get("min_cash_reserve_dollars", 0.0)),
        denylist_tickers=_tuple_of_strings(data.get("denylist_tickers", [])),
        denylist_event_tickers=_tuple_of_strings(data.get("denylist_event_tickers", [])),
        denylist_target_dates=_tuple_of_strings(data.get("denylist_target_dates", [])),
    )


def _parse_live_weather_settings(raw: Any) -> LiveWeatherSettings:
    data = _mapping(raw, "weather")
    observation_grid = _parse_weather_grid(
        data.get(
            "observation_grid",
            {
                "latitude": 40.808434,
                "longitude": -74.0199,
            },
        ),
        "weather.observation_grid",
    )
    forecast_grid = _parse_weather_grid(
        data.get(
            "forecast_grid",
            {
                "latitude": 40.78858,
                "longitude": -73.9661,
            },
        ),
        "weather.forecast_grid",
    )
    nws_station = _parse_nws_station_settings(data.get("nws_station", {}))
    return LiveWeatherSettings(
        provider=str(data.get("provider", "open_meteo")).strip(),
        observations_provider=str(data.get("observations_provider", "nws_station")).strip(),
        forecast_base_url=str(
            data.get("forecast_base_url", "https://api.open-meteo.com/v1/forecast")
        ).rstrip("/"),
        timezone=str(data.get("timezone", "America/New_York")).strip(),
        temperature_unit=str(data.get("temperature_unit", "fahrenheit")).strip(),
        wind_speed_unit=str(data.get("wind_speed_unit", "mph")).strip(),
        precipitation_unit=str(data.get("precipitation_unit", "inch")).strip(),
        observation_grid=observation_grid,
        forecast_grid=forecast_grid,
        nws_station=nws_station,
        observed_past_hours=int(data.get("observed_past_hours", 12)),
        forecast_past_hours=int(data.get("forecast_past_hours", 12)),
        forecast_days=int(data.get("forecast_days", 2)),
        max_observation_age_minutes=int(data.get("max_observation_age_minutes", 90)),
        max_forecast_age_minutes=int(data.get("max_forecast_age_minutes", 180)),
        max_unverified_observed_high_minutes=int(
            data.get("max_unverified_observed_high_minutes", 20)
        ),
        require_forecast_issue_time=bool(data.get("require_forecast_issue_time", False)),
    )


def _parse_nws_station_settings(raw: Any) -> NwsStationSettings:
    data = _mapping(raw, "weather.nws_station")
    return NwsStationSettings(
        station_id=str(data.get("station_id", "KNYC")).strip(),
        station_name=str(data.get("station_name", "NY City Central Park")).strip(),
        ghcn_station_id=str(data.get("ghcn_station_id", "GHCND:USW00094728")).strip(),
        latitude=float(data.get("latitude", 40.77898)),
        longitude=float(data.get("longitude", -73.96925)),
        base_url=str(data.get("base_url", "https://api.weather.gov")).rstrip("/"),
        user_agent=str(
            data.get("user_agent", "KalshiWeatherTrading/0.1 (local research; contact user)")
        ).strip(),
    )


def _parse_weather_grid(raw: Any, name: str) -> WeatherGridSettings:
    data = _mapping(raw, name)
    if "latitude" not in data or "longitude" not in data:
        raise TradingConfigError(f"{name} must include latitude and longitude")
    return WeatherGridSettings(
        latitude=float(data["latitude"]),
        longitude=float(data["longitude"]),
    )


def _repo_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _normalize_env(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"prod", "production", "live"}:
        return "production"
    if normalized in {"demo", "sandbox"}:
        return "demo"
    return normalized


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TradingConfigError(f"{name} must be a mapping")
    return value


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, (list, tuple)):
        raise TradingConfigError("Expected a string or list of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _parse_string_tuple_mapping(value: Any) -> dict[str, tuple[str, ...]]:
    mapping = _mapping(value, "string tuple mapping")
    return {str(key).strip(): _tuple_of_strings(items) for key, items in mapping.items()}
