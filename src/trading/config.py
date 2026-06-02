from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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


@dataclass(frozen=True)
class RiskSettings:
    kill_switch_path: Path


@dataclass(frozen=True)
class TradingConfig:
    mode: str
    trading_enabled: bool
    live_auto_enabled: bool
    kalshi: KalshiSettings
    markets: MarketSettings
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
    outputs = _parse_output_settings(raw.get("outputs", {}))
    risk = _parse_risk_settings(raw.get("risk", {}))

    config = TradingConfig(
        mode=mode,
        trading_enabled=trading_enabled,
        live_auto_enabled=live_auto_enabled,
        kalshi=kalshi,
        markets=markets,
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
    )


def _parse_risk_settings(raw: Any) -> RiskSettings:
    data = _mapping(raw, "risk")
    return RiskSettings(
        kill_switch_path=_repo_path(data.get("kill_switch_path", "runtime/KILL_SWITCH_TRADING"))
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
