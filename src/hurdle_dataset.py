from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATIONS = REPO_ROOT / "data" / "processed" / "knyc_asos_5min_2020_2026.csv.gz"
DEFAULT_DAILY = REPO_ROOT / "data" / "processed" / "daily_clean.csv"
DEFAULT_FORECASTS = REPO_ROOT / "data" / "processed" / "ndfd_knyc_daily_high_forecasts_2018_2026.csv"
DEFAULT_HOURLY = REPO_ROOT / "data" / "processed" / "hourly_clean.csv"

NY_TZ = "America/New_York"
HURDLE_TARGET_COL = "will_increase"
REMAINING_COL = "remaining_increase"
FINAL_HIGH_COL = "final_daily_high"
CURRENT_MAX_COL = "current_max_so_far"


@dataclass(frozen=True)
class HurdleDatasetConfig:
    frequency: str = "5min"
    start_time: str = "09:00"
    end_time: str = "21:55"
    observation_tolerance: str = "10min"
    atmospheric_tolerance: str = "90min"
    require_forecast: bool = True
    minimum_daily_observations: int = 200
    latest_first_observation_minute: int = 60
    earliest_last_observation_minute: int = 1380
    violation_policy: Literal["raise", "quarantine"] = "quarantine"


def settlement_round_f(values: pd.Series | np.ndarray) -> pd.Series:
    """Apply the settlement's explicit whole-Fahrenheit, half-up convention."""
    series = pd.Series(values, copy=False, dtype=float)
    return np.floor(series + 0.5)


def _decision_grid(dates: pd.Series, config: HurdleDatasetConfig) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    parsed_dates = pd.to_datetime(dates, errors="coerce").dropna()
    for day in sorted(parsed_dates.dt.date.unique()):
        start = pd.Timestamp(f"{day} {config.start_time}", tz=NY_TZ)
        end = pd.Timestamp(f"{day} {config.end_time}", tz=NY_TZ)
        local = pd.date_range(start, end, freq=config.frequency)
        chunks.append(
            pd.DataFrame(
                {
                    "prediction_time": local.tz_localize(None),
                    "prediction_time_utc": local.tz_convert("UTC"),
                    "target_date": pd.Timestamp(day),
                    "date": str(day),
                }
            )
        )
    if not chunks:
        return pd.DataFrame(columns=["prediction_time", "prediction_time_utc", "target_date", "date"])
    return pd.concat(chunks, ignore_index=True)


def _prepare_observations(path: Path, config: HurdleDatasetConfig) -> pd.DataFrame:
    obs = pd.read_csv(path)
    required = {"station", "timestamp_utc", "temp_f", "dewpoint_f"}
    missing = required.difference(obs.columns)
    if missing:
        raise ValueError(f"5-minute observation file is missing columns: {sorted(missing)}")
    obs = obs.loc[obs["station"].astype(str).str.upper().eq("KNYC")].copy()
    obs["observation_time_utc"] = pd.to_datetime(obs["timestamp_utc"], utc=True, errors="coerce")
    obs["observation_time_local"] = obs["observation_time_utc"].dt.tz_convert(NY_TZ)
    obs["target_date"] = obs["observation_time_local"].dt.tz_localize(None).dt.normalize()
    obs["current_temp"] = pd.to_numeric(obs["temp_f"], errors="coerce")
    obs["dew_point"] = pd.to_numeric(obs["dewpoint_f"], errors="coerce")
    obs = obs.dropna(subset=["observation_time_utc", "target_date", "current_temp"])
    obs = obs.sort_values(["target_date", "observation_time_utc"]).drop_duplicates(
        "observation_time_utc", keep="last"
    )
    obs["observation_minute_local"] = (
        obs["observation_time_local"].dt.hour * 60 + obs["observation_time_local"].dt.minute
    )
    obs["settlement_temp"] = settlement_round_f(obs["current_temp"]).to_numpy()
    obs["current_max_so_far"] = obs.groupby("target_date")["settlement_temp"].cummax()
    at_max = obs["settlement_temp"].eq(obs["current_max_so_far"])
    obs["current_max_time_utc"] = obs["observation_time_utc"].where(at_max)
    obs["current_max_time_utc"] = obs.groupby("target_date")["current_max_time_utc"].ffill()
    grouped = obs.groupby("target_date", sort=False)
    obs["daily_observation_count"] = grouped["observation_time_utc"].transform("size")
    obs["daily_first_observation_minute"] = grouped["observation_minute_local"].transform("min")
    obs["daily_last_observation_minute"] = grouped["observation_minute_local"].transform("max")
    obs["same_feed_final_daily_high"] = grouped["settlement_temp"].transform("max")
    obs["complete_source_day"] = (
        obs["daily_observation_count"].ge(config.minimum_daily_observations)
        & obs["daily_first_observation_minute"].le(config.latest_first_observation_minute)
        & obs["daily_last_observation_minute"].ge(config.earliest_last_observation_minute)
    )
    obs = obs.loc[obs["complete_source_day"]].copy()
    return obs[
        [
            "target_date",
            "observation_time_utc",
            "current_max_time_utc",
            "current_temp",
            "dew_point",
            "settlement_temp",
            "current_max_so_far",
            "same_feed_final_daily_high",
            "daily_observation_count",
            "daily_first_observation_minute",
            "daily_last_observation_minute",
            "complete_source_day",
        ]
    ]


def _merge_asof_by_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    tolerance: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Date-scoped as-of merge that is robust to pandas global sort rules."""
    pieces: list[pd.DataFrame] = []
    right_groups = {key: value for key, value in right.groupby("target_date", sort=False)}
    for day, left_day in left.groupby("target_date", sort=False):
        right_day = right_groups.get(day)
        left_day = left_day.sort_values(left_on)
        if right_day is None or right_day.empty:
            pieces.append(left_day.copy())
            continue
        pieces.append(
            pd.merge_asof(
                left_day,
                right_day.drop(columns="target_date").sort_values(right_on),
                left_on=left_on,
                right_on=right_on,
                direction="backward",
                tolerance=pd.Timedelta(tolerance) if tolerance is not None else None,
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(["target_date", left_on]).reset_index(drop=True)


def _attach_forecasts(grid: pd.DataFrame, path: Path) -> pd.DataFrame:
    forecasts = pd.read_csv(path)
    required = {"date", "forecast_high", "forecast_issue_time", "forecast_source"}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast archive is missing columns: {sorted(missing)}")
    forecasts["target_date"] = pd.to_datetime(forecasts["date"], errors="coerce").dt.normalize()
    forecasts["forecast_issue_time_utc"] = pd.to_datetime(
        forecasts["forecast_issue_time"], utc=True, errors="coerce"
    )
    forecasts["forecast_high"] = pd.to_numeric(forecasts["forecast_high"], errors="coerce")
    forecasts = forecasts.loc[
        forecasts["forecast_source"].astype(str).eq("nws_ndfd_historical_forecast")
        & forecasts["forecast_issue_time_utc"].notna()
        & forecasts["forecast_high"].notna()
    ].copy()
    keep = [
        "target_date",
        "forecast_issue_time_utc",
        "forecast_high",
        "forecast_source",
        "ndfd_valid_time_utc",
        "ndfd_lead_hours",
    ]
    keep = [column for column in keep if column in forecasts.columns]
    result = _merge_asof_by_date(
        grid,
        forecasts[keep],
        left_on="prediction_time_utc",
        right_on="forecast_issue_time_utc",
    )
    result["forecast_age_minutes"] = (
        result["prediction_time_utc"] - result["forecast_issue_time_utc"]
    ).dt.total_seconds() / 60.0
    result["forecast_available"] = result["forecast_high"].notna().astype(int)
    return result


def _attach_atmosphere(grid: pd.DataFrame, path: Path, tolerance: str) -> pd.DataFrame:
    hourly = pd.read_csv(path)
    if "timestamp" not in hourly.columns:
        raise ValueError("Hourly observation file is missing 'timestamp'")
    local = pd.to_datetime(hourly["timestamp"], errors="coerce")
    hourly["atmosphere_time_utc"] = local.dt.tz_localize(
        NY_TZ, ambiguous="NaT", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    hourly["target_date"] = local.dt.normalize()
    aliases = {
        "relative_humidity": "nws_relative_humidity",
        "cloud_cover": "nws_cloud_cover_pct",
        "wind_speed": "nws_wind_speed_kt",
        "wind_direction": "nws_wind_dir",
        "precipitation": "nws_precip_1h",
        "surface_pressure": "nws_mslp",
    }
    available = {out: src for out, src in aliases.items() if src in hourly.columns}
    keep = ["target_date", "atmosphere_time_utc", *available.values()]
    hourly = hourly[keep].dropna(subset=["atmosphere_time_utc"]).copy()
    hourly = hourly.sort_values("atmosphere_time_utc").drop_duplicates(
        "atmosphere_time_utc", keep="last"
    )
    hourly = hourly.rename(columns={src: out for out, src in available.items()})
    result = _merge_asof_by_date(
        grid,
        hourly,
        left_on="prediction_time_utc",
        right_on="atmosphere_time_utc",
        tolerance=tolerance,
    )
    direction = np.deg2rad(pd.to_numeric(result.get("wind_direction"), errors="coerce"))
    result["wind_dir_sin"] = np.sin(direction)
    result["wind_dir_cos"] = np.cos(direction)
    return result


def _attach_temperature_features(grid: pd.DataFrame, frequency: str) -> pd.DataFrame:
    step_minutes = int(pd.Timedelta(frequency).total_seconds() // 60)
    if step_minutes <= 0:
        raise ValueError(f"Unsupported prediction frequency: {frequency}")
    grouped = grid.groupby("target_date", sort=False)
    for window in (5, 15, 30, 60, 120, 180):
        if window % step_minutes:
            raise ValueError(f"Window {window}m is not divisible by frequency {frequency}")
        lag = window // step_minutes
        grid[f"temp_change_{window}m"] = grid["current_temp"] - grouped["current_temp"].shift(lag)
    grid["temp_slope_15m"] = grid["temp_change_15m"] / 0.25
    grid["temp_slope_30m"] = grid["temp_change_30m"] / 0.5
    grid["temp_slope_60m"] = grid["temp_change_60m"]
    grid["max_change_last_30m"] = grid["current_max_so_far"] - grouped["current_max_so_far"].shift(
        30 // step_minutes
    )
    grid["max_change_last_60m"] = grid["current_max_so_far"] - grouped["current_max_so_far"].shift(
        60 // step_minutes
    )
    grid["temp_acceleration_60m"] = grid["temp_change_60m"] - grouped["temp_change_60m"].shift(
        60 // step_minutes
    )
    grid["current_temp_minus_max_so_far"] = grid["current_temp"] - grid["current_max_so_far"]
    grid["minutes_since_max_temp_so_far"] = (
        grid["prediction_time_utc"] - grid["current_max_time_utc"]
    ).dt.total_seconds() / 60.0
    grid["hour_of_max_temp_so_far"] = grid["current_max_time_utc"].dt.tz_convert(NY_TZ).dt.hour
    grid["temp_minus_dew_point"] = grid["current_temp"] - grid["dew_point"]
    return grid


def _attach_forecast_features(grid: pd.DataFrame, frequency: str) -> pd.DataFrame:
    step_minutes = int(pd.Timedelta(frequency).total_seconds() // 60)
    grouped = grid.groupby("target_date", sort=False)
    grid["forecast_gap"] = grid["forecast_high"] - grid["current_max_so_far"]
    grid["forecast_high_minus_current_max"] = grid["forecast_gap"]
    grid["forecast_high_minus_current_temp"] = grid["forecast_high"] - grid["current_temp"]
    grid["max_so_far_minus_forecast_high"] = -grid["forecast_gap"]
    grid["forecast_revision_1h"] = grid["forecast_high"] - grouped["forecast_high"].shift(
        60 // step_minutes
    )
    grid["forecast_revision_3h"] = grid["forecast_high"] - grouped["forecast_high"].shift(
        180 // step_minutes
    )
    return grid


def build_hurdle_dataset(
    observations_path: str | Path | None = None,
    daily_path: str | Path | None = None,
    forecasts_path: str | Path | None = None,
    hourly_path: str | Path | None = None,
    *,
    config: HurdleDatasetConfig | None = None,
    prediction_frequency: str | None = None,
    strict_invariant: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build leakage-safe live-decision rows from the native 5-minute feed.

    The physical future-high target is source-consistent: both the running
    maximum and end-of-day maximum use the rounded KNYC five-minute feed.
    Official daily settlement is attached only as a reconciliation audit.
    """
    cfg = config or HurdleDatasetConfig()
    if prediction_frequency is not None:
        cfg = HurdleDatasetConfig(**{**cfg.__dict__, "frequency": prediction_frequency})
    if strict_invariant is True:
        cfg = HurdleDatasetConfig(**{**cfg.__dict__, "violation_policy": "raise"})

    obs_path = Path(observations_path) if observations_path else DEFAULT_OBSERVATIONS
    daily_file = Path(daily_path) if daily_path else DEFAULT_DAILY
    forecast_file = Path(forecasts_path) if forecasts_path else DEFAULT_FORECASTS
    hourly_file = Path(hourly_path) if hourly_path else DEFAULT_HOURLY
    for path in (obs_path, daily_file, forecast_file, hourly_file):
        if not path.exists():
            raise FileNotFoundError(path)

    observations = _prepare_observations(obs_path, cfg)
    if observations.empty:
        raise ValueError("No observation days satisfy the configured full-day coverage rules")
    grid = _decision_grid(observations["target_date"], cfg)
    rows_on_grid = len(grid)
    grid = _merge_asof_by_date(
        grid,
        observations,
        left_on="prediction_time_utc",
        right_on="observation_time_utc",
        tolerance=cfg.observation_tolerance,
    )
    fresh_observation = grid[["observation_time_utc", "current_temp", "current_max_so_far"]].notna().all(axis=1)
    rows_with_observations = int(fresh_observation.sum())
    grid["observation_age_minutes"] = (
        grid["prediction_time_utc"] - grid["observation_time_utc"]
    ).dt.total_seconds() / 60.0
    # Keep the complete clock grid while calculating lags. A missing 09:30
    # observation must not turn the 09:35 -> 09:25 difference into a fake 5m lag.
    grid = _attach_temperature_features(grid, cfg.frequency)
    grid = _attach_forecasts(grid, forecast_file)
    grid = _attach_forecast_features(grid, cfg.frequency)
    grid = _attach_atmosphere(grid, hourly_file, cfg.atmospheric_tolerance)

    daily = pd.read_csv(daily_file)
    required_daily = {"date", "actual_high", "actual_source", "source_station"}
    missing_daily = required_daily.difference(daily.columns)
    if missing_daily:
        raise ValueError(f"Daily settlement file is missing columns: {sorted(missing_daily)}")
    daily["target_date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["official_final_daily_high"] = pd.to_numeric(daily["actual_high"], errors="coerce")
    daily_keep = [
        "target_date",
        "official_final_daily_high",
        "actual_source",
        "source_station",
        "source_station_name",
        "source_file",
    ]
    daily_keep = [column for column in daily_keep if column in daily.columns]
    grid = grid.merge(daily[daily_keep], on="target_date", how="left", validate="many_to_one")
    grid["final_daily_high"] = grid["same_feed_final_daily_high"]
    grid["official_settlement_gap"] = grid["official_final_daily_high"] - grid["final_daily_high"]
    grid["official_settlement_agrees"] = grid["official_settlement_gap"].eq(0)
    rows_with_target = int((fresh_observation & grid["final_daily_high"].notna()).sum())
    grid = grid.dropna(
        subset=["observation_time_utc", "current_temp", "current_max_so_far", "final_daily_high"]
    ).copy()

    if cfg.require_forecast:
        grid = grid.loc[grid["forecast_available"].eq(1)].copy()
    rows_after_forecast_filter = len(grid)

    if (grid["observation_time_utc"] > grid["prediction_time_utc"]).any():
        raise AssertionError("Observation leakage: observation_time is after prediction_time")
    forecast_mask = grid["forecast_issue_time_utc"].notna()
    if (grid.loc[forecast_mask, "forecast_issue_time_utc"] > grid.loc[forecast_mask, "prediction_time_utc"]).any():
        raise AssertionError("Forecast leakage: forecast_issue_time is after prediction_time")
    atmosphere_mask = grid["atmosphere_time_utc"].notna()
    if (grid.loc[atmosphere_mask, "atmosphere_time_utc"] > grid.loc[atmosphere_mask, "prediction_time_utc"]).any():
        raise AssertionError("Atmospheric leakage: source observation is after prediction_time")

    grid["remaining_increase"] = grid["final_daily_high"] - grid["current_max_so_far"]
    violation_rows = grid.loc[grid["remaining_increase"] < 0].copy()
    if not violation_rows.empty:
        raise AssertionError(
            "Same-feed target invariant failed; final_daily_high must be the end-of-day "
            "maximum of the same rounded observation series"
        )
    violating_dates: list[str] = []
    grid["remaining_increase"] = grid["final_daily_high"] - grid["current_max_so_far"]
    grid["will_increase"] = (grid["remaining_increase"] > 0).astype("int8")
    grid["current_temperature"] = grid["current_temp"]
    grid["timestamp"] = grid["prediction_time"]
    grid = grid.sort_values(["target_date", "prediction_time"]).reset_index(drop=True)

    violation_columns = [
        "date",
        "prediction_time",
        "observation_time_utc",
        "current_temp",
        "current_max_so_far",
        "final_daily_high",
        "remaining_increase",
        "actual_source",
        "source_station",
    ]
    grid.attrs["invariant_violations"] = violation_rows[
        [column for column in violation_columns if column in violation_rows.columns]
    ].reset_index(drop=True)
    disagreement_columns = [
        "date",
        "final_daily_high",
        "official_final_daily_high",
        "official_settlement_gap",
        "actual_source",
        "source_station",
        "daily_observation_count",
        "daily_first_observation_minute",
        "daily_last_observation_minute",
    ]
    official_audit = grid.loc[
        grid["official_final_daily_high"].notna() & ~grid["official_settlement_agrees"]
    ]
    grid.attrs["official_settlement_disagreements"] = official_audit[
        [column for column in disagreement_columns if column in official_audit.columns]
    ].drop_duplicates("date").reset_index(drop=True)

    official_day_rows = grid.loc[grid["official_final_daily_high"].notna()].drop_duplicates("date")
    official_disagreement_days = int((~official_day_rows["official_settlement_agrees"]).sum())

    summary: dict[str, Any] = {
        "dataset_version": "hurdle_v3_knyc_same_feed_future_high",
        "prediction_frequency": cfg.frequency,
        "decision_window_local": f"{cfg.start_time}-{cfg.end_time}",
        "total_rows": int(len(grid)),
        "total_days": int(grid["target_date"].nunique()),
        "grid_rows": int(rows_on_grid),
        "rows_without_fresh_observation": int(rows_on_grid - rows_with_observations),
        "rows_without_same_feed_target": int(rows_with_observations - rows_with_target),
        "rows_without_canonical_forecast": int(rows_with_target - rows_after_forecast_filter),
        "will_increase_mean": float(grid["will_increase"].mean()),
        "n_violation_rows": int(len(violation_rows)),
        "n_violation_dates": int(len(violating_dates)),
        "violating_dates": violating_dates,
        "violation_policy": "raise_fail_closed_same_feed",
        "invariant_holds_in_training_data": True,
        "complete_day_rules": {
            "minimum_daily_observations": cfg.minimum_daily_observations,
            "latest_first_observation_minute": cfg.latest_first_observation_minute,
            "earliest_last_observation_minute": cfg.earliest_last_observation_minute,
        },
        "official_settlement_days_available": int(len(official_day_rows)),
        "official_settlement_disagreement_days": official_disagreement_days,
        "settlement_rounding": "whole Fahrenheit, half-up (floor(x + 0.5))",
        "weather_station": "KNYC / USW00094728 Central Park",
        "target_source": "KNYC five-minute observations for both current and end-of-day maxima",
        "target_definition": "same-feed rounded end-of-day maximum > same-feed rounded current maximum",
        "official_settlement_role": "reconciliation audit only; never used in will_increase or remaining_increase",
        "forecast_source": "nws_ndfd_historical_forecast",
        "forecast_availability_verified": True,
    }
    return grid, summary


def validate_hurdle_dataset(df: pd.DataFrame, allow_violations_clipped: bool = False) -> None:
    """Validate the target identity, invariant, and timestamp provenance."""
    required = {
        "target_date",
        "prediction_time",
        "current_max_so_far",
        "final_daily_high",
        "remaining_increase",
        "will_increase",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Hurdle dataset is missing columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Required hurdle columns contain missing values")
    if not set(df["will_increase"].unique()).issubset({0, 1}):
        raise ValueError("will_increase must contain only 0 and 1")
    if (df["remaining_increase"] < 0).any():
        raise AssertionError("remaining_increase is negative; do not clip settlement violations")
    expected = (df["final_daily_high"] > df["current_max_so_far"]).astype(int)
    if not np.array_equal(expected.to_numpy(), df["will_increase"].astype(int).to_numpy()):
        raise AssertionError("will_increase does not match the hurdle target definition")
    if not np.allclose(
        df["remaining_increase"], df["final_daily_high"] - df["current_max_so_far"]
    ):
        raise AssertionError("remaining_increase does not match final minus current maximum")
    for source_col in ("observation_time_utc", "forecast_issue_time_utc", "atmosphere_time_utc"):
        if source_col in df.columns:
            mask = df[source_col].notna()
            if (df.loc[mask, source_col] > df.loc[mask, "prediction_time_utc"]).any():
                raise AssertionError(f"Leakage detected in {source_col}")


def hurdle_label_summary(df: pd.DataFrame) -> dict[str, Any]:
    validate_hurdle_dataset(df)
    positive = df.loc[df["will_increase"].eq(1), "remaining_increase"]
    return {
        "rows": int(len(df)),
        "days": int(pd.to_datetime(df["target_date"]).nunique()),
        "positive_rate": float(df["will_increase"].mean()),
        "remaining_increase_mean": float(df["remaining_increase"].mean()),
        "remaining_increase_positive_mean": float(positive.mean()),
    }
