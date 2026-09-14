from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TWC_SWITCH_DATE = pd.Timestamp("2026-08-14")
DEFAULT_NDFD = Path("data/processed/ndfd_knyc_hourly_asof_forecasts_2018_2026.csv")
DEFAULT_ASOS = Path("data/raw/NYC_nws_hourly_2018_2026.csv")
DEFAULT_CLI = Path("data/processed/knyc_cli_daily_2018_2026.csv")
DEFAULT_OUTPUT = Path("data/processed/production_replay_base_2018_2026.csv")
DEFAULT_REPORT = Path("outputs/data/production_replay_base_coverage.json")

ASOS_FIELDS = [
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "gust",
    "alti",
    "mslp",
    "p01i",
    "skyc1",
    "skyc2",
    "skyc3",
]


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return frame


def build_base(ndfd: pd.DataFrame, asos: pd.DataFrame, cli: pd.DataFrame) -> pd.DataFrame:
    rows = ndfd.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    rows["prediction_timestamp_utc"] = pd.to_datetime(
        rows["prediction_timestamp_utc"], errors="raise", utc=True
    )
    rows["forecast_issue_time"] = pd.to_datetime(
        rows["forecast_issue_time"], errors="coerce", utc=True
    )
    rows["forecast_high"] = pd.to_numeric(rows["forecast_high"], errors="coerce")

    obs = asos.copy()
    if "valid" not in obs.columns:
        raise ValueError("ASOS file must include valid timestamp")
    obs["asos_observation_time_utc"] = pd.to_datetime(obs["valid"], errors="coerce", utc=True)
    obs = obs[obs["asos_observation_time_utc"].notna()].copy()
    for field in ASOS_FIELDS:
        if field in obs.columns and not field.startswith("skyc"):
            obs[field] = pd.to_numeric(obs[field], errors="coerce")
    obs = obs.sort_values("asos_observation_time_utc")

    keep_obs = ["asos_observation_time_utc"] + [field for field in ASOS_FIELDS if field in obs.columns]
    left = rows.sort_values("prediction_timestamp_utc")
    merged = pd.merge_asof(
        left,
        obs.loc[:, keep_obs],
        left_on="prediction_timestamp_utc",
        right_on="asos_observation_time_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["asos_age_minutes"] = (
        merged["prediction_timestamp_utc"] - merged["asos_observation_time_utc"]
    ).dt.total_seconds() / 60.0
    merged["asos_observation_stale"] = merged["asos_age_minutes"] > 90

    targets = cli.copy()
    targets["date"] = pd.to_datetime(targets["date"], errors="coerce").dt.normalize()
    target_columns = [
        "date",
        "actual_high",
        "official_daily_high_f",
        "actual_source",
        "source_station",
        "source_station_name",
        "cli_high_time_local",
        "cli_product_id",
        "cli_wfo",
    ]
    target_columns = [column for column in target_columns if column in targets.columns]
    targets = targets.loc[:, target_columns].drop_duplicates(subset=["date"], keep="last")
    merged = merged.merge(targets, on="date", how="left")
    merged["actual_high"] = pd.to_numeric(merged.get("actual_high"), errors="coerce")
    merged["forecast_error"] = merged["actual_high"] - merged["forecast_high"]

    merged["settlement_regime"] = "nws_cli_legacy"
    current_regime = merged["date"] >= TWC_SWITCH_DATE
    merged.loc[current_regime, "settlement_regime"] = "twc_current_cli_proxy"
    merged["target_exact_for_kalshi_regime"] = ~current_regime
    merged["training_target_eligible"] = (
        merged["actual_high"].notna()
        & merged["forecast_high"].notna()
        & merged["target_exact_for_kalshi_regime"]
    )

    future_obs = (
        merged["asos_observation_time_utc"].notna()
        & (merged["asos_observation_time_utc"] > merged["prediction_timestamp_utc"])
    )
    future_forecast = (
        merged["forecast_issue_time"].notna()
        & (merged["forecast_issue_time"] > merged["prediction_timestamp_utc"])
    )
    if future_obs.any():
        raise ValueError(f"{int(future_obs.sum())} rows use future ASOS observations")
    if future_forecast.any():
        raise ValueError(f"{int(future_forecast.sum())} rows use future NDFD forecasts")

    return merged.sort_values(["date", "prediction_time"]).reset_index(drop=True)


def report(frame: pd.DataFrame) -> dict[str, object]:
    target_dates = pd.to_datetime(frame["date"])
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "unique_days": int(target_dates.nunique()),
        "first_date": target_dates.min().date().isoformat(),
        "last_date": target_dates.max().date().isoformat(),
        "ndfd_asof_rows": int(frame["forecast_high"].notna().sum()),
        "ndfd_asof_fraction": float(frame["forecast_high"].notna().mean()),
        "asos_rows_with_observation": int(frame["asos_observation_time_utc"].notna().sum()),
        "asos_fraction": float(frame["asos_observation_time_utc"].notna().mean()),
        "asos_stale_rows": int(frame["asos_observation_stale"].fillna(True).sum()),
        "rows_with_cli_target": int(frame["actual_high"].notna().sum()),
        "training_target_eligible_rows": int(frame["training_target_eligible"].sum()),
        "training_target_eligible_days": int(
            frame.loc[frame["training_target_eligible"], "date"].nunique()
        ),
        "twc_proxy_rows": int(frame["settlement_regime"].eq("twc_current_cli_proxy").sum()),
        "notes": [
            "forecast_error and actual_high are supervision/audit columns and must not enter model features.",
            "NDFD rows are selected as-of prediction time; ASOS rows are selected as-of prediction time.",
            "Rows on/after 2026-08-14 use NWS CLI only as a TWC-regime proxy and are not marked training-target eligible.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the timestamp-safe KNYC production-replay base table.")
    parser.add_argument("--ndfd", type=Path, default=DEFAULT_NDFD)
    parser.add_argument("--asos", type=Path, default=DEFAULT_ASOS)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_base(
        _read_required(args.ndfd),
        _read_required(args.asos),
        _read_required(args.cli),
    )
    summary = report(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.report_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
