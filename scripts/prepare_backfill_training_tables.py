"""Prepare newly merged settlement-aligned data for training use.

Run from anywhere: python scripts/prepare_backfill_training_tables.py

Creates (new files only - existing pipeline outputs are untouched):
1. data/processed/hourly_asos_normalized_2018_2026.csv
   - IEM ASOS hourly obs with TRUE UTC timestamps (the raw archive's `valid`
     column is genuine UTC, unlike hourly_clean.csv whose stamps are local time
     mislabeled +00:00), nws_* schema mirroring hourly_clean.csv so Day-6-style
     feature builders can consume it, cloud-cover percent mapped from METAR
     sky-layer codes, plus America/New_York `date` / obs_time_local grouping keys.
2. data/processed/knyc_daily_actuals_combined.csv
   - 2018-2026 daily actual-high targets: union of NWS CLI reports
     (settlement-aligned proxy, preferred where available) and legacy
     NOAA/GHCN TMAX rows (fills days the CLI archive lacks); explicit source
     labels, cross-source agreement flag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
SKY_PCT = {"CLR": 0.0, "SKC": 0.0, "NSC": 0.0, "FEW": 12.5, "SCT": 37.5, "BKN": 87.5, "OVC": 100.0}

out = []


def p(*a):
    out.append(" ".join(str(x) for x in a))


# ---------------------------------------------------------------- hourly ----
a = pd.read_csv(REPO / "data/raw/NYC_nws_hourly_2018_2026.csv", low_memory=False)
a["timestamp"] = pd.to_datetime(a["valid"], errors="coerce", utc=True)
a["tmpf"] = pd.to_numeric(a["tmpf"], errors="coerce")

h = pd.DataFrame({
    "timestamp": a["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z"),
    "location": "NYC",
    "station": a["station"],
    "nws_current_temp_f": pd.to_numeric(a["tmpf"], errors="coerce"),
    "nws_dew_point_f": pd.to_numeric(a["dwpf"], errors="coerce"),
    "nws_relative_humidity": pd.to_numeric(a["relh"], errors="coerce"),
    "nws_wind_dir": pd.to_numeric(a["drct"], errors="coerce"),
    "nws_wind_speed_kt": pd.to_numeric(a["sknt"], errors="coerce"),
    "nws_wind_gust_kt": pd.to_numeric(a["gust"], errors="coerce"),
    "nws_altimeter": pd.to_numeric(a["alti"], errors="coerce"),
    "nws_mslp": pd.to_numeric(a["mslp"], errors="coerce"),
    "nws_precip_1h": pd.to_numeric(a["p01i"], errors="coerce"),
})
for c in ("skyc1", "skyc2", "skyc3"):
    h[f"nws_{c}"] = a[c]
h["nws_cloud_cover_pct"] = a["skyc1"].map(SKY_PCT)
h["nws_cloud_cover_pct"] = h["nws_cloud_cover_pct"].fillna(
    a["skyc2"].map(SKY_PCT)).fillna(a["skyc3"].map(SKY_PCT))
h["nws_metar"] = a["metar"]
h["observation_source"] = "iem_nws_asos"
# Local (America/New_York) calendar date of the observation = grouping key the
# feature builder uses for target-day paths.
local = a["timestamp"].dt.tz_convert("America/New_York")
h["date"] = local.dt.strftime("%Y-%m-%d")
h["obs_time_local"] = local.dt.strftime("%H:%M")

h = h.sort_values("timestamp").reset_index(drop=True)
dest_h = REPO / "data/processed/hourly_asos_normalized_2018_2026.csv"
h.to_csv(dest_h, index=False)

p("HOURLY NORMALIZED ->", dest_h.name)
p("rows:", len(h))
ts = pd.to_datetime(h["timestamp"], utc=True)
p("UTC range:", ts.min(), "->", ts.max())
p("temp nulls:", int(h["nws_current_temp_f"].isna().sum()),
  "| cloud pct mapped:", int(h["nws_cloud_cover_pct"].notna().sum()))

# ------------------------------------------------------------- daily -------
cli = pd.read_csv(REPO / "data/processed/knyc_cli_daily_2018_2026.csv", parse_dates=["date"])
noaa = pd.read_csv(REPO / "data/processed/daily_clean.csv", parse_dates=["date"])

cli_min = cli[[
    "date", "location", "actual_high", "official_daily_high_f",
    "actual_source", "source_station", "cli_high_time_local",
    "cli_low_f", "cli_precip_in",
]].rename(columns={"actual_high": "actual_high_cli"}).copy()

noaa_min = noaa[["date", "location", "actual_high", "official_daily_high_f",
                 "actual_source", "source_station"]].rename(
    columns={"actual_high": "actual_high_noaa",
             "official_daily_high_f": "official_daily_high_f_noaa",
             "actual_source": "noaa_source",
             "source_station": "noaa_station"})

combined = cli_min.merge(noaa_min.drop(columns="location"), on="date", how="outer")
both = combined["actual_high_cli"].notna() & combined["actual_high_noaa"].notna()
combined["sources_agree"] = np.where(both, combined["actual_high_cli"] == combined["actual_high_noaa"], pd.NA)
# Precedence: CLI is the settlement-aligned proxy for the historical regime;
# NOAA/GHCN fills days the CLI archive lacks (e.g., 2025-06-02/03).
combined["actual_source_used"] = np.where(
    combined["actual_high_cli"].notna(), "iem_nws_cli_daily_high",
    np.where(combined["actual_high_noaa"].notna(), "noaa_nws_daily_tmax", None),
)
combined["actual_high"] = combined["actual_high_cli"].fillna(combined["actual_high_noaa"])
combined["official_daily_high_f"] = combined["official_daily_high_f"].fillna(
    combined["official_daily_high_f_noaa"])
combined = combined.sort_values("date").reset_index(drop=True)

dest_d = REPO / "data/processed/knyc_daily_actuals_combined.csv"
combined.to_csv(dest_d, index=False)

p("DAILY COMBINED ->", dest_d.name)
p("rows:", len(combined), "| range:", combined['date'].min().date(), "->", combined['date'].max().date())
p("actual_source_used:", combined["actual_source_used"].value_counts(dropna=False).to_dict())
p("days with both sources:", int(both.sum()),
  "| agree:", int((combined['sources_agree'] == True).sum()),
  "| disagree:", int((combined['sources_agree'] == False).sum()))
p("null actual_high:", int(combined['actual_high'].isna().sum()))


if __name__ == "__main__":
    print("\n".join(out))