from __future__ import annotations

import argparse
import io
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backfill_knyc_iem import (
    ASOS_COLUMNS,
    ASOS_MIN_REQUEST_INTERVAL_SECONDS,
    IEM_ASOS_URL,
    _date,
    _get_with_retry,
    _session,
    _yearly_windows,
)

STATIONS = {"knyc": "NYC", "klga": "LGA", "kjfk": "JFK", "kewr": "EWR"}
DEFAULT_OUTPUT = Path("data/processed/regional_asos_hourly_features_2018_2026.csv")
DEFAULT_RAW = Path("outputs/data/regional_asos_observations_2018_2026.csv")
DEFAULT_REPORT = Path("outputs/data/regional_asos_feature_coverage.json")
NUMERIC = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "mslp", "p01i"]


def fetch_station(station: str, start: date, end: date) -> pd.DataFrame:
    session = _session()
    frames: list[pd.DataFrame] = []
    windows = list(_yearly_windows(start, end))
    for i, (lo, hi) in enumerate(windows, 1):
        params: list[tuple[str, str]] = [
            ("station", station), ("tz", "Etc/UTC"), ("format", "onlycomma"),
            ("latlon", "no"), ("elev", "no"), ("missing", "M"), ("trace", "T"),
            ("direct", "no"), ("report_type", "3"), ("report_type", "4"),
            ("sts", f"{lo.isoformat()}T00:00:00Z"), ("ets", f"{hi.isoformat()}T00:00:00Z"),
        ]
        params.extend(("data", c) for c in ASOS_COLUMNS if c != "metar")
        r = _get_with_retry(session, IEM_ASOS_URL, params=params, timeout=180)
        text = r.text.strip()
        if text and not text.lower().startswith("error"):
            f = pd.read_csv(io.StringIO(text), na_values=["M", "null", ""])
            if not f.empty:
                frames.append(f)
        print(f"{station} {i}/{len(windows)} {lo} -> {hi}", flush=True)
        time.sleep(ASOS_MIN_REQUEST_INTERVAL_SECONDS)
    if not frames:
        raise RuntimeError(f"No ASOS observations returned for {station}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce", utc=True)
    df = df[df["valid"].notna()].copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("valid", keep="last").sort_values("valid").reset_index(drop=True)


def cloud_fraction(df: pd.DataFrame) -> pd.Series:
    codes = {"CLR": 0.0, "SKC": 0.0, "FEW": .125, "SCT": .375, "BKN": .75, "OVC": 1.0, "VV": 1.0}
    layers = [df[c].astype("string").str.upper().map(codes) for c in ("skyc1", "skyc2", "skyc3") if c in df]
    return pd.concat(layers, axis=1).max(axis=1, skipna=True) if layers else pd.Series(np.nan, index=df.index)


def station_table(df: pd.DataFrame, p: str) -> pd.DataFrame:
    out = pd.DataFrame({f"{p}_observation_time_utc": df["valid"]})
    names = {
        "tmpf": "temp_f", "dwpf": "dewpoint_f", "relh": "relative_humidity_pct",
        "drct": "wind_dir_deg", "sknt": "wind_speed_kt", "gust": "wind_gust_kt",
        "alti": "altimeter_inhg", "mslp": "mslp_mb", "p01i": "precip_1h_in",
    }
    for raw, name in names.items():
        out[f"{p}_{name}"] = df[raw] if raw in df else np.nan
    out[f"{p}_cloud_fraction"] = cloud_fraction(df)
    rad = np.deg2rad(out[f"{p}_wind_dir_deg"])
    speed = out[f"{p}_wind_speed_kt"]
    out[f"{p}_wind_u_kt"] = -speed * np.sin(rad)
    out[f"{p}_wind_v_kt"] = -speed * np.cos(rad)
    return out


def build_hourly(raw: dict[str, pd.DataFrame], start: date, end: date, tolerance_min: int) -> pd.DataFrame:
    end_hour = pd.Timestamp(end + timedelta(days=1), tz="UTC") - pd.Timedelta(hours=1)
    out = pd.DataFrame({"prediction_time_utc": pd.date_range(pd.Timestamp(start, tz="UTC"), end_hour, freq="h")})
    for p, df in raw.items():
        t = station_table(df, p).sort_values(f"{p}_observation_time_utc")
        out = pd.merge_asof(
            out.sort_values("prediction_time_utc"), t,
            left_on="prediction_time_utc", right_on=f"{p}_observation_time_utc",
            direction="backward", tolerance=pd.Timedelta(minutes=tolerance_min),
        )
        out[f"{p}_observation_age_min"] = (
            out["prediction_time_utc"] - out[f"{p}_observation_time_utc"]
        ).dt.total_seconds() / 60
    out["prediction_time_local"] = out["prediction_time_utc"].dt.tz_convert("America/New_York")
    return out


def mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].mean(axis=1, skipna=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ext, coast = ["klga", "kjfk", "kewr"], ["klga", "kjfk"]
    for v, suffix in [("temp_f", "temp"), ("dewpoint_f", "dewpoint"), ("mslp_mb", "mslp")]:
        cols = [f"{s}_{v}" for s in ext]
        x[f"regional_{suffix}_mean"] = mean(x, cols)
        x[f"regional_{suffix}_std"] = x[cols].std(axis=1, ddof=0)
    x["regional_temp_max_f"] = x[[f"{s}_temp_f" for s in ext]].max(axis=1)
    x["regional_temp_min_f"] = x[[f"{s}_temp_f" for s in ext]].min(axis=1)
    x["regional_cloud_fraction_mean"] = mean(x, [f"{s}_cloud_fraction" for s in ext])
    x["regional_wind_u_mean_kt"] = mean(x, [f"{s}_wind_u_kt" for s in ext])
    x["regional_wind_v_mean_kt"] = mean(x, [f"{s}_wind_v_kt" for s in ext])
    x["knyc_minus_regional_temp_mean_f"] = x.knyc_temp_f - x.regional_temp_mean
    x["knyc_minus_regional_dewpoint_mean_f"] = x.knyc_dewpoint_f - x.regional_dewpoint_mean
    x["knyc_minus_regional_mslp_mean_mb"] = x.knyc_mslp_mb - x.regional_mslp_mean
    for s in ext:
        x[f"knyc_minus_{s}_temp_f"] = x.knyc_temp_f - x[f"{s}_temp_f"]
        x[f"knyc_minus_{s}_dewpoint_f"] = x.knyc_dewpoint_f - x[f"{s}_dewpoint_f"]
    coastal_temp = mean(x, [f"{s}_temp_f" for s in coast])
    coastal_dew = mean(x, [f"{s}_dewpoint_f" for s in coast])
    coastal_p = mean(x, [f"{s}_mslp_mb" for s in coast])
    x["ewr_minus_coastal_temp_f"] = x.kewr_temp_f - coastal_temp
    x["ewr_minus_coastal_dewpoint_f"] = x.kewr_dewpoint_f - coastal_dew
    x["ewr_minus_coastal_mslp_mb"] = x.kewr_mslp_mb - coastal_p
    x["coastal_easterly_component_kt"] = -mean(x, [f"{s}_wind_u_kt" for s in coast])
    x["coastal_southerly_component_kt"] = mean(x, [f"{s}_wind_v_kt" for s in coast])
    for s in ["knyc", *ext]:
        for v, unit in [("temp_f", "f"), ("dewpoint_f", "f"), ("mslp_mb", "mb")]:
            x[f"{s}_{v.split('_')[0]}_change_1h_{unit}"] = x[f"{s}_{v}"].diff()
    x["regional_temp_change_1h_f"] = x.regional_temp_mean.diff()
    x["regional_dewpoint_change_1h_f"] = x.regional_dewpoint_mean.diff()
    x["regional_mslp_change_1h_mb"] = x.regional_mslp_mean.diff()
    x["knyc_minus_regional_warming_1h_f"] = x.knyc_temp_change_1h_f - x.regional_temp_change_1h_f
    x["regional_station_count_available"] = x[[f"{s}_temp_f" for s in STATIONS]].notna().sum(axis=1)
    x["regional_max_observation_age_min"] = x[[f"{s}_observation_age_min" for s in STATIONS]].max(axis=1)
    x["date_local"] = x.prediction_time_local.dt.date.astype(str)
    x["hour_local"] = x.prediction_time_local.dt.hour
    return x


def report(x: pd.DataFrame, raw: pd.DataFrame, start: date, end: date, tol: int) -> dict:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "requested_start": start.isoformat(), "requested_end": end.isoformat(),
        "hourly_rows": len(x), "raw_observation_rows": len(raw),
        "first_prediction_time_utc": str(x.prediction_time_utc.min()),
        "last_prediction_time_utc": str(x.prediction_time_utc.max()),
        "all_four_station_temp_fraction": round(float((x.regional_station_count_available == 4).mean()), 6),
        "at_least_three_station_temp_fraction": round(float((x.regional_station_count_available >= 3).mean()), 6),
        "temperature_missing_fraction": {s: round(float(x[f"{s}_temp_f"].isna().mean()), 6) for s in STATIONS},
        "max_observation_age_minutes": tol,
        "stations": STATIONS,
        "underlying_product": "NOAA/NWS/FAA ASOS/METAR",
        "transport": "Iowa Environmental Mesonet archive and real-time ingest",
        "asof_rule": "observation_time <= prediction_time; backward-only merge",
        "caveat": "Historical IEM does not preserve exact network ingest time; valid-time safety is enforced, but minute-level dissemination latency is not reconstructable.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--end-date", default=date.today().isoformat())
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    p.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--max-age-minutes", type=int, default=90)
    a = p.parse_args()
    start, end = _date(a.start_date), _date(a.end_date)
    if end < start or a.max_age_minutes <= 0:
        raise ValueError("Invalid date range or max age")
    station_raw = {pfx: fetch_station(stn, start, end) for pfx, stn in STATIONS.items()}
    raw = pd.concat([d.assign(station_key=pfx) for pfx, d in station_raw.items()], ignore_index=True)
    x = engineer(build_hourly(station_raw, start, end, a.max_age_minutes))
    for path in (a.output, a.raw_output, a.report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(a.output, index=False)
    raw.to_csv(a.raw_output, index=False)
    r = report(x, raw, start, end, a.max_age_minutes)
    a.report_output.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
