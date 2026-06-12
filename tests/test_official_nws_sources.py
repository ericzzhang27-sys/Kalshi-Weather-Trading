from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features import add_observed_weather_features
from src.target_builder import build_daily_forecast_error_targets
from src.weather_data import load_nws_hourly_observations, load_official_daily_highs


def test_official_daily_loader_prefers_central_park_tmax(tmp_path: Path) -> None:
    (tmp_path / "other_daily.csv").write_text(
        "STATION,NAME,DATE,TMAX,TMIN\n"
        "USW00000000,OTHER STATION,2026-06-01,77,60\n",
        encoding="utf-8",
    )
    (tmp_path / "central_park.csv").write_text(
        "STATION,NAME,DATE,AWND,TMAX,TMIN\n"
        "USW00094728,\"NY CITY CENTRAL PARK, NY US\",2026-06-01,3.1,81,65\n"
        "USW00094728,\"NY CITY CENTRAL PARK, NY US\",2026-06-02,4.2,79,63\n",
        encoding="utf-8",
    )

    loaded = load_official_daily_highs(tmp_path)

    assert loaded["official_daily_high_f"].tolist() == [81.0, 79.0]
    assert loaded["actual_high"].tolist() == [81.0, 79.0]
    assert loaded["actual_source"].unique().tolist() == ["noaa_nws_daily_tmax"]
    assert loaded["source_station"].unique().tolist() == ["USW00094728"]


def test_official_daily_loader_rejects_unresolved_duplicate_dates(tmp_path: Path) -> None:
    (tmp_path / "central_park.csv").write_text(
        "STATION,NAME,DATE,TMAX,TMIN\n"
        "USW00094728,\"NY CITY CENTRAL PARK, NY US\",2026-06-01,81,65\n"
        "USW00094728,\"NY CITY CENTRAL PARK, NY US\",2026-06-01,82,65\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved duplicate"):
        load_official_daily_highs(tmp_path)


def test_nws_hourly_loader_maps_asos_columns_and_trace_precip(tmp_path: Path) -> None:
    (tmp_path / "NYC_nws_hourly.csv").write_text(
        "station,valid,tmpf,dwpf,relh,drct,sknt,gust,alti,mslp,p01i,skyc1,skyc2,skyc3,metar\n"
        "NYC,2026-06-01 08:51,70,60,70,180,5,M,29.92,1012.3,T,SCT,BKN,M,KNYC sample\n"
        "NYC,2026-06-01 09:12,72,61,69,190,6,12,29.91,M,0.01,BKN,M,M,KNYC special\n",
        encoding="utf-8",
    )

    loaded = load_nws_hourly_observations(tmp_path)

    assert loaded["timestamp"].tolist() == [
        pd.Timestamp("2026-06-01 08:51"),
        pd.Timestamp("2026-06-01 09:12"),
    ]
    assert loaded["nws_current_temp_f"].tolist() == [70.0, 72.0]
    assert loaded["temperature_2m"].tolist() == [70.0, 72.0]
    assert loaded["nws_precip_1h"].tolist() == [0.0, 0.01]
    assert loaded["nws_cloud_cover_pct"].tolist() == [40.0, 75.0]
    assert loaded["nws_metar"].str.contains("KNYC").all()


def test_target_builder_uses_official_high_over_openmeteo_actual_column() -> None:
    daily = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "location": ["NYC"],
            "actual_high": [75.0],
            "official_daily_high_f": [81.0],
            "actual_source": ["noaa_nws_daily_tmax"],
        }
    )
    forecast = pd.DataFrame(
        {
            "date": ["2026-06-01"],
            "location": ["NYC"],
            "forecast_high": [78.0],
        }
    )

    targets = build_daily_forecast_error_targets(daily, forecast)

    assert targets.loc[0, "actual_high"] == 81.0
    assert targets.loc[0, "official_daily_high_f"] == 81.0
    assert targets.loc[0, "forecast_error"] == 3.0


def test_nws_max_temp_so_far_uses_only_observations_at_or_before_prediction_time() -> None:
    rows = pd.DataFrame(
        {
            "target_date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
            "location": ["NYC", "NYC"],
            "prediction_time": pd.to_datetime(["2026-06-01 09:00", "2026-06-01 10:00"]),
            "actual_high": [95.0, 95.0],
            "forecast_high": [90.0, 90.0],
            "forecast_error": [5.0, 5.0],
        }
    )
    hourly = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-01 08:51", "2026-06-01 09:51", "2026-06-01 10:51"]
            ),
            "date": pd.to_datetime(["2026-06-01"] * 3),
            "target_date": pd.to_datetime(["2026-06-01"] * 3),
            "location": ["NYC"] * 3,
            "nws_current_temp_f": [70.0, 85.0, 95.0],
            "nws_dew_point_f": [60.0, 61.0, 62.0],
        }
    )

    featured = add_observed_weather_features(rows, hourly)

    assert featured["nws_current_temp_f"].tolist() == [70.0, 85.0]
    assert featured["nws_max_temp_so_far_f"].tolist() == [70.0, 85.0]
    assert pd.to_datetime(featured["max_temp_so_far_source_time"]).tolist() == [
        pd.Timestamp("2026-06-01 08:51"),
        pd.Timestamp("2026-06-01 09:51"),
    ]
