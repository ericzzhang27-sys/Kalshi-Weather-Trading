from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ndfd_extract import (
    build_daily_high_forecast_archive,
    maxt_valid_time_to_target_date,
    normalize_target_lon,
    select_nearest_point,
    temperature_to_fahrenheit,
)
from src.data.ndfd_fetch import (
    CatalogEntry,
    _parse_aws_wmo_listing,
    build_aws_wmo_prefix,
    build_catalog_url,
    download_entries,
    infer_issue_hour,
    parse_catalog,
    write_manifest,
)


MINI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">
  <service name="HTTPServer" serviceType="HTTPServer" base="/thredds/fileServer/" />
  <dataset name="access/202306/20230606/">
    <dataset name="YAAZ98_KWBN_202306061851" ID="NDFD/access/202306/20230606/YAAZ98_KWBN_202306061851" urlPath="model-ndfd-file/access/202306/20230606/YAAZ98_KWBN_202306061851" />
  </dataset>
</catalog>
"""


MINI_HTML = """
<html><body>
<a href="/thredds/catalog/model-ndfd-file/access/202306/20230606/catalog.html?dataset=NDFD/access/202306/20230606/YAAZ98_KWBN_202306061851">YAAZ98_KWBN_202306061851</a>
</body></html>
"""


MINI_S3 = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>noaa-ndfd-pds</Name>
  <Prefix>wmo/maxt/2023/06/06/</Prefix>
  <Contents>
    <Key>wmo/maxt/2023/06/06/YGUZ98_KWBN_202306060052</Key>
    <Size>123</Size>
  </Contents>
</ListBucketResult>
"""


def test_build_catalog_url() -> None:
    assert (
        build_catalog_url("2023-06-01")
        == "https://www.ncei.noaa.gov/thredds/catalog/model-ndfd-file/access/202306/20230601/catalog.xml"
    )


def test_parse_xml_catalog_url_paths() -> None:
    entries = parse_catalog(
        MINI_XML,
        "https://www.ncei.noaa.gov/thredds/catalog/model-ndfd-file/access/202306/20230606/catalog.xml",
        "2023-06-06",
    )
    assert len(entries) == 1
    assert entries[0].filename == "YAAZ98_KWBN_202306061851"
    assert entries[0].center == "KWBN"
    assert entries[0].issue_time == "2023-06-06T18:51:00"
    assert entries[0].file_url.endswith("/thredds/fileServer/model-ndfd-file/access/202306/20230606/YAAZ98_KWBN_202306061851")


def test_parse_html_catalog_dataset_links() -> None:
    entries = parse_catalog(
        MINI_HTML,
        "https://www.ncei.noaa.gov/thredds/catalog/model-ndfd-file/access/202306/20230606/catalog.html",
        "2023-06-06",
    )
    assert len(entries) == 1
    assert entries[0].file_url.endswith("model-ndfd-file/access/202306/20230606/YAAZ98_KWBN_202306061851")


def test_build_aws_wmo_prefix_and_parse_listing() -> None:
    assert build_aws_wmo_prefix("2023-06-06", variable="maxt") == "wmo/maxt/2023/06/06/"

    entries, next_token = _parse_aws_wmo_listing(
        MINI_S3,
        "https://noaa-ndfd-pds.s3.amazonaws.com/?list-type=2&prefix=wmo/maxt/2023/06/06/",
        pd.Timestamp("2023-06-06").date(),
    )

    assert next_token is None
    assert len(entries) == 1
    assert entries[0].filename == "YGUZ98_KWBN_202306060052"
    assert entries[0].issue_time == "2023-06-06T00:52:00"
    assert entries[0].center == "KWBN"
    assert entries[0].file_url.endswith("/wmo/maxt/2023/06/06/YGUZ98_KWBN_202306060052")
    assert infer_issue_hour(entries[0].filename) == 0


def test_longitude_conversion() -> None:
    assert normalize_target_lon(-73.9665, np.array([280.0, 290.0])) == pytest.approx(286.0335)
    assert normalize_target_lon(-73.9665, np.array([-80.0, -70.0])) == pytest.approx(-73.9665)


def test_select_nearest_grid_point_with_synthetic_xarray() -> None:
    xr = pytest.importorskip("xarray")
    dataset = xr.Dataset(
        {"temp": (("latitude", "longitude"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={"latitude": [40.0, 41.0], "longitude": [285.0, 286.0]},
    )
    selected, point = select_nearest_point(dataset, 40.7812, -73.9665)
    assert point.grid_lat == pytest.approx(41.0)
    assert point.grid_lon == pytest.approx(286.0)
    assert float(selected["temp"].values) == pytest.approx(4.0)


def test_temperature_conversion() -> None:
    assert temperature_to_fahrenheit(300, "K") == pytest.approx(80.33)
    assert temperature_to_fahrenheit(20, "Celsius") == pytest.approx(68)
    assert temperature_to_fahrenheit(70, "F") == pytest.approx(70)
    assert temperature_to_fahrenheit(10, "m s**-1") is None


def test_maxt_valid_time_maps_to_new_york_target_date() -> None:
    assert maxt_valid_time_to_target_date("2023-06-07T00:00:00") == pd.Timestamp("2023-06-06")


def test_daily_high_archive_uses_local_target_date_for_maxt() -> None:
    point_df = pd.DataFrame(
        {
            "source": ["NDFD"],
            "station": ["KNYC"],
            "target_lat": [40.7812],
            "target_lon": [-73.9665],
            "grid_lat": [40.78],
            "grid_lon": [286.03],
            "grid_distance_km": [0.7],
            "forecast_issue_time": ["2023-06-06T00:52:00"],
            "valid_time": ["2023-06-07T00:00:00"],
            "variable": ["maxt"],
            "value_raw": [299.8],
            "units_raw": ["K"],
            "value_f": [79.97],
            "value_standardized": [79.97],
            "nws_forecast_high_f": [79.97],
            "nws_forecast_temp_f": [None],
            "file_url": ["https://example.test/YGUZ98_KWBN_202306060052"],
            "local_file": ["data/raw/ndfd/202306/20230606/YGUZ98_KWBN_202306060052"],
            "ingest_date": ["2026-06-07T00:00:00+00:00"],
            "extraction_status": ["extracted"],
        }
    )

    archive = build_daily_high_forecast_archive(point_df)

    assert len(archive) == 1
    assert archive.loc[0, "date"] == pd.Timestamp("2023-06-06")
    assert archive.loc[0, "location"] == "NYC"
    assert archive.loc[0, "forecast_high"] == pytest.approx(79.97)
    assert archive.loc[0, "forecast_source"] == "nws_ndfd_historical_forecast"


def test_manifest_writing(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    entry = CatalogEntry(
        date="2023-06-06",
        catalog_url="catalog",
        file_url="file",
        filename="YAAZ98_KWBN_202306061851",
    )
    write_manifest([entry], path)
    df = pd.read_csv(path)
    assert list(df.columns)[:4] == ["date", "catalog_url", "file_url", "filename"]
    assert df.loc[0, "filename"] == "YAAZ98_KWBN_202306061851"


def test_dry_run_mode_sets_local_path_without_download(tmp_path: Path) -> None:
    entry = CatalogEntry(
        date="2023-06-06",
        catalog_url="catalog",
        file_url="https://example.test/file.grib2",
        filename="file.grib2",
    )
    result = download_entries([entry], cache_dir=tmp_path, dry_run=True)[0]
    assert result.download_status == "dry_run"
    assert result.local_path is not None
    assert result.local_path.endswith("202306/20230606/file.grib2") or result.local_path.endswith("202306\\20230606\\file.grib2")
    assert not Path(result.local_path).exists()
