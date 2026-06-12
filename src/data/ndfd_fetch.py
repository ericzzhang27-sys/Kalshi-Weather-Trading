from __future__ import annotations

import csv
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse
from xml.etree import ElementTree

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)

NCEI_THREDDS_ROOT = "https://www.ncei.noaa.gov/thredds"
NDFD_DATASET_ROOT = "model-ndfd-file"
AWS_NDFD_BUCKET_ROOT = "https://noaa-ndfd-pds.s3.amazonaws.com"
DEFAULT_CACHE_DIR = Path("data/raw/ndfd")
REQUEST_TIMEOUT = (8, 20)
MANIFEST_COLUMNS = [
    "date",
    "catalog_url",
    "file_url",
    "filename",
    "inferred_variable",
    "issue_time",
    "center",
    "local_path",
    "download_status",
    "skip_reason",
]

VARIABLE_ALIASES = {
    "maxt": {"maxt", "maximum temperature", "max temperature", "tmax"},
    "temp": {"temp", "temperature", "2 metre temperature", "2 m temperature", "t"},
    "sky": {"sky", "sky cover", "total cloud cover", "cloud cover", "tcc"},
    "wspd": {"wspd", "wind speed", "10 metre wind speed", "si10"},
    "wdir": {"wdir", "wind direction", "10 metre wind direction"},
    "pop12": {"pop12", "probability of precipitation", "precipitation probability", "pop"},
}


@dataclass
class CatalogEntry:
    date: str
    catalog_url: str
    file_url: str
    filename: str
    inferred_variable: str | None = None
    issue_time: str | None = None
    center: str | None = None
    local_path: str | None = None
    download_status: str = "discovered"
    skip_reason: str | None = None


class _CatalogHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start_date: str | date, end_date: str | date, limit_days: int | None = None) -> Iterable[date]:
    current = parse_date(start_date)
    end = parse_date(end_date)
    if end < current:
        raise ValueError("end_date must be on or after start_date")

    emitted = 0
    while current <= end:
        if limit_days is not None and emitted >= limit_days:
            break
        yield current
        emitted += 1
        current += timedelta(days=1)


def build_catalog_url(day: str | date, archive_segment: str = "access", xml: bool = True) -> str:
    parsed = parse_date(day)
    yyyymm = parsed.strftime("%Y%m")
    yyyymmdd = parsed.strftime("%Y%m%d")
    suffix = "catalog.xml" if xml else "catalog.html"
    segment = archive_segment.strip("/")
    parts = [NCEI_THREDDS_ROOT, "catalog", NDFD_DATASET_ROOT]
    if segment:
        parts.append(segment)
    parts.extend([yyyymm, yyyymmdd, suffix])
    return "/".join(parts)


def candidate_catalog_urls(day: str | date) -> list[str]:
    return [
        build_catalog_url(day, "access", xml=True),
        build_catalog_url(day, "", xml=True),
        build_catalog_url(day, "historical", xml=True),
    ]


def create_retry_session(total_retries: int = 1) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Kalshi-Weather-Trading NDFD pipeline"})
    return session


def infer_issue_time(filename: str) -> str | None:
    match = re.search(r"_(\d{12})$", filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M").isoformat()


def infer_center(filename: str) -> str | None:
    match = re.search(r"_([A-Z]{4})_\d{12}$", filename)
    return match.group(1) if match else None


def infer_issue_hour(filename: str) -> int | None:
    issue_time = infer_issue_time(filename)
    if issue_time is None:
        return None
    return datetime.fromisoformat(issue_time).hour


def infer_variable_from_text(text: str) -> str | None:
    normalised = re.sub(r"[_\-.]+", " ", text.lower())
    for canonical, aliases in VARIABLE_ALIASES.items():
        if any(alias in normalised for alias in aliases):
            return canonical
    return None


def _file_url_from_url_path(url_path: str) -> str:
    return f"{NCEI_THREDDS_ROOT}/fileServer/{url_path.lstrip('/')}"


def _entry_from_url_path(url_path: str, catalog_url: str, day: date) -> CatalogEntry:
    filename = Path(urlparse(url_path).path).name
    return CatalogEntry(
        date=day.isoformat(),
        catalog_url=catalog_url,
        file_url=_file_url_from_url_path(url_path),
        filename=filename,
        inferred_variable=infer_variable_from_text(filename),
        issue_time=infer_issue_time(filename),
        center=infer_center(filename),
    )


def _parse_xml_catalog(content: str, catalog_url: str, day: date) -> list[CatalogEntry]:
    root = ElementTree.fromstring(content)
    entries: list[CatalogEntry] = []
    for dataset in root.iter():
        url_path = dataset.attrib.get("urlPath")
        if not url_path:
            continue
        entries.append(_entry_from_url_path(url_path, catalog_url, day))
    return entries


def _parse_html_catalog(content: str, catalog_url: str, day: date) -> list[CatalogEntry]:
    parser = _CatalogHTMLParser()
    parser.feed(content)
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        href = unquote(href)
        file_url: str | None = None
        if "/thredds/fileServer/" in href:
            file_url = urljoin(catalog_url, href)
        elif "urlPath=" in href:
            match = re.search(r"urlPath=([^&\"']+)", href)
            if match:
                file_url = _file_url_from_url_path(match.group(1))
        elif "dataset=" in href:
            match = re.search(r"dataset=([^&\"']+)", href)
            if match:
                dataset_id = match.group(1)
                for prefix in ("NDFD/", "NDFD_kwbn-old/"):
                    if dataset_id.startswith(prefix):
                        dataset_id = dataset_id[len(prefix) :]
                file_url = _file_url_from_url_path(f"{NDFD_DATASET_ROOT}/{dataset_id}")

        if not file_url or file_url in seen or file_url.endswith("/"):
            continue
        seen.add(file_url)
        filename = Path(urlparse(file_url).path).name
        entries.append(
            CatalogEntry(
                date=day.isoformat(),
                catalog_url=catalog_url,
                file_url=file_url,
                filename=filename,
                inferred_variable=infer_variable_from_text(filename),
                issue_time=infer_issue_time(filename),
                center=infer_center(filename),
            )
        )
    return entries


def parse_catalog(content: str, catalog_url: str, day: str | date) -> list[CatalogEntry]:
    parsed_day = parse_date(day)
    stripped = content.lstrip()
    if stripped.startswith("<?xml") or stripped.startswith("<catalog"):
        return _parse_xml_catalog(content, catalog_url, parsed_day)
    return _parse_html_catalog(content, catalog_url, parsed_day)


def build_aws_wmo_prefix(day: str | date, variable: str = "maxt") -> str:
    parsed_day = parse_date(day)
    variable_path = variable.strip().lower().strip("/")
    if not variable_path:
        raise ValueError("variable must not be empty")
    return f"wmo/{variable_path}/{parsed_day:%Y/%m/%d}/"


def _xml_text(node: ElementTree.Element, name: str) -> str | None:
    namespace = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    child = node.find(f"{namespace}{name}")
    if child is None:
        child = node.find(name)
    return child.text if child is not None else None


def _parse_aws_wmo_listing(content: str, listing_url: str, day: date) -> tuple[list[CatalogEntry], str | None]:
    root = ElementTree.fromstring(content)
    entries: list[CatalogEntry] = []
    for contents in root.findall("{http://s3.amazonaws.com/doc/2006-03-01/}Contents") or root.findall("Contents"):
        key = _xml_text(contents, "Key")
        if not key or key.endswith("/"):
            continue
        filename = Path(key).name
        entries.append(
            CatalogEntry(
                date=day.isoformat(),
                catalog_url=listing_url,
                file_url=f"{AWS_NDFD_BUCKET_ROOT}/{key}",
                filename=filename,
                inferred_variable=infer_variable_from_text(key),
                issue_time=infer_issue_time(filename),
                center=infer_center(filename),
            )
        )

    next_token = _xml_text(root, "NextContinuationToken")
    return entries, next_token


def fetch_aws_wmo_entries(
    day: str | date,
    variable: str = "maxt",
    wmo_prefixes: Iterable[str] | None = None,
    issue_hours: Iterable[int] | None = None,
    session: requests.Session | None = None,
) -> list[CatalogEntry]:
    session = session or create_retry_session(total_retries=2)
    parsed_day = parse_date(day)
    object_prefix = build_aws_wmo_prefix(parsed_day, variable=variable)
    allowed_wmo_prefixes = {
        prefix.strip().upper()
        for prefix in wmo_prefixes or []
        if prefix and prefix.strip()
    }
    allowed_issue_hours = {
        int(hour)
        for hour in issue_hours or []
        if 0 <= int(hour) <= 23
    }

    entries: list[CatalogEntry] = []
    continuation_token: str | None = None
    listing_url = f"{AWS_NDFD_BUCKET_ROOT}/?list-type=2&prefix={object_prefix}"
    while True:
        params = {"list-type": "2", "prefix": object_prefix, "max-keys": "1000"}
        if continuation_token:
            params["continuation-token"] = continuation_token
        try:
            response = session.get(AWS_NDFD_BUCKET_ROOT, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return [_missing_catalog_entry(parsed_day, listing_url, "aws_prefix_not_found")]
            response.raise_for_status()
            page_entries, continuation_token = _parse_aws_wmo_listing(response.text, listing_url, parsed_day)
        except requests.RequestException as exc:
            LOGGER.warning("Failed to fetch NDFD AWS listing %s: %s", object_prefix, exc)
            return [_missing_catalog_entry(parsed_day, listing_url, f"aws_listing_error: {exc}")]
        except ElementTree.ParseError as exc:
            LOGGER.warning("Failed to parse NDFD AWS listing %s: %s", object_prefix, exc)
            return [_missing_catalog_entry(parsed_day, listing_url, f"aws_listing_parse_error: {exc}")]

        for entry in page_entries:
            filename_prefix = entry.filename.split("_", 1)[0].upper()
            if allowed_wmo_prefixes and filename_prefix not in allowed_wmo_prefixes:
                continue
            issue_hour = infer_issue_hour(entry.filename)
            if allowed_issue_hours and issue_hour not in allowed_issue_hours:
                continue
            entries.append(entry)

        if not continuation_token:
            break

    if not entries:
        return [
            _missing_catalog_entry(
                parsed_day,
                listing_url,
                "no AWS WMO files matched variable/prefix/hour filters",
            )
        ]
    return entries


def _missing_catalog_entry(day: date, catalog_url: str, reason: str | None) -> CatalogEntry:
    return CatalogEntry(
        date=day.isoformat(),
        catalog_url=catalog_url,
        file_url="",
        filename="",
        download_status="missing_catalog",
        skip_reason=reason or "no catalog entries discovered",
    )


def fetch_catalog_entries(
    day: str | date,
    session: requests.Session | None = None,
    catalog_urls: Iterable[str] | None = None,
) -> list[CatalogEntry]:
    session = session or create_retry_session()
    last_error: str | None = None
    parsed_day = parse_date(day)
    urls = list(catalog_urls or candidate_catalog_urls(parsed_day))

    for catalog_url in urls:
        try:
            response = session.get(catalog_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                last_error = "catalog_not_found"
                continue
            response.raise_for_status()
            entries = parse_catalog(response.text, catalog_url, parsed_day)
            if entries:
                return entries
            last_error = "catalog_empty"
        except requests.RequestException as exc:
            last_error = f"catalog_error: {exc}"
            LOGGER.warning("Failed to fetch NDFD catalog %s: %s", catalog_url, exc)
        except ElementTree.ParseError as exc:
            last_error = f"catalog_parse_error: {exc}"
            LOGGER.warning("Failed to parse NDFD catalog %s: %s", catalog_url, exc)

    return [_missing_catalog_entry(parsed_day, urls[0] if urls else build_catalog_url(parsed_day), last_error)]


def local_cache_path(entry: CatalogEntry, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    parsed_day = parse_date(entry.date)
    return cache_dir / parsed_day.strftime("%Y%m") / parsed_day.strftime("%Y%m%d") / entry.filename


def filter_entries(
    entries: Iterable[CatalogEntry],
    variables: Iterable[str] | None = None,
    centers: Iterable[str] = ("KWBN",),
) -> list[CatalogEntry]:
    requested = {variable.strip().lower() for variable in variables or [] if variable.strip()}
    allowed_centers = {center.strip().upper() for center in centers if center.strip()}
    filtered: list[CatalogEntry] = []

    for entry in entries:
        if not entry.file_url:
            filtered.append(entry)
            continue
        if allowed_centers and entry.center and entry.center.upper() not in allowed_centers:
            entry.download_status = "skipped"
            entry.skip_reason = f"center {entry.center} not in {sorted(allowed_centers)}"
            continue
        if requested and entry.inferred_variable and entry.inferred_variable not in requested:
            entry.download_status = "skipped"
            entry.skip_reason = f"inferred variable {entry.inferred_variable} not requested"
            continue
        filtered.append(entry)

    return filtered


def discover_range(
    start_date: str | date,
    end_date: str | date,
    variables: Iterable[str] | None = None,
    limit_days: int | None = None,
    session: requests.Session | None = None,
) -> list[CatalogEntry]:
    session = session or create_retry_session()
    entries: list[CatalogEntry] = []
    for day in iter_dates(start_date, end_date, limit_days=limit_days):
        day_entries = fetch_catalog_entries(day, session=session)
        entries.extend(filter_entries(day_entries, variables=variables))
    return entries


def discover_aws_wmo_range(
    start_date: str | date,
    end_date: str | date,
    variable: str = "maxt",
    wmo_prefixes: Iterable[str] | None = None,
    issue_hours: Iterable[int] | None = None,
    limit_days: int | None = None,
    session: requests.Session | None = None,
) -> list[CatalogEntry]:
    session = session or create_retry_session(total_retries=2)
    entries: list[CatalogEntry] = []
    for day in iter_dates(start_date, end_date, limit_days=limit_days):
        entries.extend(
            fetch_aws_wmo_entries(
                day,
                variable=variable,
                wmo_prefixes=wmo_prefixes,
                issue_hours=issue_hours,
                session=session,
            )
        )
    return entries


def download_entry(
    entry: CatalogEntry,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    session: requests.Session | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> CatalogEntry:
    if not entry.file_url:
        return entry

    local_path = local_cache_path(entry, cache_dir=cache_dir)
    entry.local_path = str(local_path)

    if dry_run:
        entry.download_status = "dry_run"
        return entry

    if local_path.exists() and local_path.stat().st_size > 0 and not overwrite:
        entry.download_status = "cached"
        return entry

    session = session or create_retry_session()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(entry.file_url, timeout=REQUEST_TIMEOUT, stream=True) as response:
            response.raise_for_status()
            with local_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        output.write(chunk)
        entry.download_status = "downloaded"
    except requests.RequestException as exc:
        entry.download_status = "failed"
        entry.skip_reason = str(exc)
        LOGGER.warning("Failed to download %s: %s", entry.file_url, exc)
    return entry


def download_entries(
    entries: Iterable[CatalogEntry],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
    dry_run: bool = False,
    session: requests.Session | None = None,
) -> list[CatalogEntry]:
    session = session or create_retry_session()
    return [
        download_entry(
            entry,
            cache_dir=cache_dir,
            session=session,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        for entry in entries
    ]


def write_manifest(entries: Iterable[CatalogEntry], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(entry) for entry in entries]
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in MANIFEST_COLUMNS})
    return output_path


def write_missing_dates_report(entries: Iterable[CatalogEntry], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "date": entry.date,
            "catalog_url": entry.catalog_url,
            "reason": entry.skip_reason,
        }
        for entry in entries
        if entry.download_status == "missing_catalog"
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["date", "catalog_url", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def manifest_counts(entries: Iterable[CatalogEntry]) -> dict[str, int]:
    materialized = list(entries)
    return {
        "files_discovered": sum(1 for entry in materialized if bool(entry.file_url)),
        "downloaded": sum(1 for entry in materialized if entry.download_status == "downloaded"),
        "cached": sum(1 for entry in materialized if entry.download_status == "cached"),
        "failed": sum(1 for entry in materialized if entry.download_status == "failed"),
        "missing_catalog": sum(1 for entry in materialized if entry.download_status == "missing_catalog"),
    }
