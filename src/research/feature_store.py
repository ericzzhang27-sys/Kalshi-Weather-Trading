from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

import pandas as pd

from .interfaces import canonical_json_hash


KEY_COLUMNS = (
    "target_date",
    "as_of_utc",
    "location",
    "forecast_issue_utc",
    "valid_time_utc",
    "station_or_grid",
    "source",
    "source_version",
)


def _iso_utc(value: object, name: str) -> str:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.tz_convert("UTC").isoformat()


class PointInTimeFeatureStore:
    """Immutable feature snapshots with strict issue/as-of availability checks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    target_date TEXT NOT NULL,
                    as_of_utc TEXT NOT NULL,
                    location TEXT NOT NULL,
                    forecast_issue_utc TEXT NOT NULL,
                    valid_time_utc TEXT NOT NULL,
                    station_or_grid TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL,
                    PRIMARY KEY (
                        target_date, as_of_utc, location, forecast_issue_utc,
                        valid_time_utc, station_or_grid, source, source_version
                    )
                );
                CREATE INDEX IF NOT EXISTS feature_asof_lookup
                ON feature_snapshots(target_date, location, source, as_of_utc, forecast_issue_utc);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def insert(self, row: Mapping[str, Any], features: Mapping[str, Any]) -> str:
        missing = [name for name in KEY_COLUMNS if name not in row]
        if missing:
            raise ValueError(f"feature snapshot missing key fields: {missing}")
        target_date = pd.Timestamp(row["target_date"]).date().isoformat()
        as_of = _iso_utc(row["as_of_utc"], "as_of_utc")
        issue = _iso_utc(row["forecast_issue_utc"], "forecast_issue_utc")
        valid = _iso_utc(row["valid_time_utc"], "valid_time_utc")
        if pd.Timestamp(issue) > pd.Timestamp(as_of):
            raise ValueError("forecast issue time cannot be later than the feature as-of time")
        if str(row["location"]).upper() != "NYC":
            raise ValueError("point-in-time feature store is scoped to NYC")
        payload = dict(features)
        content_hash = canonical_json_hash(payload)
        values = (
            target_date,
            as_of,
            "NYC",
            issue,
            valid,
            str(row["station_or_grid"]),
            str(row["source"]),
            str(row["source_version"]),
            content_hash,
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
            datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as connection:
            existing = connection.execute(
                """SELECT content_hash FROM feature_snapshots WHERE
                target_date=? AND as_of_utc=? AND location=? AND forecast_issue_utc=?
                AND valid_time_utc=? AND station_or_grid=? AND source=? AND source_version=?""",
                values[:8],
            ).fetchone()
            if existing is not None:
                if existing["content_hash"] != content_hash:
                    raise ValueError("source revision attempted to overwrite an immutable point-in-time snapshot")
                return content_hash
            connection.execute("INSERT INTO feature_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
            connection.commit()
        return content_hash

    def as_of(
        self,
        *,
        target_date: object,
        as_of_utc: object,
        location: str = "NYC",
        sources: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        cutoff = _iso_utc(as_of_utc, "as_of_utc")
        target = pd.Timestamp(target_date).date().isoformat()
        query = """SELECT * FROM feature_snapshots
                   WHERE target_date=? AND location=? AND as_of_utc<=? AND forecast_issue_utc<=?"""
        params: list[Any] = [target, location.upper(), cutoff, cutoff]
        source_list = list(sources or [])
        if source_list:
            query += " AND source IN (" + ",".join("?" for _ in source_list) + ")"
            params.extend(source_list)
        query += " ORDER BY source, station_or_grid, valid_time_utc, as_of_utc DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            key = (row["source"], row["station_or_grid"], row["valid_time_utc"])
            if key in seen:
                continue
            seen.add(key)
            item = dict(row)
            item["features"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def integrity_report(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT as_of_utc, forecast_issue_utc, payload_json, content_hash FROM feature_snapshots"
            ).fetchall()
        bad_hashes = 0
        future_issues = 0
        for row in rows:
            payload = json.loads(row["payload_json"])
            bad_hashes += canonical_json_hash(payload) != row["content_hash"]
            future_issues += pd.Timestamp(row["forecast_issue_utc"]) > pd.Timestamp(row["as_of_utc"])
        return {"rows": len(rows), "bad_hashes": int(bad_hashes), "future_issue_times": int(future_issues), "passed": not bad_hashes and not future_issues}
