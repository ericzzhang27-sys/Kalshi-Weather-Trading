from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Mapping

from .interfaces import ExperimentRecord, canonical_json_hash


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_artifacts(paths: Iterable[str | Path], destination: str | Path, *, label: str) -> dict[str, Any]:
    files = []
    for raw_path in sorted((Path(item).resolve() for item in paths), key=str):
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        files.append({"path": str(raw_path), "size": raw_path.stat().st_size, "sha256": sha256_file(raw_path)})
    manifest = {
        "schema_version": 1,
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest["content_hash"] = canonical_json_hash(files)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("content_hash") != manifest["content_hash"]:
            raise FileExistsError(f"immutable manifest already exists with different content: {target}")
        return existing
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


class ExperimentRegistry:
    def __init__(self, database_path: str | Path, manifest_dir: str | Path) -> None:
        self.database_path = Path(database_path)
        self.manifest_dir = Path(manifest_dir)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    family TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evidence_label TEXT NOT NULL,
                    promotion_decision TEXT NOT NULL,
                    record_hash TEXT NOT NULL UNIQUE,
                    manifest_path TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trials (
                    experiment_id TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    elapsed_seconds REAL NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    failure_reason TEXT,
                    PRIMARY KEY (experiment_id, trial_number),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );
                """
            )

    def register(self, record: ExperimentRecord) -> Path:
        payload = record.as_json_dict()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        manifest_path = self.manifest_dir / f"{record.experiment_id}.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if canonical_json_hash(existing) != record.record_hash:
                raise FileExistsError(f"experiment manifest is immutable: {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.experiment_id,
                    record.created_at_utc.isoformat(),
                    record.family,
                    record.status,
                    record.evidence_label,
                    record.promotion_decision,
                    record.record_hash,
                    str(manifest_path.resolve()),
                    payload_json,
                ),
            )
        return manifest_path

    def register_trial(
        self,
        experiment_id: str,
        trial_number: int,
        *,
        seed: int,
        state: str,
        elapsed_seconds: float,
        params: Mapping[str, Any],
        metrics: Mapping[str, Any],
        failure_reason: str | None = None,
    ) -> None:
        if state == "failed" and not failure_reason:
            raise ValueError("failed trials require a failure reason")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    int(trial_number),
                    int(seed),
                    state,
                    float(elapsed_seconds),
                    json.dumps(dict(params), sort_keys=True, default=str),
                    json.dumps(dict(metrics), sort_keys=True, default=str),
                    failure_reason,
                ),
            )

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT experiment_id, created_at_utc, family, status, evidence_label, promotion_decision, record_hash, manifest_path FROM experiments ORDER BY created_at_utc"
            ).fetchall()
        return [dict(row) for row in rows]
