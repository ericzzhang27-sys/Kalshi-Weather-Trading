from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re
import subprocess
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
STRICT_SEVERITIES = {"P0", "P1"}
LOCAL_TIMEZONE = "America/New_York"


@dataclass(frozen=True)
class Finding:
    finding_id: str
    severity: str
    category: str
    summary: str
    details: str
    path: str = ""
    status: str = "open"


@dataclass(frozen=True)
class AuditResult:
    generated_at_utc: str
    repository_root: str
    git: dict[str, Any]
    environment: dict[str, str]
    artifact_hashes: dict[str, str]
    findings: list[Finding]
    checks: dict[str, Any]

    @property
    def strict_failure_count(self) -> int:
        return sum(
            finding.severity in STRICT_SEVERITIES and finding.status == "open"
            for finding in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["strict_failure_count"] = self.strict_failure_count
        return payload


def run_repository_audit(repo_root: str | Path) -> AuditResult:
    root = Path(repo_root).resolve()
    findings: list[Finding] = []
    checks: dict[str, Any] = {}

    git = _git_snapshot(root)
    environment = _environment_versions()
    artifact_hashes = _artifact_hashes(root)

    _audit_feature_and_model_contract(root, findings, checks)
    _audit_processed_data(root, findings, checks)
    _audit_probability_outputs(root, findings, checks)
    _audit_trading_safety(root, findings, checks)
    _audit_secrets(root, findings, checks)
    _audit_backtest_surface(root, findings, checks)

    findings.sort(key=lambda item: (SEVERITY_ORDER.get(item.severity, 99), item.finding_id))
    return AuditResult(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        repository_root=str(root),
        git=git,
        environment=environment,
        artifact_hashes=artifact_hashes,
        findings=findings,
        checks=checks,
    )


def write_audit_artifacts(result: AuditResult, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Audit output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    payload = result.to_dict()
    (destination / "audit_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    pd.DataFrame([asdict(item) for item in result.findings]).to_csv(
        destination / "findings.csv", index=False
    )
    (destination / "repository_integrity_report.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    (destination / "artifact_status.json").write_text(
        json.dumps(_artifact_status(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def default_audit_output_dir(repo_root: str | Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(repo_root) / "outputs" / "repository_audit" / run_id


def _finding(
    findings: list[Finding],
    finding_id: str,
    severity: str,
    category: str,
    summary: str,
    details: str,
    path: Path | str = "",
) -> None:
    findings.append(
        Finding(
            finding_id=finding_id,
            severity=severity,
            category=category,
            summary=summary,
            details=details,
            path=str(path),
        )
    )


def _git_snapshot(root: Path) -> dict[str, Any]:
    def call(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    return {
        "commit": call("rev-parse", "HEAD"),
        "branch": call("branch", "--show-current"),
        "status_porcelain": call("status", "--porcelain=v1").splitlines(),
    }


def _environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ["pandas", "numpy", "scikit-learn", "scipy", "ngboost", "PyYAML"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _artifact_hashes(root: Path) -> dict[str, str]:
    paths = [
        "config/model_config.yaml",
        "config/trading_config.yaml",
        "models/production_model_bundle.json",
        "outputs/final_feature_list.json",
        "data/processed/daily_clean.csv",
        "data/processed/hourly_clean.csv",
        "data/processed/modeling_rows_v1.csv",
    ]
    result: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if path.exists() and path.is_file():
            result[relative] = _sha256(path)
    return result


def _audit_feature_and_model_contract(
    root: Path, findings: list[Finding], checks: dict[str, Any]
) -> None:
    feature_path = root / "outputs/final_feature_list.json"
    bundle_path = root / "models/production_model_bundle.json"
    if not feature_path.exists() or not bundle_path.exists():
        _finding(
            findings,
            "MODEL_BUNDLE_MISSING",
            "P1",
            "model",
            "The authoritative model bundle or feature contract is missing",
            "Production scoring must resolve one manifest that pins both artifacts.",
            bundle_path,
        )
        return

    feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
    features = feature_payload.get("features", [])
    if not features or len(features) != len(set(features)):
        _finding(
            findings,
            "FEATURE_CONTRACT_INVALID",
            "P0",
            "model",
            "Feature contract is empty or contains duplicates",
            f"feature_count={len(features)}, unique_count={len(set(features))}",
            feature_path,
        )
        return
    if int(feature_payload.get("feature_count", -1)) != len(features):
        _finding(
            findings,
            "FEATURE_COUNT_MISMATCH",
            "P1",
            "model",
            "Feature contract count metadata is incorrect",
            f"declared={feature_payload.get('feature_count')}, actual={len(features)}",
            feature_path,
        )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model_path = (bundle_path.parent / str(bundle.get("model_path", ""))).resolve()
    bundle_feature_path = (
        bundle_path.parent / str(bundle.get("feature_list_path", ""))
    ).resolve()
    hashes = bundle.get("sha256", {})
    hash_checks: dict[str, bool] = {}
    for label, path in (("model", model_path), ("feature_list", bundle_feature_path)):
        expected = str(hashes.get(label, "")).lower()
        actual = _sha256(path) if path.exists() else "missing"
        hash_checks[label] = expected == actual
        if expected != actual:
            _finding(
                findings,
                f"MODEL_BUNDLE_{label.upper()}_HASH",
                "P0",
                "model",
                f"Model bundle {label} hash does not match",
                f"expected={expected or 'missing'}, actual={actual}",
                path,
            )

    source_hashes = bundle.get("source_sha256", {})
    source_paths = bundle.get("source_paths", {})
    default_source_paths = {
        "modeling_table": root / "data/processed/modeling_rows_v1.csv",
        "training_config": root / "config/model_config.yaml",
    }
    resolved_source_paths = {
        label: (
            (bundle_path.parent / str(source_paths[label])).resolve()
            if source_paths.get(label)
            else default_path.resolve()
        )
        for label, default_path in default_source_paths.items()
    }
    for label, path in resolved_source_paths.items():
        expected = str(source_hashes.get(label, "")).lower()
        actual = _sha256(path) if path.exists() else "missing"
        hash_checks[f"source_{label}"] = bool(expected and expected == actual)
        if not expected or expected != actual:
            _finding(
                findings,
                f"MODEL_BUNDLE_SOURCE_{label.upper()}_HASH",
                "P1",
                "reproducibility",
                f"Model bundle source hash does not match: {label}",
                f"expected={expected or 'missing'}, actual={actual}",
                path,
            )

    artifact_features: list[str] = []
    if model_path.exists() and hash_checks.get("model"):
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
        if isinstance(artifact, dict):
            artifact_features = [str(value) for value in artifact.get("feature_columns", [])]
        if artifact_features != features:
            _finding(
                findings,
                "MODEL_FEATURE_ORDER_MISMATCH",
                "P0",
                "model",
                "Model artifact and feature contract differ",
                f"model_count={len(artifact_features)}, contract_count={len(features)}",
                model_path,
            )
    checks["model_bundle"] = {
        "status": bundle.get("status"),
        "model_path": str(model_path),
        "feature_count": len(features),
        "hash_checks": hash_checks,
        "source_paths": {label: str(path) for label, path in resolved_source_paths.items()},
        "artifact_feature_count": len(artifact_features),
    }


def _audit_processed_data(
    root: Path, findings: list[Finding], checks: dict[str, Any]
) -> None:
    processed = root / "data/processed"
    summary: dict[str, Any] = {}
    table_specs = {
        "hourly_clean.csv": ["timestamp", "location"],
        "daily_clean.csv": ["date", "location"],
        "modeling_rows_v1.csv": ["date", "location", "prediction_time"],
    }
    for name, keys in table_specs.items():
        path = processed / name
        if not path.exists():
            _finding(
                findings,
                f"DATA_{name.upper().replace('.', '_')}_MISSING",
                "P1",
                "data",
                f"Required processed table is missing: {name}",
                "The pipeline cannot be reproduced without this table.",
                path,
            )
            continue
        df = pd.read_csv(path)
        missing = [key for key in keys if key not in df.columns]
        duplicates = int(df.duplicated(keys).sum()) if not missing else None
        summary[name] = {"rows": len(df), "columns": len(df.columns), "duplicates": duplicates}
        if missing:
            _finding(
                findings,
                f"DATA_{name.upper().replace('.', '_')}_SCHEMA",
                "P0",
                "data",
                f"{name} is missing canonical key columns",
                f"missing={missing}",
                path,
            )
        elif duplicates:
            _finding(
                findings,
                f"DATA_{name.upper().replace('.', '_')}_DUPLICATES",
                "P1",
                "data",
                f"{name} contains duplicate canonical keys",
                f"duplicate_rows={duplicates}, keys={keys}",
                path,
            )

        if name == "modeling_rows_v1.csv":
            _audit_modeling_rows(df, path, findings, summary[name])
        elif name == "hourly_clean.csv":
            _audit_hourly_rows(df, path, findings, summary[name])
    checks["processed_data"] = summary


def _audit_modeling_rows(
    df: pd.DataFrame,
    path: Path,
    findings: list[Finding],
    summary: dict[str, Any],
) -> None:
    if "prediction_timestamp" in df.columns:
        prediction_utc = _source_time_to_utc(df["prediction_timestamp"])
    elif "prediction_time" in df.columns and _contains_clock_only(df["prediction_time"]):
        _finding(
            findings,
            "PREDICTION_TIMESTAMP_INCOMPLETE",
            "P0",
            "data",
            "Clock-only prediction_time values lack a complete prediction_timestamp",
            "A full exchange-local date and time is required before UTC conversion.",
            path,
        )
        prediction_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    else:
        prediction_utc = _source_time_to_utc(df.get("prediction_time", pd.Series(index=df.index)))

    if {"actual_high", "forecast_high", "forecast_error"}.issubset(df.columns):
        delta = (
            pd.to_numeric(df["actual_high"], errors="coerce")
            - pd.to_numeric(df["forecast_high"], errors="coerce")
            - pd.to_numeric(df["forecast_error"], errors="coerce")
        )
        inconsistent = int((delta.abs() > 1e-8).fillna(True).sum())
        summary["target_inconsistent"] = inconsistent
        if inconsistent:
            _finding(
                findings,
                "TARGET_EQUATION_INVALID",
                "P0",
                "data",
                "Modeling targets violate actual_high - forecast_high = forecast_error",
                f"inconsistent_rows={inconsistent}",
                path,
            )

    future_counts: dict[str, int] = {}
    for column in [
        value
        for value in df.columns
        if value.endswith("_source_time") or value.endswith("_issue_time")
    ]:
        source = _source_time_to_utc(df[column])
        future = int((source.notna() & prediction_utc.notna() & (source > prediction_utc)).sum())
        future_counts[column] = future
        if future:
            _finding(
                findings,
                f"FUTURE_SOURCE_{column.upper()}",
                "P0",
                "leakage",
                f"{column} is later than prediction_time",
                f"future_rows={future}",
                path,
            )
    summary["future_source_rows"] = future_counts


def _audit_hourly_rows(
    df: pd.DataFrame,
    path: Path,
    findings: list[Finding],
    summary: dict[str, Any],
) -> None:
    station_values = sorted(df["station"].dropna().astype(str).unique()) if "station" in df else []
    summary["stations"] = station_values
    allowed_station_aliases = {"NYC", "KNYC"}
    unexpected = sorted(set(station_values) - allowed_station_aliases)
    metar_station_mismatch = 0
    if "nws_metar" in df.columns:
        metar = df["nws_metar"].dropna().astype(str).str.strip()
        metar_station_mismatch = int(
            (~metar.str.match(r"^(?:SPECI\s+)?KNYC\s", case=False)).sum()
        )
    summary["metar_station_mismatch"] = metar_station_mismatch
    if unexpected or metar_station_mismatch:
        _finding(
            findings,
            "HOURLY_STATION_SCOPE",
            "P1",
            "data",
            "Canonical hourly data contains unexpected stations",
            f"stations={station_values}, unexpected={unexpected}, "
            f"metar_station_mismatch={metar_station_mismatch}",
            path,
        )


def _audit_probability_outputs(
    root: Path, findings: list[Finding], checks: dict[str, Any]
) -> None:
    candidates = [
        root / "outputs/ngboost_bucket_probabilities_calibrated.csv",
        root / "outputs/final_bucket_probability_predictions.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        checks["probability_output"] = {"status": "not_present"}
        return
    df = pd.read_csv(path)
    probability_column = next(
        (name for name in ["probability", "model_probability"] if name in df), None
    )
    group_columns = [
        name
        for name in ["row_id", "date", "prediction_time", "prediction_timestamp"]
        if name in df
    ]
    if probability_column is None or not group_columns:
        _finding(
            findings,
            "PROBABILITY_SCHEMA_INVALID",
            "P1",
            "model",
            "Probability output lacks probability or prediction keys",
            f"columns={list(df.columns)}",
            path,
        )
        return
    values = pd.to_numeric(df[probability_column], errors="coerce")
    invalid = int((values.isna() | ~np.isfinite(values) | (values < 0) | (values > 1)).sum())
    sums = df.assign(_probability=values).groupby(group_columns, dropna=False)["_probability"].sum()
    bad_sums = int(((sums - 1.0).abs() > 1e-6).sum())
    if invalid or bad_sums:
        _finding(
            findings,
            "PROBABILITY_NORMALIZATION_INVALID",
            "P0",
            "model",
            "Saved bucket probabilities are invalid",
            f"invalid_values={invalid}, bad_prediction_sums={bad_sums}",
            path,
        )
    checks["probability_output"] = {
        "path": str(path),
        "rows": len(df),
        "invalid_values": invalid,
        "bad_prediction_sums": bad_sums,
    }


def _audit_trading_safety(
    root: Path, findings: list[Finding], checks: dict[str, Any]
) -> None:
    path = root / "config/trading_config.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    flags = {
        "mode": payload.get("mode"),
        "trading_enabled": payload.get("trading_enabled"),
        "live_auto_enabled": payload.get("live_auto_enabled"),
    }
    if flags != {"mode": "shadow", "trading_enabled": False, "live_auto_enabled": False}:
        _finding(
            findings,
            "LIVE_TRADING_NOT_FAIL_CLOSED",
            "P0",
            "live_safety",
            "Default trading configuration is not fully shadow-disabled",
            f"flags={flags}",
            path,
        )
    locations = payload.get("markets", {}).get("supported_locations", [])
    if locations != ["NYC"]:
        _finding(
            findings,
            "UNVALIDATED_CITY_SCOPE",
            "P1",
            "live_safety",
            "Production trading scope includes cities without validated model bundles",
            f"supported_locations={locations}",
            path,
        )
    risk = payload.get("risk", {})
    conservative_maxima = {
        "max_contracts_per_order": 1,
        "max_contracts_per_market": 5,
        "max_dollars_per_order": 5,
        "max_dollars_per_market": 5,
        "max_dollars_per_event": 10,
        "max_total_exposure": 50,
        "max_daily_loss_dollars": 20,
    }
    loosened = {
        key: risk.get(key)
        for key, ceiling in conservative_maxima.items()
        if not isinstance(risk.get(key), (int, float)) or float(risk[key]) > ceiling
    }
    if loosened:
        _finding(
            findings,
            "SHARED_RISK_LIMITS_LOOSENED",
            "P0",
            "live_safety",
            "Shared shadow/live limits exceed the conservative production envelope",
            f"loosened_or_missing={loosened}",
            path,
        )
    checks["trading_safety"] = {
        **flags,
        "supported_locations": locations,
        "risk_limits": {key: risk.get(key) for key in conservative_maxima},
        "loosened_risk_limits": loosened,
    }


def _audit_secrets(root: Path, findings: list[Finding], checks: dict[str, Any]) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    tracked = completed.stdout.splitlines()
    suspicious = [
        name
        for name in tracked
        if re.search(r"(^|/)(\.env($|\.)|.*private.*key|.*\.pem$|id_rsa$)", name, re.I)
        and not name.endswith(".env.example")
    ]
    if suspicious:
        _finding(
            findings,
            "TRACKED_SECRET_FILE",
            "P0",
            "security",
            "Potential credential or private-key files are tracked",
            f"files={suspicious}",
            root,
        )
    checks["tracked_secret_candidates"] = suspicious


def _audit_backtest_surface(
    root: Path, findings: list[Finding], checks: dict[str, Any]
) -> None:
    required = [
        root / "src/backtest/align_probabilities.py",
        root / "src/backtest/engine.py",
        root / "src/backtest/fees.py",
        root / "src/backtest/sizing.py",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        _finding(
            findings,
            "BACKTEST_INTEGRITY_SURFACE_MISSING",
            "P1",
            "backtest",
            "Causal backtest implementation is incomplete",
            f"missing={missing}",
            root / "src/backtest",
        )
    checks["backtest_surface"] = {"missing": missing}


def _source_time_to_utc(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(
            LOCAL_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward"
        )
    return parsed.dt.tz_convert("UTC")


def _contains_clock_only(values: pd.Series) -> bool:
    text = values.dropna().astype(str).str.strip()
    return bool(text.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?").any())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_markdown(result: AuditResult) -> str:
    counts = {
        severity: sum(item.severity == severity for item in result.findings)
        for severity in SEVERITY_ORDER
    }
    lines = [
        "# Repository Integrity Audit",
        "",
        f"Generated: `{result.generated_at_utc}`",
        f"Commit: `{result.git.get('commit', '')}`",
        f"Branch: `{result.git.get('branch', '')}`",
        f"Strict failures: **{result.strict_failure_count}**",
        "",
        "## Finding Summary",
        "",
        "| P0 | P1 | P2 | P3 |",
        "|---:|---:|---:|---:|",
        f"| {counts['P0']} | {counts['P1']} | {counts['P2']} | {counts['P3']} |",
        "",
        "## Findings",
        "",
    ]
    if not result.findings:
        lines.append("No open findings.")
    else:
        for item in result.findings:
            lines.extend(
                [
                    f"### {item.severity} — {item.finding_id}",
                    "",
                    item.summary,
                    "",
                    f"- Category: `{item.category}`",
                    f"- Path: `{item.path or 'n/a'}`",
                    f"- Details: {item.details}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Snapshot",
            "",
            f"- Dirty paths recorded: {len(result.git.get('status_porcelain', []))}",
            f"- Hashed authoritative artifacts: {len(result.artifact_hashes)}",
            "- Existing files were inspected read-only; audit outputs are immutable per run.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_status(result: AuditResult) -> dict[str, Any]:
    valid = result.strict_failure_count == 0
    backtest_root = Path(result.repository_root) / "outputs/backtests"
    corrected_runs = []
    if backtest_root.exists():
        for manifest_path in sorted(backtest_root.glob("*/run_manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("split_policy", {}).get("test_used_for_selection") is False:
                corrected_runs.append({
                    "run_id": manifest.get("run_id", manifest_path.parent.name),
                    "status": manifest.get("status", "unknown"),
                    "manifest_sha256": _sha256(manifest_path),
                })
    return {
        "generated_at_utc": result.generated_at_utc,
        "repository_commit": result.git.get("commit"),
        "production_model_bundle": "validated" if valid else "blocked",
        "processed_data": "validated" if valid else "blocked",
        "historical_backtest_outputs": (
            "corrected_diagnostic_available" if corrected_runs else "invalidated_pending_regeneration"
        ),
        "legacy_backtest_outputs": "invalidated_assumptions_failed",
        "corrected_backtest_runs": corrected_runs,
        "legacy_outputs": "legacy_unverified",
        "strict_failure_count": result.strict_failure_count,
    }
