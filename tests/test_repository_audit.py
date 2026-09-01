from __future__ import annotations

import json
from pathlib import Path

from src.repository_audit import AuditResult, Finding, write_audit_artifacts


def _result(findings: list[Finding]) -> AuditResult:
    return AuditResult(
        generated_at_utc="2026-01-01T00:00:00+00:00",
        repository_root="/repo",
        git={"commit": "abc", "branch": "main", "status_porcelain": []},
        environment={},
        artifact_hashes={},
        findings=findings,
        checks={},
    )


def test_strict_failure_count_only_counts_open_p0_p1() -> None:
    result = _result(
        [
            Finding("A", "P0", "data", "bad", "details"),
            Finding("B", "P1", "model", "fixed", "details", status="resolved"),
            Finding("C", "P2", "docs", "minor", "details"),
        ]
    )
    assert result.strict_failure_count == 1


def test_audit_outputs_are_immutable(tmp_path: Path) -> None:
    destination = tmp_path / "audit"
    write_audit_artifacts(_result([]), destination)
    payload = json.loads((destination / "audit_report.json").read_text())
    assert payload["strict_failure_count"] == 0
    assert (destination / "repository_integrity_report.md").exists()

    try:
        write_audit_artifacts(_result([]), destination)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Audit writer overwrote an existing run")
