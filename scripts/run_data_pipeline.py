"""Run the full offline data production pipeline in one command.

Chains the four canonical data-production stages in order and fails closed:
any stage that exits nonzero stops the pipeline immediately.

Stages
------
1. Clean + audit raw data          scripts/run_day6_data_verification.py
2. Build supervised targets        scripts/build_day7_supervised_table.py
3. Build timestamp-safe features   scripts/build_features.py
4. Verify feature integrity        scripts/verify_feature_integrity.py

A Markdown run report is written to outputs/reports/data_pipeline_run.md.

Usage (from the repository root):

    python scripts/run_data_pipeline.py [--skip-stage STAGE_NAME]

Stage names for --skip-stage: day6_verification, day7_targets,
day8_features, feature_integrity.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "outputs" / "reports" / "data_pipeline_run.md"

NL = chr(10)

STAGES = [
    ("day6_verification", "Clean + audit raw data", "scripts/run_day6_data_verification.py"),
    ("day7_targets", "Build supervised forecast-error targets", "scripts/build_day7_supervised_table.py"),
    ("day8_features", "Build timestamp-safe modeling features", "scripts/build_features.py"),
    ("feature_integrity", "Verify feature provenance/integrity", "scripts/verify_feature_integrity.py"),
]


def _run_stage(script_rel_path: str) -> tuple[bool, float, str]:
    """Run one stage script; return (success, elapsed_seconds, tail_of_output)."""
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, script_rel_path],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    combined = (process.stdout or "") + NL + (process.stderr or "")
    tail_lines = [line for line in combined.strip().splitlines() if line.strip()][-15:]
    return process.returncode == 0, elapsed, NL.join(tail_lines)


def _write_report(rows: list[dict[str, object]], skipped: list[str], overall_ok: bool) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Data Pipeline Run Report")
    lines.append("")
    lines.append(f"- Run finished (local): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Overall result: {'SUCCESS' if overall_ok else 'FAILED'}")
    if skipped:
        lines.append(f"- Skipped stages: {', '.join(skipped)}")
    lines.append("")
    lines.append("| Stage | Script | Result | Elapsed (s) |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['stage']} | `{row['script']}` | {row['result']} | {row['elapsed']:.1f} |"
        )
    lines.append("")
    for row in rows:
        if row["output_tail"]:
            lines.append(f"## Output tail — {row['stage']}")
            lines.append("")
            lines.append("```text")
            lines.append(str(row["output_tail"]))
            lines.append("```")
            lines.append("")
    REPORT_PATH.write_text(NL.join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-stage",
        action="append",
        default=[],
        choices=[name for name, _, _ in STAGES],
        help="Skip a named stage (repeatable).",
    )
    args = parser.parse_args()

    print(f"Data pipeline start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    overall_ok = True

    for name, description, script in STAGES:
        if name in args.skip_stage:
            print(f"[SKIP] {description} ({script})")
            skipped.append(name)
            rows.append(
                {
                    "stage": f"{name} (skipped)",
                    "script": script,
                    "result": "SKIPPED",
                    "elapsed": 0.0,
                    "output_tail": "",
                }
            )
            continue

        print(f"[RUN ] {description} ({script}) ...", flush=True)
        success, elapsed, output_tail = _run_stage(script)
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {description} in {elapsed:.1f}s")
        rows.append(
            {
                "stage": name,
                "script": script,
                "result": status,
                "elapsed": elapsed,
                "output_tail": output_tail,
            }
        )
        if not success:
            overall_ok = False
            print(f"[STOP] Failing closed after failed stage: {name}")
            break

    _write_report(rows, skipped, overall_ok)
    print(f"Run report written to {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Data pipeline {'SUCCEEDED' if overall_ok else 'FAILED'}.")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())