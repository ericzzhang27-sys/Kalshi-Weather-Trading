from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.repository_audit import (  # noqa: E402
    default_audit_output_dir,
    run_repository_audit,
    write_audit_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit repository research integrity")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when any open P0/P1 correctness or safety finding exists",
    )
    args = parser.parse_args()

    result = run_repository_audit(args.repo_root)
    destination = args.output_dir or default_audit_output_dir(args.repo_root)
    write_audit_artifacts(result, destination)
    print(f"Audit written to {destination}")
    print(f"Findings: {len(result.findings)}; strict failures: {result.strict_failure_count}")
    return 2 if args.strict and result.strict_failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
