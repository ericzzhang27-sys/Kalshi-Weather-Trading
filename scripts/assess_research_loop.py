from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.loop import assess_research_loop, load_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess closed-loop competence, Pareto, and plateau state")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "outputs/research/runs")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/research/loop")
    args = parser.parse_args(argv)
    paths = sorted(args.runs_root.glob("*/report.json"))
    assessment = assess_research_loop(load_reports(paths))
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / f"{run_id}.json"
    if output.exists():
        raise FileExistsError(f"loop assessment is immutable: {output}")
    output.write_text(json.dumps(assessment, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **assessment}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
