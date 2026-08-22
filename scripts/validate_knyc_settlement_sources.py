from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_CLI = Path("data/processed/knyc_cli_daily_2018_2026.csv")
DEFAULT_NCEI = Path("data/raw/nws_daily).csv")
DEFAULT_OUTPUT = Path("outputs/data/knyc_cli_vs_ncei_validation.json")
DEFAULT_MISMATCHES = Path("outputs/data/knyc_cli_vs_ncei_mismatches.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IEM-parsed NWS CLI highs with NCEI Central Park TMAX.")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--ncei", type=Path, default=DEFAULT_NCEI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mismatches", type=Path, default=DEFAULT_MISMATCHES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cli = pd.read_csv(args.cli)
    ncei = pd.read_csv(args.ncei)

    cli["date"] = pd.to_datetime(cli["date"], errors="raise").dt.normalize()
    cli["cli_high_f"] = pd.to_numeric(cli["actual_high"], errors="coerce")

    ncei["date"] = pd.to_datetime(ncei["DATE"], errors="raise").dt.normalize()
    ncei["ncei_tmax_f"] = pd.to_numeric(ncei["TMAX"], errors="coerce")
    if "STATION" in ncei.columns:
        ncei = ncei[ncei["STATION"].astype(str).eq("USW00094728")]

    merged = cli[["date", "cli_high_f"]].merge(
        ncei[["date", "ncei_tmax_f"]], on="date", how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise RuntimeError("CLI and NCEI datasets have no overlapping dates")

    merged["difference_f"] = merged["cli_high_f"] - merged["ncei_tmax_f"]
    merged["exact_match"] = merged["difference_f"].abs() < 1e-9
    mismatches = merged[~merged["exact_match"]].copy()

    report = {
        "overlap_rows": int(len(merged)),
        "overlap_start": merged["date"].min().date().isoformat(),
        "overlap_end": merged["date"].max().date().isoformat(),
        "exact_match_rows": int(merged["exact_match"].sum()),
        "exact_match_fraction": round(float(merged["exact_match"].mean()), 6),
        "mismatch_rows": int(len(mismatches)),
        "mean_difference_f": round(float(merged["difference_f"].mean()), 6),
        "mean_absolute_difference_f": round(float(merged["difference_f"].abs().mean()), 6),
        "max_absolute_difference_f": round(float(merged["difference_f"].abs().max()), 6),
        "interpretation": (
            "CLI is the NWS daily climate-report value historically relevant to Kalshi settlement; "
            "NCEI TMAX is used here as an independent overlap check, not assumed identical a priori."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.mismatches.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    mismatches.to_csv(args.mismatches, index=False)

    print(json.dumps(report, indent=2))
    print(f"Mismatch rows: {args.mismatches}")


if __name__ == "__main__":
    main()
