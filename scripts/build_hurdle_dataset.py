from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hurdle_dataset import HurdleDatasetConfig, build_hurdle_dataset, validate_hurdle_dataset
from src.hurdle_features import add_hurdle_core_features


OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "hurdle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the same-feed five-minute future-high dataset")
    parser.add_argument("--strict-invariant", action="store_true", help="retained for CLI compatibility; same-feed invariant always fails closed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = HurdleDatasetConfig(violation_policy="raise" if args.strict_invariant else "quarantine")
    dataset, summary = build_hurdle_dataset(config=config)
    violations = dataset.attrs.get("invariant_violations")
    disagreements = dataset.attrs.get("official_settlement_disagreements")
    dataset = add_hurdle_core_features(dataset)
    validate_hurdle_dataset(dataset)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_CSV, index=False)
    hourly_rates = (
        dataset.assign(hour_local=dataset["prediction_time"].dt.hour)
        .groupby("hour_local")
        .agg(
            rows=("will_increase", "size"),
            days=("target_date", "nunique"),
            future_high_rate=("will_increase", "mean"),
            mean_remaining_increase_f=("remaining_increase", "mean"),
        )
        .reset_index()
    )
    hourly_rates["future_high_percent"] = 100.0 * hourly_rates["future_high_rate"]
    hourly_rates.to_csv(OUTPUT_DIR / "hourly_future_high_base_rates.csv", index=False)
    (OUTPUT_DIR / "hurdle_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    violations.to_csv(OUTPUT_DIR / "hurdle_invariant_violations.csv", index=False)
    disagreements.to_csv(OUTPUT_DIR / "hurdle_official_settlement_disagreements.csv", index=False)

    print(f"Built {len(dataset):,} rows across {dataset['target_date'].nunique():,} whole weather days")
    print(f"Prediction cadence: {summary['prediction_frequency']} ({summary['decision_window_local']} local)")
    print(f"Same-feed invariant violations: {summary['n_violation_dates']}")
    print(f"Official-vs-five-minute disagreement days (audit only): {summary['official_settlement_disagreement_days']}")
    print(f"Saved hourly future-high base rates for {len(hourly_rates)} clock hours")
    print(f"Saved {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
