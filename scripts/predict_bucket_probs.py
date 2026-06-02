from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict_distribution import (  # noqa: E402
    DEFAULT_CALIBRATION_CONFIG_PATH,
    DEFAULT_FEATURE_LIST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SCHEMA_PATH,
    load_bucket_schema_optional,
    load_probability_engine,
    save_prediction_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final NGBoost bucket probabilities from processed feature rows."
    )
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--feature-list-path", default=str(DEFAULT_FEATURE_LIST_PATH))
    parser.add_argument(
        "--calibration-config-path",
        default=str(DEFAULT_CALIBRATION_CONFIG_PATH),
        help="Set to empty string to disable post-hoc calibration.",
    )
    parser.add_argument(
        "--bucket-schema-path",
        default=None,
        help="Optional fixed final-temperature bucket schema CSV.",
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--schema-output-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of input rows to score. Use 0 or a negative value for all rows.",
    )
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument(
        "--forecast-rounding",
        choices=["nearest", "floor", "ceil"],
        default="nearest",
    )
    return parser.parse_args(argv)


def load_prediction_rows(path: str | Path, limit: int, row_start: int) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input file not found: {input_path}")
    if int(row_start) < 0:
        raise ValueError("row-start must be nonnegative")

    rows = pd.read_csv(input_path)
    if int(row_start) > 0:
        rows = rows.iloc[int(row_start) :].reset_index(drop=True)
    if int(limit) > 0:
        rows = rows.head(int(limit)).copy()
    if rows.empty:
        raise ValueError("No prediction rows selected")
    return rows.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    calibration_path = (
        None
        if str(args.calibration_config_path).strip() == ""
        else args.calibration_config_path
    )
    rows = load_prediction_rows(args.input_path, args.limit, args.row_start)
    buckets = load_bucket_schema_optional(args.bucket_schema_path)
    engine = load_probability_engine(
        model_path=args.model_path,
        feature_list_path=args.feature_list_path,
        calibration_config_path=calibration_path,
    )
    result = engine.predict(
        rows,
        buckets=buckets,
        forecast_rounding=args.forecast_rounding,
    )
    save_prediction_outputs(
        result,
        output_path=args.output_path,
        schema_path=args.schema_output_path,
    )
    diagnostics = result.diagnostics
    print(
        "Saved final bucket probability predictions: "
        f"{diagnostics.probability_row_count:,} rows across "
        f"{diagnostics.prediction_row_count:,} prediction states "
        f"using {diagnostics.model_name} ({diagnostics.distribution_type})."
    )
    print(f"Prediction CSV: {Path(args.output_path)}")
    print(f"Schema docs: {Path(args.schema_output_path)}")


if __name__ == "__main__":
    main(sys.argv[1:])
