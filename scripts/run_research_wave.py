from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.wave import run_baseline_research_wave  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic closed-loop NYC/Kalshi research wave")
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--hypothesis", default="A regularized cross-fitted weather/market log-odds pool improves tradeable proper scores without sacrificing calibration.")
    parser.add_argument("--regularization-c", type=float, default=0.25)
    parser.add_argument(
        "--pool-kind",
        choices=["binary_log_odds", "linear", "logarithmic", "regime_logarithmic"],
        default="binary_log_odds",
    )
    parser.add_argument(
        "--weather-ensemble-kind",
        choices=["none", "linear", "logarithmic"],
        default="none",
        help="Optional cross-fitted challenger/frozen-champion ensemble before market pooling",
    )
    parser.add_argument(
        "--uncertainty-method",
        choices=["effective_day_wilson", "wilson_plus_disagreement"],
        default="wilson_plus_disagreement",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/research/runs")
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=REPO_ROOT / "models/production_model_bundle.json",
        help="Immutable model or composite-probability manifest hashed into the experiment record",
    )
    args = parser.parse_args(argv)
    output, report = run_baseline_research_wave(
        args.aligned,
        output_root=args.output_root,
        registry_path=REPO_ROOT / "outputs/research/experiments.sqlite",
        manifest_dir=REPO_ROOT / "outputs/research/manifests",
        model_path=args.model_manifest,
        requirements_path=REPO_ROOT / "requirements.txt",
        hypothesis=args.hypothesis,
        seed=args.seed,
        regularization_c=args.regularization_c,
        pool_kind=args.pool_kind,
        weather_ensemble_kind=(
            None if args.weather_ensemble_kind == "none" else args.weather_ensemble_kind
        ),
        uncertainty_method=args.uncertainty_method,
    )
    trading = report["trading_metrics"]
    print(f"Research wave written to {output}")
    print(f"Competence gates passed: {report['competence_gates']['passed']}")
    print(f"Sharpe/Sortino/Calmar: {trading.get('sharpe')}/{trading.get('sortino')}/{trading.get('calmar')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
