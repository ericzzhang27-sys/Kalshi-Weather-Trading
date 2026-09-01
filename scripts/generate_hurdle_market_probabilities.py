from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conditional_increase_model import load_conditional_increase_model  # noqa: E402
from src.distributional_model import distribution_cdf  # noqa: E402
from src.hurdle_distribution import hurdle_cdf, integer_delta_probabilities  # noqa: E402
from src.hurdle_model import load_hurdle_predictor  # noqa: E402
from src.predict_distribution import load_probability_engine  # noqa: E402
from src.research.registry import sha256_file  # noqa: E402


def _load_components() -> tuple[object, dict, dict, dict]:
    exceedance_path = REPO_ROOT / "models/exceedance_model_bundle.json"
    conditional_path = REPO_ROOT / "models/remaining_increase_bundle.json"
    exceedance = json.loads(exceedance_path.read_text(encoding="utf-8"))
    conditional = json.loads(conditional_path.read_text(encoding="utf-8"))
    if exceedance.get("status") not in {"frozen_validated", "frozen_validated_user_override"}:
        raise ValueError("exceedance model is not frozen and validated")
    if conditional.get("status") != "frozen_validated":
        raise ValueError("conditional model is not frozen and validated")
    models = REPO_ROOT / "models"
    predictor = load_hurdle_predictor(
        models / exceedance["paths"]["classifier"],
        models / exceedance["paths"]["features"],
        models / exceedance["paths"]["calibrator"],
        exceedance["calibration"],
    )
    conditional_model = load_conditional_increase_model(models / "remaining_increase_ngboost.pkl")
    return predictor, conditional_model, exceedance, conditional


def _attach_frozen_champion(
    hurdle: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Attach point-in-time predictions from the frozen production champion.

    The comparison is made only on exact shared timestamps.  It is deliberately
    calculated before Kalshi alignment, so later market data cannot influence the
    weather-model baseline.
    """
    bundle_path = REPO_ROOT / "models/production_model_bundle.json"
    modeling_path = REPO_ROOT / "data/processed/modeling_rows_v1.csv"
    engine = load_probability_engine(model_bundle_path=bundle_path)
    legacy = pd.read_csv(modeling_path, low_memory=False)
    legacy["target_date"] = pd.to_datetime(legacy["target_date"], errors="raise").dt.date.astype(str)
    legacy["prediction_time"] = pd.to_datetime(legacy["prediction_time"], errors="raise")
    required = ["target_date", "prediction_time", "forecast_high", "actual_high", *engine.feature_columns]
    missing = sorted(set(required) - set(legacy.columns))
    if missing:
        raise ValueError(f"frozen champion source rows are missing features: {missing}")
    legacy = legacy[required].drop_duplicates(["target_date", "prediction_time"], keep="last")
    renamed = {
        column: f"champion__{column}"
        for column in legacy.columns
        if column not in {"target_date", "prediction_time"}
    }
    aligned = hurdle.merge(
        legacy.rename(columns=renamed),
        on=["target_date", "prediction_time"],
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        raise ValueError("no exact timestamps align with the frozen production champion")
    champion_input = pd.DataFrame(index=aligned.index)
    for feature in engine.feature_columns:
        champion_input[feature] = aligned[f"champion__{feature}"]
    champion_input["date"] = aligned["target_date"]
    champion_input["prediction_time"] = aligned["prediction_time"]
    champion_input["forecast_high"] = aligned["champion__forecast_high"]
    champion_input["actual_high"] = aligned["champion__actual_high"]
    params = engine.predict_distribution_params(champion_input)
    aligned["champion_mu"] = pd.to_numeric(params["mu"], errors="raise").to_numpy(float)
    aligned["champion_sigma"] = pd.to_numeric(params["sigma"], errors="raise").to_numpy(float)
    aligned["champion_distribution_type"] = params["distribution_type"].astype(str).to_numpy()
    aligned["champion_df"] = pd.to_numeric(params.get("df", np.nan), errors="coerce")
    aligned["champion_skew"] = pd.to_numeric(params.get("skew", np.nan), errors="coerce")
    aligned["champion_forecast_high"] = pd.to_numeric(
        aligned["champion__forecast_high"], errors="raise"
    )
    return aligned, sha256_file(bundle_path)


def _central_coverage_hits(
    hurdle: pd.DataFrame,
    p_increase: np.ndarray,
    conditional_model: dict,
    *,
    gap_support: np.ndarray,
    gap_weights: np.ndarray,
    max_delta: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    same_feed_probability = integer_delta_probabilities(
        p_increase, conditional_model, hurdle, max_delta=max_delta
    )
    if gap_weights.shape != (len(hurdle), len(gap_support)):
        raise ValueError("reconciliation weights are not aligned with hurdle rows")
    minimum = int(gap_support.min())
    maximum = max_delta + int(gap_support.max())
    values = np.arange(minimum, maximum + 1)
    probability = np.zeros((len(hurdle), len(values) + 1), dtype=float)
    for gap_index, gap in enumerate(gap_support.astype(int)):
        weight = gap_weights[:, gap_index]
        for delta in range(max_delta + 1):
            destination = delta + gap - minimum
            probability[:, destination] += weight * same_feed_probability[:, delta]
        probability[:, -1] += weight * same_feed_probability[:, -1]
    probability /= probability.sum(axis=1, keepdims=True)
    realized = (
        pd.to_numeric(hurdle["official_final_daily_high"], errors="raise").to_numpy(int)
        - pd.to_numeric(hurdle["current_max_so_far"], errors="raise").to_numpy(int)
    )
    realized_index = realized - minimum
    realized_index = np.where(realized > maximum, len(values), realized_index)
    if (realized_index < 0).any():
        raise ValueError("official daily high falls below reconciliation support")
    cdf = np.cumsum(probability, axis=1)
    row_index = np.arange(len(hurdle))
    cdf_at_truth = cdf[row_index, realized_index]
    cdf_below_truth = np.where(
        realized_index > 0,
        cdf[row_index, np.maximum(0, realized_index - 1)],
        0.0,
    )
    hits: list[np.ndarray] = []
    for level in (0.80, 0.90):
        tail = (1.0 - level) / 2.0
        lower = np.argmax(cdf >= tail, axis=1)
        upper = np.argmax(cdf >= 1.0 - tail, axis=1)
        hits.append(((realized_index >= lower) & (realized_index <= upper)).astype(float))
    return hits[0], hits[1], cdf_below_truth, cdf_at_truth


def _expanding_reconciliation_weights(
    all_hurdle: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    support: np.ndarray,
    window_days: int | None,
    zero_gap_prior_strength: float,
) -> np.ndarray:
    """Estimate official-DCR minus same-feed error using prior target-days only."""
    daily = all_hurdle.groupby("target_date", sort=True).agg(
        same_feed=("same_feed_final_daily_high", "first"),
        official=("official_final_daily_high", "first"),
    ).dropna()
    daily.index = pd.to_datetime(daily.index, errors="raise")
    daily["gap"] = np.rint(daily["official"] - daily["same_feed"]).astype(int)
    support_set = set(support.astype(int).tolist())
    outside = sorted(set(daily["gap"].tolist()) - support_set)
    if outside:
        raise ValueError(f"observed reconciliation gaps exceed configured support: {outside}")
    dates = pd.to_datetime(selected["target_date"], errors="raise")
    result = np.zeros((len(selected), len(support)), dtype=float)
    cache: dict[str, np.ndarray] = {}
    for row_index, target in enumerate(dates):
        key = target.date().isoformat()
        weights = cache.get(key)
        if weights is None:
            history = daily[daily.index < target]
            if window_days is not None:
                history = history[history.index >= target - pd.Timedelta(days=window_days)]
            if len(history) < 180:
                raise ValueError(
                    f"fewer than 180 prior reconciliation days are available for {key}"
                )
            counts = np.full(len(support), 0.25, dtype=float)
            counts[np.flatnonzero(support == 0)[0]] += float(zero_gap_prior_strength)
            observed = history["gap"].value_counts()
            for support_index, gap in enumerate(support):
                counts[support_index] += float(observed.get(int(gap), 0.0))
            weights = counts / counts.sum()
            cache[key] = weights
        result[row_index] = weights
    return result


def _champion_cdf(long: pd.DataFrame, boundary: np.ndarray) -> np.ndarray:
    forecast = pd.to_numeric(long["champion_forecast_high"], errors="raise").to_numpy(float)
    distribution_types = long["champion_distribution_type"].dropna().astype(str).unique()
    if len(distribution_types) != 1:
        raise ValueError("frozen champion must use one distribution family per export")
    return distribution_cdf(
        np.asarray(boundary, dtype=float) - forecast,
        mu=pd.to_numeric(long["champion_mu"], errors="raise").to_numpy(float),
        sigma=pd.to_numeric(long["champion_sigma"], errors="raise").to_numpy(float),
        distribution=str(distribution_types[0]),
        df=(pd.to_numeric(long["champion_df"], errors="coerce").to_numpy(float)
            if long["champion_df"].notna().any() else None),
        skew=(pd.to_numeric(long["champion_skew"], errors="coerce").to_numpy(float)
              if long["champion_skew"].notna().any() else None),
    )


def generate_probabilities(
    *,
    hurdle_dataset_path: Path,
    canonical_path: Path,
    output_path: Path,
    test_start: str,
    reconciliation_window_days: int | None = 365,
    zero_gap_prior_strength: float = 10.0,
) -> dict[str, object]:
    predictor, conditional_model, exceedance, conditional = _load_components()
    all_hurdle = pd.read_csv(hurdle_dataset_path)
    all_hurdle["target_date"] = pd.to_datetime(all_hurdle["target_date"], errors="raise").dt.date.astype(str)
    all_hurdle["prediction_time"] = pd.to_datetime(all_hurdle["prediction_time"], errors="raise")
    hurdle = all_hurdle[all_hurdle["target_date"] >= pd.Timestamp(test_start).date().isoformat()].copy()
    if hurdle.empty:
        raise ValueError("no hurdle rows in requested held-out period")
    hurdle, champion_bundle_hash = _attach_frozen_champion(hurdle)
    p_increase = predictor.predict_proba(hurdle)
    gap_support = np.arange(-10, 11, dtype=int)
    gap_weights = _expanding_reconciliation_weights(
        all_hurdle,
        hurdle,
        support=gap_support,
        window_days=reconciliation_window_days,
        zero_gap_prior_strength=zero_gap_prior_strength,
    )
    coverage_80_hit, coverage_90_hit, cdf_below_truth, cdf_at_truth = _central_coverage_hits(
        hurdle,
        p_increase,
        conditional_model,
        gap_support=gap_support,
        gap_weights=gap_weights,
    )
    hurdle_probability = hurdle[["target_date", "prediction_time"]].copy()
    hurdle_probability["p_increase"] = p_increase
    hurdle_probability["coverage_80_hit"] = coverage_80_hit
    hurdle_probability["coverage_90_hit"] = coverage_90_hit
    hurdle_probability["weather_cdf_below_truth"] = cdf_below_truth
    hurdle_probability["weather_cdf_at_truth"] = cdf_at_truth
    for gap_index, gap in enumerate(gap_support):
        hurdle[f"reconciliation_gap_{gap}"] = gap_weights[:, gap_index]
    canonical = pd.read_parquet(canonical_path)
    schemas = canonical[
        ["target_date", "event_ticker", "market_ticker", "bucket_lower", "bucket_upper", "bucket_label"]
    ].drop_duplicates("market_ticker", keep="last")
    schemas["target_date"] = pd.to_datetime(schemas["target_date"], errors="raise").dt.date.astype(str)
    complete_days = schemas.groupby("target_date").filter(lambda group: len(group) >= 2)
    long = hurdle.merge(complete_days, on="target_date", how="inner", validate="many_to_many")
    if long.empty:
        raise ValueError("no held-out hurdle rows match the canonical Kalshi schemas")
    long = long.merge(
        hurdle_probability,
        on=["target_date", "prediction_time"],
        how="left",
        validate="many_to_one",
    )
    current_max = pd.to_numeric(long["current_max_so_far"], errors="raise").to_numpy(float)
    probability_increase = pd.to_numeric(long["p_increase"], errors="raise").to_numpy(float)
    lower_bound = pd.to_numeric(long["bucket_lower"], errors="coerce").to_numpy(float)
    upper_bound = pd.to_numeric(long["bucket_upper"], errors="coerce").to_numpy(float)
    lower_cdf = np.zeros(len(long), dtype=float)
    upper_cdf = np.zeros(len(long), dtype=float)
    finite_lower = np.isfinite(lower_bound)
    finite_upper = np.isfinite(upper_bound)
    for gap in gap_support:
        weight = pd.to_numeric(
            long[f"reconciliation_gap_{gap}"], errors="raise"
        ).to_numpy(float)
        if finite_lower.any():
            lower_cdf[finite_lower] += weight[finite_lower] * hurdle_cdf(
                probability_increase[finite_lower], conditional_model,
                long.loc[finite_lower],
                lower_bound[finite_lower] - current_max[finite_lower] - gap,
            )
        if finite_upper.any():
            upper_cdf[finite_upper] += weight[finite_upper] * hurdle_cdf(
                probability_increase[finite_upper], conditional_model,
                long.loc[finite_upper],
                upper_bound[finite_upper] - current_max[finite_upper] - gap,
            )
        else:
            upper_cdf += weight
    upper_cdf[~finite_upper] = 1.0
    long["probability"] = np.clip(upper_cdf - lower_cdf, 0.0, 1.0)
    champion_lower_cdf = np.zeros(len(long), dtype=float)
    champion_upper_cdf = np.ones(len(long), dtype=float)
    if finite_lower.any():
        champion_lower_cdf[finite_lower] = _champion_cdf(
            long.loc[finite_lower], lower_bound[finite_lower]
        )
    if finite_upper.any():
        champion_upper_cdf[finite_upper] = _champion_cdf(
            long.loc[finite_upper], upper_bound[finite_upper]
        )
    long["baseline_probability"] = np.clip(
        champion_upper_cdf - champion_lower_cdf, 0.0, 1.0
    )
    keys = ["target_date", "prediction_time", "event_ticker"]
    totals = long.groupby(keys)["probability"].transform("sum")
    if (totals <= 0).any() or not np.isfinite(totals).all():
        raise ValueError("hurdle event probabilities have invalid totals")
    long["probability"] = long["probability"] / totals
    baseline_totals = long.groupby(keys)["baseline_probability"].transform("sum")
    if (baseline_totals <= 0).any() or not np.isfinite(baseline_totals).all():
        raise ValueError("frozen champion event probabilities have invalid totals")
    long["baseline_probability"] = long["baseline_probability"] / baseline_totals
    deviations = long.groupby(keys)["probability"].sum().sub(1.0).abs()
    if float(deviations.max()) > 1e-8:
        raise ValueError("hurdle event probabilities do not sum to one")
    export = long[
        [
            "target_date", "prediction_time", "event_ticker", "market_ticker",
            "bucket_lower", "bucket_upper", "bucket_label", "probability",
            "baseline_probability", "current_max_so_far", "final_daily_high",
            "remaining_increase", "p_increase", "coverage_80_hit", "coverage_90_hit",
            "weather_cdf_below_truth", "weather_cdf_at_truth",
            "forecast_issue_time_utc", "observation_time_utc",
        ]
    ].copy()
    export["location"] = "NYC"
    export["model_name"] = "intraday_lightgbm_hurdle_shifted_poisson"
    export["model_version"] = f"{exceedance.get('winner')}+{conditional.get('winner')}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"hurdle probability output is immutable: {output_path}")
    export.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "rows": int(len(export)),
        "event_days": int(export["target_date"].nunique()),
        "prediction_timestamps": int(export[["target_date", "prediction_time"]].drop_duplicates().shape[0]),
        "test_start": test_start,
        "source_hurdle_dataset_sha256": sha256_file(hurdle_dataset_path),
        "canonical_sha256": sha256_file(canonical_path),
        "exceedance_bundle_sha256": sha256_file(REPO_ROOT / "models/exceedance_model_bundle.json"),
        "conditional_bundle_sha256": sha256_file(REPO_ROOT / "models/remaining_increase_bundle.json"),
        "frozen_champion_bundle_sha256": champion_bundle_hash,
        "baseline": "frozen_production_ngboost_exact_shared_timestamps_and_kalshi_buckets",
        "reconciliation": {
            "target": "official_daily_climate_report_minus_same_feed_intraday_max",
            "method": "expanding_prior_day_empirical_convolution",
            "window_days": reconciliation_window_days,
            "zero_gap_prior_strength": zero_gap_prior_strength,
            "support": gap_support.tolist(),
        },
        "test_period_previously_inspected": True,
        "evidence_label": "historical_challenger_previously_inspected",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert the frozen intraday hurdle model to Kalshi bucket probabilities")
    parser.add_argument("--hurdle-dataset", type=Path, default=REPO_ROOT / "data/processed/hurdle_dataset.csv")
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/research/hurdle_market_probabilities.parquet")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--reconciliation-window-days", type=int, default=365)
    parser.add_argument("--zero-gap-prior-strength", type=float, default=10.0)
    args = parser.parse_args(argv)
    manifest = generate_probabilities(
        hurdle_dataset_path=args.hurdle_dataset,
        canonical_path=args.canonical,
        output_path=args.output,
        test_start=args.test_start,
        reconciliation_window_days=args.reconciliation_window_days,
        zero_gap_prior_strength=args.zero_gap_prior_strength,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
