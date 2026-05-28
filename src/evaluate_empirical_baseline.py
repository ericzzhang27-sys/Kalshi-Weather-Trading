from __future__ import annotations

from pathlib import Path
import math
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.empirical_error_model import EmpiricalErrorModel, REQUIRED_COLUMNS  # noqa: E402
from src.interval_probs import (  # noqa: E402
    Interval,
    cdf_to_interval_probs,
    normalize_probs,
    validate_interval_probs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_TABLE_PATH = REPO_ROOT / "data" / "processed" / "modeling_rows_v1.csv"


def assign_interval(
    value: float,
    intervals: list[Interval],
) -> Interval:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"Cannot assign non-finite value to an interval: {value!r}")

    for lower, upper in intervals:
        if lower is None and upper is None:
            raise ValueError("Interval cannot have both boundaries open")
        if lower is None and numeric_value <= float(upper):
            return (lower, upper)
        if upper is None and numeric_value > float(lower):
            return (lower, upper)
        if lower is not None and upper is not None and float(lower) < numeric_value <= float(upper):
            return (lower, upper)

    raise ValueError(f"No interval contains forecast_error value {numeric_value:g}")


def run_chronological_empirical_validation(
    feature_df: pd.DataFrame,
    intervals: list[Interval],
    train_end_date: str,
    output_predictions_path: str = "outputs/day9_empirical_baseline/empirical_baseline_predictions.csv",
    output_report_path: str = "outputs/day9_empirical_baseline/empirical_baseline_report.md",
    model_output_path: str = "models/empirical_error_baseline.pkl",
    min_samples: int = 30,
    doy_window: int = 30,
    horizon_window_hours: float = 6.0,
    smoothing_alpha: float = 1.0,
) -> pd.DataFrame:
    if not intervals:
        raise ValueError("At least one forecast-error interval is required")

    working = _prepare_feature_df(feature_df)
    preparation_notes = list(working.attrs.get("empirical_preparation_notes", []))
    _validate_intervals_cover_errors(working["forecast_error"], intervals)

    train_end = pd.to_datetime(train_end_date, errors="raise").normalize()
    train_df = working[working["date"] <= train_end].copy()
    test_df = working[working["date"] > train_end].copy()
    if train_df.empty:
        raise ValueError(f"Training split is empty for train_end_date={train_end_date!r}")
    if test_df.empty:
        raise ValueError(f"Test split is empty for train_end_date={train_end_date!r}")

    model = EmpiricalErrorModel(
        min_samples=min_samples,
        doy_window=doy_window,
        horizon_window_hours=horizon_window_hours,
        smoothing_alpha=smoothing_alpha,
    ).fit(train_df)

    max_train_date = model.error_library_["date"].max()
    min_test_date = test_df["date"].min()
    if max_train_date >= min_test_date:
        raise AssertionError("Chronological split allows same-date or future rows into training history")

    interval_columns = _probability_column_map(intervals)
    prediction_records: list[dict[str, Any]] = []

    test_df = test_df.sort_values(["date", "station_id", "prediction_hour"]).reset_index(drop=True)
    for _, row in test_df.iterrows():
        prediction = model.predict_interval_probs(row, intervals)
        probs = prediction["probs"]
        validate_interval_probs(probs)
        normal_prediction = _predict_normal_interval_probs(model, row, intervals)
        normal_probs = normal_prediction["probs"]
        validate_interval_probs(normal_probs)

        true_interval = assign_interval(float(row["forecast_error"]), intervals)
        prob_true_interval = float(probs[true_interval])
        nll = -math.log(max(prob_true_interval, 1e-12))
        predicted_interval = max(probs.items(), key=lambda item: item[1])[0]
        normal_prob_true_interval = float(normal_probs[true_interval])
        normal_nll = -math.log(max(normal_prob_true_interval, 1e-12))
        normal_predicted_interval = max(normal_probs.items(), key=lambda item: item[1])[0]

        record: dict[str, Any] = {
            "date": row["date"].date().isoformat(),
            "station_id": row["station_id"],
            "prediction_hour": int(row["prediction_hour"]),
            "forecast_horizon_hours": float(row["forecast_horizon_hours"]),
            "is_post_peak": bool(row["is_post_peak"]),
            "abs_hours_from_peak": float(row["abs_hours_from_peak"]),
            "forecast_error": float(row["forecast_error"]),
            "true_interval": interval_to_label(true_interval),
            "predicted_interval": interval_to_label(predicted_interval),
            "prob_true_interval": prob_true_interval,
            "nll": float(nll),
            "is_top_interval_correct": bool(predicted_interval == true_interval),
            "fallback_level": prediction["fallback_level"],
            "sample_size": int(prediction["sample_size"]),
            "normal_mu": float(normal_prediction["mu"]),
            "normal_sigma": float(normal_prediction["sigma"]),
            "normal_predicted_interval": interval_to_label(normal_predicted_interval),
            "normal_prob_true_interval": normal_prob_true_interval,
            "normal_nll": float(normal_nll),
            "normal_is_top_interval_correct": bool(normal_predicted_interval == true_interval),
        }
        for interval, column in interval_columns.items():
            record[column] = float(probs[interval])
            record[f"normal_{column}"] = float(normal_probs[interval])
        prediction_records.append(record)

    predictions = pd.DataFrame(prediction_records)
    _validate_predictions(predictions, list(interval_columns.values()))
    _validate_normal_predictions(
        predictions,
        [f"normal_{column}" for column in interval_columns.values()],
    )

    predictions_path = _resolve_output_path(output_predictions_path)
    report_path = _resolve_output_path(output_report_path)
    model_path = _resolve_output_path(model_output_path)
    for path in [predictions_path, report_path, model_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(predictions_path, index=False)
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    report_text = _build_report(
        predictions=predictions,
        train_df=train_df,
        test_df=test_df,
        intervals=intervals,
        min_samples=min_samples,
        doy_window=doy_window,
        horizon_window_hours=horizon_window_hours,
        smoothing_alpha=smoothing_alpha,
        probability_columns=list(interval_columns.values()),
        normal_probability_columns=[f"normal_{column}" for column in interval_columns.values()],
        preparation_notes=preparation_notes,
    )
    report_path.write_text(report_text, encoding="utf-8")

    for path in [predictions_path, report_path, model_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Expected output was not written: {path}")

    return predictions


def _prepare_feature_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        raise ValueError("Feature dataframe is empty")

    df = feature_df.copy()
    if "station_id" not in df.columns:
        if "location" not in df.columns:
            raise ValueError("Feature dataframe must contain station_id or location")
        df["station_id"] = df["location"]

    if "date" not in df.columns:
        if "target_date" not in df.columns:
            raise ValueError("Feature dataframe must contain date or target_date")
        df["date"] = df["target_date"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("Feature dataframe contains unparseable date values")

    if "prediction_hour" not in df.columns:
        df["prediction_hour"] = _derive_prediction_hour(df)

    if "day_of_year" not in df.columns:
        df["day_of_year"] = df["date"].dt.dayofyear

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    if "season" not in df.columns:
        month = pd.to_numeric(df["month"], errors="coerce")
        df["season"] = np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            [0, 1, 2, 3],
            default=np.nan,
        )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Feature dataframe cannot support empirical baseline; missing {missing_columns}")

    for column in ["forecast_error", "prediction_hour", "forecast_horizon_hours", "day_of_year", "month"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[REQUIRED_COLUMNS].isna().any().any():
        missing = df[REQUIRED_COLUMNS].isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"Feature dataframe has missing empirical baseline fields: {missing}")

    if not np.isfinite(df["forecast_horizon_hours"]).all():
        raise ValueError("forecast_horizon_hours contains non-finite values")
    if "is_post_peak" not in df.columns:
        df["is_post_peak"] = df["forecast_horizon_hours"] < 0.0
    if "abs_hours_from_peak" not in df.columns:
        df["abs_hours_from_peak"] = df["forecast_horizon_hours"].abs()

    preparation_notes = [
        "Because the dataset uses all 24 prediction hours, forecast_horizon_hours is signed. "
        "Positive values indicate prediction times before the typical daily high period, zero "
        "indicates near-peak times, and negative values indicate post-peak prediction states. "
        "Negative values are preserved rather than clipped, because post-peak rows represent "
        "a different information regime."
    ]

    df["station_id"] = df["station_id"].astype(str)
    df = df.sort_values(["date", "station_id", "prediction_hour"]).reset_index(drop=True)
    df.attrs["empirical_preparation_notes"] = preparation_notes
    return df


def _derive_prediction_hour(df: pd.DataFrame) -> pd.Series:
    if "prediction_time" in df.columns:
        raw = df["prediction_time"].astype(str).str.strip()
        clock_only = raw.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$").all()
        if clock_only:
            return pd.to_numeric(raw.str.split(":").str[0], errors="coerce")
        parsed = pd.to_datetime(df["prediction_time"], errors="coerce")
        if parsed.notna().all():
            return parsed.dt.hour

    if "prediction_clock_time" in df.columns:
        raw = df["prediction_clock_time"].astype(str).str.strip()
        return pd.to_numeric(raw.str.split(":").str[0], errors="coerce")

    raise ValueError("Cannot derive prediction_hour without prediction_time or prediction_clock_time")


def _validate_intervals_cover_errors(errors: pd.Series, intervals: list[Interval]) -> None:
    for value in pd.to_numeric(errors, errors="coerce").dropna():
        assign_interval(float(value), intervals)


def _probability_column_map(intervals: list[Interval]) -> dict[Interval, str]:
    raw_names = {interval: _probability_column_name(interval) for interval in intervals}
    counts: dict[str, int] = {}
    unique_names: dict[Interval, str] = {}
    for interval, name in raw_names.items():
        counts[name] = counts.get(name, 0) + 1
        unique_names[interval] = name if counts[name] == 1 else f"{name}_{counts[name]}"
    return unique_names


def _probability_column_name(interval: Interval) -> str:
    lower, upper = interval
    if lower is None and upper is None:
        raise ValueError("Interval cannot have both boundaries open")
    if lower is None:
        return f"prob_error_le_{_format_boundary_for_column(upper)}"
    if upper is None:
        return f"prob_error_gt_{_format_boundary_for_column(lower)}"
    return (
        f"prob_error_{_format_boundary_for_column(lower)}"
        f"_to_{_format_boundary_for_column(upper)}"
    )


def _format_boundary_for_column(value: float | None) -> str:
    if value is None:
        raise ValueError("Cannot format open interval boundary as a finite column suffix")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Interval boundary must be finite, got {value!r}")
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6g}".replace(".", "p")


def interval_to_label(interval: Interval) -> str:
    lower, upper = interval
    if lower is None and upper is None:
        raise ValueError("Interval cannot have both boundaries open")
    if lower is None:
        return f"(-inf, {_format_boundary_for_label(upper)}]"
    if upper is None:
        return f"({_format_boundary_for_label(lower)}, inf)"
    return f"({_format_boundary_for_label(lower)}, {_format_boundary_for_label(upper)}]"


def _format_boundary_for_label(value: float | None) -> str:
    if value is None:
        raise ValueError("Cannot format None as a finite boundary")
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _validate_predictions(predictions: pd.DataFrame, probability_columns: list[str]) -> None:
    required_columns = [
        "true_interval",
        "predicted_interval",
        "prob_true_interval",
        "nll",
        "fallback_level",
        "sample_size",
    ]
    for column in required_columns:
        if column not in predictions.columns:
            raise AssertionError(f"Predictions are missing required column: {column}")
        if predictions[column].isna().any():
            raise AssertionError(f"Predictions contain missing values in {column}")

    if not probability_columns:
        raise AssertionError("Predictions contain no interval probability columns")

    probabilities = predictions[probability_columns].astype(float)
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6):
        raise AssertionError("Interval probability columns do not sum to 1 for every row")
    if (probabilities < -1e-9).any().any():
        raise AssertionError("Predictions contain negative interval probabilities")
    if (probabilities > 1.0 + 1e-9).any().any():
        raise AssertionError("Predictions contain interval probabilities above 1")
    if (pd.to_numeric(predictions["sample_size"], errors="coerce") <= 0).any():
        raise AssertionError("Predictions contain nonpositive sample_size values")
    if not np.isfinite(pd.to_numeric(predictions["nll"], errors="coerce")).all():
        raise AssertionError("Predictions contain non-finite NLL values")


def _validate_normal_predictions(
    predictions: pd.DataFrame,
    normal_probability_columns: list[str],
) -> None:
    required_columns = [
        "normal_mu",
        "normal_sigma",
        "normal_predicted_interval",
        "normal_prob_true_interval",
        "normal_nll",
        "normal_is_top_interval_correct",
    ]
    for column in required_columns:
        if column not in predictions.columns:
            raise AssertionError(f"Predictions are missing required normal column: {column}")
        if predictions[column].isna().any():
            raise AssertionError(f"Predictions contain missing normal values in {column}")

    probabilities = predictions[normal_probability_columns].astype(float)
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6):
        raise AssertionError("Normal interval probability columns do not sum to 1 for every row")
    if (probabilities < -1e-9).any().any():
        raise AssertionError("Normal predictions contain negative interval probabilities")
    if (probabilities > 1.0 + 1e-9).any().any():
        raise AssertionError("Normal predictions contain interval probabilities above 1")
    if (pd.to_numeric(predictions["normal_sigma"], errors="coerce") <= 0).any():
        raise AssertionError("Normal predictions contain nonpositive sigma values")
    if not np.isfinite(pd.to_numeric(predictions["normal_nll"], errors="coerce")).all():
        raise AssertionError("Normal predictions contain non-finite NLL values")


def _predict_normal_interval_probs(
    model: EmpiricalErrorModel,
    row: pd.Series,
    intervals: list[Interval],
    min_sigma: float = 1e-6,
) -> dict[str, Any]:
    errors, fallback_level, sample_size = model._candidate_errors(row)
    mu = float(np.mean(errors))
    sigma = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
    sigma = max(sigma, min_sigma)

    finite_boundaries = sorted(
        {
            float(boundary)
            for interval in intervals
            for boundary in interval
            if boundary is not None
        }
    )
    cdf_values = {
        boundary: _normal_cdf(boundary, mu=mu, sigma=sigma)
        for boundary in finite_boundaries
    }
    probs = cdf_to_interval_probs(cdf_values, intervals)
    probs = normalize_probs(probs)
    validate_interval_probs(probs)

    return {
        "probs": probs,
        "cdf_values": cdf_values,
        "fallback_level": fallback_level,
        "sample_size": sample_size,
        "mu": mu,
        "sigma": sigma,
    }


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError(f"Normal sigma must be positive and finite, got {sigma!r}")
    z = (float(x) - mu) / (sigma * math.sqrt(2.0))
    return float(0.5 * (1.0 + math.erf(z)))


def _build_report(
    predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    intervals: list[Interval],
    min_samples: int,
    doy_window: int,
    horizon_window_hours: float,
    smoothing_alpha: float,
    probability_columns: list[str],
    normal_probability_columns: list[str],
    preparation_notes: list[str],
) -> str:
    fallback_counts = predictions["fallback_level"].value_counts().sort_index()
    fallback_percentages = fallback_counts / len(predictions) * 100.0
    nll_by_fallback = predictions.groupby("fallback_level")["nll"].mean().sort_index()
    sample_by_fallback = predictions.groupby("fallback_level")["sample_size"].mean().sort_index()

    brier_score = _brier_score(predictions, intervals, probability_columns)
    normal_brier_score = _brier_score_from_columns(
        predictions,
        intervals,
        normal_probability_columns,
        true_interval_column="true_interval",
    )
    empirical_mean_nll = float(predictions["nll"].mean())
    normal_mean_nll = float(predictions["normal_nll"].mean())
    uniform_nll = -math.log(1.0 / len(intervals))
    nll_improvement = uniform_nll - empirical_mean_nll
    normal_nll_improvement = uniform_nll - normal_mean_nll

    lines = [
        "# Empirical Historical Forecast-Error Baseline",
        "",
        "## Goal",
        "This baseline estimates forecast-error probabilities from similar historical forecast errors.",
        "",
        "## Method",
        "For each prediction row, the model gathers eligible historical forecast errors and forms an empirical CDF.",
        "",
        "F(c) = count(error <= c) / n",
        "",
        "The implementation uses the smoothed version:",
        "",
        "F(c) = (count(error <= c) + alpha) / (n + 2 alpha)",
        "",
        "Forecast-error interval probabilities are then computed as:",
        "",
        "P(a < error <= b) = F(b) - F(a)",
        "",
        "Open-ended intervals use F(b) for error <= b and 1 - F(a) for error > a. "
        "Interval probabilities are normalized and validated to sum to one.",
        "",
        "## No-Future-Leakage Rule",
        "Every prediction only used rows with date < prediction row date. Same-date and future rows are excluded.",
        "",
        "## Fallback Hierarchy",
        "1. same_station_doy_hour_horizon: same station, circular day-of-year window, same prediction hour, similar horizon.",
        "2. same_season_hour_horizon: same season, same prediction hour, similar horizon.",
        "3. same_station_month: same station and month.",
        "4. all_past: all rows before the prediction row date.",
        "",
        "## Validation Setup",
        f"- Train rows: {len(train_df):,}",
        f"- Test rows: {len(test_df):,}",
        f"- Train date range: {_date_range_text(train_df['date'])}",
        f"- Test date range: {_date_range_text(test_df['date'])}",
        f"- Intervals: {', '.join(interval_to_label(interval) for interval in intervals)}",
        f"- min_samples: {min_samples}",
        f"- doy_window: {doy_window}",
        f"- horizon_window_hours: {horizon_window_hours:g}",
        f"- smoothing_alpha: {smoothing_alpha:g}",
    ]
    if preparation_notes:
        lines.extend(["", "Forecast horizon note:"])
        lines.extend(f"- {note}" for note in preparation_notes)
    lines.extend([
        "",
        "## Forecast Horizon Diagnostics",
        "| Split | Min | 25th pct | Median | 75th pct | Max | Negative | Zero | Positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _horizon_diagnostics_row("Train", train_df),
        _horizon_diagnostics_row("Test", test_df),
        "",
        "## Candidate Sample Size Diagnostics",
        "| Min | 25th pct | Median | 75th pct | Max |",
        "| ---: | ---: | ---: | ---: | ---: |",
        _sample_size_diagnostics_row(predictions),
        "",
        "## Uniform Baseline Comparison",
        f"- Empirical mean NLL: {empirical_mean_nll:.6f}",
        f"- Normal mean NLL: {normal_mean_nll:.6f}",
        f"- Uniform baseline NLL: {uniform_nll:.6f}",
        f"- Empirical NLL improvement vs uniform: {nll_improvement:.6f}",
        f"- Normal NLL improvement vs uniform: {normal_nll_improvement:.6f}",
        "",
        "## Normal Distribution Baseline",
        "This comparator uses the same leakage-safe historical candidate rows as the empirical baseline, "
        "then fits a normal distribution with the candidate mean and sample standard deviation.",
        f"- Mean NLL: {normal_mean_nll:.6f}",
        f"- Median NLL: {predictions['normal_nll'].median():.6f}",
        f"- Top-interval accuracy: {predictions['normal_is_top_interval_correct'].mean():.6f}",
        f"- Average probability assigned to true interval: {predictions['normal_prob_true_interval'].mean():.6f}",
        f"- Average fitted sigma: {predictions['normal_sigma'].mean():.6f}",
        f"- Brier score over interval probabilities: {normal_brier_score:.6f}",
        "",
        "## Results",
        f"- Number of test rows: {len(predictions):,}",
        f"- Mean NLL: {empirical_mean_nll:.6f}",
        f"- Median NLL: {predictions['nll'].median():.6f}",
        f"- Top-interval accuracy: {predictions['is_top_interval_correct'].mean():.6f}",
        f"- Average probability assigned to true interval: {predictions['prob_true_interval'].mean():.6f}",
        f"- Average sample size: {predictions['sample_size'].mean():.2f}",
        f"- Brier score over interval probabilities: {brier_score:.6f}",
        "",
        "## Fallback Usage",
        "| Fallback level | Count | Percent | Mean NLL | Average sample size |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for fallback_level in fallback_counts.index:
        lines.append(
            "| "
            f"{fallback_level} | "
            f"{int(fallback_counts[fallback_level]):,} | "
            f"{fallback_percentages[fallback_level]:.2f}% | "
            f"{nll_by_fallback[fallback_level]:.6f} | "
            f"{sample_by_fallback[fallback_level]:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "- Sparse samples can make some fallback levels noisy.",
            "- Estimates may be unstable for extreme weather.",
            "- Weather and forecast systems can change over time, creating possible regime changes.",
            "- Station-specific differences may remain even after fallback filtering.",
            "- This is a benchmark, not the final model.",
            "",
            "## Role in Project",
            "NGBoost/DGBM should beat this baseline on chronological validation.",
            "",
        ]
    )
    return "\n".join(lines)


def _horizon_diagnostics_row(label: str, df: pd.DataFrame) -> str:
    horizon = pd.to_numeric(df["forecast_horizon_hours"], errors="coerce").dropna()
    if horizon.empty:
        return f"| {label} | NA | NA | NA | NA | NA | 0 | 0 | 0 |"
    negative = int((horizon < 0.0).sum())
    zero = int((horizon == 0.0).sum())
    positive = int((horizon > 0.0).sum())
    return (
        f"| {label} | "
        f"{horizon.min():.2f} | "
        f"{horizon.quantile(0.25):.2f} | "
        f"{horizon.median():.2f} | "
        f"{horizon.quantile(0.75):.2f} | "
        f"{horizon.max():.2f} | "
        f"{negative:,} | "
        f"{zero:,} | "
        f"{positive:,} |"
    )


def _sample_size_diagnostics_row(predictions: pd.DataFrame) -> str:
    sample_size = pd.to_numeric(predictions["sample_size"], errors="coerce").dropna()
    if sample_size.empty:
        return "| NA | NA | NA | NA | NA |"
    return (
        f"| {sample_size.min():.0f} | "
        f"{sample_size.quantile(0.25):.0f} | "
        f"{sample_size.median():.0f} | "
        f"{sample_size.quantile(0.75):.0f} | "
        f"{sample_size.max():.0f} |"
    )


def _brier_score(
    predictions: pd.DataFrame,
    intervals: list[Interval],
    probability_columns: list[str],
) -> float:
    return _brier_score_from_columns(
        predictions,
        intervals,
        probability_columns,
        true_interval_column="true_interval",
    )


def _brier_score_from_columns(
    predictions: pd.DataFrame,
    intervals: list[Interval],
    probability_columns: list[str],
    true_interval_column: str,
) -> float:
    total = 0.0
    for _, row in predictions.iterrows():
        true_interval = row[true_interval_column]
        row_score = 0.0
        for interval, column in zip(intervals, probability_columns):
            observed = 1.0 if interval_to_label(interval) == true_interval else 0.0
            row_score += (float(row[column]) - observed) ** 2
        total += row_score
    return total / len(predictions)


def _date_range_text(series: pd.Series) -> str:
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return "not available"
    return f"{values.min().date()} to {values.max().date()}"


def _resolve_output_path(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = REPO_ROOT / output
    return output


if __name__ == "__main__":
    feature_df = pd.read_csv(DEFAULT_FEATURE_TABLE_PATH)

    intervals = [
        (None, -3),
        (-3, -1),
        (-1, 1),
        (1, 3),
        (3, None),
    ]

    run_chronological_empirical_validation(
        feature_df=feature_df,
        intervals=intervals,
        train_end_date="2024-12-31",
    )
