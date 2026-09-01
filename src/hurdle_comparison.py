from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def brier_skill_score(
    y_true: np.ndarray | pd.Series,
    model_probability: np.ndarray | pd.Series,
    reference_probability: np.ndarray | pd.Series,
) -> float:
    y = np.asarray(y_true, dtype=float)
    model = np.asarray(model_probability, dtype=float)
    reference = np.asarray(reference_probability, dtype=float)
    model_brier = float(np.mean((model - y) ** 2))
    reference_brier = float(np.mean((reference - y) ** 2))
    if reference_brier <= 0:
        return float("nan")
    return float(1.0 - model_brier / reference_brier)


def time_of_day_bucket(prediction_time: pd.Series) -> pd.Categorical:
    timestamp = pd.to_datetime(prediction_time)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    labels = ["before 12 PM", "12–2 PM", "2–4 PM", "4–5 PM", "5–6 PM", "6–7 PM", "after 7 PM"]
    return pd.cut(
        hour,
        bins=[-np.inf, 12, 14, 16, 17, 18, 19, np.inf],
        labels=labels,
        right=False,
        ordered=True,
    )


def choose_exceedance_winner(
    comparison: pd.DataFrame,
    *,
    incumbent: str = "ngboost_bernoulli",
    minimum_bss_gain: float = 0.02,
    minimum_late_calibration_gain: float = 0.02,
) -> tuple[str, dict[str, Any]]:
    """Apply the explicit replacement hurdle without considering AUC."""
    if incumbent not in set(comparison["model"]):
        raise ValueError(f"Incumbent {incumbent!r} is absent from comparison")
    indexed = comparison.set_index("model")
    incumbent_row = indexed.loc[incumbent]
    eligible: list[tuple[str, str]] = []
    reasons: dict[str, dict[str, Any]] = {}
    for model, row in indexed.iterrows():
        if model == incumbent:
            continue
        bss_gain = float(row["brier_skill_score"] - incumbent_row["brier_skill_score"])
        late_calibration_gain = float(
            incumbent_row["late_day_calibration_gap"] - row["late_day_calibration_gap"]
        )
        bss_replacement = (
            bss_gain >= minimum_bss_gain
            and row["log_loss"] <= incumbent_row["log_loss"] + 0.001
        )
        late_replacement = (
            late_calibration_gain >= minimum_late_calibration_gain
            and row["late_day_brier"] < incumbent_row["late_day_brier"]
            and row["brier_skill_score"] >= incumbent_row["brier_skill_score"] - 0.005
            and row["log_loss"] <= incumbent_row["log_loss"] + 0.005
        )
        if bss_replacement:
            eligible.append((model, "meaningful_bss_gain"))
        elif late_replacement:
            eligible.append((model, "clearly_better_late_day_calibration"))
        reasons[model] = {
            "bss_gain_vs_incumbent": bss_gain,
            "late_day_calibration_gain_vs_incumbent": late_calibration_gain,
            "passes_bss_replacement_rule": bool(bss_replacement),
            "passes_late_day_replacement_rule": bool(late_replacement),
        }
    if not eligible:
        return incumbent, {
            "decision": "retain_incumbent",
            "reason": "no challenger cleared the replacement threshold",
            "thresholds": {
                "minimum_bss_gain": minimum_bss_gain,
                "minimum_late_calibration_gain": minimum_late_calibration_gain,
            },
            "challengers": reasons,
        }
    eligible_names = [name for name, _ in eligible]
    winner = (
        comparison.loc[comparison["model"].isin(eligible_names)]
        .sort_values(["brier_skill_score", "log_loss"], ascending=[False, True])
        .iloc[0]["model"]
    )
    winning_reason = dict(eligible)[str(winner)]
    return str(winner), {
        "decision": "replace_incumbent",
        "reason": winning_reason,
        "thresholds": {
            "minimum_bss_gain": minimum_bss_gain,
            "minimum_late_calibration_gain": minimum_late_calibration_gain,
        },
        "challengers": reasons,
    }
