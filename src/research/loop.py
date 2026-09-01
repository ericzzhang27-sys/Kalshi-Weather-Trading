from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .gates import pareto_frontier


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _failure_attribution(report: dict[str, Any]) -> dict[str, str]:
    failed = set(report.get("competence_gates", {}).get("failed_gates", []))
    granularity = report.get("robustness", {}).get("candle_granularity", {})
    if not granularity.get("passed", False):
        return {"component": "data", "next_wave": "repair point-in-time one-minute coverage before any model search"}
    if failed & {
        "weather_log_loss_skill_positive",
        "weather_log_loss_skill_statistically_positive",
        "weather_rps_skill_positive",
        "weather_rps_skill_statistically_positive",
    }:
        return {
            "component": "probability_model",
            "next_wave": "test a preregistered weather family or acquire an operational forecast source targeted at the residual regime",
        }
    if failed & {"calibration_ece", "coverage_80", "coverage_90"}:
        return {"component": "calibration", "next_wave": "cross-fit a stricter calibration or conformal interval candidate"}
    if "hybrid_skill_positive" in failed:
        return {
            "component": "market_combination",
            "next_wave": "test coherent validation-optimized linear, logarithmic, and time-regime pools",
        }
    if failed & {"trade_count", "event_day_count"}:
        return {
            "component": "execution_evidence",
            "next_wave": "collect more event-days or improve calibrated edge; do not lower uncertainty or cost assumptions",
        }
    if failed & {"pbo", "deflated_sharpe_confidence", "bootstrap_net_pnl_lower"}:
        return {"component": "strategy", "next_wave": "reduce search multiplicity and test stability-preserving event-level strategies"}
    if failed:
        return {"component": "sizing_or_risk", "next_wave": "diagnose the remaining risk and stress failures"}
    return {"component": "none", "next_wave": "run the three mandatory post-floor research waves"}


def assess_research_loop(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(reports, key=lambda row: str(row.get("generated_at_utc", "")))
    if not ordered:
        raise ValueError("at least one research report is required")
    objective_rows: list[dict[str, float]] = []
    objective_report_indices: list[int] = []
    for index, report in enumerate(ordered):
        probability = report.get("probability_metrics", {})
        hybrid = probability.get("hybrid", {})
        trading = report.get("trading_metrics", {})
        row = {
            "hybrid_log_loss": hybrid.get("log_loss"),
            "ece": probability.get("ece"),
            "net_pnl": trading.get("net_pnl"),
            "calmar": trading.get("calmar"),
        }
        if all(_finite(value) for value in row.values()):
            objective_rows.append({key: float(value) for key, value in row.items()})
            objective_report_indices.append(index)
    local_frontier = pareto_frontier(
        objective_rows,
        {"hybrid_log_loss": "min", "ece": "min", "net_pnl": "max", "calmar": "max"},
    ) if objective_rows else []
    frontier_ids = [
        ordered[objective_report_indices[index]].get("run_id") for index in local_frontier
    ]
    passing_indices = [
        index for index, report in enumerate(ordered)
        if report.get("competence_gates", {}).get("passed") is True
    ]
    latest = ordered[-1]
    if not passing_indices:
        attribution = _failure_attribution(latest)
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "continue_research_floors_not_met",
            "continue_required": True,
            "competent_champion_exists": False,
            "reports_assessed": len(ordered),
            "latest_run_id": latest.get("run_id"),
            "pareto_run_ids": frontier_ids,
            "dominant_failure": attribution,
            "plateau_rule": "a plateau cannot stop research before every competence floor passes",
        }
    first_pass = passing_indices[0]
    later = ordered[first_pass + 1 :]
    if len(later) < 3:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "continue_three_post_floor_waves",
            "continue_required": True,
            "competent_champion_exists": True,
            "first_passing_run_id": ordered[first_pass].get("run_id"),
            "post_floor_waves_completed": len(later),
            "pareto_run_ids": frontier_ids,
            "dominant_failure": _failure_attribution(latest),
        }

    improvements: list[dict[str, Any]] = []
    incumbent = ordered[first_pass]
    for candidate in later:
        incumbent_probability = incumbent["probability_metrics"]["hybrid"]
        candidate_probability = candidate["probability_metrics"]["hybrid"]
        incumbent_trading = incumbent["trading_metrics"]
        candidate_trading = candidate["trading_metrics"]
        proper_score_improvement = max(
            1.0 - float(candidate_probability["log_loss"]) / float(incumbent_probability["log_loss"]),
            1.0 - float(candidate_probability["ranked_probability_score"]) / float(incumbent_probability["ranked_probability_score"]),
        )
        dsr_improvement = (
            float(candidate_trading["deflated_sharpe"]["deflated_sharpe"])
            - float(incumbent_trading["deflated_sharpe"]["deflated_sharpe"])
        )
        calmar_improvement = float(candidate_trading["calmar"]) - float(incumbent_trading["calmar"])
        material = proper_score_improvement >= 0.01 or dsr_improvement >= 0.1 or calmar_improvement >= 0.1
        improvements.append({
            "run_id": candidate.get("run_id"),
            "proper_score_improvement": proper_score_improvement,
            "deflated_sharpe_improvement": dsr_improvement,
            "calmar_improvement": calmar_improvement,
            "material": material,
        })
        if material and candidate.get("competence_gates", {}).get("passed"):
            incumbent = candidate
    last_three = improvements[-3:]
    plateau = len(last_three) == 3 and not any(item["material"] for item in last_three)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_floor_plateau_stop_eligible" if plateau else "continue_research_material_improvement",
        "continue_required": not plateau,
        "competent_champion_exists": True,
        "first_passing_run_id": ordered[first_pass].get("run_id"),
        "post_floor_waves_completed": len(later),
        "last_three_improvements": last_three,
        "pareto_run_ids": frontier_ids,
        "dominant_failure": _failure_attribution(latest),
    }


def load_reports(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
