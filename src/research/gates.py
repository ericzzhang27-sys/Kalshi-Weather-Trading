from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class CompetenceThresholds:
    minimum_log_loss_skill: float = 0.0
    minimum_rps_skill: float = 0.0
    maximum_ece: float = 0.02
    maximum_coverage_error: float = 0.03
    minimum_hybrid_skill: float = 0.0
    minimum_trades: int = 250
    minimum_event_days: int = 180
    minimum_bootstrap_pnl_lower: float = 0.0
    minimum_profit_factor: float = 1.25
    minimum_dsr_confidence: float = 0.95
    maximum_pbo: float = 0.20
    minimum_sharpe: float = 1.5
    minimum_sortino: float = 2.0
    minimum_calmar: float = 1.5
    maximum_drawdown_fraction: float = 0.10
    minimum_positive_fold_fraction: float = 0.75
    minimum_doubled_cost_pnl: float = 0.0
    minimum_two_tick_pnl: float = 0.0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_competence_gates(
    probability: Mapping[str, Any],
    trading: Mapping[str, Any],
    robustness: Mapping[str, Any],
    thresholds: CompetenceThresholds | None = None,
) -> dict[str, Any]:
    floor = thresholds or CompetenceThresholds()
    checks = {
        "weather_log_loss_skill_positive": _finite(probability.get("log_loss_skill")) and float(probability["log_loss_skill"]) > floor.minimum_log_loss_skill,
        "weather_rps_skill_positive": _finite(probability.get("rps_skill")) and float(probability["rps_skill"]) > floor.minimum_rps_skill,
        "weather_log_loss_skill_statistically_positive": _finite(probability.get("log_loss_skill_ci_2_5")) and float(probability["log_loss_skill_ci_2_5"]) > floor.minimum_log_loss_skill,
        "weather_rps_skill_statistically_positive": _finite(probability.get("rps_skill_ci_2_5")) and float(probability["rps_skill_ci_2_5"]) > floor.minimum_rps_skill,
        "calibration_ece": _finite(probability.get("ece")) and float(probability["ece"]) <= floor.maximum_ece,
        "coverage_80": _finite(probability.get("coverage_error_80")) and abs(float(probability["coverage_error_80"])) <= floor.maximum_coverage_error,
        "coverage_90": _finite(probability.get("coverage_error_90")) and abs(float(probability["coverage_error_90"])) <= floor.maximum_coverage_error,
        "hybrid_skill_positive": _finite(probability.get("hybrid_log_loss_skill")) and float(probability["hybrid_log_loss_skill"]) > floor.minimum_hybrid_skill,
        "trade_count": int(trading.get("n_trades", 0)) >= floor.minimum_trades,
        "event_day_count": int(trading.get("n_event_days", 0)) >= floor.minimum_event_days,
        "bootstrap_net_pnl_lower": _finite(trading.get("net_pnl_ci_2_5")) and float(trading["net_pnl_ci_2_5"]) > floor.minimum_bootstrap_pnl_lower,
        "profit_factor": _finite(trading.get("profit_factor")) and float(trading["profit_factor"]) >= floor.minimum_profit_factor,
        "deflated_sharpe_confidence": _finite(trading.get("dsr_confidence")) and float(trading["dsr_confidence"]) >= floor.minimum_dsr_confidence,
        "pbo": _finite(trading.get("pbo")) and float(trading["pbo"]) < floor.maximum_pbo,
        "sharpe": _finite(trading.get("sharpe")) and float(trading["sharpe"]) >= floor.minimum_sharpe,
        "sortino": _finite(trading.get("sortino")) and float(trading["sortino"]) >= floor.minimum_sortino,
        "calmar": _finite(trading.get("calmar")) and float(trading["calmar"]) >= floor.minimum_calmar,
        "drawdown": _finite(trading.get("max_drawdown_fraction")) and abs(float(trading["max_drawdown_fraction"])) <= floor.maximum_drawdown_fraction,
        "rolling_folds": _finite(robustness.get("positive_fold_fraction")) and float(robustness["positive_fold_fraction"]) >= floor.minimum_positive_fold_fraction,
        "doubled_costs": _finite(robustness.get("doubled_cost_net_pnl")) and float(robustness["doubled_cost_net_pnl"]) >= floor.minimum_doubled_cost_pnl,
        "two_adverse_ticks": _finite(robustness.get("two_tick_net_pnl")) and float(robustness["two_tick_net_pnl"]) >= floor.minimum_two_tick_pnl,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_gates": failed,
        "thresholds": asdict(floor),
        "promotion_decision": "eligible_for_manual_promotion" if not failed else "retain_champion",
    }


def pareto_frontier(rows: list[Mapping[str, Any]], objectives: Mapping[str, str]) -> list[int]:
    """Return indices not dominated under min/max objective directions."""
    if any(direction not in {"min", "max"} for direction in objectives.values()):
        raise ValueError("objective directions must be 'min' or 'max'")
    frontier: list[int] = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            weak = []
            strict = []
            for key, direction in objectives.items():
                left, right = float(row[key]), float(other[key])
                weak.append(right <= left if direction == "min" else right >= left)
                strict.append(right < left if direction == "min" else right > left)
            if all(weak) and any(strict):
                dominated = True
                break
        if not dominated:
            frontier.append(index)
    return frontier
