from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.interfaces import ExperimentRecord, canonical_json_hash  # noqa: E402
from src.research.registry import ExperimentRegistry, sha256_file  # noqa: E402
from src.research.return_optimizer import (  # noqa: E402
    build_candidates,
    candidate_daily_return_matrix,
    exact_strategy_metrics,
    filter_ledger,
    fixed_signal_stress,
    walk_forward_filter_selection,
    walk_forward_kelly_selection,
)
from src.research.statistics import (  # noqa: E402
    block_bootstrap_pnl,
    daily_returns,
    deflated_sharpe_confidence,
    probability_of_backtest_overfitting,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _evaluation_dates(aligned_path: Path, folds: list[dict]) -> list[str]:
    available = pd.read_parquet(aligned_path, columns=["target_date"])["target_date"]
    dates = pd.to_datetime(available, errors="raise").dt.normalize().drop_duplicates().sort_values()
    selected: list[str] = []
    for fold in folds:
        mask = dates.between(pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"]))
        fold_dates = dates[mask].dt.date.astype(str).tolist()
        if len(fold_dates) != int(fold["n_validation_days"]):
            raise ValueError(
                f"evaluation calendar mismatch for {fold['fold_id']}: "
                f"expected {fold['n_validation_days']}, found {len(fold_dates)}"
            )
        selected.extend(fold_dates)
    return sorted(set(selected))


def _report_markdown(report: dict) -> str:
    primary = report["primary_oos_trading_metrics"]
    baseline = report["baseline_oos_trading_metrics"]
    sizing = report["sizing_sensitivity"]["metrics"]
    probability = report["probability_metrics"]
    rows = [
        "# Return Optimization Report",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "## Decision",
        "",
        "The primary result remains a one-contract historical proxy. The larger sizing result is a "
        "counterfactual sensitivity only because historical top-of-book depth is unavailable. Live trading remains disabled.",
        "",
        "## Strictly out-of-sample trading metrics",
        "",
        "| Metric | Walk-forward optimized | Unfiltered baseline | No-depth sizing sensitivity |",
        "|---|---:|---:|---:|",
    ]
    for label, key, percent in [
        ("CAGR", "cagr", True),
        ("Total return", "total_return", True),
        ("Net P&L", "net_pnl", False),
        ("Maximum drawdown", "max_drawdown_fraction", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Calmar", "calmar", False),
        ("Profit factor", "profit_factor", False),
        ("Trades", "n_trades", False),
        ("Traded event-days", "n_event_days", False),
    ]:
        values = [primary.get(key), baseline.get(key), sizing.get(key)]
        if percent:
            formatted = ["n/a" if value is None else f"{100 * value:.4f}%" for value in values]
        elif key == "net_pnl":
            formatted = [f"${value:.2f}" for value in values]
        elif key in {"n_trades", "n_event_days"}:
            formatted = [str(int(value)) for value in values]
        else:
            formatted = ["n/a" if value is None else f"{value:.6f}" for value in values]
        rows.append(f"| {label} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")
    rows.extend(
        [
            "",
            "## Exact probability metrics",
            "",
            f"- Weather NLL: {probability['weather']['log_loss']:.12f}; frozen champion NLL: "
            f"{probability['frozen_champion']['log_loss']:.12f}; skill: {probability['log_loss_skill']:.12f}.",
            f"- Weather RPS: {probability['weather']['ranked_probability_score']:.12f}; frozen champion RPS: "
            f"{probability['frozen_champion']['ranked_probability_score']:.12f}; skill: {probability['rps_skill']:.12f}.",
            f"- Hybrid NLL: {probability['hybrid']['log_loss']:.12f}; coherent-market NLL: "
            f"{probability['coherent_market']['log_loss']:.12f}; skill: {probability['hybrid_log_loss_skill']:.12f}.",
            f"- Hybrid RPS: {probability['hybrid']['ranked_probability_score']:.12f}; coherent-market RPS: "
            f"{probability['coherent_market']['ranked_probability_score']:.12f}; skill: {probability['hybrid_rps_skill']:.12f}.",
            f"- Trading-distribution ECE: {probability['ece']:.12f}; weather 80%/90% coverage: "
            f"{probability['weather']['coverage_80']:.12f}/{probability['weather']['coverage_90']:.12f}.",
            "",
            "## Walk-forward selections",
            "",
        ]
    )
    for fold in report["folds"]:
        rows.append(
            f"- {fold['fold_id']}: `{fold['selected_candidate']}`; CAGR {100 * fold['metrics']['cagr']:.4f}%; "
            f"drawdown {100 * fold['metrics']['max_drawdown_fraction']:.4f}%; P&L ${fold['metrics']['net_pnl']:.2f}."
        )
    rows.extend(
        [
            "",
            "## Evidence limits",
            "",
            f"- Evidence label: `{report['evidence_label']}`.",
            f"- Drawdown constraint: {100 * report['hard_max_drawdown_fraction']:.1f}% maximum; observed "
            f"{100 * primary['max_drawdown_fraction']:.4f}%.",
            f"- Fixed-signal doubled-fee P&L: ${report['robustness']['fixed_signal_doubled_fee_net_pnl']:.2f}; "
            f"two-extra-tick P&L: ${report['robustness']['fixed_signal_two_extra_tick_net_pnl']:.2f}.",
            f"- PBO across the preregistered filter grid: {report['robustness']['pbo']['pbo']}. "
            f"The upstream threshold search PBO remains {report['source_threshold_search_pbo']}; the two estimates "
            "measure different search layers and the lower filter-grid estimate does not erase the upstream failure.",
            "- No historical depth exists, so the multi-contract CAGR is not executable-fill evidence and cannot support promotion.",
        ]
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Causally optimize OOS CAGR under a hard drawdown ceiling.")
    parser.add_argument("--config", default="config/return_optimization.yaml")
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_dir = (ROOT / config["input_run"]).resolve()
    report_path = input_dir / "report.json"
    ledger_path = input_dir / "ledger.csv"
    aligned_path = (ROOT / config["evaluation_calendar_source"]).resolve()
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    ledger = pd.read_csv(ledger_path)
    folds = source_report["folds"]
    evaluation_dates = _evaluation_dates(aligned_path, folds)
    candidates = build_candidates(config["candidate_space"])
    starting_equity = float(config["starting_equity"])
    max_drawdown = float(config["hard_max_drawdown_fraction"])

    optimized, selection_table, fold_reports = walk_forward_filter_selection(
        ledger,
        folds,
        evaluation_dates,
        candidates,
        starting_equity=starting_equity,
        hard_max_drawdown_fraction=max_drawdown,
        first_fold_candidate=str(config["selection_policy"]["first_fold_candidate"]),
        minimum_prior_trades=int(config["selection_policy"]["minimum_prior_trades"]),
        require_positive_each_prior_fold=bool(config["selection_policy"]["require_positive_each_prior_fold"]),
    )
    primary_metrics = exact_strategy_metrics(optimized, evaluation_dates, starting_equity=starting_equity)
    baseline_metrics = exact_strategy_metrics(ledger, evaluation_dates, starting_equity=starting_equity)
    if abs(float(primary_metrics["max_drawdown_fraction"])) > max_drawdown + 1e-12:
        raise RuntimeError("selected OOS strategy breached the hard 15% drawdown constraint")

    matrix = candidate_daily_return_matrix(
        ledger, candidates, evaluation_dates, starting_equity=starting_equity
    )
    returns = daily_returns(optimized, starting_cash=starting_equity, evaluation_dates=evaluation_dates)
    dsr = deflated_sharpe_confidence(
        returns,
        trials=len(candidates),
        observed_sharpe=primary_metrics["sharpe"],
    )
    pbo = probability_of_backtest_overfitting(matrix)
    bootstrap = block_bootstrap_pnl(
        optimized,
        block_days=15,
        n_resamples=4000,
        seed=20260831,
        evaluation_dates=evaluation_dates,
    )

    sensitivity_config = config["sizing_sensitivity"]
    sized_ledger, sizing_folds, selected_fractions = walk_forward_kelly_selection(
        optimized,
        folds,
        evaluation_dates,
        [float(value) for value in sensitivity_config["lower_confidence_kelly_fractions"]],
        starting_equity=starting_equity,
        hard_max_drawdown_fraction=max_drawdown,
        fee_rate=float(config["execution"]["fee_rate"]),
        max_market_fraction=float(sensitivity_config["max_market_fraction"]),
        max_event_fraction=float(sensitivity_config["max_event_fraction"]),
        max_total_exposure_fraction=float(sensitivity_config["max_total_exposure_fraction"]),
        max_daily_loss_fraction=float(sensitivity_config["max_daily_loss_fraction"]),
    )
    sizing_metrics = exact_strategy_metrics(sized_ledger, evaluation_dates, starting_equity=starting_equity)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-return-optimization"
    output_dir = ROOT / "outputs" / "research" / "return_optimization" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    optimized.to_csv(output_dir / "primary_oos_ledger.csv", index=False)
    sized_ledger.to_csv(output_dir / "sizing_sensitivity_ledger.csv", index=False)
    selection_table.to_csv(output_dir / "candidate_selection.csv", index=False)

    robustness = {
        "block_bootstrap": bootstrap,
        "deflated_sharpe": dsr,
        "pbo": pbo,
        "fixed_signal_doubled_fee_net_pnl": fixed_signal_stress(
            optimized, fee_rate=2.0 * float(config["execution"]["fee_rate"])
        ),
        "fixed_signal_two_extra_tick_net_pnl": fixed_signal_stress(
            optimized,
            fee_rate=float(config["execution"]["fee_rate"]),
            extra_adverse_ticks=2,
        ),
        "drawdown_constraint_passed": abs(float(primary_metrics["max_drawdown_fraction"])) <= max_drawdown,
    }
    report = {
        "run_id": run_id,
        "generated_at_utc": now.isoformat(),
        "hypothesis": "A causal meta-filter and lower-bound sizing policy can increase OOS CAGR without exceeding 15% drawdown.",
        "objective": config["objective"],
        "evidence_label": "historical_proxy_validated_one_contract",
        "hard_max_drawdown_fraction": max_drawdown,
        "selection_trial_count": len(candidates),
        "evaluation_start": min(evaluation_dates),
        "evaluation_end": max(evaluation_dates),
        "n_evaluation_days": len(evaluation_dates),
        "primary_oos_trading_metrics": primary_metrics,
        "baseline_oos_trading_metrics": baseline_metrics,
        "folds": fold_reports,
        "probability_metrics": source_report["probability_metrics"],
        "robustness": robustness,
        "source_threshold_search_pbo": source_report["trading_metrics"].get("pbo"),
        "sizing_sensitivity": {
            "evidence_label": sensitivity_config["evidence_label"],
            "historical_depth_available": False,
            "selected_fraction_by_fold": selected_fractions,
            "fold_selection": sizing_folds,
            "metrics": sizing_metrics,
        },
        "safety": {
            "mode": "shadow",
            "trading_enabled": False,
            "live_auto_enabled": False,
            "promotion_decision": "retain_champion",
            "reason": "Historical depth is unavailable and the competence gates still include an excessive PBO and insufficient event-days.",
        },
        "input_hashes": {
            "source_report": sha256_file(report_path),
            "source_ledger": sha256_file(ledger_path),
            "evaluation_calendar": sha256_file(aligned_path),
            "config": sha256_file(config_path),
        },
    }
    report = _json_safe(report)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8")

    elapsed = time.perf_counter() - started
    champion_manifest = ROOT / "outputs" / "research" / "baseline" / "champion_manifest_v2.json"
    requirements = ROOT / "requirements.txt"
    record = ExperimentRecord(
        experiment_id=run_id,
        created_at_utc=now,
        hypothesis=report["hypothesis"],
        family="walk_forward_strategy_filter_and_sizing",
        seed=20260831,
        data_hash=canonical_json_hash(report["input_hashes"]),
        model_hash=sha256_file(champion_manifest),
        config_hash=sha256_file(config_path),
        package_hash=sha256_file(requirements),
        folds=tuple(fold_reports),
        trial_count=len(candidates),
        elapsed_seconds=elapsed,
        probability_metrics=source_report["probability_metrics"],
        trading_metrics=primary_metrics,
        robustness_tests=robustness,
        promotion_decision="retain_champion",
        evidence_label="historical_proxy_validated_one_contract",
    )
    registry = ExperimentRegistry(
        ROOT / "outputs" / "research" / "experiments.sqlite",
        ROOT / "outputs" / "research" / "manifests",
    )
    registry.register(record)
    print(output_dir)


if __name__ == "__main__":
    main()
