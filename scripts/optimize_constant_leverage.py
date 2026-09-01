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
    calendar_for_fold,
    exact_strategy_metrics,
    replay_constant_contract_multiplier,
)
from src.research.statistics import (  # noqa: E402
    block_bootstrap_pnl,
    daily_returns,
    deflated_sharpe_confidence,
)


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _all_evaluation_dates(aligned_path: Path, folds: list[dict]) -> list[str]:
    available = pd.read_parquet(aligned_path, columns=["target_date"])["target_date"]
    dates = pd.to_datetime(available, errors="raise").dt.normalize().drop_duplicates().sort_values()
    result = []
    for fold in folds:
        fold_dates = calendar_for_fold(dates, fold)
        if len(fold_dates) != int(fold["n_evaluation_days"]):
            raise ValueError(f"calendar mismatch for {fold['fold_id']}")
        result.extend(fold_dates)
    return sorted(set(result))


def _daily_correlation(left: pd.DataFrame, right: pd.DataFrame, dates: list[str]) -> float:
    a = daily_returns(left, starting_cash=1.0, evaluation_dates=dates)
    b = daily_returns(right, starting_cash=1.0, evaluation_dates=dates)
    return float(a.corr(b))


def _markdown(report: dict) -> str:
    base = report["baseline_metrics"]
    chosen = report["selected_metrics"]
    stress = report["stress_tests"]
    return f"""# Constant-Leverage Optimization

Run: `{report['run_id']}`

## Result

The smallest fully funded constant-contract multiplier satisfying all requested
constraints is **{report['selected_contract_multiplier']}x**. It changes no signal,
side, price filter, or relative contract weight.

| Metric | One contract | {report['selected_contract_multiplier']}x contracts |
|---|---:|---:|
| CAGR | {100 * base['cagr']:.6f}% | {100 * chosen['cagr']:.6f}% |
| Total return | {100 * base['total_return']:.6f}% | {100 * chosen['total_return']:.6f}% |
| Net P&L | ${base['net_pnl']:.2f} | ${chosen['net_pnl']:.2f} |
| Maximum drawdown | {100 * base['max_drawdown_fraction']:.6f}% | {100 * chosen['max_drawdown_fraction']:.6f}% |
| Sharpe | {base['sharpe']:.6f} | {chosen['sharpe']:.6f} |
| Sortino | {base['sortino']:.6f} | {chosen['sortino']:.6f} |
| Calmar | {base['calmar']:.6f} | {chosen['calmar']:.6f} |
| Profit factor | {base['profit_factor']:.6f} | {chosen['profit_factor']:.6f} |

## Constraints and execution diagnostics

- Target CAGR: {100 * report['constraints']['target_cagr']:.1f}%.
- Minimum Sharpe: {report['constraints']['minimum_sharpe']:.1f}.
- Maximum drawdown: {100 * report['constraints']['maximum_drawdown_fraction']:.1f}%.
- Signals preserved: {report['execution_diagnostics']['executed_signals']} of {report['execution_diagnostics']['input_signals']}.
- Maximum concurrent cash committed: ${report['execution_diagnostics']['maximum_concurrent_cash_committed']:.2f} ({100 * report['execution_diagnostics']['maximum_concurrent_cash_committed_fraction']:.2f}% of initial equity).
- Daily P&L correlation with one-contract strategy: {report['daily_pnl_correlation']:.12f}.
- Doubled-fee CAGR/P&L: {100 * stress['doubled_fee']['metrics']['cagr']:.6f}% / ${stress['doubled_fee']['metrics']['net_pnl']:.2f}.
- Two-additional-tick CAGR/P&L: {100 * stress['two_extra_ticks']['metrics']['cagr']:.6f}% / ${stress['two_extra_ticks']['metrics']['net_pnl']:.2f}.

## Evidence limit

This is a constant-size sensitivity on previously inspected OOS signals, not a
new untouched test. Historical top-of-book depth is unavailable, so {report['selected_contract_multiplier']}
contracts per signal cannot be claimed executable. Live trading remains disabled.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Find constant leverage that preserves OOS Sharpe.")
    parser.add_argument("--config", default="config/constant_leverage_optimization.yaml")
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_dir = (ROOT / config["input_run"]).resolve()
    input_report_path = input_dir / "report.json"
    input_ledger_path = input_dir / config["input_ledger"]
    aligned_path = (ROOT / config["evaluation_calendar_source"]).resolve()
    source_report = json.loads(input_report_path.read_text(encoding="utf-8"))
    source_ledger = pd.read_csv(input_ledger_path)
    folds = source_report["folds"]
    dates = _all_evaluation_dates(aligned_path, folds)
    starting_equity = float(config["starting_equity"])
    constraints = config["constraints"]
    fee_rate = float(config["execution"]["fee_rate"])

    grid_rows = []
    ledgers = {}
    diagnostics = {}
    for multiplier in [int(value) for value in config["candidate_contract_multipliers"]]:
        ledger, diagnostic = replay_constant_contract_multiplier(
            source_ledger,
            starting_equity=starting_equity,
            contract_multiplier=multiplier,
            fee_rate=fee_rate,
            fully_funded_only=bool(config["execution"]["fully_funded_only"]),
        )
        metrics = exact_strategy_metrics(ledger, dates, starting_equity=starting_equity)
        fold_pnl = {
            str(fold["fold_id"]): float(
                ledger[ledger["outer_fold_id"].astype(str).eq(str(fold["fold_id"]))]["net_pnl"].sum()
            )
            for fold in folds
        }
        preserves_signals = int(diagnostic["executed_signals"]) == int(diagnostic["input_signals"])
        eligible = (
            float(metrics["cagr"]) >= float(constraints["target_cagr"])
            and float(metrics["sharpe"]) >= float(constraints["minimum_sharpe"])
            and abs(float(metrics["max_drawdown_fraction"])) <= float(constraints["maximum_drawdown_fraction"])
            and (not bool(constraints["require_positive_each_outer_fold"]) or all(value > 0 for value in fold_pnl.values()))
            and (not bool(config["execution"]["preserve_every_signal_or_reject_candidate"]) or preserves_signals)
        )
        grid_rows.append(
            {
                "contract_multiplier": multiplier,
                "eligible": eligible,
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "calmar": metrics["calmar"],
                "max_drawdown_fraction": metrics["max_drawdown_fraction"],
                "net_pnl": metrics["net_pnl"],
                "profit_factor": metrics["profit_factor"],
                "executed_signals": diagnostic["executed_signals"],
                "rejected_for_cash": diagnostic["rejected_for_cash"],
                "max_concurrent_cash_committed": diagnostic["maximum_concurrent_cash_committed"],
                "fold_pnl": fold_pnl,
            }
        )
        ledgers[multiplier] = ledger
        diagnostics[multiplier] = diagnostic
    grid = pd.DataFrame(grid_rows)
    eligible = grid[grid["eligible"]]
    if eligible.empty:
        nearest = grid[grid["sharpe"].ge(float(constraints["minimum_sharpe"]))].sort_values(
            ["cagr", "contract_multiplier"], ascending=[False, True]
        )
        if nearest.empty:
            raise RuntimeError("no leverage candidate retained the minimum Sharpe")
        selected_multiplier = int(nearest.iloc[0]["contract_multiplier"])
        target_reached = False
    else:
        selected_multiplier = int(eligible["contract_multiplier"].min())
        target_reached = True
    selected_ledger = ledgers[selected_multiplier]
    selected_metrics = exact_strategy_metrics(selected_ledger, dates, starting_equity=starting_equity)
    baseline_metrics = exact_strategy_metrics(source_ledger, dates, starting_equity=starting_equity)

    doubled_ledger, doubled_diag = replay_constant_contract_multiplier(
        source_ledger,
        starting_equity=starting_equity,
        contract_multiplier=selected_multiplier,
        fee_rate=float(config["stress_tests"]["doubled_fee_rate"]),
        fully_funded_only=True,
    )
    tick_ledger, tick_diag = replay_constant_contract_multiplier(
        source_ledger,
        starting_equity=starting_equity,
        contract_multiplier=selected_multiplier,
        fee_rate=fee_rate,
        additional_adverse_ticks=int(config["stress_tests"]["additional_adverse_ticks"]),
        fully_funded_only=True,
    )
    stress = {
        "doubled_fee": {
            "metrics": exact_strategy_metrics(doubled_ledger, dates, starting_equity=starting_equity),
            "diagnostics": doubled_diag,
        },
        "two_extra_ticks": {
            "metrics": exact_strategy_metrics(tick_ledger, dates, starting_equity=starting_equity),
            "diagnostics": tick_diag,
        },
    }
    returns = daily_returns(selected_ledger, starting_cash=starting_equity, evaluation_dates=dates)
    dsr = deflated_sharpe_confidence(
        returns,
        trials=len(grid),
        observed_sharpe=selected_metrics["sharpe"],
    )
    bootstrap = block_bootstrap_pnl(
        selected_ledger,
        block_days=15,
        n_resamples=4000,
        seed=20260831,
        evaluation_dates=dates,
    )
    fold_metrics = []
    for fold in folds:
        fold_dates = calendar_for_fold(dates, fold)
        part = selected_ledger[selected_ledger["outer_fold_id"].astype(str).eq(str(fold["fold_id"]))]
        fold_metrics.append(
            {
                "fold_id": fold["fold_id"],
                "metrics": exact_strategy_metrics(part, fold_dates, starting_equity=starting_equity),
            }
        )

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-constant-leverage"
    output_dir = ROOT / "outputs" / "research" / "leverage_optimization" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_ledger.to_csv(output_dir / "selected_leverage_ledger.csv", index=False)
    grid.to_csv(output_dir / "leverage_grid.csv", index=False)
    report = _safe(
        {
            "run_id": run_id,
            "generated_at_utc": now.isoformat(),
            "objective": config["objective"],
            "constraints": constraints,
            "target_reached": target_reached,
            "selected_contract_multiplier": selected_multiplier,
            "baseline_metrics": baseline_metrics,
            "selected_metrics": selected_metrics,
            "execution_diagnostics": diagnostics[selected_multiplier],
            "daily_pnl_correlation": _daily_correlation(source_ledger, selected_ledger, dates),
            "folds": fold_metrics,
            "stress_tests": stress,
            "block_bootstrap": bootstrap,
            "deflated_sharpe": dsr,
            "probability_metrics": source_report["probability_metrics"],
            "evidence_label": config["reporting"]["evidence_label"],
            "safety": {
                "mode": "shadow",
                "trading_enabled": False,
                "live_auto_enabled": False,
                "promotion_decision": "retain_champion",
            },
            "input_hashes": {
                "source_report": sha256_file(input_report_path),
                "source_ledger": sha256_file(input_ledger_path),
                "evaluation_calendar": sha256_file(aligned_path),
                "config": sha256_file(config_path),
            },
        }
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")

    record = ExperimentRecord(
        experiment_id=run_id,
        created_at_utc=now,
        hypothesis="Constant-contract scaling can reach 40% CAGR without changing the six-Sharpe signal shape.",
        family="constant_contract_leverage",
        seed=20260831,
        data_hash=canonical_json_hash(report["input_hashes"]),
        model_hash=sha256_file(ROOT / "outputs" / "research" / "baseline" / "champion_manifest_v2.json"),
        config_hash=sha256_file(config_path),
        package_hash=sha256_file(ROOT / "requirements.txt"),
        folds=tuple(fold_metrics),
        trial_count=len(grid),
        elapsed_seconds=time.perf_counter() - started,
        probability_metrics=source_report["probability_metrics"],
        trading_metrics=selected_metrics,
        robustness_tests={"stress_tests": stress, "block_bootstrap": bootstrap, "deflated_sharpe": dsr},
        promotion_decision="retain_champion",
        evidence_label=config["reporting"]["evidence_label"],
    )
    ExperimentRegistry(
        ROOT / "outputs" / "research" / "experiments.sqlite",
        ROOT / "outputs" / "research" / "manifests",
    ).register(record)
    print(output_dir)


if __name__ == "__main__":
    main()
