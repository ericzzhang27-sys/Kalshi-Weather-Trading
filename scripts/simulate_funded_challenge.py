"""Sweep Kelly sizing against the funded-challenge rules and report the
config with the best pass probability.

Run from the repo root::

    python scripts/simulate_funded_challenge.py

Reads rules from config/funded_challenge.yaml, rebuilds the aligned dataset
once, runs the backtest engine per Kelly fraction on the challenge bankroll,
replays each ledger through src.backtest.challenge, and bootstraps pass
probability by resampling trading days. Writes a Markdown report to
outputs/backtests/funded_challenge_report.md.

Simulation only - never places orders.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from src.backtest.align_probabilities import (  # noqa: E402
    align_probabilities_with_markets,
    load_model_probabilities,
)
from src.backtest.challenge import ChallengeRules, bootstrap_pass_rate, simulate_challenge  # noqa: E402
from src.backtest.engine import EngineConfig, run_backtest  # noqa: E402
from src.backtest.sizing import SizingConfig  # noqa: E402
from src.backtest.strategies import StrategyConfig  # noqa: E402
from src.kalshi.normalize_markets import normalize_historical_markets  # noqa: E402

RULES_PATH = REPO_ROOT / "config" / "funded_challenge.yaml"
REPORT_PATH = REPO_ROOT / "outputs" / "backtests" / "funded_challenge_report.md"
KELLY_SWEEP = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.25, 0.40]
BOOTSTRAP_PATHS = 500
# Challenge has a 30-day time limit -> ~21 trading days.
BOOTSTRAP_MAX_DAYS = 21


def load_rules() -> ChallengeRules:
    raw = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    acct = raw["account"]
    r = raw["rules"]
    return ChallengeRules(
        starting_balance=float(acct["starting_balance"]),
        profit_target_pct=float(r["profit_target_pct"]),
        max_daily_loss_pct=float(r["max_daily_loss_pct"]),
        breach_action=str(r.get("max_daily_loss_breach_action", "fail")),
        drawdown_mode=str(r.get("drawdown_mode", "static_from_initial")),
        min_trading_days=int(r.get("min_trading_days", 5)),
        per_market_notional_cap_pct=float(r["per_market_notional_cap_pct"]),
    )


def build_aligned(city: str = "NYC") -> pd.DataFrame:
    markets = pd.read_csv(REPO_ROOT / "data/kalshi/processed/historical_markets_processed.csv")
    candles = pd.read_csv(REPO_ROOT / "data/kalshi/processed/historical_candles_processed.csv")
    canonical = normalize_historical_markets(markets, candles,
                                             processed_dir=REPO_ROOT / "data/kalshi/processed")
    prob_df = load_model_probabilities()
    return align_probabilities_with_markets(prob_df, canonical, city=city)


def ledger_for_fraction(aligned: pd.DataFrame, rules: ChallengeRules,
                        fraction: float) -> pd.DataFrame:
    sizing = SizingConfig(
        method="kelly_fractional",
        bankroll=rules.starting_balance,
        kelly_fraction=fraction,
        max_dollars_per_market=rules.per_market_notional_cap,
        max_dollars_per_event=rules.per_market_notional_cap * 1.5,
        max_contracts_per_market=1_000_000.0,   # notional cap is the binding limit
        max_daily_exposure=float("inf"),
    )
    eng = EngineConfig(strategy="A", threshold=0.05,
                       strategy_config=StrategyConfig(threshold=0.05),
                       sizing_config=sizing, trade_policy="one_position_per_market",
                       entry_cutoff_hour_ny=16)
    return run_backtest(aligned.copy(), eng)


def main() -> None:
    rules = load_rules()
    logger.info("Rules: %s", rules)
    aligned = build_aligned()
    rows = []
    for frac in KELLY_SWEEP:
        ledger = ledger_for_fraction(aligned, rules, frac)
        if ledger.empty:
            logger.warning("No trades for fraction %.2f", frac)
            continue
        outcome = simulate_challenge(ledger, rules)
        boot = bootstrap_pass_rate(ledger, rules, n_paths=BOOTSTRAP_PATHS,
                                   max_days=BOOTSTRAP_MAX_DAYS)
        rows.append({
            "kelly_fraction": frac, **outcome.__dict__,
            "boot_pass_rate": round(boot["pass_rate"], 3),
            "boot_median_days": boot["median_days_to_resolution"],
            "boot_outcomes": boot["outcome_counts"],
        })
        logger.info("frac=%.2f -> %s (boot pass rate %.1f%%)",
                    frac, outcome.result, 100 * boot["pass_rate"])

    table = pd.DataFrame(rows)
    passed = table[table["result"] == "passed"]
    best = (passed.sort_values("boot_pass_rate", ascending=False).iloc[0]
            if len(passed) else table.sort_values("boot_pass_rate", ascending=False).iloc[0])

    lines = [
        "# Funded challenge simulation report",
        "",
        f"- Rules: target +{rules.profit_target_pct:.0%}, daily loss "
        f"-{rules.max_daily_loss_pct:.0%} ({rules.breach_action}), drawdown "
        f"-{rules.max_total_drawdown_pct:.0%} ({rules.drawdown_mode}), "
        f"min trading days {rules.min_trading_days}, per-market cap "
        f"{rules.per_market_notional_cap_pct:.0%}",
        f"- Bankroll ${rules.starting_balance:,.0f}; strategy A, threshold 0.05, "
        "entry cutoff 4 PM ET; single historical path plus day-bootstrap",
        f"- Bootstrap paths per config: {BOOTSTRAP_PATHS}",
        "",
        "| Kelly frac | Result | Days | Final equity | Max DD | Worst day | Trades | Boot pass % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['kelly_fraction']:.2f} | {r['result']} | {r['days_traded']} "
            f"| ${r['final_equity']:,.0f} | ${r['max_drawdown']:,.0f} "
            f"| ${r['worst_day_pnl']:,.0f} | {r['trades_taken']} "
            f"| {100 * r['boot_pass_rate']:.1f}% |")
    lines += ["", f"**Recommended kelly_fraction: {best['kelly_fraction']}** "
              f"(historical: {best['result']}, bootstrap pass rate "
              f"{100 * best['boot_pass_rate']:.1f}%, median resolution "
              f"{best['boot_median_days']} days)", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(table.drop(columns=["boot_outcomes"]).to_string(index=False))
    print(f"\nRecommended kelly_fraction: {best['kelly_fraction']}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
