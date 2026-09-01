from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.backtest.fees import kalshi_taker_fee
from src.backtest.metrics import ledger_metrics
from src.research.statistics import (
    block_bootstrap_pnl,
    daily_returns,
    deflated_sharpe_confidence,
    probability_of_backtest_overfitting,
)


@dataclass(frozen=True)
class StrategyCandidate:
    minimum_lower_confidence_edge: float
    local_time_regime: str
    side: str
    maximum_trades_per_event: int | None

    @property
    def candidate_id(self) -> str:
        edge = format(self.minimum_lower_confidence_edge, ".6f").rstrip("0").rstrip(".")
        edge = edge or "0"
        cap = "unlimited" if self.maximum_trades_per_event is None else str(self.maximum_trades_per_event)
        return f"edge_{edge}_{self.local_time_regime}_{self.side}_{cap}"

    @property
    def complexity(self) -> int:
        return int(self.minimum_lower_confidence_edge > 0) + int(self.local_time_regime != "all") + int(
            self.side != "both"
        ) + int(self.maximum_trades_per_event is not None)


def build_candidates(candidate_space: Mapping[str, Iterable[Any]]) -> list[StrategyCandidate]:
    caps = [None if str(value).lower() == "unlimited" else int(value) for value in candidate_space["maximum_trades_per_event"]]
    return [
        StrategyCandidate(float(edge), str(regime), str(side), cap)
        for edge, regime, side, cap in product(
            candidate_space["minimum_lower_confidence_edge"],
            candidate_space["local_time_regime"],
            candidate_space["side"],
            caps,
        )
    ]


def filter_ledger(ledger: pd.DataFrame, candidate: StrategyCandidate) -> pd.DataFrame:
    result = ledger.copy()
    result = result[
        pd.to_numeric(result["lower_confidence_edge"], errors="raise")
        >= candidate.minimum_lower_confidence_edge - 1e-12
    ]
    if candidate.side == "buy_yes":
        result = result[result["side"].eq("BUY_YES")]
    elif candidate.side == "buy_no":
        result = result[result["side"].eq("BUY_NO")]
    elif candidate.side != "both":
        raise ValueError(f"unknown side filter: {candidate.side}")

    local_hour = pd.to_datetime(result["execution_timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    ).dt.hour
    masks = {
        "all": pd.Series(True, index=result.index),
        "before_12": local_hour < 12,
        "12_to_16": (local_hour >= 12) & (local_hour < 16),
        "after_16": local_hour >= 16,
        "after_12": local_hour >= 12,
    }
    if candidate.local_time_regime not in masks:
        raise ValueError(f"unknown time regime: {candidate.local_time_regime}")
    result = result[masks[candidate.local_time_regime]]
    if candidate.maximum_trades_per_event is not None and not result.empty:
        result = (
            result.sort_values(
                ["event_ticker", "lower_confidence_edge", "execution_timestamp", "market_ticker"],
                ascending=[True, False, True, True],
                kind="stable",
            )
            .groupby("event_ticker", sort=False)
            .head(candidate.maximum_trades_per_event)
        )
    return result.sort_values(["execution_timestamp", "market_ticker"], kind="stable").reset_index(drop=True)


def calendar_for_fold(all_dates: Iterable[Any], fold: Mapping[str, Any]) -> list[str]:
    dates = pd.to_datetime(pd.Series(list(all_dates)), errors="raise").dt.normalize()
    start = pd.Timestamp(str(fold["validation_start"]))
    end = pd.Timestamp(str(fold["validation_end"]))
    return dates[(dates >= start) & (dates <= end)].dt.date.astype(str).tolist()


def exact_strategy_metrics(
    ledger: pd.DataFrame,
    evaluation_dates: Iterable[Any],
    *,
    starting_equity: float,
) -> dict[str, Any]:
    dates = list(evaluation_dates)
    result = ledger_metrics(ledger, starting_cash=starting_equity, evaluation_dates=dates)
    result["max_drawdown_fraction"] = result.pop("max_drawdown_pct")
    result["sharpe"] = result.pop("sharpe_ratio")
    result["sortino"] = result.pop("sortino_ratio")
    result["calmar"] = result.pop("calmar_ratio")
    return result


def walk_forward_filter_selection(
    ledger: pd.DataFrame,
    folds: list[Mapping[str, Any]],
    all_evaluation_dates: Iterable[Any],
    candidates: list[StrategyCandidate],
    *,
    starting_equity: float,
    hard_max_drawdown_fraction: float,
    first_fold_candidate: str,
    minimum_prior_trades: int,
    require_positive_each_prior_fold: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if first_fold_candidate not in candidate_by_id:
        raise ValueError(f"first-fold candidate is not in the preregistered space: {first_fold_candidate}")
    fold_calendars = [calendar_for_fold(all_evaluation_dates, fold) for fold in folds]
    selected_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []

    for fold_index, fold in enumerate(folds):
        fold_id = str(fold["fold_id"])
        current_calendar = fold_calendars[fold_index]
        if fold_index == 0:
            selected_candidate = candidate_by_id[first_fold_candidate]
            selection_basis = "preregistered_default_no_prior_oos_fold"
        else:
            prior_calendar = [date for dates in fold_calendars[:fold_index] for date in dates]
            prior_start = min(prior_calendar)
            prior_end = max(prior_calendar)
            prior_rows = ledger[
                pd.to_datetime(ledger["target_date"]).between(prior_start, prior_end)
            ]
            eligible: list[tuple[tuple[float, float, int, int], StrategyCandidate]] = []
            for candidate in candidates:
                filtered = filter_ledger(prior_rows, candidate)
                metrics = exact_strategy_metrics(filtered, prior_calendar, starting_equity=starting_equity)
                fold_pnl = []
                for prior_fold_index in range(fold_index):
                    prior_fold_dates = fold_calendars[prior_fold_index]
                    date_set = set(prior_fold_dates)
                    subset = filtered[filtered["target_date"].astype(str).isin(date_set)]
                    fold_pnl.append(float(subset["net_pnl"].sum()))
                drawdown_pass = abs(float(metrics["max_drawdown_fraction"])) <= hard_max_drawdown_fraction + 1e-12
                trades_pass = int(metrics["n_trades"]) >= minimum_prior_trades
                folds_pass = not require_positive_each_prior_fold or all(value > 0.0 for value in fold_pnl)
                is_eligible = drawdown_pass and trades_pass and folds_pass and float(metrics["cagr"]) > 0.0
                selection_rows.append(
                    {
                        "selection_for_fold": fold_id,
                        "candidate_id": candidate.candidate_id,
                        "eligible": is_eligible,
                        "prior_cagr": metrics["cagr"],
                        "prior_max_drawdown_fraction": metrics["max_drawdown_fraction"],
                        "prior_net_pnl": metrics["net_pnl"],
                        "prior_n_trades": metrics["n_trades"],
                        "prior_fold_pnl": fold_pnl,
                        "complexity": candidate.complexity,
                    }
                )
                if is_eligible:
                    rank_key = (
                        float(metrics["cagr"]),
                        -abs(float(metrics["max_drawdown_fraction"])),
                        int(metrics["n_trades"]),
                        -candidate.complexity,
                    )
                    eligible.append((rank_key, candidate))
            if not eligible:
                selected_candidate = candidate_by_id[first_fold_candidate]
                selection_basis = "fallback_default_no_eligible_prior_candidate"
            else:
                selected_candidate = max(eligible, key=lambda item: item[0])[1]
                selection_basis = "maximum_prior_oos_cagr_subject_to_preregistered_constraints"

        current_dates = set(current_calendar)
        current_rows = ledger[ledger["target_date"].astype(str).isin(current_dates)]
        selected = filter_ledger(current_rows, selected_candidate)
        selected["outer_fold_id"] = fold_id
        selected["selected_filter_candidate"] = selected_candidate.candidate_id
        selected_parts.append(selected)
        fold_metrics = exact_strategy_metrics(selected, current_calendar, starting_equity=starting_equity)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "n_evaluation_days": len(current_calendar),
                "selected_candidate": selected_candidate.candidate_id,
                "selection_basis": selection_basis,
                "metrics": fold_metrics,
            }
        )

    combined = pd.concat(selected_parts, ignore_index=True) if selected_parts else ledger.iloc[:0].copy()
    return combined, pd.DataFrame(selection_rows), fold_reports


def candidate_daily_return_matrix(
    ledger: pd.DataFrame,
    candidates: list[StrategyCandidate],
    evaluation_dates: Iterable[Any],
    *,
    starting_equity: float,
) -> np.ndarray:
    dates = list(evaluation_dates)
    columns = [
        daily_returns(filter_ledger(ledger, candidate), starting_cash=starting_equity, evaluation_dates=dates).to_numpy(float)
        for candidate in candidates
    ]
    return np.column_stack(columns)


def fixed_signal_stress(
    ledger: pd.DataFrame,
    *,
    fee_rate: float,
    extra_adverse_ticks: int = 0,
    tick_size: float = 0.01,
) -> float:
    total = 0.0
    for row in ledger.itertuples(index=False):
        contracts = int(row.contracts)
        price = min(0.99, float(row.entry_price) + extra_adverse_ticks * tick_size)
        fee = kalshi_taker_fee(price, contracts, fee_rate=fee_rate)
        total += float(row.settlement) * contracts - price * contracts - fee
    return float(total)


def replay_lower_bound_kelly(
    signals: pd.DataFrame,
    *,
    starting_equity: float,
    kelly_fraction_by_fold: Mapping[str, float],
    fee_rate: float,
    max_market_fraction: float,
    max_event_fraction: float,
    max_total_exposure_fraction: float,
    max_daily_loss_fraction: float,
) -> pd.DataFrame:
    cash = float(starting_equity)
    realized_pnl = 0.0
    active: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    realized_pnl_by_day: dict[str, float] = {}

    def release(now: pd.Timestamp) -> None:
        nonlocal cash, realized_pnl
        remaining = []
        for position in active:
            if position["settlement_timestamp"] <= now:
                cash += position["payout"]
                realized_pnl += position["net_pnl"]
                settlement_day = position["settlement_timestamp"].tz_convert(
                    "America/New_York"
                ).date().isoformat()
                realized_pnl_by_day[settlement_day] = (
                    realized_pnl_by_day.get(settlement_day, 0.0) + position["net_pnl"]
                )
                position["cash_after_settlement"] = cash
                completed.append(position)
            else:
                remaining.append(position)
        active[:] = remaining

    ordered = signals.copy()
    ordered["execution_timestamp"] = pd.to_datetime(ordered["execution_timestamp"], utc=True)
    ordered["settlement_timestamp"] = pd.to_datetime(ordered["settlement_timestamp"], utc=True)
    ordered = ordered.sort_values(["execution_timestamp", "market_ticker"], kind="stable")
    for row in ordered.itertuples(index=False):
        release(row.execution_timestamp)
        fraction = float(kelly_fraction_by_fold[str(row.outer_fold_id)])
        price = float(row.entry_price)
        net_edge = max(0.0, float(row.lower_confidence_edge))
        if net_edge <= 0 or not 0 < price < 1:
            continue
        equity_reference = max(0.0, starting_equity + realized_pnl)
        execution_day = row.execution_timestamp.tz_convert("America/New_York").date().isoformat()
        if realized_pnl_by_day.get(execution_day, 0.0) <= -equity_reference * max_daily_loss_fraction:
            continue
        full_kelly = net_edge / (1.0 - price)
        desired_notional = equity_reference * fraction * full_kelly
        market_cap = equity_reference * max_market_fraction
        event_cap = equity_reference * max_event_fraction
        total_cap = equity_reference * max_total_exposure_fraction
        active_event = sum(item["gross_cost"] for item in active if item["event_ticker"] == row.event_ticker)
        active_total = sum(item["gross_cost"] for item in active)
        notional_cap = min(
            desired_notional,
            market_cap,
            max(0.0, event_cap - active_event),
            max(0.0, total_cap - active_total),
            cash,
        )
        contracts = int(math.floor(notional_cap / price))
        if contracts <= 0:
            continue
        fee = kalshi_taker_fee(price, contracts, fee_rate=fee_rate)
        while contracts > 0 and price * contracts + fee > cash + 1e-12:
            contracts -= 1
            if contracts:
                fee = kalshi_taker_fee(price, contracts, fee_rate=fee_rate)
        if contracts <= 0:
            continue
        gross_cost = price * contracts
        committed = gross_cost + fee
        payout = float(row.settlement) * contracts
        cash -= committed
        record = row._asdict()
        record.update(
            {
                "contracts": contracts,
                "gross_cost": gross_cost,
                "fees": fee,
                "cash_committed": committed,
                "gross_pnl": payout - gross_cost,
                "net_pnl": payout - committed,
                "kelly_fraction": fraction,
                "evidence_label": "counterfactual_sizing_sensitivity_no_historical_depth",
                "supports_profitability_claim": False,
                "payout": payout,
            }
        )
        active.append(record)
    release(pd.Timestamp.max.tz_localize("UTC"))
    if not completed:
        return signals.iloc[:0].copy()
    return pd.DataFrame(completed).drop(columns=["payout"]).sort_values(
        ["settlement_timestamp", "market_ticker"], kind="stable"
    ).reset_index(drop=True)


def walk_forward_kelly_selection(
    signals: pd.DataFrame,
    folds: list[Mapping[str, Any]],
    all_evaluation_dates: Iterable[Any],
    fractions: list[float],
    *,
    starting_equity: float,
    hard_max_drawdown_fraction: float,
    fee_rate: float,
    max_market_fraction: float,
    max_event_fraction: float,
    max_total_exposure_fraction: float,
    max_daily_loss_fraction: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    fold_calendars = [calendar_for_fold(all_evaluation_dates, fold) for fold in folds]
    selected: dict[str, float] = {}
    reports: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        fold_id = str(fold["fold_id"])
        if fold_index == 0:
            choice = max(fractions)
            basis = "preregistered_quarter_kelly_default"
        else:
            prior_fold_ids = {str(item["fold_id"]) for item in folds[:fold_index]}
            prior_signals = signals[signals["outer_fold_id"].astype(str).isin(prior_fold_ids)]
            prior_dates = [date for dates in fold_calendars[:fold_index] for date in dates]
            ranked = []
            for fraction in fractions:
                mapping = {prior_id: fraction for prior_id in prior_fold_ids}
                replay = replay_lower_bound_kelly(
                    prior_signals,
                    starting_equity=starting_equity,
                    kelly_fraction_by_fold=mapping,
                    fee_rate=fee_rate,
                    max_market_fraction=max_market_fraction,
                    max_event_fraction=max_event_fraction,
                    max_total_exposure_fraction=max_total_exposure_fraction,
                    max_daily_loss_fraction=max_daily_loss_fraction,
                )
                metrics = exact_strategy_metrics(replay, prior_dates, starting_equity=starting_equity)
                if float(metrics["net_pnl"]) > 0 and abs(float(metrics["max_drawdown_fraction"])) <= hard_max_drawdown_fraction:
                    ranked.append((float(metrics["cagr"]), -abs(float(metrics["max_drawdown_fraction"])), fraction))
            choice = max(ranked)[2] if ranked else min(fractions)
            basis = "maximum_prior_oos_cagr_under_drawdown_cap" if ranked else "fallback_smallest_fraction"
        selected[fold_id] = float(choice)
        reports.append({"fold_id": fold_id, "selected_kelly_fraction": float(choice), "selection_basis": basis})
    replay = replay_lower_bound_kelly(
        signals,
        starting_equity=starting_equity,
        kelly_fraction_by_fold=selected,
        fee_rate=fee_rate,
        max_market_fraction=max_market_fraction,
        max_event_fraction=max_event_fraction,
        max_total_exposure_fraction=max_total_exposure_fraction,
        max_daily_loss_fraction=max_daily_loss_fraction,
    )
    return replay, reports, selected


def candidate_to_dict(candidate: StrategyCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["candidate_id"] = candidate.candidate_id
    payload["complexity"] = candidate.complexity
    return payload


def replay_constant_contract_multiplier(
    signals: pd.DataFrame,
    *,
    starting_equity: float,
    contract_multiplier: int,
    fee_rate: float,
    additional_adverse_ticks: int = 0,
    tick_size: float = 0.01,
    fully_funded_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay the same signals with an identical whole-contract multiplier.

    Unlike Kelly sizing, this preserves every signal's contract weight and
    therefore preserves the one-contract strategy shape as closely as bulk-fee
    rounding permits. Positions remain fully funded and capital stays locked
    until the recorded settlement timestamp.
    """
    if int(contract_multiplier) != contract_multiplier or contract_multiplier < 1:
        raise ValueError("contract_multiplier must be a positive whole number")
    cash = float(starting_equity)
    active: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    max_committed = 0.0
    minimum_cash = cash
    rejected_for_cash = 0

    def release(now: pd.Timestamp) -> None:
        nonlocal cash
        remaining = []
        for position in active:
            if position["settlement_timestamp"] <= now:
                cash += position["payout"]
                position["cash_after_settlement"] = cash
                completed.append(position)
            else:
                remaining.append(position)
        active[:] = remaining

    ordered = signals.copy()
    ordered["execution_timestamp"] = pd.to_datetime(ordered["execution_timestamp"], utc=True)
    ordered["settlement_timestamp"] = pd.to_datetime(ordered["settlement_timestamp"], utc=True)
    ordered = ordered.sort_values(["execution_timestamp", "market_ticker"], kind="stable")
    for row in ordered.itertuples(index=False):
        release(row.execution_timestamp)
        price = min(0.99, float(row.entry_price) + additional_adverse_ticks * tick_size)
        contracts = int(contract_multiplier)
        fee = kalshi_taker_fee(price, contracts, fee_rate=fee_rate)
        gross_cost = price * contracts
        committed = gross_cost + fee
        if fully_funded_only and committed > cash + 1e-12:
            rejected_for_cash += 1
            continue
        cash -= committed
        payout = float(row.settlement) * contracts
        record = row._asdict()
        record.update(
            {
                "entry_price": price,
                "contracts": contracts,
                "gross_cost": gross_cost,
                "fees": fee,
                "cash_committed": committed,
                "gross_pnl": payout - gross_cost,
                "net_pnl": payout - committed,
                "contract_multiplier": contracts,
                "evidence_label": "counterfactual_constant_leverage_no_historical_depth",
                "supports_profitability_claim": False,
                "payout": payout,
            }
        )
        active.append(record)
        current_committed = sum(float(item["cash_committed"]) for item in active)
        max_committed = max(max_committed, current_committed)
        minimum_cash = min(minimum_cash, cash)
    release(pd.Timestamp.max.tz_localize("UTC"))
    if completed:
        ledger = pd.DataFrame(completed).drop(columns=["payout"]).sort_values(
            ["settlement_timestamp", "market_ticker"], kind="stable"
        ).reset_index(drop=True)
    else:
        ledger = signals.iloc[:0].copy()
    diagnostics = {
        "contract_multiplier": int(contract_multiplier),
        "input_signals": int(len(signals)),
        "executed_signals": int(len(ledger)),
        "rejected_for_cash": int(rejected_for_cash),
        "maximum_concurrent_cash_committed": float(max_committed),
        "maximum_concurrent_cash_committed_fraction": float(max_committed / starting_equity),
        "minimum_free_cash": float(minimum_cash),
        "ending_cash_after_all_settlements": float(cash),
        "fully_funded_only": bool(fully_funded_only),
        "additional_adverse_ticks": int(additional_adverse_ticks),
        "fee_rate": float(fee_rate),
    }
    return ledger, diagnostics
