from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.metrics import ledger_metrics
from src.research.return_optimizer import (
    StrategyCandidate,
    build_candidates,
    filter_ledger,
    replay_constant_contract_multiplier,
    walk_forward_filter_selection,
)


def _ledger() -> pd.DataFrame:
    rows = []
    for index, (date, pnl, edge, hour) in enumerate(
        [
            ("2025-01-01", 0.50, 0.01, 10),
            ("2025-01-02", -0.20, 0.03, 14),
            ("2025-01-03", 0.70, 0.05, 17),
            ("2025-01-04", 0.60, 0.06, 10),
        ]
    ):
        rows.append(
            {
                "trade_id": index + 1,
                "target_date": date,
                "event_ticker": f"E{index}",
                "market_ticker": f"M{index}",
                "signal_timestamp": f"{date}T{hour + 5:02d}:00:00Z",
                "execution_timestamp": f"{date}T{hour + 5:02d}:00:00Z",
                "settlement_timestamp": f"{date}T23:00:00Z",
                "side": "BUY_YES",
                "entry_price": 0.40,
                "contracts": 1,
                "gross_cost": 0.40,
                "cash_committed": 0.42,
                "fees": 0.02,
                "gross_pnl": pnl + 0.02,
                "net_pnl": pnl,
                "predicted_edge": edge + 0.01,
                "lower_confidence_edge": edge,
                "settlement": int(pnl > 0),
            }
        )
    return pd.DataFrame(rows)


def test_ledger_metrics_reports_calendar_cagr() -> None:
    metrics = ledger_metrics(
        _ledger(),
        starting_cash=1000.0,
        evaluation_dates=["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
    )
    expected = (1001.6 / 1000.0) ** (365.2425 / 4.0) - 1.0
    assert metrics["cagr"] == pytest.approx(expected)
    assert metrics["total_return"] == pytest.approx(0.0016)
    assert metrics["elapsed_calendar_days"] == 4


def test_filter_ledger_applies_edge_time_and_event_cap() -> None:
    ledger = _ledger()
    duplicate = ledger.iloc[[0]].copy()
    duplicate["market_ticker"] = "M-extra"
    duplicate["lower_confidence_edge"] = 0.08
    ledger = pd.concat([ledger, duplicate], ignore_index=True)
    candidate = StrategyCandidate(0.02, "before_12", "buy_yes", 1)
    filtered = filter_ledger(ledger, candidate)
    assert filtered["market_ticker"].tolist() == ["M-extra", "M3"]


def test_walk_forward_selection_never_uses_current_fold_for_tuning() -> None:
    ledger = _ledger()
    candidates = build_candidates(
        {
            "minimum_lower_confidence_edge": [0.0, 0.05],
            "local_time_regime": ["all"],
            "side": ["both"],
            "maximum_trades_per_event": ["unlimited"],
        }
    )
    folds = [
        {
            "fold_id": "f0",
            "validation_start": "2025-01-01",
            "validation_end": "2025-01-02",
        },
        {
            "fold_id": "f1",
            "validation_start": "2025-01-03",
            "validation_end": "2025-01-04",
        },
    ]
    selected, _, reports = walk_forward_filter_selection(
        ledger,
        folds,
        ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        candidates,
        starting_equity=1000.0,
        hard_max_drawdown_fraction=0.15,
        first_fold_candidate="edge_0_all_both_unlimited",
        minimum_prior_trades=1,
        require_positive_each_prior_fold=True,
    )
    assert reports[0]["selected_candidate"] == "edge_0_all_both_unlimited"
    assert reports[1]["selected_candidate"] == "edge_0_all_both_unlimited"
    assert len(selected) == 4


def test_constant_multiplier_preserves_signals_and_recalculates_bulk_fees() -> None:
    ledger, diagnostics = replay_constant_contract_multiplier(
        _ledger(),
        starting_equity=1000.0,
        contract_multiplier=20,
        fee_rate=0.07,
    )
    assert len(ledger) == len(_ledger())
    assert set(ledger["contracts"]) == {20}
    assert diagnostics["rejected_for_cash"] == 0
    # At 40 cents the exact 20-contract fee is 34 cents, not 20 times
    # the one-contract rounded fee.
    assert set(ledger["fees"].round(2)) == {0.34}


def test_constant_multiplier_reports_cash_rejections_fail_closed() -> None:
    ledger, diagnostics = replay_constant_contract_multiplier(
        _ledger(),
        starting_equity=1.0,
        contract_multiplier=20,
        fee_rate=0.07,
    )
    assert ledger.empty
    assert diagnostics["rejected_for_cash"] == len(_ledger())
