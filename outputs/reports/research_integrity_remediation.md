# Repository Research Integrity Remediation

## Outcome

The strict repository audit reports zero P0/P1 findings. The complete test suite
passes. Shadow/live defaults remain disabled and production remains NYC-only.

## Reproducibility evidence

Two independent rebuilds from the same raw ASOS hash produced identical hashes:

| Artifact | SHA-256 |
|---|---|
| Raw KNYC ASOS | `494885fb95416a607b7267e62ea817fde420e4511b24bb73c3222fd974aac5d2` |
| Daily clean | `8315da3a3d8114a48503a988331558e67bd10f93f60b6c3d990c55ecbf67e381` |
| Hourly clean | `8bf556f3ec03393eef6b743c6495b3f008cd98aadba7516bbe643b232365e9cf` |
| Forecasts clean | `4f4e1093f29d97a25605f1e7627c0d749b642ebd5223d6d1bd49b44e19af919c` |
| Supervised rows | `b0eb1fa4556b3cbb8e4a8b968e5bbb76c229fae37ee7b0e86273f279c5e186a6` |
| Modeling rows | `bbc5556ede8616c5b4baa148f5e5c0828e49a6b3d8ae3266572f465dfbe78a1a` |
| Generated feature specification | `9c63baae0a80d9d14225db98669621c7270d178d92860d04eb770c79bc622bb7` |

The production model artifact contains 36 ordered features and split row counts
that exactly match the rebuilt table: 16,791 train, 8,396 validation, and 11,610
test rows. Its manifest pins the model, feature list, source table, configuration,
calibration, split boundaries, and package versions.

## Material backtest correction

The legacy `outputs/backtests_real/summary.csv` reported 620 trades, $24.24 net
P&L, Sharpe 2.78, and maximum drawdown of -$5.07. Those figures are invalidated
because the old simulation allowed assumptions now prohibited by the integrity
contract, including same-candle pricing, non-causal threshold research, and
settlement fallbacks.

The corrected immutable run `20260826T014118Z-511c1aa4` selected a 0.15 threshold
on validation only and produced zero executable 2025+ trades. Historical prices
are hourly candles, so no first-subsequent candle open exists within the required
five-minute execution window. Net P&L and drawdown are therefore zero; Sharpe,
Sortino, and Calmar are undefined. The run is labeled
`diagnostic_no_test_trades` and supports no profitability claim.

This is a correctness result, not evidence that the strategy is unprofitable.
Executable sub-five-minute quote history and order-book depth are required before
profitability can be evaluated under the declared assumptions.
