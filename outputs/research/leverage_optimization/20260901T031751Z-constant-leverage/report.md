# Constant-Leverage Optimization

Run: `20260901T031751Z-constant-leverage`

## Result

The smallest fully funded constant-contract multiplier satisfying all requested
constraints is **20x**. It changes no signal,
side, price filter, or relative contract weight.

| Metric | One contract | 20x contracts |
|---|---:|---:|
| CAGR | 2.020900% | 41.794580% |
| Total return | 2.109000% | 43.947000% |
| Net P&L | $21.09 | $439.47 |
| Maximum drawdown | -0.114922% | -1.646971% |
| Sharpe | 6.085650 | 6.288525 |
| Sortino | 15.215039 | 16.223975 |
| Calmar | 20.283286 | 29.492314 |
| Profit factor | 2.784264 | 2.959208 |

## Constraints and execution diagnostics

- Target CAGR: 40.0%.
- Minimum Sharpe: 6.0.
- Maximum drawdown: 15.0%.
- Signals preserved: 172 of 172.
- Maximum concurrent cash committed: $89.53 (8.95% of initial equity).
- Daily P&L correlation with one-contract strategy: 0.999775866599.
- Doubled-fee CAGR/P&L: 38.892379% / $408.75.
- Two-additional-tick CAGR/P&L: 35.086825% / $368.51.

## Evidence limit

This is a constant-size sensitivity on previously inspected OOS signals, not a
new untouched test. Historical top-of-book depth is unavailable, so 20
contracts per signal cannot be claimed executable. Live trading remains disabled.
