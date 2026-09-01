# Return Optimization Report

Run: `20260901T030427Z-return-optimization`

## Decision

The primary result remains a one-contract historical proxy. The larger sizing result is a counterfactual sensitivity only because historical top-of-book depth is unavailable. Live trading remains disabled.

## Strictly out-of-sample trading metrics

| Metric | Walk-forward optimized | Unfiltered baseline | No-depth sizing sensitivity |
|---|---:|---:|---:|
| CAGR | 2.0209% | 2.0113% | 17.7748% |
| Total return | 2.1090% | 2.0990% | 18.6090% |
| Net P&L | $21.09 | $20.99 | $186.09 |
| Maximum drawdown | -0.1149% | -0.1613% | -5.7631% |
| Sharpe | 6.085650 | 5.908174 | 1.687730 |
| Sortino | 15.215039 | 11.039348 | 4.766580 |
| Calmar | 20.283286 | 14.384240 | 3.568860 |
| Profit factor | 2.784264 | 1.916594 | 1.484786 |
| Trades | 172 | 283 | 163 |
| Traded event-days | 104 | 104 | 104 |

## Exact probability metrics

- Weather NLL: 0.803499183385; frozen champion NLL: 0.858669590429; skill: 0.064251031665.
- Weather RPS: 0.079894818610; frozen champion RPS: 0.087210897458; skill: 0.083889503050.
- Hybrid NLL: 0.581851551892; coherent-market NLL: 0.582308334691; skill: 0.000784434589.
- Hybrid RPS: 0.052535491679; coherent-market RPS: 0.052695803040; skill: 0.003042203592.
- Trading-distribution ECE: 0.019852386867; weather 80%/90% coverage: 0.795107581109/0.902386319710.

## Walk-forward selections

- trading_outer_00: `edge_0_all_both_unlimited`; CAGR 2.6036%; drawdown -0.0536%; P&L $12.32.
- trading_outer_01: `edge_0.01_all_buy_yes_2`; CAGR 1.5072%; drawdown -0.1163%; P&L $6.08.
- trading_outer_02: `edge_0_all_both_1`; CAGR 1.6769%; drawdown -0.0479%; P&L $2.69.

## Evidence limits

- Evidence label: `historical_proxy_validated_one_contract`.
- Drawdown constraint: 15.0% maximum; observed -0.1149%.
- Fixed-signal doubled-fee P&L: $19.52; two-extra-tick P&L: $17.65.
- PBO across the preregistered filter grid: 0.0. The upstream threshold search PBO remains 0.9714285714285714; the two estimates measure different search layers and the lower filter-grid estimate does not erase the upstream failure.
- No historical depth exists, so the multi-contract CAGR is not executable-fill evidence and cannot support promotion.
