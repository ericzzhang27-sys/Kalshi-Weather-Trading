# Closed-loop research wave

Evidence label: `historical_proxy_validated`
Promotion: `retain_champion`

## Probability evidence

- Weather NLL: 1.652608; RPS: 0.085473; Brier: 0.480469
- Market NLL: 0.582308; RPS: 0.052696
- Hybrid NLL: 0.588514; RPS: 0.053129; ECE: 0.024060
- Hybrid log-loss skill versus market: -1.0659%

## Trading evidence

- Trades/event-days: 67/40
- Net P&L: 8.87; profit factor: 2.5896057347670247
- Sharpe/Sortino/Calmar: 10.720738247201416/31.35396533779751/117.0951429375336
- DSR confidence/PBO: 0.9954614069409556/0.7285714285714285

## Gate result

Passed: **False**
Failed gates: weather_log_loss_skill_positive, weather_log_loss_skill_statistically_positive, weather_rps_skill_statistically_positive, calibration_ece, coverage_80, coverage_90, hybrid_skill_positive, trade_count, event_day_count, pbo

This is paper research, not a profitability guarantee or executable-depth proof.