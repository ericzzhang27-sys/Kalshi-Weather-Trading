# Closed-loop research wave

Evidence label: `historical_proxy_validated`
Promotion: `retain_champion`

## Probability evidence

- Weather NLL: 0.866359; RPS: 0.083840; Brier: 0.467241
- Market NLL: 0.582308; RPS: 0.052696
- Hybrid NLL: 0.583263; RPS: 0.052802; ECE: 0.020775
- Hybrid log-loss skill versus market: -0.1640%

## Trading evidence

- Trades/event-days: 98/55
- Net P&L: 19.65; profit factor: 4.174474959612277
- Sharpe/Sortino/Calmar: 19.74021063621312/60.88793746007029/190.69681909093188
- DSR confidence/PBO: 0.9999915157473065/0.04285714285714286

## Gate result

Passed: **False**
Failed gates: weather_log_loss_skill_positive, weather_log_loss_skill_statistically_positive, weather_rps_skill_statistically_positive, calibration_ece, hybrid_skill_positive, trade_count, event_day_count

This is paper research, not a profitability guarantee or executable-depth proof.