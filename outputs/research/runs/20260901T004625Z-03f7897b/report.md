# Closed-loop research wave

Evidence label: `historical_proxy_validated`
Promotion: `retain_champion`

## Probability evidence

- Weather NLL: 1.597839; RPS: 0.215225; Brier: 0.671422
- Market NLL: 0.686695; RPS: 0.126221
- Hybrid NLL: 0.684571; RPS: 0.126294; ECE: 0.054895
- Hybrid log-loss skill versus market: 0.3093%

## Trading evidence

- Trades/event-days: 205/114
- Net P&L: 26.53; profit factor: 2.369643779039752
- Sharpe/Sortino/Calmar: 10.497242867542756/27.486202420160865/47.23289494736738
- DSR confidence/PBO: 0.9999982065624384/0.05714285714285714

## Gate result

Passed: **False**
Failed gates: weather_log_loss_skill_positive, weather_rps_skill_positive, calibration_ece, coverage_80, coverage_90, trade_count, event_day_count

This is paper research, not a profitability guarantee or executable-depth proof.