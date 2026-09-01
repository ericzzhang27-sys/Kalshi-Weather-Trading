# Closed-loop research wave

Evidence label: `historical_proxy_validated`
Promotion: `retain_champion`

## Probability evidence

- Weather NLL: 0.792972; RPS: 0.152039; Brier: 0.488539
- Market NLL: 0.723878; RPS: 0.136769
- Hybrid NLL: 0.720935; RPS: 0.135587; ECE: 0.076245
- Hybrid log-loss skill versus market: 0.4065%

## Trading evidence

- Trades/event-days: 88/56
- Net P&L: 18.85; profit factor: 3.4966887417218544
- Sharpe/Sortino/Calmar: 16.143689597517422/46.87179502645993/182.67695425535226
- DSR confidence/PBO: 0.9999790175384885/0.5142857142857142

## Gate result

Passed: **False**
Failed gates: weather_log_loss_skill_positive, weather_rps_skill_positive, calibration_ece, coverage_80, coverage_90, trade_count, event_day_count, pbo

This is paper research, not a profitability guarantee or executable-depth proof.