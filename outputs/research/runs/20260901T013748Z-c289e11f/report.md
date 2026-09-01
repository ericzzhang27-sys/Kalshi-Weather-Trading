# Closed-loop research wave

Evidence label: `historical_proxy_validated`
Promotion: `retain_champion`

## Probability evidence

- Weather NLL: 0.866359; RPS: 0.083840; Brier: 0.467241
- Market NLL: 0.582308; RPS: 0.052696
- Hybrid NLL: 0.597875; RPS: 0.052898; ECE: 0.030231
- Hybrid log-loss skill versus market: -2.6733%

## Trading evidence

- Trades/event-days: 128/87
- Net P&L: 27.44; profit factor: 4.421446384039899
- Sharpe/Sortino/Calmar: 15.979638892507863/34.462988411686574/89.67265839080491
- DSR confidence/PBO: 0.9999894458909063/0.0

## Gate result

Passed: **False**
Failed gates: weather_log_loss_skill_positive, weather_log_loss_skill_statistically_positive, weather_rps_skill_statistically_positive, calibration_ece, hybrid_skill_positive, trade_count, event_day_count

This is paper research, not a profitability guarantee or executable-depth proof.