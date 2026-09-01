# Closed-loop research wave

Evidence label: `blocked_hourly_candles_no_proxy_fills`
Promotion: `retain_champion`

## Probability evidence

- Weather NLL: 2.203146; RPS: 0.225747; Brier: 0.793265
- Market NLL: 0.618270; RPS: 0.094839
- Hybrid NLL: 0.610917; RPS: 0.093811; ECE: 0.041105
- Hybrid log-loss skill versus market: 1.1891%

## Trading evidence

- Trades/event-days: 0/0
- Net P&L: 0.00; profit factor: None
- Sharpe/Sortino/Calmar: None/None/None
- DSR confidence/PBO: 0.0/1.0

## Gate result

Passed: **False**
Failed gates: weather_log_loss_skill_positive, weather_rps_skill_positive, calibration_ece, coverage_80, coverage_90, trade_count, event_day_count, bootstrap_net_pnl_lower, profit_factor, deflated_sharpe_confidence, pbo, sharpe, sortino, calmar, rolling_folds

This is paper research, not a profitability guarantee or executable-depth proof.