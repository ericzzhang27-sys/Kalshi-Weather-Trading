# Reproducible command for updating Kalshi history and rerunning entire backtest
# Usage (PowerShell, from repo root):
#   .\scripts\update_kalshi_history_and_rerun_backtest.ps1
#   .\scripts\update_kalshi_history_and_rerun_backtest.ps1 -City NYC -Threshold 0.05 -Strategy A
param(
    [string]$City = "NYC",
    [double]$Threshold = 0.05,
    [string]$Strategy = "A"
)
Write-Host "=== Updating Kalshi history (tries real API, falls back to synthetic) ===" -ForegroundColor Cyan
# Try real API download; if no credentials or no history, synthetic fallback is automatic in run_kalshi_backtest
python -m src.kalshi.download_weather_history --city $City
if ($LASTEXITCODE -ne 0) {
    Write-Host "Real API download failed or no credentials — generating synthetic history for offline reproducibility" -ForegroundColor Yellow
    python -m src.kalshi.synthetic
}
Write-Host "=== Normalizing markets ===" -ForegroundColor Cyan
python -m src.kalshi.normalize_markets
Write-Host "=== Running backtest pipeline ===" -ForegroundColor Cyan
python scripts/run_kalshi_backtest.py --city $City --threshold $Threshold --strategy $Strategy
Write-Host "=== Done. Outputs in data/kalshi/ and outputs/backtests/. Open notebook: notebooks/kalshi_historical_backtest.ipynb ===" -ForegroundColor Green
