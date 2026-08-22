# One-time elevated setup: converts the Kalshi orderbook scraper to true 24/7.
#
# Run from an ADMINISTRATOR PowerShell:
#   powershell -ExecutionPolicy Bypass -File "c:\Weather Trading\Kalshi-Weather-Trading\scripts\setup_orderbook_scraper_24x7.ps1"
#
# What it does:
#   * Replaces the logon-triggered loop task with an hourly single-cycle task
#     that runs WHETHER YOU ARE LOGGED ON OR NOT (S4U logon type; no password
#     is stored on disk).
#   * Fires at every top of the hour, forever.
#   * StartWhenAvailable: if the PC was asleep/off at fire time, the missed
#     run executes as soon as Windows next can (catch-up after downtime).
#   * Each run is a fresh process -> a crash can never take collection down;
#     the next hour's run simply happens.

$ErrorActionPreference = "Stop"

$repoRoot = "c:\Weather Trading\Kalshi-Weather-Trading"
$python = "C:\Users\fredd\miniconda3\python.exe"
$taskName = "KalshiOrderbookScraper"

# Stop + remove any existing version of the task.
Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "scripts\run_orderbook_scraper.py" `
    -WorkingDirectory $repoRoot

$nextHour = (Get-Date).Date.AddHours((Get-Date).Hour + 1)
$trigger = New-ScheduledTaskTrigger -Once -At $nextHour `
    -RepetitionInterval (New-TimeSpan -Minutes 60) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

# S4U = run whether user is logged on or not, without storing a password.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName `
    -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description "24/7 hourly Kalshi NYC daily-high orderbook scraper" | Out-Null

Write-Host ""
Write-Host "Done. Task '$taskName' now runs every hour whether you are logged on or not."
Write-Host "First scheduled run: $nextHour"
Start-ScheduledTask -TaskName $taskName
Write-Host "A verification cycle was started now - check data\raw\kalshi_orderbooks\scrape_log.csv in a minute."