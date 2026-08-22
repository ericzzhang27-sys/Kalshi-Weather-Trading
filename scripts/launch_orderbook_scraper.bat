@echo off
REM Launcher for the hourly Kalshi orderbook scraper (used by Task Scheduler).
REM Runs the continuous loop; output is appended to a log under the data store.
cd /d "c:\Weather Trading\Kalshi-Weather-Trading"
if not exist "data\raw\kalshi_orderbooks" mkdir "data\raw\kalshi_orderbooks"
"C:\Users\fredd\miniconda3\python.exe" scripts\run_orderbook_scraper.py --loop >> "data\raw\kalshi_orderbooks\scheduler_stdout.log" 2>&1