# Kalshi Orderbook Scraper (NYC Daily High Markets)

Systematically downloads full order books for the NYC daily-high temperature
markets (Kalshi series `KXHIGHNY`) every hour and stores them in an
append-only, quality-validated format designed for backtesting.

## Usage

```bash
cd "c:\Weather Trading\Kalshi-Weather-Trading"

# Single cycle now (fetch + validate + store)
python scripts/run_orderbook_scraper.py

# Continuous hourly loop, aligned to the top of the hour (UTC)
python scripts/run_orderbook_scraper.py --loop

# Fetch + validate only, write nothing
python scripts/run_orderbook_scraper.py --dry-run

# Loop limited to N cycles (useful for smoke tests)
python scripts/run_orderbook_scraper.py --loop --max-cycles 3

# Options
#   --depth N                 order book depth (default 20)
#   --include-ineligible      also scrape markets failing trading eligibility
#   --auth                    sign requests (public market data works unsigned)
#   --config-path PATH        scraper config (config/orderbook_scraper_config.yaml)
#   --trading-config-path P   trading config (series tickers, discovery filters)
```

Exit code is `1` when a single cycle's validation status is `FAIL`.

## How it works (per cycle)

1. **Discover** open NYC daily-high markets via the existing read-only market
   discovery pipeline (`src/trading/market_discovery.py`, series `KXHIGHNY`).
2. **Fetch** `GET /markets/{ticker}/orderbook?depth=N` for each eligible
   market, with a configurable pause between requests.
3. **Validate** (fail closed — see below).
4. **Store** snapshots append-only; rejected cycles are quarantined instead.

## Storage layout (`data/raw/kalshi_orderbooks/`)

| Path | Contents |
|---|---|
| `orderbook_levels_YYYYMM.csv` | Normalized book levels (one row per side/level), enriched with `event_ticker`, `close_time`, `market_status`, `eligible`. Append-only. |
| `orderbook_summary_YYYYMM.csv` | Per-ticker per-cycle summaries (best bid/ask, spreads, midpoints, depths, status). |
| `raw/YYYY/MM/DD/<cycle_id>.jsonl` | Full raw API payloads per cycle (audit trail / re-normalization). |
| `scrape_log.csv` | One row per cycle: status, counts, violation summary, output paths. |
| `state/last_fetch_by_ticker.json` | Continuity watermark (latest stored `fetched_at` per ticker). Falls back to scanning monthly partitions if lost. |
| `quarantine/<cycle_id>/` | Rejected cycles (levels, summary, raw payload) — never merged into canonical partitions. |
| `latest_quality_report.md` | Human-readable report for the most recent cycle. |

## Data-quality checks (fail closed)

Any **FAIL** violation quarantines the entire cycle; nothing is appended to
the canonical CSV partitions.

Levels checks:
- Prices within `[0, 1]`; sizes finite and nonnegative.
- **Monotonicity:** bid prices non-increasing across levels; ask prices
  non-decreasing across levels.
- Cumulative size monotonically non-decreasing across levels.
- Complementary YES/NO quotes derived from the same raw book side sum to 1.
- No duplicate `(fetched_at, ticker, side, level)` rows.

Summary checks:
- No crossed books (best bid ≤ best ask); spreads within `[0, 1]`;
  midpoints within `[0, 1]`; depths/sizes nonnegative.

Cross-cycle checks:
- **Timestamp monotonicity:** a new snapshot for a ticker may not be older
  than the last stored snapshot (guards against clock skew / replayed data).

Statuses: `OK` (no violations), `WARN` (e.g., empty book / no matching
markets — still stored), `FAIL` (quarantined), `ERROR` (unexpected exception,
logged by the loop without crashing).

## Automatic execution (Windows Task Scheduler)

Two modes are supported:

### Mode A — while logged in (already active, no admin needed)

A scheduled task `KalshiOrderbookScraper` runs at user logon and launches
`scripts/launch_orderbook_scraper.bat`, which runs the continuous `--loop`
mode: one scrape at startup, then one every top of the hour while you are
logged in. Console output is appended to
`data/raw/kalshi_orderbooks/scheduler_stdout.log`.

### Mode B — true 24/7 (one-time elevated setup)

To collect even when you are logged out, run this **once** from an
*Administrator* PowerShell (creating a "run whether logged on or not" task
requires elevation; no password is stored on disk):

```powershell
powershell -ExecutionPolicy Bypass -File "c:\Weather Trading\Kalshi-Weather-Trading\scripts\setup_orderbook_scraper_24x7.ps1"
```

This replaces Mode A with an hourly single-cycle task that:
- runs whether you are logged on or not (S4U logon type),
- fires at every top of the hour, forever,
- catches up missed runs after sleep/reboot (`StartWhenAvailable`),
- runs each cycle as a fresh process, so a crash can never stop collection.

Verify it worked:

```powershell
Get-ScheduledTask -TaskName 'KalshiOrderbookScraper'   # State should be Ready
Get-Content "c:\Weather Trading\Kalshi-Weather-Trading\data\raw\kalshi_orderbooks\scrape_log.csv" -Tail 5
```

Note: the PC must be powered on for collection to happen. If it sleeps,
missed hours are backfilled once at wake (one snapshot, not per-missed-hour).

### Mode C — free 24/7 without your PC (GitHub Actions)

`.github/workflows/orderbook_scraper.yml` runs one scrape cycle per hour on
GitHub's servers and commits the snapshots back to this repository — your PC
can be off entirely.

To activate:
1. Commit and push the workflow file: `git add .github/workflows/orderbook_scraper.yml && git push`
2. Open the repo on GitHub → **Actions** tab → enable workflows if prompted.
   The schedule starts automatically; you can also trigger a test run via
   **Run workflow** (`workflow_dispatch`).
3. Collected data lands in `data/raw/kalshi_orderbooks/` as commits on
   `main`; `git pull` locally to merge it into your backtesting store.

Free-tier fit: ~730 min/month vs 2,000 free minutes for private repos
(unlimited for public repos). Caveats: GitHub cron can fire a few minutes
late at busy times (fetch timestamps in the data remain exact), and GitHub
disables schedules after ~60 days of repo inactivity — any manual commit or
push resets that clock.

If both Mode B/C and a local task collect simultaneously, prefer one primary
collector and `git pull --rebase` before running locally to avoid CSV merge
conflicts.

Other genuinely-free 24/7 options if you outgrow Actions: Oracle Cloud's
Always Free VM tier (a real always-on Linux box — run the `--loop` mode under
systemd), or a spare laptop / Raspberry Pi running the same loop.

### Managing the task

Manage it from PowerShell:

```powershell
Get-ScheduledTask -TaskName 'KalshiOrderbookScraper'      # check state (Ready/Running)
Start-ScheduledTask -TaskName 'KalshiOrderbookScraper'    # start now
Stop-ScheduledTask  -TaskName 'KalshiOrderbookScraper'    # stop
Unregister-ScheduledTask -TaskName 'KalshiOrderbookScraper' -Confirm:$false  # remove
```

## Configuration

Scraper cadence/storage/pacing live in `config/orderbook_scraper_config.yaml`.
Market discovery settings (series tickers, location/weather terms, eligibility)
are reused from `config/trading_config.yaml`. This tool is read-only and does
not touch trading safety flags.

## Backtesting notes

- Load all partitions: `pd.concat(map(pd.read_csv, glob(".../orderbook_levels_*.csv")))`.
- Rows are keyed by `(fetched_at, ticker, outcome_side, quote_type, level)`;
  `fetched_at` is UTC ISO-8601 and monotonic per ticker by construction.
- Use `raw/*.jsonl` to re-derive levels with different normalization rules
  without re-fetching history.

## Tests

```bash
pytest tests/trading/test_orderbook_quality.py tests/trading/test_orderbook_scraper.py
```
