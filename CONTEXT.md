# Project Context

This document is the working context and handoff notes for the Kalshi Weather Trading project. It captures the current state, key decisions, conventions, and known gotchas so that any engineer (or agent) can resume work without re-deriving history.

## What This Project Is

A two-layer system for NYC daily high-temperature markets on Kalshi (`KXHIGHNY` series):

1. **Probability engine** — an NGBoost distributional model that predicts the distribution of `forecast_error = actual_high - forecast_high` and converts it into bucket probabilities via CDF differences.
2. **Live trading foundation** — a fail-closed trading loop (currently shadow mode) that connects the probability engine to live Kalshi market data with contract mapping, edge computation, risk limits, and a paper broker.

## Current State

- The probability engine is trained, evaluated, and calibrated. It is demo-ready.
- The trading loop runs end-to-end in shadow/paper mode against live Kalshi public market data. No orders are placed by default:
  - `config/trading_config.yaml`: `mode: shadow`, `trading_enabled: false`, `live_auto_enabled: false`.
- Historical backtesting of execution is intentionally absent: there is no historical executable order-book data. Validation is forward paper trading instead.
- A Streamlit dashboard (`apps/live_trading_dashboard.py`) visualizes live cycle outputs.

## Key Decisions and Rationale

| Decision | Rationale |
|---|---|
| Model target is forecast error, not raw temperature | The NWS/NDFD forecast high is a strong baseline; modeling residuals isolates learnable signal. |
| Official daily high = NOAA/NWS daily TMAX for Central Park | Matches Kalshi settlement station (`KNYC` / GHCND `USW00094728`). Open-Meteo historical daily max was rejected as the official source. |
| Forecast anchor = timestamp-safe NWS/NDFD MaxT archive with issue-time filtering | Prevents leakage from forecast updates issued after `prediction_time`. Open-Meteo forecast history rejected as training anchor. |
| Observed intraday features = IEM/NWS ASOS hourly | Station observations expose true intra-hour highs via periodic max-temp remarks; Open-Meteo allowed only as explicit emergency fallback. |
| Chronological splits (val starts 2024-01-01, test starts 2025-01-01) | Time-series problem; random splits would leak. |
| Normal distribution selected over Laplace for the standard training path | Won validation NLL, interval log loss, bucket Brier, and 90% coverage error on the frozen current36 feature contract without touching test. |
| Bucket convention: lower-open, upper-closed `[a, b)` style bounds | Must match exchange settlement semantics exactly; ambiguous contracts become `NO_TRADE`. |
| Fail-closed everywhere in trading code | Missing credentials, stale data, failed mapping, or failed validation must stop order generation, never guess. |
| Forward paper trading instead of backtesting | No historical executable books exist; forward evidence replaces unavailable historical fills. |

## Canonical Data Sources

See also `docs/data_sources.md` and the priority lists in `config/model_config.yaml` (`data_sources:`).

| Role | Source | Notes |
|---|---|---|
| Actual daily high | NOAA/NWS official daily TMAX CSV (Central Park) | Canonical target anchor. |
| Intraday observations | IEM/NWS ASOS hourly CSV (KNYC) | Feature source; Open-Meteo fallback only for live display. |
| Forecast high | NWS/NDFD historical MaxT archive (`data/raw/ndfd/`) | Issue-time filtered; rebuilt via `scripts/build_ndfd_daily_high_archive.py` / `build_ndfd_point_forecasts.py`. |
| Legacy auxiliary | Open-Meteo forecast history | Not used in the current feature contract. |

## Key Artifacts

| Artifact | Purpose |
|---|---|
| `data/processed/modeling_rows_v1.csv` | Final modeling table (timestamp-safe features + targets). |
| `outputs/final_feature_list.json` | Frozen feature contract for the selected model. |
| `models/ngboost_laplace_current36_default.pkl` | Default live probability engine artifact. |
| `src/predict_distribution.py` → `load_probability_engine()` | Final prediction interface used by the trading loop. |
| `config/model_config.yaml` | Authoritative training configuration (selected candidate: `official_migration_depth3_subsample_15`). |
| `config/trading_config.yaml` | All trading behavior: modes, market scope, weather providers, settlement gating, risk limits, edge rules, output paths. |
| `outputs/live_trading/*` | Every trading-cycle artifact (discovery, mapping, features, probabilities, books, edge, risk, intents, paper PnL). |

## Conventions

- Timestamps are local America/New_York unless stated otherwise; temperatures are Fahrenheit.
- Every modeling row has a `prediction_time`; nothing after it may inform features.
- Bucket probabilities must be finite, nonnegative, and sum to 1 per row — validated at pricing time.
- Trading outputs are append-style CSVs under `outputs/live_trading/` so cycles are auditable.
- Scripts are standalone CLIs run from the repo root; modules under `src/` are importable libraries.

## Known Gotchas

1. **Artifact names lie about distribution.** Files named `ngboost_normal_v0.pkl` or `...laplace...` reflect historical experiments. Trust `config/model_config.yaml` and the metadata JSON files, not filenames.
2. **NDFD archive is daily MaxT only.** Hourly forecast-relative features are reproduced from the as-of-available daily-high forecast to preserve the 36-feature contract. Do not "fix" this by pulling future-looking hourly forecasts.
3. **Observed-high verification window.** After `max_unverified_observed_high_minutes` (20 min), `observed_high_so_far` is treated as a lower bound and the dashboard marks weather `NO_TRADE`.
4. **Settlement gating.** After peak window end (18:00 local), a falling temperature path can force diagnostic-only `POST_PEAK_NO_TRADE` unless final settlement data is available.
5. **Kalshi API specifics.** Auth uses RSA-PSS/SHA256 over `timestamp + METHOD + path` (query params stripped). YES asks are inferred from NO bids. New orders should use `/trade-api/v2/portfolio/events/orders`. See `docs/trading/LIVE_TRADING_IMPLEMENTATION_PLAN.md` for the full reference list.
6. **Fees/rounding.** Edge must survive fee rate (7%), fee rounding increments, slippage buffer, spread cap, and minimum liquidity before a trade is considered.
7. **Risk limits are tight by design** (e.g., max 5 contracts/market, $50 total exposure, $25 daily loss). Do not loosen them casually.

## Where Things Stand / Next Steps

Remaining work tracks the plan in `docs/trading/LIVE_TRADING_IMPLEMENTATION_PLAN.md`:

1. Run forward paper trading for multiple weeks and review weekly reports.
2. Complete manual-live readiness items (review queue tooling, order manager, fills reconciliation, monitoring alerts, trading tests).
3. Only then consider limited manual-live rollout; keep `live_auto_enabled: false`.

## Documentation Index

- `README.md` — overview, commands, layout.
- `AGENTS.md` — how agents should work in this repo.
- `docs/project/PROJECT_SPEC.md` — original project spec.
- `docs/trading/LIVE_TRADING_IMPLEMENTATION_PLAN.md` — full trading build plan.
- `docs/data_sources.md`, `docs/ndfd_pipeline.md` — data provenance details.
- `outputs/README.md` — outputs guide.