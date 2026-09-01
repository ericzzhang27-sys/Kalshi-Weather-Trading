"""Tiny backtest over recent multi-level orderbook snapshots (last few days).

Strategy (illustrative, NOT the production NGBoost engine):
- At each orderbook snapshot time t (local America/New_York), build a nowcast of
  the day's final high:  expected_high = max_temp_so_far + typical_remaining_gain(hour)
  where typical gain stats come from historical KNYC hourly obs vs daily highs.
- Bucket probability = Normal CDF around expected_high (sigma from same history).
- Trade when model prob beats the executable price by >= MIN_NET_EDGE after fees,
  walking the YES/NO ask side of the book for a realistic 10-contract fill.
- One entry per market; settle against actual daily highs (fail-closed: days
  without a settled actual high are excluded and reported as pending).

Outputs -> outputs/backtests/tiny_orderbook_recent/
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.backtest.fees import kalshi_fee  # noqa: E402

LEVELS_PATH = REPO_ROOT / "data/raw/kalshi_orderbooks/orderbook_levels_202608.csv"
OUT_DIR = REPO_ROOT / "outputs/backtests/tiny_orderbook_recent"

SERIES_PREFIX = "KXHIGHNY"
ORDER_SIZE = 10
MIN_NET_EDGE = 0.04          # model_prob - ask - fee_per_contract >= this
MAX_WALK_LEVELS = 5          # depth used when walking the book
SIGMA_FLOOR_F = 1.0          # deg F floor on nowcast uncertainty

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def parse_bucket(ticker: str):
    """KXHIGHNY-26AUG23-B84.5 -> (83.5, 85.5] (= '84 to 85'); T81 -> (-inf, 80.5]; T89 -> (88.5, inf).

    Matches repo convention (see contract_bucket_mapping.csv): floor = c-0.5,
    cap = c+0.5, boundaries lower-open/upper-closed at half-degree marks.
    """
    tail = ticker.split("-")[-1]
    if tail.startswith("B"):
        center = float(tail[1:])
        return center - 1.0, center + 1.0
    if tail.startswith("T"):
        cap = float(tail[1:])
        return -np.inf, cap - 0.5
    raise ValueError(f"unparseable ticker strike: {ticker}")


def target_date_from_ticker(ticker: str, year: int) -> pd.Timestamp:
    token = ticker.split("-")[1]  # e.g. 26AUG23
    return pd.to_datetime(f"{token[2:]} {year}", format="%b%d %Y")


def fetch_knyc_hourly() -> pd.DataFrame:
    params = dict(
        station="KNYC", data="tmpf",
        year1="2026", month1="8", day1="20",
        year2="2026", month2="8", day2="26",
        tz="America/New_York",
        format="onlycomma,latlon=0,elevation=0,missing=M,trace=T",
    )
    r = requests.get(IEM_ASOS_URL, params=params, timeout=60)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["valid", "tmpf"]).sort_values("valid")
    return df.rename(columns={"valid": "ts_local", "tmpf": "temp_f"})[["ts_local", "temp_f"]]


def build_gain_climatology() -> pd.DataFrame:
    """Per local hour: mean/std of (actual_daily_high - max_temp_so_far_at_hour)."""
    hourly = pd.read_csv(REPO_ROOT / "data/processed/hourly_clean.csv")
    hourly = hourly[hourly["location"] == "NYC"].copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    hourly["d"] = hourly["timestamp"].dt.date
    hourly["h"] = hourly["timestamp"].dt.hour

    daily = pd.read_csv(REPO_ROOT / "data/processed/knyc_daily_actuals_combined.csv")
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    highs = daily.set_index("date")["actual_high"].to_dict()

    hourly["daily_high"] = hourly["d"].map(highs)
    hourly = hourly.dropna(subset=["daily_high", "nws_current_temp_f"])
    run_max = hourly.groupby("d")["nws_current_temp_f"].cummax()
    residual = hourly["daily_high"] - run_max
    out = pd.DataFrame({"h": hourly["h"], "resid": residual})
    g = out.groupby("h")["resid"].agg(["mean", "std", "count"])
    return g[g["count"] >= 200]


def load_books() -> pd.DataFrame:
    lv = pd.read_csv(LEVELS_PATH)
    lv = lv[
        lv["ticker"].str.startswith(SERIES_PREFIX)
        & (lv["eligible"] == True)  # noqa: E712
        & (lv["market_status"] == "active")
    ].copy()
    lv["fetched_at"] = pd.to_datetime(lv["fetched_at"], utc=True)
    return lv


def walk_book(rows: pd.DataFrame, need: int) -> tuple[float | None, float]:
    """Return (VWAP price for `need` contracts, filled contracts); None if book too thin."""
    rows = rows.sort_values("level")
    cost, filled = 0.0, 0
    for _, r in rows.head(MAX_WALK_LEVELS).iterrows():
        take = min(need - filled, float(r["size_contracts"]))
        if take <= 0 or not np.isfinite(r["price_dollars"]):
            continue
        cost += take * float(r["price_dollars"])
        filled += take
        if filled >= need:
            break
    if filled < need:
        return None, filled
    return cost / filled, filled


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    books = load_books()
    obs = fetch_knyc_hourly()
    clim = build_gain_climatology()

    # Actual highs so far (settled days only; today stays pending -> fail closed)
    obs_by_day = obs.groupby(obs["ts_local"].dt.date)["temp_f"].max()
    last_obs_day = obs["ts_local"].dt.date.max()
    settled_days = sorted(d for d in obs_by_day.index if d < last_obs_day)

    trades = []
    evaluated_snapshots = 0
    for snap_ts, snap in books.groupby("fetched_at"):
        local_ts = snap_ts.tz_convert("America/New_York").tz_localize(None)
        day = local_ts.date()
        if day not in settled_days:
            continue  # unsettled or pre-window day
        if day not in [pd.Timestamp(d).date() for d in settled_days]:
            continue

        h = local_ts.hour
        if h not in clim.index:
            continue
        gain_mu = float(clim.loc[h, "mean"])
        sigma = max(SIGMA_FLOOR_F, float(clim.loc[h, "std"]))
        tod_obs = obs[(obs["ts_local"].dt.date == day) & (obs["ts_local"] <= local_ts)]
        if tod_obs.empty:
            continue
        max_so_far = float(tod_obs["temp_f"].max())
        exp_high = max_so_far + gain_mu
        evaluated_snapshots += 1

        def bucket_prob(lo: float, hi: float) -> float:
            z_hi = np.inf if np.isinf(hi) else (hi - exp_high) / sigma
            z_lo = -np.inf if np.isinf(lo) or lo == -np.inf else (lo - exp_high) / sigma
            from scipy.stats import norm
            return float(norm.cdf(z_hi) - norm.cdf(z_lo))

        seen_tickers = {t["ticker"] for t in trades}
        for ticker, tk in snap.groupby("ticker"):
            if ticker in seen_tickers:
                continue
            # Same-day markets only: the nowcast applies to today's settlement.
            if target_date_from_ticker(ticker, local_ts.year).date() != day:
                continue
            lo, hi = parse_bucket(ticker)
            p_yes = bucket_prob(lo, hi)

            yes_ask_rows = tk[(tk["outcome_side"] == "YES") & (tk["quote_type"] == "ask")]
            no_ask_rows = tk[(tk["outcome_side"] == "NO") & (tk["quote_type"] == "ask")]

            for side, ask_rows, p_model in (("YES", yes_ask_rows, p_yes),
                                            ("NO", no_ask_rows, 1.0 - p_yes)):
                px, filled_n = walk_book(ask_rows, ORDER_SIZE)
                if px is None or not (0.01 <= px <= 0.99):
                    continue
                fee_pc = kalshi_fee(px, 1.0)
                edge = p_model - px - fee_pc
                if edge < MIN_NET_EDGE:
                    continue
                fee_tot = kalshi_fee(px, ORDER_SIZE)
                win = (
                    (side == "YES" and lo < obs_by_day[day] <= hi)
                    or (side == "NO" and not (lo < obs_by_day[day] <= hi))
                )
                pnl = ORDER_SIZE * ((1.0 - px) if win else -px) - fee_tot
                trades.append({
                    "snapshot_utc": snap_ts, "local_time": local_ts, "target_date": day,
                    "ticker": ticker, "side": side, "fill_price": round(px, 4),
                    "contracts": ORDER_SIZE, "fee_dollars": round(fee_tot, 4),
                    "model_prob": round(p_model, 4), "net_edge": round(edge, 4),
                    "exp_high_nowcast": round(exp_high, 2), "sigma": round(sigma, 2),
                    "max_so_far": max_so_far, "actual_high": obs_by_day[day],
                    "win": win, "pnl_dollars": round(pnl, 2), "hour": h,
                })
                break  # one position per market

    tdf = pd.DataFrame(trades)
    tdf.to_csv(OUT_DIR / "trades.csv", index=False)

    lines = ["# Tiny Orderbook Backtest — recent snapshots (NYC daily-high)", ""]
    if tdf.empty:
        lines.append("No trades triggered.")
    else:
        wins = int(tdf["win"].sum())
        lines += [
            f"- Window: {books['fetched_at'].min()} .. {books['fetched_at'].max()} (UTC)",
            f"- Settled days traded: {', '.join(str(d) for d in sorted(set(tdf['target_date'])))}"
            f"  (actual highs: "
            f"{', '.join(f'{d}: {obs_by_day[d]:.0f}F' for d in sorted(set(tdf['target_date'])))})",
            f"- Snapshots evaluated: {evaluated_snapshots}",
            f"- Trades: {len(tdf)}  ({int((tdf['side']=='YES').sum())} YES / {int((tdf['side']=='NO').sum())} NO)",
            f"- Win rate: {wins}/{len(tdf)} = {wins/len(tdf):.1%}",
            f"- Gross fees paid: ${tdf['fee_dollars'].sum():.2f}",
            f"- **Total PnL: ${tdf['pnl_dollars'].sum():+.2f}**"
            f" on ~${(tdf['fill_price']*tdf['contracts']).sum():.0f} risked",
            "",
            "## By target date",
            tdf.groupby("target_date").agg(
                trades=("pnl_dollars", "size"), pnl=("pnl_dollars", "sum"),
                avg_edge=("net_edge", "mean"),
            ).round(3).to_markdown(),
            "",
            "## Trades",
            tdf.drop(columns=["snapshot_utc"]).to_markdown(index=False),
        ]
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
