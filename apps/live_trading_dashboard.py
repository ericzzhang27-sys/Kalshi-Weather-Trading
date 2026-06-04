from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.trading.config import DEFAULT_TRADING_CONFIG_PATH, load_trading_config  # noqa: E402
from src.trading.dashboard_data import (  # noqa: E402
    DashboardState,
    load_dashboard_state,
    load_dashboard_state_from_artifacts,
)


CENTRAL_PARK_STATION = {
    "name": "NY City Central Park",
    "station_id": "GHCND:USW00094728",
    "latitude": 40.77898,
    "longitude": -73.96925,
}

FEATURE_LABELS = {
    "day_of_year_sin": ("Calendar", "Seasonal position, sine encoded"),
    "hour_sin": ("Time", "Prediction hour, sine encoded"),
    "hour_cos": ("Time", "Prediction hour, cosine encoded"),
    "month": ("Calendar", "Calendar month"),
    "season": ("Calendar", "Meteorological season code"),
    "forecast_horizon_hours": ("Time", "Hours until typical 3 PM high"),
    "current_temp": ("Observed Weather", "Latest observed temperature"),
    "dew_point": ("Observed Weather", "Latest dew point"),
    "cloud_cover_now": ("Observed Weather", "Latest cloud cover"),
    "wind_speed": ("Observed Weather", "Latest wind speed"),
    "precipitation_now": ("Observed Weather", "Latest precipitation"),
    "temp_minus_dew_point": ("Observed Weather", "Temperature minus dew point"),
    "wind_dir_sin": ("Observed Weather", "Wind direction, sine encoded"),
    "wind_dir_cos": ("Observed Weather", "Wind direction, cosine encoded"),
    "max_temp_so_far": ("Observed Path", "Observed max temperature so far"),
    "temp_change_60m": ("Observed Path", "Temperature change over 60 minutes"),
    "temp_change_120m": ("Observed Path", "Temperature change over 120 minutes"),
    "temp_change_180m": ("Observed Path", "Temperature change over 180 minutes"),
    "temp_change_240m": ("Observed Path", "Temperature change over 240 minutes"),
    "temp_change_300m": ("Observed Path", "Temperature change over 300 minutes"),
    "temp_acceleration_60m": ("Observed Path", "Recent temperature acceleration"),
    "temp_change_60m_minus_3h_avg_rate": ("Observed Path", "Short-term move versus 3-hour average"),
    "forecast_temp_current_hour": ("Forecast", "Forecast temperature for current hour"),
    "current_temp_minus_forecast_temp": ("Forecast Relative", "Current observed temp minus forecast temp"),
    "forecast_max_so_far": ("Forecast Path", "Forecast max through current hour"),
    "max_so_far_minus_forecast_max_so_far": ("Forecast Relative", "Observed max minus forecast max so far"),
    "current_temp_minus_max_so_far": ("Observed Path", "Current temp below/at high so far"),
    "minutes_since_max_temp_so_far": ("Observed Path", "Minutes since current daily high"),
    "hour_of_max_temp_so_far": ("Observed Path", "Hour when current daily high occurred"),
    "max_so_far_minus_forecast_high": ("Forecast Relative", "Observed max so far minus forecast high"),
    "mean_temp_error_so_far": ("Forecast Relative", "Mean observed-vs-forecast temp error so far"),
    "max_temp_error_so_far": ("Forecast Relative", "Max observed-vs-forecast temp error so far"),
    "num_new_highs_last_3h": ("Observed Path", "Strict new highs in trailing 3 hours"),
    "area_under_temp_curve_so_far": ("Observed Path", "Temperature integral so far"),
    "near_boundary_duration_so_far": ("Observed Path", "Boundary-adjacent observations so far"),
    "minutes_until_typical_peak": ("Time", "Minutes until typical 3 PM high"),
}


def main() -> None:
    st, px, go, st_autorefresh = _dashboard_dependencies()
    st.set_page_config(
        page_title="Kalshi Weather Trading Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles(st)

    st.title("Kalshi Weather Live Trading Dashboard")

    controls = _sidebar_controls(st, st_autorefresh)
    config = load_trading_config(controls["config_path"])
    state = _load_state(st, config, controls)

    _render_status(st, state)
    tabs = st.tabs([
        "Bucket Board",
        "Probability",
        "Order Book",
        "Edge",
        "Feature Inspector",
        "Weather Inputs",
        "Audit Artifacts",
    ])
    with tabs[0]:
        _render_bucket_board(st, state)
    with tabs[1]:
        _render_probability(st, px, state)
    with tabs[2]:
        _render_orderbook(st, px, state)
    with tabs[3]:
        _render_edge(st, state)
    with tabs[4]:
        _render_features(st, px, state)
    with tabs[5]:
        _render_weather(st, px, state, config)
    with tabs[6]:
        _render_artifacts(st, state)


def _dashboard_dependencies():
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dashboard dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    try:
        from streamlit_autorefresh import st_autorefresh
    except ModuleNotFoundError:
        st_autorefresh = None
    return st, px, go, st_autorefresh


def _sidebar_controls(st, st_autorefresh) -> dict[str, Any]:
    st.sidebar.header("Refresh")
    data_mode = st.sidebar.radio(
        "Data mode",
        ["Auto-fetch live", "Saved artifacts"],
        index=0,
        help="Live mode is still read-only. Saved artifacts mode never calls external APIs.",
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_seconds = st.sidebar.slider("Refresh seconds", min_value=15, max_value=300, value=60, step=15)
    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=refresh_seconds * 1000, key="dashboard_refresh")
        else:
            st.sidebar.info("Install streamlit-autorefresh for timer refresh; use the manual button for now.")
    manual_refresh = st.sidebar.button("Refresh Now", type="primary")
    if manual_refresh:
        st.cache_data.clear()

    st.sidebar.header("Market")
    event_ticker = st.sidebar.text_input("Event ticker override", value="")
    target_date = st.sidebar.text_input("Target date override", value="", placeholder="YYYY-MM-DD")
    depth = st.sidebar.slider("Order-book depth", min_value=1, max_value=100, value=20, step=1)

    st.sidebar.header("Configuration")
    config_path = st.sidebar.text_input("Trading config path", value=str(DEFAULT_TRADING_CONFIG_PATH))
    auth_market_data = st.sidebar.checkbox("Use auth for discovery", value=False)
    auth_orderbooks = st.sidebar.checkbox("Use auth for order books", value=False)
    fallback = st.sidebar.checkbox("Fallback to saved artifacts on live error", value=True)

    return {
        "data_mode": data_mode,
        "event_ticker": event_ticker.strip() or None,
        "target_date": target_date.strip() or None,
        "depth": depth,
        "config_path": config_path,
        "auth_market_data": auth_market_data,
        "auth_orderbooks": auth_orderbooks,
        "fallback": fallback,
    }


def _load_state(st, config, controls: dict[str, Any]) -> DashboardState:
    if controls["data_mode"] == "Saved artifacts":
        return load_dashboard_state_from_artifacts(config)

    try:
        return load_dashboard_state(
            config,
            event_ticker=controls["event_ticker"],
            target_date=controls["target_date"],
            depth=controls["depth"],
            auth_market_data=controls["auth_market_data"],
            auth_orderbooks=controls["auth_orderbooks"],
        )
    except Exception as exc:
        if not controls["fallback"]:
            raise
        st.warning(f"Live refresh failed; showing saved artifacts instead. Reason: {exc}")
        return load_dashboard_state_from_artifacts(config)


def _render_status(st, state: DashboardState) -> None:
    status = state.status
    status_color = "OK" if status.get("dashboard_status") == "OK" else "WARN"
    cols = st.columns(6)
    cols[0].metric("Dashboard", status_color)
    cols[1].metric("Mode", status.get("mode", ""))
    cols[2].metric("Trading Enabled", str(status.get("trading_enabled", "")))
    cols[3].metric("Kalshi Env", status.get("kalshi_env", ""))
    cols[4].metric("Event", status.get("event_ticker", ""))
    cols[5].metric("Prob Rows", status.get("probability_rows", 0))

    st.caption(f"Refreshed at: {status.get('refreshed_at', '')} | Data source: {status.get('data_source', '')} | Read-only: {status.get('read_only', True)}")
    warnings = status.get("warnings", [])
    if warnings:
        st.warning("Warnings: " + "; ".join(str(item) for item in warnings))


def _render_bucket_board(st, state: DashboardState) -> None:
    st.subheader("Bucket Board")
    board = state.bucket_board.copy()
    if board.empty:
        st.info("No bucket board is available yet.")
        return
    st.caption("Market prices are read from Kalshi production market data and shown as executable buy buttons when available.")
    st.dataframe(_kalshi_bucket_board_display(board), use_container_width=True, hide_index=True)


def _render_probability(st, px, state: DashboardState) -> None:
    st.subheader("Predicted Probability Distribution")
    probs = state.bucket_probabilities.copy()
    if probs.empty:
        st.info("No live bucket probabilities are available yet.")
        return

    chart = px.bar(
        probs.sort_values("bucket_index") if "bucket_index" in probs.columns else probs,
        x="bucket_name",
        y="probability",
        color="probability",
        color_continuous_scale="Teal",
        labels={"probability": "Model probability", "bucket_name": "Bucket"},
    )
    chart.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=80))
    st.plotly_chart(chart, use_container_width=True)

    params = state.distribution_params
    if not params.empty:
        cols = st.columns(4)
        cols[0].metric("Mu", _fmt(params["mu"].iloc[0]) if "mu" in params else "")
        cols[1].metric("Sigma", _fmt(params["sigma"].iloc[0]) if "sigma" in params else "")
        cols[2].metric("Distribution", str(params["distribution_type"].iloc[0]) if "distribution_type" in params else "")
        cols[3].metric("Model", str(params["model_name"].iloc[0]) if "model_name" in params else "")
    st.dataframe(probs, use_container_width=True, hide_index=True)


def _render_orderbook(st, px, state: DashboardState) -> None:
    st.subheader("Full Order Book")
    orderbook = state.orderbook.copy()
    if orderbook.empty:
        st.info("No order-book snapshot is available. Run live refresh with a current event or use saved artifacts after an order-book fetch.")
        return

    tickers = sorted(orderbook["ticker"].dropna().astype(str).unique())
    selected = st.selectbox("Select bucket ticker", tickers)
    selected_book = orderbook[orderbook["ticker"] == selected].copy()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(selected_book, use_container_width=True, hide_index=True)
    with col2:
        depth = selected_book.copy()
        if not depth.empty:
            chart = px.line(
                depth,
                x="price_dollars",
                y="cumulative_size",
                color="outcome_side",
                line_dash="quote_type",
                markers=True,
                labels={"price_dollars": "Price", "cumulative_size": "Cumulative size"},
            )
            chart.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=40))
            st.plotly_chart(chart, use_container_width=True)
    if not state.orderbook_summary.empty:
        st.markdown("#### Best Prices")
        st.dataframe(state.orderbook_summary, use_container_width=True, hide_index=True)


def _render_edge(st, state: DashboardState) -> None:
    st.subheader("Edge Table")
    edge = getattr(state, "edge_table", pd.DataFrame()).copy()
    if edge.empty:
        st.info("No edge table is available yet.")
        return
    candidates = edge[edge["edge_status"].astype(str) == "CANDIDATE"].copy()
    cols = st.columns(3)
    cols[0].metric("Candidates", len(candidates))
    cols[1].metric("Rows", len(edge))
    best_net_edge = candidates["net_edge"].max() if not candidates.empty else None
    cols[2].metric("Best Net Edge", _fmt(best_net_edge) if best_net_edge is not None else "")
    display_cols = [
        column
        for column in [
            "ticker",
            "bucket_name",
            "action",
            "fair_value",
            "executable_price",
            "executable_size",
            "spread",
            "gross_edge",
            "fee_per_contract",
            "slippage_buffer",
            "net_edge",
            "edge_status",
            "no_trade_reason",
        ]
        if column in edge.columns
    ]
    st.dataframe(edge[display_cols], use_container_width=True, hide_index=True)


def _render_features(st, px, state: DashboardState) -> None:
    st.subheader("Feature Inspector")
    features = state.live_feature_rows.copy()
    freshness = state.feature_freshness.copy()
    if features.empty:
        st.info("No live feature rows are available.")
        return

    metadata = {
        "row_id",
        "date",
        "target_date",
        "prediction_time",
        "prediction_timestamp",
        "location",
        "event_ticker",
        "forecast_high",
        "bucket_count",
        "mapping_status",
        "weather_status",
        "live_feature_status",
        "no_trade_reason",
    }
    row = features.iloc[0]
    records = []
    for column in features.columns:
        if column in metadata:
            continue
        group, label = FEATURE_LABELS.get(column, ("Other", column.replace("_", " ").title()))
        records.append({"feature": column, "group": group, "label": label, "value": row[column]})
    feature_table = pd.DataFrame.from_records(records)
    if not freshness.empty and "feature" in freshness.columns:
        feature_table = feature_table.merge(freshness, on="feature", how="left")

    group_counts = feature_table.groupby("group")["feature"].count().reset_index()
    chart = px.bar(group_counts, x="group", y="feature", color="group", labels={"feature": "Feature count"})
    chart.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=30, b=80))
    st.plotly_chart(chart, use_container_width=True)
    selected_groups = st.multiselect("Feature groups", sorted(feature_table["group"].unique()), default=sorted(feature_table["group"].unique()))
    st.dataframe(
        feature_table[feature_table["group"].isin(selected_groups)],
        use_container_width=True,
        hide_index=True,
    )


def _render_weather(st, px, state: DashboardState, config) -> None:
    st.subheader("Weather Inputs")
    weather = state.live_weather.copy()
    if weather.empty:
        st.info("No weather snapshot is available.")
        return

    obs = _filter_source_role(weather, "hourly_observations")
    forecasts = _filter_source_role(weather, "hourly_forecasts")
    daily = _filter_source_role(weather, "daily_forecast")
    diagnostics = _filter_source_role(weather, "weather_diagnostics")
    target_date = _status_target_date(state)

    _render_weather_source_summary(st, config)
    _render_weather_metrics(st, obs, forecasts, daily, state, target_date)

    obs_target = _weather_for_target_date(obs, target_date)
    forecast_target = _weather_for_target_date(forecasts, target_date)
    chart_rows = []
    for frame, label in [
        (obs_target, "Observed proxy"),
        (forecast_target, "Forecast proxy"),
    ]:
        if frame.empty or not {"timestamp", "temperature_2m"}.issubset(frame.columns):
            continue
        piece = frame[["timestamp", "temperature_2m"]].copy()
        piece["timestamp"] = pd.to_datetime(piece["timestamp"], errors="coerce")
        piece["temperature_2m"] = pd.to_numeric(piece["temperature_2m"], errors="coerce")
        piece["series"] = label
        chart_rows.append(piece.dropna(subset=["timestamp", "temperature_2m"]))
    if chart_rows:
        chart_frame = pd.concat(chart_rows, ignore_index=True)
        chart = px.line(
            chart_frame,
            x="timestamp",
            y="temperature_2m",
            color="series",
            markers=True,
            title=f"Event-Date Temperature Path ({target_date or 'target date unknown'})",
            labels={"temperature_2m": "Temperature / high (F)", "timestamp": "Local time"},
        )
        high_so_far = _weather_trace_frame(
            obs_target,
            value_col="observed_high_so_far",
            series_name="Observed high so far",
        )
        if not high_so_far.empty:
            chart.add_scatter(
                x=high_so_far["timestamp"],
                y=high_so_far["value"],
                mode="lines+markers",
                name="Observed high so far",
                line=dict(dash="dash", shape="hv"),
                marker=dict(size=7),
            )
        six_hour_max = _weather_trace_frame(
            obs_target,
            value_col="nws_6h_max_temp",
            series_name="NWS 6h max remark",
        )
        if not six_hour_max.empty:
            chart.add_scatter(
                x=six_hour_max["timestamp"],
                y=six_hour_max["value"],
                mode="markers",
                name="NWS 6h max remark",
                marker=dict(symbol="diamond", size=11),
            )
        daily_max = _weather_trace_frame(
            obs_target,
            value_col="nws_24h_max_temp",
            series_name="NWS 24h max remark",
        )
        if not daily_max.empty:
            chart.add_scatter(
                x=daily_max["timestamp"],
                y=daily_max["value"],
                mode="markers",
                name="NWS 24h max remark",
                marker=dict(symbol="star", size=13),
            )
        chart.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=45))
        st.plotly_chart(chart, use_container_width=True)
    if not daily.empty:
        st.markdown("#### Daily Forecast")
        st.dataframe(
            _weather_display_columns(
                _weather_for_target_date(daily, target_date),
                [
                    "date",
                    "forecast_high",
                    "temperature_2m_min",
                    "weather_code",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "forecast_source",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )
    if not diagnostics.empty:
        st.markdown("#### Freshness and Provenance Diagnostics")
        st.dataframe(
            _weather_display_columns(
                diagnostics,
                [
                    "diagnostic_name",
                    "diagnostic_source_role",
                    "status",
                    "no_trade_reason",
                    "source_time",
                    "fetched_at",
                    "age_minutes",
                    "detail",
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_artifacts(st, state: DashboardState) -> None:
    st.subheader("Audit Artifacts")
    orderbook_summary = getattr(state, "orderbook_summary", pd.DataFrame())
    edge_table = getattr(state, "edge_table", pd.DataFrame())
    artifacts = {
        "dashboard_status.json": pd.DataFrame([state.status]).to_json(indent=2),
        "bucket_board.csv": state.bucket_board.to_csv(index=False),
        "market_discovery.csv": state.market_discovery.to_csv(index=False),
        "contract_mapping.csv": state.mapping.to_csv(index=False),
        "live_feature_rows.csv": state.live_feature_rows.to_csv(index=False),
        "live_feature_freshness.csv": state.feature_freshness.to_csv(index=False),
        "live_bucket_probabilities.csv": state.bucket_probabilities.to_csv(index=False),
        "orderbook_snapshot.csv": state.orderbook.to_csv(index=False),
        "orderbook_summary.csv": orderbook_summary.to_csv(index=False),
        "edge_table.csv": edge_table.to_csv(index=False),
    }
    for name, payload in artifacts.items():
        mime = "application/json" if name.endswith(".json") else "text/csv"
        st.download_button(
            label=f"Download {name}",
            data=payload,
            file_name=name,
            mime=mime,
        )


def _format_probability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in [
        "probability",
        "best_yes_bid",
        "best_yes_ask",
        "yes_spread",
        "best_no_bid",
        "best_no_ask",
        "no_spread",
        "best_net_edge",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(4)
    return result


def _kalshi_bucket_board_display(board: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in board.iterrows():
        yes_buy_price = _coalesce_numeric(row.get("yes_ask_dollars"), row.get("best_yes_ask"))
        no_buy_price = _coalesce_numeric(row.get("no_ask_dollars"), row.get("best_no_ask"))
        chance_price = _coalesce_numeric(
            row.get("last_price_dollars"),
            _market_chance_from_prices(row, yes_buy_price, no_buy_price),
        )
        records.append(
            {
                "Market": row.get("bucket_name", ""),
                "Chance": _format_market_chance(chance_price),
                "Yes": _format_kalshi_button("Yes", yes_buy_price),
                "No": _format_kalshi_button("No", no_buy_price),
                "Model": _format_percent(row.get("probability")),
                "Best Edge": _format_edge(row.get("best_edge_action"), row.get("best_net_edge")),
                "Status": row.get("orderbook_status", ""),
                "Ticker": row.get("ticker", ""),
            }
        )
    return pd.DataFrame.from_records(records)


def _market_chance_from_prices(
    row: pd.Series,
    yes_buy_price: float | None,
    no_buy_price: float | None,
) -> float | None:
    yes_bid = _coalesce_numeric(row.get("yes_bid_dollars"), row.get("best_yes_bid"))
    no_bid = _coalesce_numeric(row.get("no_bid_dollars"), row.get("best_no_bid"))
    if yes_bid is not None and yes_buy_price is not None:
        return (yes_bid + yes_buy_price) / 2.0
    if no_bid is not None and no_buy_price is not None:
        no_mid = (no_bid + no_buy_price) / 2.0
        return 1.0 - no_mid
    if yes_buy_price is not None:
        return yes_buy_price
    if no_buy_price is not None:
        return 1.0 - no_buy_price
    if yes_bid is not None:
        return yes_bid
    if no_bid is not None:
        return 1.0 - no_bid
    return None


def _format_market_chance(value: Any) -> str:
    numeric = _coalesce_numeric(value)
    if numeric is None:
        return "--"
    bounded = min(1.0, max(0.0, numeric))
    if bounded <= 0.01:
        return "<1%"
    if bounded >= 0.995:
        return ">99%"
    return f"{bounded * 100:.0f}%"


def _format_percent(value: Any) -> str:
    numeric = _coalesce_numeric(value)
    if numeric is None:
        return "--"
    if numeric <= 0.01:
        return "<1%"
    if numeric >= 0.995:
        return ">99%"
    return f"{numeric * 100:.0f}%"


def _format_kalshi_button(label: str, price: Any) -> str:
    numeric = _coalesce_numeric(price)
    if numeric is None:
        return f"{label} --"
    if numeric >= 0.995:
        return label
    cents = max(0, min(100, int(round(numeric * 100))))
    if cents <= 0:
        return f"{label} <1c"
    return f"{label} {cents}c"


def _format_edge(action: Any, net_edge: Any) -> str:
    action_text = str(action or "").strip()
    edge = _coalesce_numeric(net_edge)
    if not action_text or edge is None:
        return "--"
    return f"{action_text} ({edge * 100:.1f}c)"


def _coalesce_numeric(*values: Any) -> float | None:
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(numeric):
            return numeric
    return None


def _filter_source_role(frame: pd.DataFrame, source_role: str) -> pd.DataFrame:
    if frame.empty or "source_role" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame["source_role"].astype(str) == source_role].copy()


def _status_target_date(state: DashboardState) -> str:
    value = state.status.get("target_date", "")
    if value:
        return str(value)[:10]
    return ""


def _weather_for_target_date(frame: pd.DataFrame, target_date: str) -> pd.DataFrame:
    if frame.empty or not target_date:
        return frame.copy()
    result = frame.copy()
    if "date" in result.columns:
        dates = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return result[dates == target_date].copy()
    if "target_date" in result.columns:
        dates = pd.to_datetime(result["target_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return result[dates == target_date].copy()
    return result


def _weather_display_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return frame
    return frame.loc[:, present].copy()


def _weather_trace_frame(
    frame: pd.DataFrame,
    *,
    value_col: str,
    series_name: str,
) -> pd.DataFrame:
    if frame.empty or not {"timestamp", value_col}.issubset(frame.columns):
        return pd.DataFrame(columns=["timestamp", "value", "series"])
    trace = frame[["timestamp", value_col]].copy()
    trace["timestamp"] = pd.to_datetime(trace["timestamp"], errors="coerce")
    trace["value"] = pd.to_numeric(trace[value_col], errors="coerce")
    trace["series"] = series_name
    return trace.dropna(subset=["timestamp", "value"])[["timestamp", "value", "series"]]


def _render_weather_source_summary(st, config) -> None:
    forecast_grid = config.weather.forecast_grid
    if getattr(config.weather, "observations_provider", "") == "nws_station":
        station = config.weather.nws_station
        obs_label = f"{station.station_id} / {station.ghcn_station_id}"
        obs_miles = _distance_miles(
            station.latitude,
            station.longitude,
            CENTRAL_PARK_STATION["latitude"],
            CENTRAL_PARK_STATION["longitude"],
        )
    else:
        obs_grid = config.weather.observation_grid
        obs_label = "Open-Meteo proxy grid"
        obs_miles = _distance_miles(
            obs_grid.latitude,
            obs_grid.longitude,
            CENTRAL_PARK_STATION["latitude"],
            CENTRAL_PARK_STATION["longitude"],
        )
    forecast_miles = _distance_miles(
        forecast_grid.latitude,
        forecast_grid.longitude,
        CENTRAL_PARK_STATION["latitude"],
        CENTRAL_PARK_STATION["longitude"],
    )
    cols = st.columns(3)
    cols[0].metric("Observed Source", obs_label)
    cols[1].metric("Obs Grid Distance", f"{obs_miles:.2f} mi")
    cols[2].metric("Forecast Grid Distance", f"{forecast_miles:.2f} mi")
    if forecast_miles > 0.25 or getattr(config.weather, "observations_provider", "") != "nws_station":
        st.warning(
            "Forecast inputs are Open-Meteo proxy grid values; Kalshi settlement uses NWS Central Park observations. "
            f"Central Park station: {CENTRAL_PARK_STATION['latitude']:.5f}, "
            f"{CENTRAL_PARK_STATION['longitude']:.5f}; "
            f"forecast grid: {forecast_grid.latitude:.5f}, {forecast_grid.longitude:.5f}."
        )


def _render_weather_metrics(st, obs: pd.DataFrame, forecasts: pd.DataFrame, daily: pd.DataFrame, state: DashboardState, target_date: str) -> None:
    obs_target = _weather_for_target_date(obs, target_date)
    forecast_target = _weather_for_target_date(forecasts, target_date)
    daily_target = _weather_for_target_date(daily, target_date)

    current_temp = _latest_numeric(obs_target, "temperature_2m", "timestamp")
    high_so_far = _latest_numeric(obs_target, "observed_high_so_far", "timestamp")
    high_so_far = _coalesce_float(high_so_far, _max_numeric(obs_target, "temperature_2m"))
    forecast_high = _first_numeric(daily_target, "forecast_high")
    forecast_current = _latest_numeric(forecast_target, "temperature_2m", "timestamp")

    if not state.live_feature_rows.empty:
        row = state.live_feature_rows.iloc[0]
        current_temp = _coalesce_float(row.get("current_temp"), current_temp)
        high_so_far = _coalesce_float(row.get("max_temp_so_far"), high_so_far)
        forecast_high = _coalesce_float(row.get("forecast_high"), forecast_high)
        forecast_current = _coalesce_float(row.get("forecast_temp_current_hour"), forecast_current)

    cols = st.columns(4)
    cols[0].metric("Observed Current", _fmt_or_blank(current_temp))
    cols[1].metric("Observed High So Far", _fmt_or_blank(high_so_far))
    cols[2].metric("Daily Forecast High", _fmt_or_blank(forecast_high))
    cols[3].metric("Forecast Current Hour", _fmt_or_blank(forecast_current))


def _latest_numeric(frame: pd.DataFrame, value_col: str, time_col: str) -> float | None:
    if frame.empty or value_col not in frame.columns or time_col not in frame.columns:
        return None
    working = frame.copy()
    working[time_col] = pd.to_datetime(working[time_col], errors="coerce")
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[time_col, value_col]).sort_values(time_col)
    if working.empty:
        return None
    return float(working.iloc[-1][value_col])


def _max_numeric(frame: pd.DataFrame, value_col: str) -> float | None:
    if frame.empty or value_col not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _first_numeric(frame: pd.DataFrame, value_col: str) -> float | None:
    if frame.empty or value_col not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _coalesce_float(primary: Any, fallback: float | None) -> float | None:
    try:
        value = float(primary)
    except (TypeError, ValueError):
        return fallback
    if pd.isna(value):
        return fallback
    return value


def _fmt_or_blank(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1f} F"


def _distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import atan2, cos, radians, sin, sqrt

    earth_radius_miles = 3958.7613
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a = (
        sin(dlat / 2.0) ** 2
        + cos(radians(float(lat1)))
        * cos(radians(float(lat2)))
        * sin(dlon / 2.0) ** 2
    )
    return 2.0 * earth_radius_miles * atan2(sqrt(a), sqrt(1.0 - a))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _inject_styles(st) -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
