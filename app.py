from __future__ import annotations

import base64
import html as html_lib
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.chatbot import answer_question, get_suggested_questions
from src.config import APP_NAME, DEFAULT_BENCHMARK, DEFAULT_TICKERS, DISCLAIMER, MARKET_PRESETS, MARKET_SUFFIXES, PERIOD_OPTIONS
from src.data import (
    fetch_history,
    fetch_live_info,
    fetch_live_price_table,
    fetch_many_histories,
    fetch_news,
    normalize_ticker,
    parse_ticker_list,
)
from src.features import add_technical_indicators
from src.portfolio import allocate_investment, build_price_frame, optimize_weights
from src.prediction import PredictionResult, predict_next_price
from src.recommender import RecommendationResult, build_recommendation
from src.risk import risk_metrics
from src.sentiment import aggregate_sentiment, analyze_news
from src.universe import get_company_universe


NEWS_SOURCE_OPTIONS = ["Google News RSS (fast)", "Yahoo Finance + Google (slower)", "Yahoo Finance only"]

# Universe options vary by selected market – resolved dynamically below
_BASE_UNIVERSE_OPTIONS = {
    "🇮🇳 India (NSE)": ["NIFTY 50", "NSE all listed equities", "Custom tickers"],
    "🇮🇳 India (BSE)": ["NIFTY 50", "Custom tickers"],
    "🇺🇸 United States": ["US popular stocks", "Custom tickers"],
    "🇬🇧 United Kingdom": ["UK popular stocks", "Custom tickers"],
    "🇩🇪 Germany": ["DE popular stocks", "Custom tickers"],
    "🇯🇵 Japan": ["JP popular stocks", "Custom tickers"],
    "🇭🇰 Hong Kong": ["HK popular stocks", "Custom tickers"],
    "🇦🇺 Australia": ["AU popular stocks", "Custom tickers"],
    "🇨🇦 Canada": ["CA popular stocks", "Custom tickers"],
}

MARKET_INTERVAL_OPTIONS = {
    "Daily latest (fast)": ("5d", "1d"),
    "Intraday 5 minute (slower)": ("1d", "5m"),
    "Intraday 1 minute (slowest)": ("1d", "1m"),
}
BASE_DIR = Path(__file__).resolve().parent
HEADER_IMAGE_PATH = BASE_DIR / "assets" / "market-header.png"

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")

CSS_PATH = BASE_DIR / "assets" / "style.css"
if CSS_PATH.exists():
    css_content = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
else:
    st.warning("Custom stylesheet assets/style.css not found.")



@st.cache_data(ttl=900, show_spinner=False)
def cached_history(ticker: str, period: str, refresh_token: str = "static") -> pd.DataFrame:
    return fetch_history(ticker, period=period)


@st.cache_data(ttl=300, show_spinner=False)
def cached_live_info(ticker: str, refresh_token: str = "static") -> dict[str, Any]:
    return fetch_live_info(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_company_universe(universe_name: str, custom_raw: str, suffix: str) -> pd.DataFrame:
    custom_tickers = parse_ticker_list(custom_raw, suffix)
    return get_company_universe(universe_name, custom_tickers)


@st.cache_data(ttl=300, show_spinner=False)
def cached_live_price_table(tickers: tuple[str, ...], period_value: str, interval_value: str, refresh_token: str = "static") -> pd.DataFrame:
    chunk_size = 60 if interval_value in {"1m", "5m"} else 100
    return fetch_live_price_table(list(tickers), period=period_value, interval=interval_value, chunk_size=chunk_size)


@st.cache_data(ttl=900, show_spinner=False)
def cached_prediction(history_frame: pd.DataFrame, lookback: int, epochs: int, use_lstm: bool) -> PredictionResult:
    return predict_next_price(history_frame, lookback=lookback, epochs=epochs, use_lstm=use_lstm)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(ticker: str, company_query: str, limit: int, source_mode: str) -> list[dict[str, Any]]:
    return fetch_news(ticker, company_query=company_query, limit=limit, source_mode=source_mode)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news_sentiment(
    ticker: str,
    company_query: str,
    limit: int,
    source_mode: str,
    use_finbert_model: bool,
    refresh_token: str = "static",
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    news_items = fetch_news(ticker, company_query=company_query, limit=limit, source_mode=source_mode)
    sentiment_frame = analyze_news(news_items, use_finbert=use_finbert_model)
    return sentiment_frame, aggregate_sentiment(sentiment_frame)


def money(value: float | int | None, currency: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{float(value):,.2f}"


def pct(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def price_chart(feature_frame: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=feature_frame.index,
            open=feature_frame["Open"],
            high=feature_frame["High"],
            low=feature_frame["Low"],
            close=feature_frame["Close"],
            name="Price",
            increasing_line_color="#169b62",
            increasing_fillcolor="#169b62",
            decreasing_line_color="#c2413d",
            decreasing_fillcolor="#c2413d",
        )
    )
    fig.add_trace(go.Scatter(x=feature_frame.index, y=feature_frame["MA_20"], name="MA 20", line=dict(color="#2563eb", width=1.8)))
    fig.add_trace(go.Scatter(x=feature_frame.index, y=feature_frame["MA_50"], name="MA 50", line=dict(color="#b7791f", width=1.8)))
    fig.update_layout(
        title=f"{ticker} price action",
        height=420,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#17202a"),
        xaxis=dict(showgrid=True, gridcolor="#edf1f5", rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="#edf1f5"),
    )
    return fig


def recommendation_class(action: str) -> str:
    return {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(action, "signal-hold")


def asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def safe_text(value: Any) -> str:
    return html_lib.escape(str(value if value is not None else "N/A"))


def signed_pct(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value) * 100:+.2f}%"


def tone_for_number(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "neutral"
    if float(value) > 0:
        return "positive"
    if float(value) < 0:
        return "negative"
    return "neutral"


def action_tone(action: str) -> str:
    return {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}.get(action, "hold")


def metric_card(label: str, value: str, note: str = "", tone: str = "neutral") -> None:
    note_html = f"<div class='metric-note tone-{tone}'>{safe_text(note)}</div>" if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{safe_text(label)}</div>
            <div class="metric-value">{safe_text(value)}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{safe_text(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title">
            <div>
                <h2>{safe_text(title)}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    ticker: str,
    company_name: str,
    price: float | int | None,
    change: float | int | None,
    period_value: str,
    currency_value: str,
) -> None:
    image_uri = asset_data_uri(HEADER_IMAGE_PATH)
    media_style = f"background-image: url('{image_uri}')" if image_uri else ""
    change_tone = tone_for_number(change)
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="hero-copy">
                <div class="eyebrow"><span class="live-dot"></span> Live market intelligence</div>
                <h1>{safe_text(APP_NAME)}</h1>
                <p>{safe_text(company_name)} is loaded with current quote data, technical indicators, news sentiment, portfolio tools, and risk signals.</p>
                <div class="hero-stats">
                    <div class="hero-stat"><span>Symbol</span><strong>{safe_text(ticker)}</strong></div>
                    <div class="hero-stat"><span>Price</span><strong>{safe_text(money(price, currency_value))}</strong></div>
                    <div class="hero-stat"><span>Move</span><strong class="tone-{change_tone}">{safe_text(signed_pct(change))}</strong></div>
                    <div class="hero-stat"><span>Window</span><strong>{safe_text(period_value)}</strong></div>
                </div>
            </div>
            <div class="hero-media" style="{media_style}"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def profile_panel(rows: list[tuple[str, str]]) -> None:
    rendered_rows = "\n".join(
        f"<div class='profile-row'><span>{safe_text(label)}</span><strong>{safe_text(value)}</strong></div>"
        for label, value in rows
    )
    st.markdown(f"<div class='profile-panel'>{rendered_rows}</div>", unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.markdown(f"<div class='empty-state'>{safe_text(message)}</div>", unsafe_allow_html=True)


def market_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "Change %" in display.columns:
        display["Change %"] = display["Change %"] * 100
    return display


def refresh_time_label(tz: str = "UTC") -> str:
    return pd.Timestamp.now(tz=tz).strftime("%I:%M:%S %p %Z")


def render_landing_page(active_preset: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div style="padding: 2.5rem; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-lg); box-shadow: var(--shadow-md); margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem;">
                <span style="display: inline-flex; align-items: center; justify-content: center; width: 2.5rem; height: 2.5rem; border-radius: var(--radius-sm); background: var(--gradient-primary); color: white; font-weight: 900; font-size: 1.1rem; box-shadow: 0 4px 12px rgba(13,148,136,0.3);">FS</span>
                <h1 style="margin: 0; background: var(--gradient-primary); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.2rem; letter-spacing: -0.02em;">FinSight AI</h1>
            </div>
            <p style="color: var(--muted); font-size: 1.05rem; margin-top: 0.5rem; margin-bottom: 1.8rem; line-height: 1.6;">
                Welcome to FinSight AI, a deployment-ready market intelligence dashboard. Select a preset company from the active market or enter a custom ticker symbol below to begin real-time data analysis, neural forecasting, sentiment scoring, and AI financial reasoning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔍 Select a Company to Analyze")
    
    presets = active_preset.get("default_tickers", [])
    options = presets + ["Custom Ticker..."]
    
    selected_option = st.selectbox(
        "Choose a stock from the active country/preset list",
        options=options,
        index=0,
        key="landing_stock_select"
    )
    
    ticker_input = ""
    if selected_option == "Custom Ticker...":
        ticker_input = st.text_input("Enter stock ticker symbol (e.g., AAPL, RELIANCE.NS, BP.L)", placeholder="e.g. NVDA").strip()
    
    if st.button("Analyze Stock 📈", type="primary", use_container_width=True):
        final_ticker = ""
        if selected_option == "Custom Ticker...":
            if not ticker_input:
                st.warning("Please enter a custom ticker symbol.")
                return
            final_ticker = normalize_ticker(ticker_input, active_preset["suffix"])
        else:
            final_ticker = selected_option
            
        if final_ticker:
            st.session_state["selected_ticker"] = final_ticker
            st.rerun()


with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <span>FS</span>
            <strong>{safe_text(APP_NAME)}</strong>
            <small>Deployable market intelligence</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    market_name = st.selectbox("🌍 Select Market / Country", list(MARKET_PRESETS.keys()), index=0)
    active_preset = MARKET_PRESETS[market_name]
    suffix = active_preset["suffix"]
    default_benchmark = active_preset["benchmark"]
    market_tz = active_preset.get("timezone", "UTC")

    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = None
    if "current_market" not in st.session_state:
        st.session_state["current_market"] = market_name

    if st.session_state["current_market"] != market_name:
        st.session_state["current_market"] = market_name
        st.session_state["selected_ticker"] = None
        st.rerun()

    selected_ticker = st.session_state["selected_ticker"]

    if selected_ticker:
        if st.sidebar.button("🔄 Change Stock / Company", use_container_width=True):
            st.session_state["selected_ticker"] = None
            st.rerun()

        period = st.selectbox("History period", PERIOD_OPTIONS, index=2)
        benchmark = st.text_input("Benchmark", value=default_benchmark)
        use_finbert = st.checkbox("Use FinBERT when installed", value=False)
        auto_refresh = st.checkbox("Auto live refresh", value=True)
        refresh_interval = st.slider(
            "Refresh every",
            min_value=15,
            max_value=120,
            value=30,
            step=15,
            disabled=not auto_refresh,
        )
        if "manual_refresh_nonce" not in st.session_state:
            st.session_state["manual_refresh_nonce"] = "initial"
        if st.button("Refresh now"):
            st.session_state["manual_refresh_nonce"] = str(time.time())

        live_refresh_count = 0
        if auto_refresh:
            live_refresh_count = st_autorefresh(
                interval=refresh_interval * 1000,
                key="live_auto_refresh",
            )

        refresh_token = f"{live_refresh_count}-{st.session_state['manual_refresh_nonce']}"
        news_refresh_token = f"{int(time.time() // max(refresh_interval * 10, 300))}-{st.session_state['manual_refresh_nonce']}"
        st.markdown(
            f"""
            <div class="live-status{' ' if auto_refresh else ' off'}">
                <strong>{'Auto live refresh ON' if auto_refresh else 'Auto live refresh OFF'}</strong>
                Last checked: {safe_text(refresh_time_label(market_tz))}<br>
                Interval: {refresh_interval if auto_refresh else 'manual'} seconds
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        period = "6mo"
        benchmark = default_benchmark
        use_finbert = False
        auto_refresh = False
        refresh_interval = 30
        refresh_token = "static"
        news_refresh_token = "static"

    st.caption(DISCLAIMER)

# If no stock is selected, show landing page and stop
if not selected_ticker:
    render_landing_page(active_preset)
    st.stop()

try:
    history = cached_history(selected_ticker, period, refresh_token)
    live_info = cached_live_info(selected_ticker, refresh_token)
    features = add_technical_indicators(history)
except Exception as exc:
    st.error(str(exc))
    st.info("Install missing dependencies with: pip install -r requirements.txt")
    st.stop()

currency = live_info.get("currency") or ""
latest_close = float(features["Close"].iloc[-1])
current_price = live_info.get("current_price") or latest_close
previous_close = live_info.get("previous_close")
day_change = None
if current_price and previous_close:
    day_change = (float(current_price) / float(previous_close)) - 1

render_header(
    selected_ticker,
    live_info.get("name") or selected_ticker,
    current_price,
    day_change,
    period,
    currency,
)

page = st.tabs(["Dashboard", "Company Spotlight", "Market Watch", "Prediction", "News", "Portfolio", "Risk", "Chatbot"])

with page[0]:  # Dashboard
    col_title, col_switch = st.columns([3, 1])
    with col_title:
        section_title("Dashboard", live_info.get("name") or selected_ticker)
    with col_switch:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Change Stock / Company", key="dash_reset_btn_2", use_container_width=True):
            st.session_state["selected_ticker"] = None
            st.rerun()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Current Price", money(current_price, currency), "Live quote")
    with col2:
        metric_card("Latest Close", money(latest_close, currency), "Historical close")
    with col3:
        metric_card("Day Change", signed_pct(day_change), "vs previous close", tone_for_number(day_change))
    with col4:
        metric_card("20D Volatility", pct(float(features["Volatility_20"].iloc[-1])), "Daily return risk", "warning")

    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
    st.plotly_chart(price_chart(features, selected_ticker), use_container_width=True)

    left, right = st.columns([2, 1])
    with left:
        section_title("Recent Market Data")
        st.dataframe(
            features.tail(10)[["Open", "High", "Low", "Close", "Volume", "MA_20", "RSI_14", "MACD"]],
            use_container_width=True,
            hide_index=False,
            column_config={
                "Open": st.column_config.NumberColumn(format="%.2f"),
                "High": st.column_config.NumberColumn(format="%.2f"),
                "Low": st.column_config.NumberColumn(format="%.2f"),
                "Close": st.column_config.NumberColumn(format="%.2f"),
                "Volume": st.column_config.NumberColumn(format="%d"),
                "MA_20": st.column_config.NumberColumn(format="%.2f"),
                "RSI_14": st.column_config.NumberColumn(format="%.2f"),
                "MACD": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    with right:
        section_title("Company")
        profile_panel(
            [
                ("Name", live_info.get("name") or selected_ticker),
                ("Exchange", live_info.get("exchange") or "N/A"),
                ("Market Cap", money(live_info.get("market_cap"), currency)),
                ("Day High", money(live_info.get("day_high"), currency)),
                ("Day Low", money(live_info.get("day_low"), currency)),
            ]
        )

with page[2]:  # Market Watch
    section_title("Market Watch", "Live company universe scanner")

    left_controls, right_controls = st.columns([2, 1])
    universe_options = _BASE_UNIVERSE_OPTIONS.get(market_name, ["Custom tickers"])
    default_universe = active_preset.get("default_universe", universe_options[0])
    universe_name = left_controls.segmented_control("Company universe", options=universe_options, default=default_universe)
    interval_label = right_controls.selectbox("Quote mode", list(MARKET_INTERVAL_OPTIONS.keys()), index=0)
    custom_raw = ""
    if universe_name == "Custom tickers":
        custom_raw = st.text_area("Custom tickers", value=", ".join(active_preset["default_tickers"]), height=90)

    with st.spinner("Loading company symbols..."):
        universe = cached_company_universe(universe_name, custom_raw, suffix)

    if universe.empty:
        st.warning("No companies found for this universe.")
    else:
        full_universe = st.checkbox("Fetch full universe", value=(universe_name != "NSE all listed equities"))
        if full_universe:
            fetch_count = len(universe)
        else:
            default_count = min(100, len(universe))
            fetch_count = st.slider("Companies to fetch", min_value=1, max_value=len(universe), value=default_count)

        st.caption(f"Universe size: {len(universe):,} companies | Fetching: {fetch_count:,}")
        period_value, interval_value = MARKET_INTERVAL_OPTIONS[interval_label]
        selected_universe = universe.head(fetch_count).copy()
        tickers = tuple(selected_universe["Ticker"].tolist())

        market_fetch_clicked = st.button("Fetch live company data", type="primary")
        should_auto_update_market = (
            auto_refresh
            and st.session_state.get("market_live_tickers") == tickers
            and st.session_state.get("market_live_period") == period_value
            and st.session_state.get("market_live_interval") == interval_value
        )

        if market_fetch_clicked or should_auto_update_market:
            start_time = time.perf_counter()
            with st.spinner("Fetching live prices in batches..."):
                market_table = cached_live_price_table(tickers, period_value, interval_value, refresh_token)
                market_table = market_table.merge(selected_universe[["Ticker", "Symbol", "Company"]], on="Ticker", how="left")
                market_table = market_table[["Symbol", "Company", "Ticker", "Latest Price", "Reference Price", "Change %", "Volume", "Last Update"]]
            st.session_state["market_table"] = market_table
            st.session_state["market_elapsed"] = time.perf_counter() - start_time
            st.session_state["market_universe_size"] = len(universe)
            st.session_state["market_requested_count"] = fetch_count
            st.session_state["market_live_tickers"] = tickers
            st.session_state["market_live_period"] = period_value
            st.session_state["market_live_interval"] = interval_value
            st.session_state["market_live_updated_at"] = refresh_time_label(market_tz)

        market_table = st.session_state.get("market_table")
        if isinstance(market_table, pd.DataFrame) and not market_table.empty:
            loaded_count = len(market_table)
            requested_count = st.session_state.get("market_requested_count", loaded_count)
            market_display = market_display_frame(market_table)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Companies Loaded", f"{loaded_count:,}", "quotes returned")
            with c2:
                metric_card("Requested", f"{requested_count:,}", "tickers scanned")
            with c3:
                metric_card("Top Gain", f"{market_table['Change %'].max() * 100:.2f}%", "best mover", "positive")
            with c4:
                metric_card("Top Loss", f"{market_table['Change %'].min() * 100:.2f}%", "weakest mover", "negative")
            if "market_elapsed" in st.session_state:
                st.caption(
                    f"Last live update: {st.session_state.get('market_live_updated_at', refresh_time_label(market_tz))} | "
                    f"Fetch time: {st.session_state['market_elapsed']:.2f}s"
                )

            gainers, losers = st.columns(2)
            with gainers:
                section_title("Top Gainers")
                st.dataframe(
                    market_display.sort_values("Change %", ascending=False).head(10)[["Symbol", "Company", "Latest Price", "Change %", "Volume"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Latest Price": st.column_config.NumberColumn(format="%.2f"),
                        "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Volume": st.column_config.NumberColumn(format="%d"),
                    },
                )
            with losers:
                section_title("Top Losers")
                st.dataframe(
                    market_display.sort_values("Change %", ascending=True).head(10)[["Symbol", "Company", "Latest Price", "Change %", "Volume"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Latest Price": st.column_config.NumberColumn(format="%.2f"),
                        "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Volume": st.column_config.NumberColumn(format="%d"),
                    },
                )

            section_title("All Loaded Companies")
            st.dataframe(
                market_display.sort_values("Change %", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Latest Price": st.column_config.NumberColumn(format="%.2f"),
                    "Reference Price": st.column_config.NumberColumn(format="%.2f"),
                    "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Volume": st.column_config.NumberColumn(format="%d"),
                },
            )
            st.download_button(
                "Download market data CSV",
                data=market_display.to_csv(index=False).encode("utf-8"),
                file_name="live_market_data.csv",
                mime="text/csv",
            )
        else:
            empty_state("Choose a company universe and fetch live company data.")

with page[3]:  # Prediction
    section_title("Prediction Engine", "Fast technical signal with optional LSTM training")
    controls = st.columns(4)
    lookback = controls[0].slider("Lookback days", min_value=30, max_value=120, value=60, step=10)
    model_mode = controls[1].segmented_control("Model mode", options=["Fast", "LSTM"], default="Fast")
    refresh_news = controls[2].checkbox("Refresh news too", value=False)
    news_limit = controls[3].slider("News items", min_value=3, max_value=15, value=5, disabled=not refresh_news)
    epochs = 3
    if model_mode == "LSTM":
        epochs = st.slider("LSTM epochs", min_value=1, max_value=10, value=3)
        st.caption("LSTM training is slower because it trains a fresh neural network in the app. Fast mode is recommended for live demos.")

    prediction_clicked = st.button("Run prediction", type="primary")
    should_auto_update_prediction = (
        auto_refresh
        and st.session_state.get("prediction_live_enabled") is True
        and st.session_state.get("prediction_live_ticker") == selected_ticker
        and st.session_state.get("prediction_live_lookback") == lookback
        and st.session_state.get("prediction_live_mode") == "Fast"
        and model_mode == "Fast"
    )

    if prediction_clicked or should_auto_update_prediction:
        start_time = time.perf_counter()
        with st.spinner("Generating prediction signal..."):
            prediction = cached_prediction(history, lookback=lookback, epochs=epochs, use_lstm=(model_mode == "LSTM"))
            if refresh_news and prediction_clicked:
                sentiment_frame, sentiment_summary = cached_news_sentiment(
                    selected_ticker,
                    live_info.get("name") or selected_ticker,
                    news_limit,
                    "Google News RSS (fast)",
                    use_finbert,
                    news_refresh_token,
                )
                st.session_state["news_frame"] = sentiment_frame
                st.session_state["news_summary"] = sentiment_summary
            else:
                sentiment_summary = st.session_state.get("news_summary", {"label": "Neutral", "score": 0.5, "confidence": 0.0, "headline_count": 0})
            recommendation = build_recommendation(prediction, sentiment_summary)

        st.session_state["latest_prediction"] = prediction
        st.session_state["latest_sentiment"] = sentiment_summary
        st.session_state["latest_recommendation"] = recommendation
        st.session_state["latest_prediction_elapsed"] = time.perf_counter() - start_time
        st.session_state["latest_prediction_updated_at"] = refresh_time_label(market_tz)
        st.session_state["prediction_live_enabled"] = model_mode == "Fast"
        st.session_state["prediction_live_ticker"] = selected_ticker
        st.session_state["prediction_live_lookback"] = lookback
        st.session_state["prediction_live_mode"] = model_mode

    prediction = st.session_state.get("latest_prediction")
    sentiment_summary = st.session_state.get("latest_sentiment")
    recommendation = st.session_state.get("latest_recommendation")

    if isinstance(prediction, PredictionResult) and isinstance(recommendation, RecommendationResult):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Current", money(prediction.current_price, currency), "latest close")
        with c2:
            metric_card("Predicted Next Close", money(prediction.predicted_price, currency), f"{prediction.expected_return_pct:+.2f}%", tone_for_number(prediction.expected_return_pct))
        with c3:
            metric_card("Confidence", f"{prediction.confidence * 100:.1f}%", prediction.method)
        with c4:
            tone = action_tone(recommendation.action)
            st.markdown(
                f"""
                <div class="recommendation-panel">
                    <h3>Recommendation</h3>
                    <div class="big-action pill-{tone}">{safe_text(recommendation.action)}</div>
                    <span class="status-pill pill-{tone}">Hybrid score {recommendation.score:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        section_title("Model Rationale")
        st.progress(min(1.0, max(0.0, (recommendation.score + 1) / 2)))
        st.caption(f"Hybrid score: {recommendation.score:.2f} | Sentiment: {sentiment_summary.get('label')} ({float(sentiment_summary.get('score', 0.5)):.2f})")
        if "latest_prediction_elapsed" in st.session_state:
            st.caption(
                f"Last live update: {st.session_state.get('latest_prediction_updated_at', refresh_time_label(market_tz))} | "
                f"Run time: {st.session_state['latest_prediction_elapsed']:.2f}s"
            )
        for reason in recommendation.rationale:
            st.write(f"- {reason}")

        if "fallback_reason" in prediction.details:
            st.info(f"LSTM fallback used: {prediction.details['fallback_reason']}")
    else:
        empty_state("Run prediction to generate the current model signal.")

with page[4]:  # News
    section_title("News Sentiment", "Headline-level finance sentiment")
    news_source = st.segmented_control("News source", options=NEWS_SOURCE_OPTIONS, default="Google News RSS (fast)")
    limit = st.slider("Headlines to fetch", min_value=3, max_value=20, value=8)
    news_clicked = st.button("Refresh news sentiment")
    should_auto_update_news = (
        auto_refresh
        and st.session_state.get("news_live_enabled") is True
        and st.session_state.get("news_live_ticker") == selected_ticker
        and st.session_state.get("news_live_limit") == limit
        and st.session_state.get("news_live_source") == news_source
        and st.session_state.get("news_live_finbert") == use_finbert
    )

    if news_clicked or should_auto_update_news:
        start_time = time.perf_counter()
        with st.spinner("Fetching cached-friendly headlines and scoring sentiment..."):
            sentiment_frame, sentiment_summary = cached_news_sentiment(
                selected_ticker,
                live_info.get("name") or selected_ticker,
                limit,
                news_source,
                use_finbert,
                news_refresh_token,
            )
        st.session_state["news_frame"] = sentiment_frame
        st.session_state["news_summary"] = sentiment_summary
        st.session_state["news_elapsed"] = time.perf_counter() - start_time
        st.session_state["news_updated_at"] = refresh_time_label(market_tz)
        st.session_state["news_live_enabled"] = True
        st.session_state["news_live_ticker"] = selected_ticker
        st.session_state["news_live_limit"] = limit
        st.session_state["news_live_source"] = news_source
        st.session_state["news_live_finbert"] = use_finbert

    sentiment_frame = st.session_state.get("news_frame")
    sentiment_summary = st.session_state.get("news_summary")
    if isinstance(sentiment_frame, pd.DataFrame) and sentiment_summary:
        c1, c2, c3 = st.columns(3)
        sentiment_tone = tone_for_number(float(sentiment_summary["score"]) - 0.5)
        with c1:
            metric_card("Overall Sentiment", str(sentiment_summary["label"]), "aggregate label", sentiment_tone)
        with c2:
            metric_card("Score", f"{float(sentiment_summary['score']):.2f}", "0 negative | 1 positive", sentiment_tone)
        with c3:
            metric_card("Headlines", f"{int(sentiment_summary['headline_count']):,}", "articles scored")
        if "news_elapsed" in st.session_state:
            st.caption(
                f"Last news update: {st.session_state.get('news_updated_at', refresh_time_label(market_tz))} | "
                f"Fetch time: {st.session_state['news_elapsed']:.2f}s | Auto cadence: about 5 minutes"
            )
        st.dataframe(
            sentiment_frame[["headline", "source", "date", "sentiment", "sentiment_score", "confidence", "model"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "sentiment_score": st.column_config.NumberColumn(format="%.2f"),
                "confidence": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        empty_state("Refresh news sentiment to score current headlines.")

with page[5]:  # Portfolio
    section_title("Portfolio Optimizer", "Allocation by return, volatility, and Sharpe ratio")
    default_raw = ", ".join(active_preset["default_tickers"][:4])
    raw_tickers = st.text_area("Tickers", value=default_raw, height=90)
    investment_amount = st.number_input("Investment amount", min_value=1000.0, value=100000.0, step=5000.0)
    risk_preference = st.segmented_control("Risk preference", options=["Conservative", "Balanced", "Aggressive"], default="Balanced")

    if st.button("Optimize portfolio", type="primary"):
        tickers = parse_ticker_list(raw_tickers, suffix)
        with st.spinner("Fetching price history and optimizing allocation..."):
            histories = fetch_many_histories(tickers, period=period)
            price_frame = build_price_frame(histories)
            portfolio = optimize_weights(price_frame, risk_preference=risk_preference)
            current_prices = {ticker: float(histories[ticker]["Close"].dropna().iloc[-1]) for ticker in portfolio.weights.index if ticker in histories}
            allocation = allocate_investment(portfolio.weights, current_prices, investment_amount)

        st.session_state["portfolio_result"] = portfolio
        st.session_state["portfolio_allocation"] = allocation

    portfolio = st.session_state.get("portfolio_result")
    allocation = st.session_state.get("portfolio_allocation")
    if portfolio is not None and isinstance(allocation, pd.DataFrame):
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Expected Return", pct(portfolio.expected_return), "annualized", tone_for_number(portfolio.expected_return))
        with c2:
            metric_card("Volatility", pct(portfolio.volatility), "annualized risk", "warning")
        with c3:
            metric_card("Sharpe Ratio", f"{portfolio.sharpe_ratio:.2f}", portfolio.method, tone_for_number(portfolio.sharpe_ratio))
        st.caption(portfolio.method)

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=allocation["Ticker"],
                    values=allocation["Weight"],
                    hole=0.5,
                    marker=dict(colors=["#0f9f8f", "#2563eb", "#b7791f", "#169b62", "#c2413d", "#7c3aed", "#0891b2"]),
                )
            ]
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#17202a"),
        )
        st.plotly_chart(fig, use_container_width=True)
        display_allocation = allocation.copy()
        display_allocation["Weight"] = display_allocation["Weight"].map(lambda x: f"{x * 100:.2f}%")
        display_allocation["Allocation"] = display_allocation["Allocation"].map(lambda x: f"{x:,.2f}")
        display_allocation["Current Price"] = display_allocation["Current Price"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
        display_allocation["Approx Shares"] = display_allocation["Approx Shares"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        st.dataframe(display_allocation, use_container_width=True, hide_index=True)
    else:
        empty_state("Optimize a portfolio to generate allocation weights.")

with page[6]:  # Risk
    section_title("Risk Assessment", "Return, volatility, drawdown, beta, and risk level")
    try:
        benchmark_history = cached_history(benchmark, period, refresh_token)
    except Exception:
        benchmark_history = None
    metrics = risk_metrics(history, benchmark_history=benchmark_history)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Annual Return", pct(metrics["annual_return"]), "estimated", tone_for_number(metrics["annual_return"]))
    with c2:
        metric_card("Volatility", pct(metrics["annual_volatility"]), "annualized", "warning")
    with c3:
        metric_card("Sharpe", f"{float(metrics['sharpe_ratio']):.2f}", "risk-adjusted", tone_for_number(metrics["sharpe_ratio"]))
    with c4:
        metric_card("Max Drawdown", pct(metrics["max_drawdown"]), "worst peak-to-trough", "negative")
    with c5:
        risk_tone = {"Low": "positive", "Medium": "warning", "High": "negative"}.get(str(metrics["risk_level"]), "neutral")
        metric_card("Risk Level", str(metrics["risk_level"]), "classification", risk_tone)

    risk_score = {"Low": 25, "Medium": 58, "High": 88}.get(str(metrics["risk_level"]), 50)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#17202a"},
                "steps": [
                    {"range": [0, 35], "color": "#dff7eb"},
                    {"range": [35, 70], "color": "#fff3d8"},
                    {"range": [70, 100], "color": "#ffe1df"},
                ],
            },
            title={"text": f"Risk gauge vs {benchmark}"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="#ffffff", font=dict(color="#17202a"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Beta vs {benchmark}: {metrics['beta']:.2f}" if metrics["beta"] is not None else f"Beta vs {benchmark}: N/A")

with page[7]:  # Chatbot
    section_title("AI Financial Chatbot", "Deep reasoning engine with multi-factor analysis")

    # ── Extra chatbot styles ──
        # Chatbot styles are loaded globally from style.css

    # ── Build prediction / sentiment context ──
    if "latest_prediction" not in st.session_state:
        st.info("💡 Run a prediction first for the richest analysis. A fast fallback is active.")
        quick_prediction = predict_next_price(history, use_lstm=False)
        quick_sentiment: dict[str, Any] = {"label": "Neutral", "score": 0.5, "confidence": 0.0, "headline_count": 0}
        quick_recommendation = build_recommendation(quick_prediction, quick_sentiment)
    else:
        quick_prediction = st.session_state["latest_prediction"]
        quick_sentiment = st.session_state.get("latest_sentiment", {"label": "Neutral", "score": 0.5, "confidence": 0.0, "headline_count": 0})
        quick_recommendation = st.session_state["latest_recommendation"]

    metrics = risk_metrics(history)
    latest_row = features.iloc[-1]

    # ── Context status badges ──
    has_prediction = "latest_prediction" in st.session_state
    has_sentiment = "news_summary" in st.session_state
    has_risk = True  # always computed
    badge = lambda label, active: f"<span class='ctx-badge{' active' if active else ''}'>" \
                                   f"{'✅' if active else '○'} {label}</span>"
    st.markdown(
        f"""
        <div class="context-status">
            {badge("Prediction", has_prediction)}
            {badge("Sentiment", has_sentiment)}
            {badge("Risk Metrics", has_risk)}
            {badge("Technical Indicators", True)}
            {badge("Company Info", True)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Assemble rich context dict ──
    chat_context: dict[str, Any] = {
        # Core identifiers
        "ticker": selected_ticker,
        "company_name": live_info.get("name") or selected_ticker,
        "currency": currency,
        "exchange": live_info.get("exchange", ""),
        "sector": live_info.get("sector", ""),
        "industry": live_info.get("industry", ""),
        "business_summary": live_info.get("summary", ""),
        # Price data
        "current_price": current_price,
        "day_change": day_change,
        # Prediction
        "predicted_price": quick_prediction.predicted_price,
        "expected_return_pct": quick_prediction.expected_return_pct,
        "prediction_confidence": quick_prediction.confidence,
        "prediction_method": quick_prediction.method,
        "trend": quick_prediction.trend,
        # Recommendation
        "recommendation": quick_recommendation.action,
        "hybrid_score": quick_recommendation.score,
        "rationale": quick_recommendation.rationale,
        # Sentiment
        "sentiment": quick_sentiment.get("label", "Neutral"),
        "sentiment_score": float(quick_sentiment.get("score", 0.5)),
        "sentiment_confidence": float(quick_sentiment.get("confidence", 0.0)),
        "headline_count": int(quick_sentiment.get("headline_count", 0)),
        # Risk
        "risk_level": metrics["risk_level"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "annual_return": metrics["annual_return"],
        "annual_volatility": metrics["annual_volatility"],
        "max_drawdown": metrics["max_drawdown"],
        "beta": metrics.get("beta"),
        # Technical indicators
        "rsi": float(latest_row["RSI_14"]),
        "macd": float(latest_row["MACD"]),
        "macd_signal": float(latest_row.get("MACD_signal", 0)),
        "ma_20": float(latest_row["MA_20"]),
        "ma_50": float(latest_row.get("MA_50", 0)),
        "volatility": float(latest_row["Volatility_20"]),
        # Company fundamentals
        "pe_ratio": live_info.get("pe_ratio"),
        "eps": live_info.get("eps"),
        "dividend_yield": live_info.get("dividend_yield"),
        "market_cap": live_info.get("market_cap"),
        "fifty_two_week_high": live_info.get("fifty_two_week_high"),
        "fifty_two_week_low": live_info.get("fifty_two_week_low"),
        "volume": live_info.get("volume"),
        "avg_volume": live_info.get("avg_volume"),
    }

    # ── Suggested questions ──
    suggestions = get_suggested_questions(selected_ticker)
    st.markdown(
        '<div class="suggested-chips">'
        + "".join(f"<span class='suggested-chip'>{safe_text(s)}</span>" for s in suggestions[:6])
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Conversation history ──
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Input area
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "Ask anything about the stock",
            placeholder=f"e.g. Should I buy {selected_ticker}? / What's the risk? / Full analysis...",
            key="chatbot_input",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_clicked = st.button("Ask 🔍", type="primary", use_container_width=True)

    if ask_clicked and question.strip():
        answer = answer_question(question, chat_context)
        import datetime as _dt
        timestamp = _dt.datetime.now().strftime("%I:%M %p")
        st.session_state["chat_history"].append({
            "question": question.strip(),
            "answer": answer,
            "time": timestamp,
            "ticker": selected_ticker,
        })

    # Clear button
    if st.session_state["chat_history"]:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    # ── Render conversation ──
    if st.session_state["chat_history"]:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for entry in reversed(st.session_state["chat_history"]):
            st.markdown(
                f"""
                <div class="chat-bubble chat-user">
                    {safe_text(entry['question'])}
                    <div class="chat-timestamp">{safe_text(entry.get('time',''))} · {safe_text(entry.get('ticker',''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="chat-bubble chat-assistant">
{safe_text(entry['answer'])}
                    <div class="chat-timestamp">{safe_text(APP_NAME)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        empty_state(
            "Ask anything about the stock — buy/sell analysis, risk, technicals, valuation, sentiment, "
            "portfolio guidance, or request a complete analysis. The more prediction and sentiment data "
            "you've loaded, the deeper the analysis."
        )


# ───────────────────────────────────────────────────────────────────────
# Company Spotlight tab – slide through every company with live details
# ───────────────────────────────────────────────────────────────────────
with page[1]:  # Company Spotlight
    section_title("Company Spotlight", "Slide through every company with live-updated details")

    # Resolve the universe for the current market
    spotlight_universe_options = _BASE_UNIVERSE_OPTIONS.get(market_name, ["Custom tickers"])
    spotlight_default_universe = active_preset.get("default_universe", spotlight_universe_options[0])
    # Pick the first non-custom option
    spotlight_universe_name = spotlight_default_universe
    spotlight_custom_raw = ""
    if spotlight_universe_name == "Custom tickers":
        spotlight_custom_raw = ", ".join(active_preset["default_tickers"])

    with st.spinner("Loading company list..."):
        spotlight_universe = cached_company_universe(spotlight_universe_name, spotlight_custom_raw, suffix)

    if spotlight_universe.empty:
        empty_state("No companies found for this market.")
    else:
        company_list = spotlight_universe[["Symbol", "Company", "Ticker"]].reset_index(drop=True)
        total_companies = len(company_list)

        # ── Initialise session state for navigation ──
        # Reset index when market changes
        if st.session_state.get("spotlight_market") != market_name:
            st.session_state["spotlight_idx"] = 0
            st.session_state["spotlight_market"] = market_name

        if "spotlight_idx" not in st.session_state:
            st.session_state["spotlight_idx"] = 0

        # Clamp to valid range
        st.session_state["spotlight_idx"] = max(0, min(st.session_state["spotlight_idx"], total_companies - 1))

        # ── Prev / Next Buttons ──
        btn_prev, btn_counter, btn_next = st.columns([1, 2, 1])
        with btn_prev:
            if st.button("◀  Previous", key="spot_prev", disabled=st.session_state["spotlight_idx"] <= 0, use_container_width=True):
                st.session_state["spotlight_idx"] = max(0, st.session_state["spotlight_idx"] - 1)
                st.rerun()
        with btn_counter:
            st.markdown(
                f"""
                <div style="text-align:center;padding:0.55rem 0;">
                    <span style="font-size:1.05rem;font-weight:800;color:var(--ink);">
                        {st.session_state["spotlight_idx"] + 1}
                    </span>
                    <span style="color:var(--muted);font-size:0.92rem;"> / {total_companies} companies</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with btn_next:
            if st.button("Next  ▶", key="spot_next", disabled=st.session_state["spotlight_idx"] >= total_companies - 1, use_container_width=True):
                st.session_state["spotlight_idx"] = min(total_companies - 1, st.session_state["spotlight_idx"] + 1)
                st.rerun()

        # Slider to navigate companies
        st.markdown("<div class='spotlight-nav'>", unsafe_allow_html=True)
        company_index = st.slider(
            "Navigate companies",
            min_value=0,
            max_value=total_companies - 1,
            value=st.session_state["spotlight_idx"],
            format=f"Company %d of {total_companies}",
            key="spotlight_slider",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Sync slider back to session state when user drags it
        if company_index != st.session_state["spotlight_idx"]:
            st.session_state["spotlight_idx"] = company_index

        # Quick jump with selectbox
        company_options = [f"{row['Symbol']} — {row['Company']}" for _, row in company_list.iterrows()]
        selected_company_label = st.selectbox(
            "Or jump to company",
            company_options,
            index=st.session_state["spotlight_idx"],
            key="spotlight_select",
        )
        # Sync selectbox → session state
        selectbox_index = company_options.index(selected_company_label) if selected_company_label in company_options else st.session_state["spotlight_idx"]
        if selectbox_index != st.session_state["spotlight_idx"]:
            st.session_state["spotlight_idx"] = selectbox_index

        jump_index = st.session_state["spotlight_idx"]

        active_row = company_list.iloc[jump_index]
        spot_ticker = active_row["Ticker"]
        spot_symbol = active_row["Symbol"]
        spot_company = active_row["Company"]

        # Fetch live data for this company
        try:
            with st.spinner(f"Loading live data for {spot_symbol}..."):
                spot_info = cached_live_info(spot_ticker, refresh_token)
                spot_history = cached_history(spot_ticker, period, refresh_token)
                spot_features = add_technical_indicators(spot_history)

            spot_currency = spot_info.get("currency") or ""
            spot_price = spot_info.get("current_price")
            spot_prev_close = spot_info.get("previous_close")
            spot_day_change = None
            if spot_price and spot_prev_close:
                spot_day_change = (float(spot_price) / float(spot_prev_close)) - 1

            change_tone = tone_for_number(spot_day_change)
            change_text = signed_pct(spot_day_change)

            # ── Spotlight Card ──
            st.markdown(
                f"""
                <div class="spotlight-card">
                    <div class="spotlight-header">
                        <div>
                            <h2>{safe_text(spot_info.get('name') or spot_company)}</h2>
                            <div class="spotlight-meta">
                                <span>{safe_text(spot_info.get('sector'))}</span>
                                <span>{safe_text(spot_info.get('industry'))}</span>
                                <span>{safe_text(spot_info.get('exchange'))}</span>
                            </div>
                        </div>
                        <div class="ticker-badge">{safe_text(spot_ticker)}</div>
                    </div>
                    <div style="display:flex;align-items:baseline;gap:0.5rem;margin-bottom:1rem;">
                        <span class="spotlight-price-big">{safe_text(money(spot_price, spot_currency))}</span>
                        <span class="spotlight-change tone-{change_tone}">{safe_text(change_text)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Key Metrics Grid ──
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                metric_card("Open", money(spot_info.get("open_price"), spot_currency), "today")
            with m2:
                metric_card("Day High", money(spot_info.get("day_high"), spot_currency), "intraday")
            with m3:
                metric_card("Day Low", money(spot_info.get("day_low"), spot_currency), "intraday")
            with m4:
                metric_card("Prev Close", money(spot_prev_close, spot_currency), "yesterday")
            with m5:
                metric_card("Volume", f"{int(spot_info.get('volume') or 0):,}", "today")
            with m6:
                metric_card("Market Cap", money(spot_info.get("market_cap"), spot_currency), "total")

            st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

            # ── Extended Metrics ──
            e1, e2, e3, e4, e5, e6 = st.columns(6)
            with e1:
                metric_card("52W High", money(spot_info.get("fifty_two_week_high"), spot_currency), "annual")
            with e2:
                metric_card("52W Low", money(spot_info.get("fifty_two_week_low"), spot_currency), "annual")
            with e3:
                pe = spot_info.get("pe_ratio")
                metric_card("P/E Ratio", f"{float(pe):.2f}" if pe else "N/A", "valuation")
            with e4:
                eps = spot_info.get("eps")
                metric_card("EPS", money(eps, spot_currency) if eps else "N/A", "earnings")
            with e5:
                dy = spot_info.get("dividend_yield")
                metric_card("Div Yield", f"{float(dy) * 100:.2f}%" if dy else "N/A", "annual")
            with e6:
                beta_val = spot_info.get("beta")
                metric_card("Beta", f"{float(beta_val):.2f}" if beta_val else "N/A", "risk")

            st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

            # ── Technical Indicators ──
            t1, t2, t3, t4 = st.columns(4)
            latest_features = spot_features.iloc[-1]
            with t1:
                rsi_val = float(latest_features["RSI_14"])
                rsi_tone = "positive" if rsi_val < 70 else ("negative" if rsi_val > 80 else "warning")
                if rsi_val < 30:
                    rsi_tone = "negative"
                metric_card("RSI (14)", f"{rsi_val:.1f}", "overbought >70 / oversold <30", rsi_tone)
            with t2:
                metric_card("MACD", f"{float(latest_features['MACD']):.2f}", "momentum", tone_for_number(float(latest_features["MACD"])))
            with t3:
                metric_card("MA 20", money(float(latest_features["MA_20"]), spot_currency), "short-term trend")
            with t4:
                metric_card("Volatility", pct(float(latest_features["Volatility_20"])), "20-day risk", "warning")

            st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

            # ── Mini Price Chart ──
            st.plotly_chart(price_chart(spot_features, spot_ticker), use_container_width=True)

            # ── Business Summary ──
            summary_text = spot_info.get("summary", "")
            if summary_text:
                st.markdown(
                    f"<div class='spotlight-summary'>{safe_text(summary_text)}</div>",
                    unsafe_allow_html=True,
                )

            # ── Recent Data Table ──
            section_title("Recent Trading Data")
            st.dataframe(
                spot_features.tail(10)[["Open", "High", "Low", "Close", "Volume", "MA_20", "RSI_14", "MACD"]],
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Open": st.column_config.NumberColumn(format="%.2f"),
                    "High": st.column_config.NumberColumn(format="%.2f"),
                    "Low": st.column_config.NumberColumn(format="%.2f"),
                    "Close": st.column_config.NumberColumn(format="%.2f"),
                    "Volume": st.column_config.NumberColumn(format="%d"),
                    "MA_20": st.column_config.NumberColumn(format="%.2f"),
                    "RSI_14": st.column_config.NumberColumn(format="%.2f"),
                    "MACD": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            st.caption(
                f"Company {jump_index + 1} of {total_companies} | "
                f"Last refreshed: {refresh_time_label(market_tz)} | "
                f"Auto-refresh {'ON' if auto_refresh else 'OFF'}"
            )

        except Exception as spot_exc:
            st.error(f"Could not load data for {spot_ticker}: {spot_exc}")
            st.info("Try sliding to another company, or check if the ticker is valid.")

# ───────────────────────────────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="app-footer">
        <div class="footer-brand">📈 {safe_text(APP_NAME)}</div>
        <div style="margin-top:0.35rem;">
            Real-time market intelligence · Technical analysis · AI-powered predictions · Portfolio optimisation
        </div>
        <div style="margin-top:0.5rem; font-size: 0.72rem;">
            {safe_text(DISCLAIMER)}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
