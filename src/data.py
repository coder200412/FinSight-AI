"""Live market and news data access."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import re
from typing import Any
from urllib.parse import quote_plus

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - exercised when dependency is absent
    yf = None

try:
    import feedparser
except Exception:  # pragma: no cover - exercised when dependency is absent
    feedparser = None


REQUIRED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def normalize_ticker(symbol: str, exchange_suffix: str = "") -> str:
    """Normalize a user-entered symbol and optionally add an exchange suffix."""
    ticker = (symbol or "").strip().upper()
    if not ticker:
        return ticker
    if exchange_suffix and "." not in ticker and not ticker.startswith("^"):
        return f"{ticker}{exchange_suffix}"
    return ticker


def parse_ticker_list(raw: str, exchange_suffix: str = "") -> list[str]:
    """Parse comma or newline separated tickers."""
    parts = re.split(r"[\n,]+", raw or "")
    tickers = [normalize_ticker(part, exchange_suffix) for part in parts]
    return [ticker for ticker in tickers if ticker]


def _require_yfinance() -> None:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")


def fetch_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    _require_yfinance()
    data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if data is None or data.empty:
        raise ValueError(f"No market data returned for {ticker}. Check the symbol and exchange suffix.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Yahoo Finance response for {ticker} is missing: {', '.join(missing)}")

    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data = data[REQUIRED_PRICE_COLUMNS + [c for c in data.columns if c not in REQUIRED_PRICE_COLUMNS]]
    data = data.dropna(subset=["Close"])
    data["Volume"] = data["Volume"].fillna(0)
    return data


def fetch_many_histories(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch histories for many symbols, keeping successful results."""
    histories: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            histories[ticker] = fetch_history(ticker, period=period)
        except Exception:
            continue
    return histories


def _extract_downloaded_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        first_level = raw.columns.get_level_values(0)
        second_level = raw.columns.get_level_values(1)
        if ticker in first_level:
            frame = raw.xs(ticker, axis=1, level=0, drop_level=True)
        elif ticker in second_level:
            frame = raw.xs(ticker, axis=1, level=1, drop_level=True)
        else:
            return pd.DataFrame()
    else:
        frame = raw

    frame = frame.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    if "Close" not in frame.columns:
        return pd.DataFrame()
    return frame


def fetch_live_price_table(
    tickers: list[str],
    period: str = "5d",
    interval: str = "1d",
    chunk_size: int = 80,
) -> pd.DataFrame:
    """Fetch latest available prices for many companies with Yahoo Finance batch calls."""
    _require_yfinance()
    clean_tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    if not clean_tickers:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for chunk in _chunks(clean_tickers, chunk_size):
        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        for ticker in chunk:
            frame = _extract_downloaded_ticker(raw, ticker)
            if frame.empty:
                continue

            close = frame["Close"].dropna()
            if close.empty:
                continue

            latest_price = float(close.iloc[-1])
            reference_price = float(close.iloc[-2]) if len(close) > 1 else latest_price
            change_pct = ((latest_price / reference_price) - 1) if reference_price else 0.0
            volume = None
            if "Volume" in frame.columns:
                volume_values = frame["Volume"].dropna()
                if not volume_values.empty:
                    volume = int(volume_values.iloc[-1])

            rows.append(
                {
                    "Ticker": ticker,
                    "Latest Price": latest_price,
                    "Reference Price": reference_price,
                    "Change %": change_pct,
                    "Volume": volume,
                    "Last Update": str(close.index[-1]),
                }
            )

    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)


def _fast_get(info: Any, key: str, default: Any = None) -> Any:
    if info is None:
        return default
    try:
        if hasattr(info, "get"):
            return info.get(key, default)
    except Exception:
        pass
    try:
        return info[key]
    except Exception:
        return getattr(info, key, default)


def fetch_live_info(ticker: str) -> dict[str, Any]:
    """Fetch current quote-like values with fast_info and info fallbacks."""
    _require_yfinance()
    stock = yf.Ticker(ticker)
    fast = None
    info: dict[str, Any] = {}

    try:
        fast = stock.fast_info
    except Exception:
        fast = None

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    def first_value(*keys: str) -> Any:
        for key in keys:
            value = _fast_get(fast, key)
            if value is not None:
                return value
            value = info.get(key)
            if value is not None:
                return value
        return None

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": first_value("currency") or info.get("financialCurrency") or "",
        "exchange": info.get("exchange") or info.get("fullExchangeName") or "",
        "current_price": first_value("last_price", "regularMarketPrice", "currentPrice"),
        "previous_close": first_value("previous_close", "regularMarketPreviousClose", "previousClose"),
        "day_high": first_value("day_high", "regularMarketDayHigh", "dayHigh"),
        "day_low": first_value("day_low", "regularMarketDayLow", "dayLow"),
        "market_cap": first_value("market_cap", "marketCap"),
        "timezone": first_value("timezone") or info.get("exchangeTimezoneName") or "",
        # Extended fields for company spotlight
        "open_price": first_value("open", "regularMarketOpen"),
        "volume": first_value("last_volume", "regularMarketVolume", "volume"),
        "avg_volume": info.get("averageDailyVolume10Day") or info.get("averageVolume"),
        "fifty_two_week_high": first_value("year_high", "fiftyTwoWeekHigh"),
        "fifty_two_week_low": first_value("year_low", "fiftyTwoWeekLow"),
        "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "sector": info.get("sector") or "",
        "industry": info.get("industry") or "",
        "summary": info.get("longBusinessSummary") or "",
    }


def _extract_yahoo_news_item(item: dict[str, Any]) -> dict[str, Any] | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    title = item.get("title") or content.get("title")
    if not title:
        return None

    publish_time = item.get("providerPublishTime") or content.get("pubDate")
    published = None
    if isinstance(publish_time, (int, float)):
        published = datetime.fromtimestamp(publish_time, tz=timezone.utc).isoformat()
    elif publish_time:
        published = str(publish_time)

    provider = item.get("publisher") or content.get("provider", {}).get("displayName") or "Yahoo Finance"
    link = item.get("link") or content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url")

    return {
        "headline": html.unescape(str(title)).strip(),
        "date": published or "",
        "source": provider,
        "url": link or "",
    }


def fetch_yahoo_news(ticker: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch Yahoo Finance news when yfinance exposes it."""
    _require_yfinance()
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    items: list[dict[str, Any]] = []
    for raw_item in raw_items[:limit]:
        parsed = _extract_yahoo_news_item(raw_item)
        if parsed:
            items.append(parsed)
    return items


def fetch_google_news(query: str, limit: int = 10, region: str = "IN", timeout: int = 6) -> list[dict[str, Any]]:
    """Fetch headlines from Google News RSS."""
    if feedparser is None:
        return []

    encoded_query = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-{region}&gl={region}&ceid={region}:en"
    try:
        import requests

        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 AI-Financial-Assistant"},
            timeout=timeout,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception:
        return []

    items: list[dict[str, Any]] = []
    for entry in feed.entries[:limit]:
        items.append(
            {
                "headline": html.unescape(getattr(entry, "title", "")).strip(),
                "date": getattr(entry, "published", ""),
                "source": getattr(entry, "source", {}).get("title", "Google News")
                if isinstance(getattr(entry, "source", {}), dict)
                else "Google News",
                "url": getattr(entry, "link", ""),
            }
        )
    return [item for item in items if item["headline"]]


def fetch_news(
    ticker: str,
    company_query: str | None = None,
    limit: int = 15,
    source_mode: str = "Google News RSS (fast)",
) -> list[dict[str, Any]]:
    """Collect and deduplicate financial news from Yahoo and Google RSS."""
    query = company_query or ticker
    if source_mode == "Yahoo Finance only":
        candidates = fetch_yahoo_news(ticker, limit=limit)
    elif source_mode == "Yahoo Finance + Google (slower)":
        candidates = fetch_yahoo_news(ticker, limit=limit) + fetch_google_news(query, limit=limit)
    else:
        candidates = fetch_google_news(query, limit=limit)

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in candidates:
        key = re.sub(r"\W+", "", item.get("headline", "").lower())[:120]
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped
