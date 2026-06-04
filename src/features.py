"""Feature engineering for price prediction and risk analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).clip(0, 100)


def calculate_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, signal, and histogram."""
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, RSI, MACD, returns, and volatility."""
    if data.empty or "Close" not in data.columns:
        raise ValueError("A non-empty dataframe with a Close column is required.")

    frame = data.copy()
    frame["Volume"] = frame.get("Volume", pd.Series(0, index=frame.index)).fillna(0)
    frame["MA_20"] = frame["Close"].rolling(window=20, min_periods=1).mean()
    frame["MA_50"] = frame["Close"].rolling(window=50, min_periods=1).mean()
    frame["RSI_14"] = calculate_rsi(frame["Close"], window=14)
    frame["MACD"], frame["MACD_signal"], frame["MACD_hist"] = calculate_macd(frame["Close"])
    frame["Returns"] = frame["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["Volatility_20"] = frame["Returns"].rolling(window=20, min_periods=2).std().fillna(0)
    frame["Volume_MA_20"] = frame["Volume"].rolling(window=20, min_periods=1).mean()
    return frame.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def infer_trend(feature_frame: pd.DataFrame) -> str:
    """Infer a simple up, down, or sideways trend from indicators."""
    if feature_frame.empty:
        return "sideways"

    latest = feature_frame.iloc[-1]
    recent = feature_frame.tail(10)
    slope = recent["Close"].pct_change().mean()

    if latest["Close"] >= latest["MA_20"] >= latest["MA_50"] and slope > 0:
        return "up"
    if latest["Close"] <= latest["MA_20"] <= latest["MA_50"] and slope < 0:
        return "down"
    return "sideways"

