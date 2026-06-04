"""Portfolio and stock risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE


def max_drawdown(close: pd.Series) -> float:
    """Return maximum drawdown as a negative decimal."""
    running_max = close.cummax()
    drawdown = (close / running_max) - 1
    return float(drawdown.min())


def calculate_beta(stock_history: pd.DataFrame, benchmark_history: pd.DataFrame | None = None) -> float | None:
    """Calculate beta against a benchmark when available."""
    if benchmark_history is None or benchmark_history.empty:
        return None

    stock_returns = stock_history["Close"].pct_change().rename("stock")
    benchmark_returns = benchmark_history["Close"].pct_change().rename("benchmark")
    joined = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 20:
        return None

    variance = joined["benchmark"].var()
    if variance == 0 or np.isnan(variance):
        return None
    covariance = joined["stock"].cov(joined["benchmark"])
    return float(covariance / variance)


def classify_risk(volatility: float, drawdown: float) -> str:
    """Classify risk from annualized volatility and drawdown."""
    if volatility < 0.18 and drawdown > -0.15:
        return "Low"
    if volatility < 0.32 and drawdown > -0.30:
        return "Medium"
    return "High"


def risk_metrics(
    history: pd.DataFrame,
    benchmark_history: pd.DataFrame | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict[str, float | str | None]:
    """Calculate annual return, volatility, Sharpe ratio, drawdown, and beta."""
    if history.empty:
        raise ValueError("History is required to calculate risk metrics.")

    close = history["Close"].astype(float).dropna()
    returns = close.pct_change().dropna()
    annual_return = float(returns.mean() * 252) if not returns.empty else 0.0
    annual_volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else 0.0
    drawdown = max_drawdown(close)
    beta = calculate_beta(history, benchmark_history)

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": drawdown,
        "beta": beta,
        "risk_level": classify_risk(annual_volatility, drawdown),
    }

