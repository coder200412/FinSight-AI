"""Modern Portfolio Theory style allocation tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str


def build_price_frame(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a close-price matrix from per-ticker history frames."""
    closes = []
    for ticker, history in histories.items():
        if history is not None and not history.empty and "Close" in history.columns:
            closes.append(history["Close"].rename(ticker))
    if not closes:
        return pd.DataFrame()
    return pd.concat(closes, axis=1).dropna(how="all").ffill().dropna()


def _portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, covariance: pd.DataFrame) -> tuple[float, float, float]:
    expected_return = float(np.dot(weights, mean_returns))
    volatility = float(np.sqrt(np.dot(weights.T, np.dot(covariance, weights))))
    sharpe = expected_return / volatility if volatility else 0.0
    return expected_return, volatility, sharpe


def optimize_weights(price_frame: pd.DataFrame, risk_preference: str = "Balanced") -> PortfolioResult:
    """Optimize long-only weights using Sharpe ratio or risk preference objective."""
    if price_frame.empty:
        raise ValueError("At least one valid price series is required.")

    returns = price_frame.pct_change().dropna()
    if returns.empty:
        tickers = price_frame.columns
        weights = pd.Series(1 / len(tickers), index=tickers)
        return PortfolioResult(weights, 0.0, 0.0, 0.0, "Equal weight fallback")

    mean_returns = returns.mean() * 252
    covariance = returns.cov() * 252
    tickers = list(price_frame.columns)
    n_assets = len(tickers)
    initial = np.repeat(1 / n_assets, n_assets)

    try:
        from scipy.optimize import minimize

        def objective(weights: np.ndarray) -> float:
            expected_return, volatility, sharpe = _portfolio_stats(weights, mean_returns, covariance)
            if risk_preference == "Conservative":
                return volatility
            if risk_preference == "Aggressive":
                return -(0.7 * sharpe + 0.3 * expected_return)
            return -sharpe

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0, 1)] * n_assets,
            constraints=({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},),
            options={"maxiter": 500},
        )
        weights_array = result.x if result.success else initial
        method = f"MPT optimizer ({risk_preference})" if result.success else "Equal weight fallback"
    except Exception:
        weights_array = initial
        method = "Equal weight fallback"

    expected_return, volatility, sharpe = _portfolio_stats(weights_array, mean_returns, covariance)
    weights = pd.Series(weights_array, index=tickers)
    weights = weights[weights > 0.001]
    weights = weights / weights.sum()
    return PortfolioResult(weights, expected_return, volatility, sharpe, method)


def allocate_investment(weights: pd.Series, current_prices: dict[str, float], investment_amount: float) -> pd.DataFrame:
    """Convert optimized weights into currency allocation and approximate shares."""
    rows = []
    for ticker, weight in weights.items():
        allocation = investment_amount * float(weight)
        price = current_prices.get(ticker)
        shares = allocation / price if price and price > 0 else np.nan
        rows.append(
            {
                "Ticker": ticker,
                "Weight": float(weight),
                "Allocation": allocation,
                "Current Price": price,
                "Approx Shares": shares,
            }
        )
    return pd.DataFrame(rows)

