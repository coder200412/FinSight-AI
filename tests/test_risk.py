import pandas as pd

from src.risk import max_drawdown, risk_metrics


def test_max_drawdown_is_negative_when_price_falls():
    close = pd.Series([100, 120, 90, 110])

    assert max_drawdown(close) == -0.25


def test_risk_metrics_returns_core_values():
    history = pd.DataFrame({"Close": [100, 101, 103, 102, 106, 108, 107, 110]})

    metrics = risk_metrics(history)

    assert "annual_return" in metrics
    assert "annual_volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert metrics["risk_level"] in {"Low", "Medium", "High"}

