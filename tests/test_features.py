import pandas as pd

from src.features import add_technical_indicators, infer_trend


def test_add_technical_indicators_creates_expected_columns():
    data = pd.DataFrame(
        {
            "Open": range(1, 81),
            "High": range(2, 82),
            "Low": range(0, 80),
            "Close": range(1, 81),
            "Volume": [1000] * 80,
        }
    )

    result = add_technical_indicators(data)

    assert {"MA_20", "MA_50", "RSI_14", "MACD", "Returns", "Volatility_20"}.issubset(result.columns)
    assert result["RSI_14"].between(0, 100).all()
    assert infer_trend(result) == "up"

