from src.prediction import PredictionResult
from src.recommender import build_recommendation


def test_recommender_buy_when_positive_sentiment_and_uptrend():
    prediction = PredictionResult(
        current_price=100,
        predicted_price=104,
        expected_return_pct=4,
        trend="up",
        confidence=0.8,
        method="test",
    )

    result = build_recommendation(prediction, {"label": "Positive", "score": 0.8})

    assert result.action == "BUY"
    assert result.score > 0


def test_recommender_sell_on_negative_forecast():
    prediction = PredictionResult(
        current_price=100,
        predicted_price=95,
        expected_return_pct=-5,
        trend="down",
        confidence=0.8,
        method="test",
    )

    result = build_recommendation(prediction, {"label": "Neutral", "score": 0.5})

    assert result.action == "SELL"

