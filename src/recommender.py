"""Hybrid buy, sell, or hold recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass

from .prediction import PredictionResult


@dataclass
class RecommendationResult:
    action: str
    score: float
    price_signal: float
    sentiment_signal: float
    rationale: list[str]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def build_recommendation(prediction: PredictionResult, sentiment_summary: dict[str, float | str]) -> RecommendationResult:
    """Combine price forecast and sentiment into a recommendation."""
    sentiment_score = float(sentiment_summary.get("score", 0.5))
    price_signal = clamp(prediction.expected_return_pct / 5.0, -1.0, 1.0)
    sentiment_signal = clamp((sentiment_score - 0.5) * 2, -1.0, 1.0)
    hybrid_score = (0.7 * price_signal) + (0.3 * sentiment_signal)

    rationale: list[str] = [
        f"Forecast return is {prediction.expected_return_pct:.2f} percent using {prediction.method}.",
        f"Trend is {prediction.trend}.",
        f"News sentiment is {sentiment_summary.get('label', 'Neutral')} with score {sentiment_score:.2f}.",
    ]

    if sentiment_score >= 0.7 and prediction.trend == "up" and prediction.expected_return_pct > 0:
        action = "BUY"
        rationale.append("Positive news and upward trend align.")
    elif sentiment_score <= 0.3 or prediction.expected_return_pct <= -2:
        action = "SELL"
        rationale.append("Weak sentiment or material negative forecast triggered caution.")
    elif hybrid_score >= 0.25:
        action = "BUY"
        rationale.append("Hybrid score is meaningfully positive.")
    elif hybrid_score <= -0.25:
        action = "SELL"
        rationale.append("Hybrid score is meaningfully negative.")
    else:
        action = "HOLD"
        rationale.append("Signals are mixed or not strong enough.")

    return RecommendationResult(
        action=action,
        score=float(hybrid_score),
        price_signal=float(price_signal),
        sentiment_signal=float(sentiment_signal),
        rationale=rationale,
    )

