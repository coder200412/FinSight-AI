"""Financial headline sentiment analysis with FinBERT fallback support."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

import pandas as pd

POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "buyback",
    "growth",
    "gain",
    "gains",
    "higher",
    "profit",
    "profits",
    "record",
    "rises",
    "strong",
    "surge",
    "upgrade",
    "wins",
}

NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "decline",
    "default",
    "downgrade",
    "drop",
    "falls",
    "fraud",
    "loss",
    "losses",
    "miss",
    "probe",
    "risk",
    "slump",
    "weak",
    "warning",
}


@lru_cache(maxsize=1)
def _get_finbert_pipeline() -> Any:
    from transformers import pipeline

    return pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")


def _map_label_to_score(label: str, confidence: float) -> tuple[str, float, float]:
    normalized = label.lower()
    if "positive" in normalized:
        return "Positive", 1.0, confidence
    if "negative" in normalized:
        return "Negative", 0.0, confidence
    return "Neutral", 0.5, confidence


def _lexicon_sentiment(text: str) -> tuple[str, float, float]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    total = positive_count + negative_count

    if total == 0:
        return "Neutral", 0.5, 0.55

    raw = (positive_count - negative_count) / total
    score = (raw + 1) / 2
    confidence = min(0.95, 0.55 + (abs(raw) * 0.35))
    if score >= 0.62:
        return "Positive", float(score), float(confidence)
    if score <= 0.38:
        return "Negative", float(score), float(confidence)
    return "Neutral", float(score), float(confidence)


def analyze_headline(headline: str, use_finbert: bool = False) -> tuple[str, float, float, str]:
    """Return sentiment label, normalized score, confidence, and model name."""
    text = (headline or "").strip()
    if not text:
        return "Neutral", 0.5, 0.0, "empty"

    if use_finbert:
        try:
            result = _get_finbert_pipeline()(text[:512])[0]
            label, score, confidence = _map_label_to_score(result.get("label", "neutral"), float(result.get("score", 0.5)))
            return label, score, confidence, "FinBERT"
        except Exception:
            pass

    label, score, confidence = _lexicon_sentiment(text)
    return label, score, confidence, "finance lexicon"


def analyze_news(news_items: list[dict[str, Any]], use_finbert: bool = False) -> pd.DataFrame:
    """Analyze sentiment for a list of news dicts."""
    rows: list[dict[str, Any]] = []
    for item in news_items:
        headline = item.get("headline", "")
        label, score, confidence, model_name = analyze_headline(headline, use_finbert=use_finbert)
        rows.append(
            {
                "headline": headline,
                "date": item.get("date", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "sentiment": label,
                "sentiment_score": score,
                "confidence": confidence,
                "model": model_name,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["headline", "date", "source", "url", "sentiment", "sentiment_score", "confidence", "model"],
    )


def aggregate_sentiment(sentiment_frame: pd.DataFrame) -> dict[str, float | str]:
    """Aggregate headline-level sentiment into a single score."""
    if sentiment_frame.empty:
        return {"label": "Neutral", "score": 0.5, "confidence": 0.0, "headline_count": 0}

    score = float(sentiment_frame["sentiment_score"].mean())
    confidence = float(sentiment_frame["confidence"].mean())
    if score >= 0.62:
        label = "Positive"
    elif score <= 0.38:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "score": score,
        "confidence": confidence,
        "headline_count": int(len(sentiment_frame)),
    }

