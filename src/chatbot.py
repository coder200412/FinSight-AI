"""Advanced financial assistant with deep reasoning and multi-factor analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .config import DISCLAIMER


# ── Intent classification ────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("buy_sell", ["should i buy", "should i sell", "buy or sell", "is it a buy", "is it a sell",
                  "good time to buy", "good time to sell", "worth buying", "worth selling",
                  "enter position", "exit position", "accumulate", "long or short"]),
    ("recommendation", ["recommend", "suggestion", "what do you think", "your opinion",
                        "what should i do", "action", "signal", "call"]),
    ("risk", ["risk", "volatile", "volatility", "drawdown", "dangerous", "safe",
              "how risky", "downside", "beta", "max loss", "worst case"]),
    ("sentiment", ["sentiment", "news", "headlines", "market mood", "bullish or bearish",
                   "what are people saying", "media", "press", "public opinion"]),
    ("prediction", ["predict", "forecast", "price target", "next price", "tomorrow",
                    "where is it going", "future price", "expected price", "will it go up",
                    "will it go down", "price movement"]),
    ("technical", ["rsi", "macd", "moving average", "ma 20", "ma 50", "technical",
                   "indicator", "overbought", "oversold", "support", "resistance",
                   "trend", "momentum", "chart pattern", "candlestick"]),
    ("valuation", ["valuation", "pe ratio", "p/e", "eps", "earnings", "overvalued",
                   "undervalued", "fair value", "market cap", "dividend", "yield",
                   "book value", "price to book"]),
    ("portfolio", ["portfolio", "allocate", "allocation", "diversify", "diversification",
                   "how much to invest", "position size", "weight", "split investment"]),
    ("compare", ["compare", "versus", " vs ", "better than", "which is better",
                 "difference between", "outperform"]),
    ("company", ["what does", "about the company", "what is", "tell me about",
                 "company info", "business", "sector", "industry", "who is"]),
    ("summary", ["summary", "overview", "full analysis", "complete analysis",
                 "everything", "big picture", "overall", "wrap up", "brief me"]),
]


def _classify_intents(text: str) -> list[str]:
    """Return all matching intents, ordered by match quality."""
    text_lower = text.lower().strip()
    matched: list[str] = []
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if pattern in text_lower:
                if intent not in matched:
                    matched.append(intent)
                break
    if not matched:
        matched.append("summary")
    return matched


# ── Formatting helpers ───────────────────────────────────────────────────

def _fmt_money(value: float | int | None, currency: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{float(value):,.2f}"


def _fmt_pct(value: float | int | None, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    fmt = f"{float(value) * 100:+.2f}%" if signed else f"{float(value) * 100:.2f}%"
    return fmt


def _safe(value: Any, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    return str(value)


def _risk_emoji(level: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(level, "⚪")


def _trend_emoji(trend: str) -> str:
    return {"up": "📈", "down": "📉", "sideways": "↔️"}.get(trend, "↔️")


def _sentiment_emoji(label: str) -> str:
    return {"Positive": "😊", "Negative": "😟", "Neutral": "😐"}.get(label, "😐")


def _action_emoji(action: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")


# ── Analysis sections ────────────────────────────────────────────────────

def _analyze_buy_sell(ctx: dict[str, Any]) -> str:
    """Deep buy/sell analysis combining all available signals."""
    ticker = ctx.get("ticker", "the stock")
    action = ctx.get("recommendation", "HOLD")
    score = ctx.get("hybrid_score", 0.0)
    predicted = ctx.get("predicted_price")
    current = ctx.get("current_price")
    sentiment_label = ctx.get("sentiment", "Neutral")
    sentiment_score = ctx.get("sentiment_score", 0.5)
    risk_level = ctx.get("risk_level", "Unknown")
    trend = ctx.get("trend", "sideways")
    rsi = ctx.get("rsi")
    macd = ctx.get("macd")
    volatility = ctx.get("volatility")
    day_change = ctx.get("day_change")
    rationale = ctx.get("rationale", [])

    lines = [f"═══ {_action_emoji(action)} BUY / SELL ANALYSIS for {ticker} ═══\n"]

    # Verdict
    lines.append(f"▸ Model Verdict: {action} (hybrid score: {score:+.2f})")

    # Price signal
    if current and predicted:
        ret = ((predicted - current) / current) * 100
        direction = "upside" if ret > 0 else "downside"
        lines.append(f"▸ Price Signal: {_fmt_money(current, ctx.get('currency',''))} → "
                      f"{_fmt_money(predicted, ctx.get('currency',''))} ({ret:+.2f}% {direction})")
    
    # Trend context
    lines.append(f"▸ Trend: {_trend_emoji(trend)} {trend.capitalize()} trend detected")

    # Sentiment alignment
    alignment = "aligns with" if (
        (action == "BUY" and sentiment_label == "Positive") or
        (action == "SELL" and sentiment_label == "Negative")
    ) else "conflicts with" if (
        (action == "BUY" and sentiment_label == "Negative") or
        (action == "SELL" and sentiment_label == "Positive")
    ) else "is neutral relative to"
    lines.append(f"▸ Sentiment: {_sentiment_emoji(sentiment_label)} {sentiment_label} "
                  f"(score {sentiment_score:.2f}) — {alignment} the price signal")

    # Risk check
    lines.append(f"▸ Risk Level: {_risk_emoji(risk_level)} {risk_level}")

    # Technical filters
    if rsi is not None:
        if rsi > 70:
            lines.append(f"▸ ⚠️ RSI at {rsi:.1f} — overbought territory, caution on new buys")
        elif rsi < 30:
            lines.append(f"▸ 💡 RSI at {rsi:.1f} — oversold territory, potential bounce opportunity")
        else:
            lines.append(f"▸ RSI at {rsi:.1f} — in normal range")

    if macd is not None:
        macd_signal = "bullish" if macd > 0 else "bearish"
        lines.append(f"▸ MACD at {macd:.2f} — {macd_signal} momentum")

    # Confidence factors
    lines.append("\n── Confidence Assessment ──")
    factors_for = []
    factors_against = []

    if trend == "up" and action == "BUY":
        factors_for.append("uptrend supports buy signal")
    elif trend == "down" and action == "SELL":
        factors_for.append("downtrend supports sell signal")
    elif trend != "sideways":
        factors_against.append(f"{trend} trend conflicts with {action} signal")

    if sentiment_label == "Positive" and action == "BUY":
        factors_for.append("positive sentiment reinforces buy")
    elif sentiment_label == "Negative" and action == "SELL":
        factors_for.append("negative sentiment reinforces sell")
    elif sentiment_label != "Neutral":
        factors_against.append(f"{sentiment_label.lower()} sentiment diverges from {action} signal")

    if rsi is not None:
        if rsi > 70 and action == "BUY":
            factors_against.append("overbought RSI suggests limited upside near-term")
        elif rsi < 30 and action == "SELL":
            factors_against.append("oversold RSI suggests a bounce may be near")

    if risk_level == "High":
        factors_against.append("high-risk profile means larger position sizing caution")

    if volatility and volatility > 0.025:
        factors_against.append(f"elevated daily volatility ({_fmt_pct(volatility)}) increases uncertainty")

    if factors_for:
        lines.append("  ✅ Supporting: " + "; ".join(factors_for))
    if factors_against:
        lines.append("  ⚠️ Caution: " + "; ".join(factors_against))

    # Model rationale
    if rationale:
        lines.append("\n── Model Rationale ──")
        for r in rationale:
            lines.append(f"  • {r}")

    # Final guidance
    lines.append("\n── Guidance ──")
    if action == "BUY":
        lines.append("The signals lean positive. Consider scaling in rather than a lump-sum entry. "
                      "Set a stop-loss based on recent support levels and your risk tolerance.")
    elif action == "SELL":
        lines.append("Caution signals are elevated. If holding, consider tightening stops or reducing exposure. "
                      "If not yet invested, wait for clearer reversal signals before entering.")
    else:
        lines.append("Mixed signals suggest patience. Monitor for a clearer trend break, "
                      "an RSI extreme, or a sentiment shift before committing capital.")

    return "\n".join(lines)


def _analyze_risk(ctx: dict[str, Any]) -> str:
    """Detailed risk breakdown."""
    ticker = ctx.get("ticker", "the stock")
    risk_level = ctx.get("risk_level", "Unknown")
    sharpe = ctx.get("sharpe_ratio")
    annual_return = ctx.get("annual_return")
    annual_vol = ctx.get("annual_volatility")
    max_dd = ctx.get("max_drawdown")
    beta = ctx.get("beta")
    volatility = ctx.get("volatility")

    lines = [f"═══ {_risk_emoji(risk_level)} RISK ANALYSIS for {ticker} ═══\n"]
    lines.append(f"▸ Overall Risk Classification: {risk_level}\n")

    lines.append("── Key Risk Metrics ──")
    if annual_return is not None:
        lines.append(f"  • Annualised Return: {_fmt_pct(annual_return, signed=True)}")
    if annual_vol is not None:
        lines.append(f"  • Annualised Volatility: {_fmt_pct(annual_vol)}")
        if annual_vol < 0.18:
            lines.append("    → Low volatility — relatively stable price action")
        elif annual_vol < 0.32:
            lines.append("    → Moderate volatility — expect meaningful price swings")
        else:
            lines.append("    → High volatility — significant price swings, suitable for risk-tolerant investors")
    if sharpe is not None:
        lines.append(f"  • Sharpe Ratio: {sharpe:.2f}")
        if sharpe > 1.0:
            lines.append("    → Excellent risk-adjusted return")
        elif sharpe > 0.5:
            lines.append("    → Acceptable risk-adjusted return")
        elif sharpe > 0:
            lines.append("    → Marginal risk-adjusted return — risk may not be well compensated")
        else:
            lines.append("    → Negative Sharpe — returns have not compensated for the risk taken")
    if max_dd is not None:
        lines.append(f"  • Maximum Drawdown: {_fmt_pct(max_dd, signed=True)}")
        if max_dd > -0.10:
            lines.append("    → Shallow drawdown — limited peak-to-trough loss")
        elif max_dd > -0.25:
            lines.append("    → Moderate drawdown — expect temporary paper losses")
        else:
            lines.append("    → Deep drawdown — investors should be prepared for significant temporary losses")
    if beta is not None:
        lines.append(f"  • Beta: {beta:.2f}")
        if beta > 1.2:
            lines.append("    → High beta — amplifies market moves in both directions")
        elif beta < 0.8:
            lines.append("    → Low beta — more defensive than the broader market")
        else:
            lines.append("    → Near-market beta — moves roughly in line with the index")

    # Practical advice
    lines.append("\n── Risk Management Guidance ──")
    if risk_level == "High":
        lines.append("• Consider limiting this to 5–10% of total portfolio value")
        lines.append("• Use stop-losses to cap downside exposure")
        lines.append("• Pair with low-correlation assets for diversification")
    elif risk_level == "Medium":
        lines.append("• A moderate allocation (10–20%) is reasonable for balanced portfolios")
        lines.append("• Monitor technical support levels for stop-loss placement")
    else:
        lines.append("• Low-risk profile allows a larger allocation (15–30%)")
        lines.append("• Still monitor for any fundamental shifts that could alter the risk profile")

    return "\n".join(lines)


def _analyze_sentiment(ctx: dict[str, Any]) -> str:
    """Sentiment deep-dive."""
    ticker = ctx.get("ticker", "the stock")
    label = ctx.get("sentiment", "Neutral")
    score = ctx.get("sentiment_score", 0.5)
    confidence = ctx.get("sentiment_confidence", 0.0)
    headline_count = ctx.get("headline_count", 0)

    lines = [f"═══ {_sentiment_emoji(label)} SENTIMENT ANALYSIS for {ticker} ═══\n"]
    lines.append(f"▸ Overall Sentiment: {label} (score: {score:.2f})")
    lines.append(f"▸ Confidence: {confidence:.2f}")
    lines.append(f"▸ Headlines Analysed: {headline_count}\n")

    # Interpretation
    lines.append("── Interpretation ──")
    if score >= 0.75:
        lines.append("Strong positive sentiment. Markets are optimistic about near-term catalysts. "
                      "However, extreme positivity can sometimes precede 'sell the news' events.")
    elif score >= 0.62:
        lines.append("Moderately positive sentiment. There is a constructive backdrop, "
                      "but conviction is not extreme — a healthy sign for sustained moves.")
    elif score >= 0.38:
        lines.append("Neutral sentiment. Headlines lack a clear directional bias. "
                      "Price is more likely driven by technicals and fundamentals than news flow.")
    elif score >= 0.25:
        lines.append("Moderately negative sentiment. Cautious tone in headlines. "
                      "Consider whether the negativity reflects genuine fundamental risk or temporary fear.")
    else:
        lines.append("Strong negative sentiment. Markets are fearful or pessimistic. "
                      "Contrarian investors may see opportunity, but confirmation is needed before acting.")

    # Confidence assessment
    lines.append("\n── Reliability ──")
    if headline_count < 3:
        lines.append("⚠️ Very few headlines analysed — sentiment signal has low statistical reliability. "
                      "Gather more data points before making sentiment-driven decisions.")
    elif headline_count < 6:
        lines.append("Moderate headline sample. Reasonable for directional indication, "
                      "but more headlines would increase confidence.")
    else:
        lines.append("Good headline coverage. Sentiment signal is relatively robust.")

    if confidence < 0.5:
        lines.append("Low model confidence — many headlines may be ambiguous or mixed.")
    elif confidence > 0.8:
        lines.append("High model confidence — headlines show clear directional agreement.")

    return "\n".join(lines)


def _analyze_prediction(ctx: dict[str, Any]) -> str:
    """Prediction deep-dive."""
    ticker = ctx.get("ticker", "the stock")
    current = ctx.get("current_price")
    predicted = ctx.get("predicted_price")
    confidence = ctx.get("prediction_confidence", 0.5)
    method = ctx.get("prediction_method", "technical model")
    trend = ctx.get("trend", "sideways")
    expected_return = ctx.get("expected_return_pct", 0.0)
    currency = ctx.get("currency", "")

    lines = [f"═══ 🔮 PRICE FORECAST for {ticker} ═══\n"]

    if current and predicted:
        lines.append(f"▸ Current Price: {_fmt_money(current, currency)}")
        lines.append(f"▸ Predicted Next Close: {_fmt_money(predicted, currency)}")
        lines.append(f"▸ Expected Move: {expected_return:+.2f}%")
    lines.append(f"▸ Model: {method}")
    lines.append(f"▸ Confidence: {confidence * 100:.1f}%")
    lines.append(f"▸ Trend Context: {_trend_emoji(trend)} {trend.capitalize()}\n")

    # Interpretation
    lines.append("── Forecast Interpretation ──")
    if abs(expected_return) < 0.5:
        lines.append("The model predicts minimal price movement. This suggests consolidation "
                      "or a wait-and-see phase. Sideways trading strategies may be appropriate.")
    elif expected_return > 2.0:
        lines.append("The model signals a meaningful upside move. Verify with volume confirmation "
                      "and check that sentiment is not diverging from this bullish signal.")
    elif expected_return > 0:
        lines.append("A modest upside is expected. This is a mildly constructive signal — "
                      "best confirmed with supportive trend and sentiment data.")
    elif expected_return > -2.0:
        lines.append("A modest downside is expected. Consider tightening stop-losses "
                      "and monitoring for any acceleration in selling pressure.")
    else:
        lines.append("The model signals notable downside risk. Defensive positioning or "
                      "reduced exposure may be prudent until the signal stabilises.")

    # Confidence caveat
    lines.append("\n── Confidence & Limitations ──")
    if confidence < 0.4:
        lines.append("⚠️ Low model confidence. The forecast should be treated as a weak signal. "
                      "Market conditions may be too volatile or data too limited for reliable prediction.")
    elif confidence < 0.65:
        lines.append("Moderate confidence. The forecast is a useful data point but should be "
                      "cross-referenced with technicals, sentiment, and fundamental data.")
    else:
        lines.append("High confidence. The model has good recent validation accuracy. "
                      "Still, no model is perfect — use as one input among several.")

    if "LSTM" in method:
        lines.append("The LSTM neural network learns from historical price patterns. "
                      "It captures non-linear relationships but can overfit to recent trends.")
    else:
        lines.append("The momentum-based model uses recent return trends and moving averages. "
                      "It is fast but less nuanced than deep-learning approaches.")

    return "\n".join(lines)


def _analyze_technical(ctx: dict[str, Any]) -> str:
    """Technical indicator breakdown."""
    ticker = ctx.get("ticker", "the stock")
    rsi = ctx.get("rsi")
    macd = ctx.get("macd")
    macd_signal = ctx.get("macd_signal")
    ma_20 = ctx.get("ma_20")
    ma_50 = ctx.get("ma_50")
    current = ctx.get("current_price")
    volatility = ctx.get("volatility")
    trend = ctx.get("trend", "sideways")
    volume = ctx.get("volume")
    avg_volume = ctx.get("avg_volume")
    currency = ctx.get("currency", "")

    lines = [f"═══ 📊 TECHNICAL ANALYSIS for {ticker} ═══\n"]
    lines.append(f"▸ Trend: {_trend_emoji(trend)} {trend.capitalize()}\n")

    # Moving averages
    lines.append("── Moving Averages ──")
    if current and ma_20:
        pct_from_ma20 = ((current - ma_20) / ma_20) * 100
        position = "above" if current > ma_20 else "below"
        lines.append(f"  • Price vs MA(20): {_fmt_money(current, currency)} is {position} "
                      f"{_fmt_money(ma_20, currency)} ({pct_from_ma20:+.2f}%)")
    if current and ma_50:
        pct_from_ma50 = ((current - ma_50) / ma_50) * 100
        position = "above" if current > ma_50 else "below"
        lines.append(f"  • Price vs MA(50): {position} {_fmt_money(ma_50, currency)} ({pct_from_ma50:+.2f}%)")
    if ma_20 and ma_50:
        if ma_20 > ma_50:
            lines.append("  • MA(20) > MA(50) → Bullish crossover (golden cross tendency)")
        else:
            lines.append("  • MA(20) < MA(50) → Bearish crossover (death cross tendency)")

    # RSI
    lines.append("\n── RSI (14-period) ──")
    if rsi is not None:
        lines.append(f"  • Current RSI: {rsi:.1f}")
        if rsi > 80:
            lines.append("  • 🔴 Extremely overbought — high probability of a pullback")
        elif rsi > 70:
            lines.append("  • 🟠 Overbought zone — momentum may be exhausting")
        elif rsi > 55:
            lines.append("  • 🟢 Bullish momentum — healthy upside bias")
        elif rsi > 45:
            lines.append("  • ⚪ Neutral zone — no strong directional bias")
        elif rsi > 30:
            lines.append("  • 🟡 Mild bearish pressure — watch for further deterioration")
        elif rsi > 20:
            lines.append("  • 🔵 Oversold zone — potential reversal area")
        else:
            lines.append("  • 🔵 Extremely oversold — strong bounce probability, but confirm with volume")

    # MACD
    lines.append("\n── MACD ──")
    if macd is not None:
        lines.append(f"  • MACD Line: {macd:.4f}")
        if macd_signal is not None:
            lines.append(f"  • Signal Line: {macd_signal:.4f}")
            if macd > macd_signal:
                lines.append("  • 🟢 MACD above signal — bullish momentum")
            else:
                lines.append("  • 🔴 MACD below signal — bearish momentum")
        if macd > 0:
            lines.append("  • MACD positive — short-term trend is above long-term average")
        else:
            lines.append("  • MACD negative — short-term trend is below long-term average")

    # Volatility
    lines.append("\n── Volatility ──")
    if volatility is not None:
        lines.append(f"  • 20-day Daily Volatility: {_fmt_pct(volatility)}")
        annualised = volatility * (252 ** 0.5)
        lines.append(f"  • Annualised (approx): {_fmt_pct(annualised)}")

    # Volume
    if volume and avg_volume:
        vol_ratio = volume / avg_volume if avg_volume else 1.0
        lines.append(f"\n── Volume ──")
        lines.append(f"  • Today's Volume: {int(volume):,}")
        lines.append(f"  • Average Volume: {int(avg_volume):,}")
        lines.append(f"  • Volume Ratio: {vol_ratio:.2f}x average")
        if vol_ratio > 1.5:
            lines.append("  • 📢 Elevated volume — confirms current price move")
        elif vol_ratio < 0.5:
            lines.append("  • 🤫 Low volume — move has weak participation, may reverse")

    return "\n".join(lines)


def _analyze_valuation(ctx: dict[str, Any]) -> str:
    """Valuation breakdown from live info."""
    ticker = ctx.get("ticker", "the stock")
    pe = ctx.get("pe_ratio")
    eps = ctx.get("eps")
    dividend_yield = ctx.get("dividend_yield")
    market_cap = ctx.get("market_cap")
    beta = ctx.get("beta")
    high_52 = ctx.get("fifty_two_week_high")
    low_52 = ctx.get("fifty_two_week_low")
    current = ctx.get("current_price")
    currency = ctx.get("currency", "")

    lines = [f"═══ 💰 VALUATION ANALYSIS for {ticker} ═══\n"]

    # P/E
    lines.append("── Price-to-Earnings ──")
    if pe is not None:
        lines.append(f"  • P/E Ratio: {pe:.2f}")
        if pe < 0:
            lines.append("  • Negative P/E indicates the company is currently unprofitable")
        elif pe < 12:
            lines.append("  • Low P/E — potentially undervalued or market expects lower growth")
        elif pe < 22:
            lines.append("  • Moderate P/E — in line with typical market valuations")
        elif pe < 40:
            lines.append("  • High P/E — market expects above-average growth")
        else:
            lines.append("  • Very high P/E — priced for exceptional growth, high expectations risk")
    else:
        lines.append("  • P/E data not available")

    # EPS
    if eps is not None:
        lines.append(f"\n── Earnings Per Share ──")
        lines.append(f"  • Trailing EPS: {_fmt_money(eps, currency)}")
        if eps > 0:
            lines.append("  • Positive earnings — the company is profitable")
        else:
            lines.append("  • Negative EPS — the company is reporting losses")

    # Dividend
    if dividend_yield is not None:
        lines.append(f"\n── Dividend ──")
        lines.append(f"  • Dividend Yield: {float(dividend_yield) * 100:.2f}%")
        if dividend_yield > 0.04:
            lines.append("  • High yield — attractive for income investors, but verify sustainability")
        elif dividend_yield > 0.02:
            lines.append("  • Moderate yield — balanced income and growth potential")
        elif dividend_yield > 0:
            lines.append("  • Low yield — growth-oriented, minimal income component")
    else:
        lines.append(f"\n  • No dividend data available (may not pay dividends)")

    # 52-week range
    if current and high_52 and low_52 and high_52 > low_52:
        range_position = (current - low_52) / (high_52 - low_52) * 100
        lines.append(f"\n── 52-Week Range Position ──")
        lines.append(f"  • 52W High: {_fmt_money(high_52, currency)}")
        lines.append(f"  • 52W Low: {_fmt_money(low_52, currency)}")
        lines.append(f"  • Current: {_fmt_money(current, currency)} ({range_position:.0f}% of range)")
        if range_position > 90:
            lines.append("  • Trading near 52-week highs — momentum is strong but breakout risk exists")
        elif range_position > 60:
            lines.append("  • Upper half of range — constructive positioning")
        elif range_position > 40:
            lines.append("  • Mid-range — no strong positional bias")
        elif range_position > 10:
            lines.append("  • Lower half of range — potential value opportunity or ongoing weakness")
        else:
            lines.append("  • Near 52-week lows — deep value territory or fundamental concerns")

    # Market cap
    if market_cap:
        lines.append(f"\n── Market Cap ──")
        cap_val = float(market_cap)
        if cap_val > 200e9:
            cap_class = "Mega-cap"
        elif cap_val > 10e9:
            cap_class = "Large-cap"
        elif cap_val > 2e9:
            cap_class = "Mid-cap"
        elif cap_val > 300e6:
            cap_class = "Small-cap"
        else:
            cap_class = "Micro-cap"
        lines.append(f"  • Market Cap: {_fmt_money(market_cap, currency)}")
        lines.append(f"  • Classification: {cap_class}")

    return "\n".join(lines)


def _analyze_portfolio(ctx: dict[str, Any]) -> str:
    """Portfolio allocation guidance."""
    ticker = ctx.get("ticker", "the stock")
    risk_level = ctx.get("risk_level", "Unknown")
    sharpe = ctx.get("sharpe_ratio")
    volatility = ctx.get("volatility")
    action = ctx.get("recommendation", "HOLD")

    lines = [f"═══ 📁 PORTFOLIO GUIDANCE for {ticker} ═══\n"]

    # Position sizing
    lines.append("── Suggested Position Sizing ──")
    if risk_level == "High":
        lines.append("  • Recommended allocation: 3–8% of total portfolio")
        lines.append("  • High volatility requires smaller positions to manage drawdown risk")
    elif risk_level == "Medium":
        lines.append("  • Recommended allocation: 8–15% of total portfolio")
        lines.append("  • Moderate risk allows a meaningful but controlled position")
    else:
        lines.append("  • Recommended allocation: 12–25% of total portfolio")
        lines.append("  • Low-risk profile supports a larger weighting")

    # Entry strategy
    lines.append("\n── Entry Strategy ──")
    if action == "BUY":
        lines.append("  • Signal is bullish — consider scaling in over 2–3 tranches")
        lines.append("  • First tranche: 40% of intended position at current levels")
        lines.append("  • Second tranche: 35% on a minor pullback or confirmation")
        lines.append("  • Final tranche: 25% on trend confirmation")
    elif action == "SELL":
        lines.append("  • Signal is bearish — consider reducing exposure")
        lines.append("  • If holding: trim 30–50% and tighten stop-loss on remainder")
        lines.append("  • If not holding: wait for reversal signals before entering")
    else:
        lines.append("  • Signal is neutral — no urgent action needed")
        lines.append("  • If holding: maintain position with existing stop-loss levels")
        lines.append("  • If not holding: wait for a clearer signal before committing capital")

    # Diversification
    lines.append("\n── Diversification Notes ──")
    lines.append("  • Use the Portfolio tab to optimise across multiple tickers")
    lines.append("  • Mix asset types (growth, value, income) and sectors")
    lines.append("  • Consider geographical diversification to reduce concentration risk")

    if sharpe is not None:
        lines.append(f"\n  ℹ️ Current Sharpe ratio ({sharpe:.2f}) indicates "
                      f"{'good' if sharpe > 0.5 else 'weak'} risk-adjusted returns. "
                      f"{'Consider increasing allocation.' if sharpe > 1.0 else 'Weight accordingly.'}")

    return "\n".join(lines)


def _analyze_company(ctx: dict[str, Any]) -> str:
    """Company info summary."""
    ticker = ctx.get("ticker", "the stock")
    name = ctx.get("company_name", ticker)
    sector = ctx.get("sector", "N/A")
    industry = ctx.get("industry", "N/A")
    exchange = ctx.get("exchange", "N/A")
    summary = ctx.get("business_summary", "")
    market_cap = ctx.get("market_cap")
    currency = ctx.get("currency", "")

    lines = [f"═══ 🏢 COMPANY PROFILE: {name} ({ticker}) ═══\n"]
    lines.append(f"  • Sector: {_safe(sector)}")
    lines.append(f"  • Industry: {_safe(industry)}")
    lines.append(f"  • Exchange: {_safe(exchange)}")
    if market_cap:
        lines.append(f"  • Market Cap: {_fmt_money(market_cap, currency)}")

    if summary:
        lines.append(f"\n── Business Overview ──")
        # Truncate long summaries
        if len(summary) > 600:
            summary = summary[:597] + "..."
        lines.append(f"  {summary}")

    return "\n".join(lines)


def _build_full_summary(ctx: dict[str, Any]) -> str:
    """Comprehensive multi-section analysis."""
    ticker = ctx.get("ticker", "the stock")
    action = ctx.get("recommendation", "HOLD")
    current = ctx.get("current_price")
    predicted = ctx.get("predicted_price")
    risk_level = ctx.get("risk_level", "Unknown")
    sentiment = ctx.get("sentiment", "Neutral")
    trend = ctx.get("trend", "sideways")
    currency = ctx.get("currency", "")

    lines = [f"═══ 📋 COMPLETE ANALYSIS SUMMARY for {ticker} ═══\n"]

    # Quick snapshot
    lines.append("── At a Glance ──")
    lines.append(f"  {_action_emoji(action)} Signal: {action}")
    if current:
        lines.append(f"  💲 Price: {_fmt_money(current, currency)}")
    if predicted:
        lines.append(f"  🔮 Forecast: {_fmt_money(predicted, currency)}")
    lines.append(f"  {_trend_emoji(trend)} Trend: {trend.capitalize()}")
    lines.append(f"  {_sentiment_emoji(sentiment)} Sentiment: {sentiment}")
    lines.append(f"  {_risk_emoji(risk_level)} Risk: {risk_level}")

    # Add each section
    lines.append("\n" + "─" * 50)
    lines.append(_analyze_prediction(ctx))
    lines.append("\n" + "─" * 50)
    lines.append(_analyze_technical(ctx))
    lines.append("\n" + "─" * 50)
    lines.append(_analyze_risk(ctx))
    lines.append("\n" + "─" * 50)
    lines.append(_analyze_sentiment(ctx))
    lines.append("\n" + "─" * 50)
    lines.append(_analyze_portfolio(ctx))

    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────────

_INTENT_HANDLERS: dict[str, Any] = {
    "buy_sell": _analyze_buy_sell,
    "recommendation": _analyze_buy_sell,
    "risk": _analyze_risk,
    "sentiment": _analyze_sentiment,
    "prediction": _analyze_prediction,
    "technical": _analyze_technical,
    "valuation": _analyze_valuation,
    "portfolio": _analyze_portfolio,
    "company": _analyze_company,
    "compare": _analyze_buy_sell,  # fallback to buy/sell with full context
    "summary": _build_full_summary,
}


def answer_question(question: str, context: dict[str, Any]) -> str:
    """Generate a deep, context-aware financial analysis response.

    Parameters
    ----------
    question:
        The user's natural-language question.
    context:
        A rich dict populated from live data, prediction, sentiment,
        risk metrics, technical indicators, and company info.
    """
    text = (question or "").strip()
    if not text:
        return "Ask me anything about the stock — try 'Should I buy?', 'What's the risk?', or 'Give me a full analysis'."

    intents = _classify_intents(text)

    # Build response from all matched intents
    sections: list[str] = []
    seen_handlers: set[str] = set()

    for intent in intents:
        handler = _INTENT_HANDLERS.get(intent, _build_full_summary)
        handler_name = handler.__name__
        if handler_name not in seen_handlers:
            seen_handlers.add(handler_name)
            sections.append(handler(context))

    response = "\n\n".join(sections)
    response += f"\n\n{'─' * 50}\n⚠️ {DISCLAIMER}"

    return response


# ── Suggested questions ──────────────────────────────────────────────────

def get_suggested_questions(ticker: str) -> list[str]:
    """Return a curated list of suggested questions for the chatbot."""
    return [
        f"Should I buy {ticker}?",
        f"What's the risk level of {ticker}?",
        f"Give me a full analysis of {ticker}",
        f"What do the technical indicators say about {ticker}?",
        f"How is the market sentiment for {ticker}?",
        f"What's the price forecast for {ticker}?",
        f"Is {ticker} overvalued or undervalued?",
        f"How should I allocate {ticker} in my portfolio?",
        f"Tell me about the company behind {ticker}",
        f"What's the Sharpe ratio and drawdown risk?",
    ]
