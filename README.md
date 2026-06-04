# FinSight AI

Live Streamlit dashboard for market data, technical forecasting, news sentiment, recommendation signals, portfolio optimization, and risk analysis.

## Features

- Live Yahoo Finance stock data through `yfinance`
- Deployment-ready Streamlit frontend with branded header, KPI cards, styled charts, and responsive layout
- Market Watch page for NIFTY 50, custom lists, US demo stocks, or NSE all listed equities
- Technical indicators: moving averages, RSI, MACD, returns, volatility
- LSTM next-close prediction when TensorFlow is available
- Momentum fallback predictor so the app still runs in lightweight environments
- Financial news aggregation from Yahoo Finance and Google News RSS
- FinBERT-ready sentiment analysis with a finance lexicon fallback
- Hybrid BUY / SELL / HOLD recommendation engine
- Modern Portfolio Theory style allocation with Sharpe ratio optimization
- Risk metrics: annual return, volatility, Sharpe ratio, beta, maximum drawdown
- Streamlit dashboard and rule-based financial chatbot

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, deploy `app.py` and keep `requirements.txt`, `.streamlit/config.toml`, `src/`, and `assets/` in the repository.

Optional AI dependencies:

```bash
pip install -r requirements-ai.txt
```

If you are on Python 3.13 and TensorFlow installation is unavailable from pip, use Python 3.11 or 3.12 for the LSTM environment. The app will still run with the fallback predictor.

## Suggested Demo Flow

1. Open the Dashboard page and inspect live prices for `RELIANCE.NS`.
2. Open Market Watch and fetch NIFTY 50 or NSE all listed equities.
3. Run Prediction in `Fast` mode for a quick live signal.
4. Switch Prediction to `LSTM` only when you want to demonstrate neural-network training.
5. Refresh News with `Google News RSS (fast)` to score current headlines.
6. Use `Yahoo Finance + Google (slower)` when you want broader news coverage.
7. Optimize a portfolio with `RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS`.
8. Ask the chatbot: `Should I buy Reliance?`

## Notes

This is an educational project, not financial advice. The recommendation engine is deliberately transparent and rule-based so it can be explained in interviews and improved with stronger models later.
