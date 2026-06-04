"""Shared configuration for FinSight AI."""

APP_NAME = "FinSight AI"

# ─── Country / Market Presets ─────────────────────────────────────────────
# Each preset defines:  suffix, benchmark, default tickers, default universe, currency symbol
MARKET_PRESETS = {
    "🇮🇳 India (NSE)": {
        "suffix": ".NS",
        "benchmark": "^NSEI",
        "default_ticker": "RELIANCE.NS",
        "default_tickers": [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
        ],
        "default_universe": "NIFTY 50",
        "currency": "₹",
        "timezone": "Asia/Kolkata",
    },
    "🇮🇳 India (BSE)": {
        "suffix": ".BO",
        "benchmark": "^BSESN",
        "default_ticker": "RELIANCE.BO",
        "default_tickers": [
            "RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO",
            "ICICIBANK.BO", "SBIN.BO", "ITC.BO", "LT.BO",
        ],
        "default_universe": "NIFTY 50",
        "currency": "₹",
        "timezone": "Asia/Kolkata",
    },
    "🇺🇸 United States": {
        "suffix": "",
        "benchmark": "^GSPC",
        "default_ticker": "AAPL",
        "default_tickers": [
            "AAPL", "MSFT", "NVDA", "GOOGL",
            "AMZN", "META", "TSLA", "JPM",
        ],
        "default_universe": "US popular stocks",
        "currency": "$",
        "timezone": "America/New_York",
    },
    "🇬🇧 United Kingdom": {
        "suffix": ".L",
        "benchmark": "^FTSE",
        "default_ticker": "SHEL.L",
        "default_tickers": [
            "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L",
            "BP.L", "RIO.L", "GSK.L", "DGE.L",
        ],
        "default_universe": "UK popular stocks",
        "currency": "£",
        "timezone": "Europe/London",
    },
    "🇩🇪 Germany": {
        "suffix": ".DE",
        "benchmark": "^GDAXI",
        "default_ticker": "SAP.DE",
        "default_tickers": [
            "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE",
            "BAS.DE", "MBG.DE", "BMW.DE", "MUV2.DE",
        ],
        "default_universe": "DE popular stocks",
        "currency": "€",
        "timezone": "Europe/Berlin",
    },
    "🇯🇵 Japan": {
        "suffix": ".T",
        "benchmark": "^N225",
        "default_ticker": "7203.T",
        "default_tickers": [
            "7203.T", "6758.T", "9984.T", "8306.T",
            "6861.T", "9432.T", "6501.T", "4502.T",
        ],
        "default_universe": "JP popular stocks",
        "currency": "¥",
        "timezone": "Asia/Tokyo",
    },
    "🇭🇰 Hong Kong": {
        "suffix": ".HK",
        "benchmark": "^HSI",
        "default_ticker": "0700.HK",
        "default_tickers": [
            "0700.HK", "9988.HK", "1299.HK", "0005.HK",
            "2318.HK", "0941.HK", "3690.HK", "1810.HK",
        ],
        "default_universe": "HK popular stocks",
        "currency": "HK$",
        "timezone": "Asia/Hong_Kong",
    },
    "🇦🇺 Australia": {
        "suffix": ".AX",
        "benchmark": "^AXJO",
        "default_ticker": "BHP.AX",
        "default_tickers": [
            "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX",
            "WBC.AX", "ANZ.AX", "MQG.AX", "WES.AX",
        ],
        "default_universe": "AU popular stocks",
        "currency": "A$",
        "timezone": "Australia/Sydney",
    },
    "🇨🇦 Canada": {
        "suffix": ".TO",
        "benchmark": "^GSPTSE",
        "default_ticker": "RY.TO",
        "default_tickers": [
            "RY.TO", "TD.TO", "ENB.TO", "CNR.TO",
            "BN.TO", "BMO.TO", "CP.TO", "SU.TO",
        ],
        "default_universe": "CA popular stocks",
        "currency": "C$",
        "timezone": "America/Toronto",
    },
}

# Legacy references for backward compatibility
DEFAULT_TICKERS = MARKET_PRESETS["🇮🇳 India (NSE)"]["default_tickers"]
DEFAULT_BENCHMARK = MARKET_PRESETS["🇮🇳 India (NSE)"]["benchmark"]

MARKET_SUFFIXES = {name: preset["suffix"] for name, preset in MARKET_PRESETS.items()}

PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

RISK_FREE_RATE = 0.03

DISCLAIMER = (
    "Educational project only. Signals are model-based estimates, not financial advice. "
    "Verify with a qualified advisor before investing."
)
