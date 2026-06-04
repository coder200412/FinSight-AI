"""Company universe helpers for market-wide quote screens."""

from __future__ import annotations

from io import StringIO

import pandas as pd


NIFTY_50_COMPANIES = [
    ("ADANIENT", "Adani Enterprises"),
    ("ADANIPORTS", "Adani Ports and SEZ"),
    ("APOLLOHOSP", "Apollo Hospitals"),
    ("ASIANPAINT", "Asian Paints"),
    ("AXISBANK", "Axis Bank"),
    ("BAJAJ-AUTO", "Bajaj Auto"),
    ("BAJFINANCE", "Bajaj Finance"),
    ("BAJAJFINSV", "Bajaj Finserv"),
    ("BEL", "Bharat Electronics"),
    ("BHARTIARTL", "Bharti Airtel"),
    ("CIPLA", "Cipla"),
    ("COALINDIA", "Coal India"),
    ("DRREDDY", "Dr. Reddy's Laboratories"),
    ("EICHERMOT", "Eicher Motors"),
    ("ETERNAL", "Eternal"),
    ("GRASIM", "Grasim Industries"),
    ("HCLTECH", "HCL Technologies"),
    ("HDFCBANK", "HDFC Bank"),
    ("HDFCLIFE", "HDFC Life Insurance"),
    ("HEROMOTOCO", "Hero MotoCorp"),
    ("HINDALCO", "Hindalco Industries"),
    ("HINDUNILVR", "Hindustan Unilever"),
    ("ICICIBANK", "ICICI Bank"),
    ("INDUSINDBK", "IndusInd Bank"),
    ("INFY", "Infosys"),
    ("ITC", "ITC"),
    ("JIOFIN", "Jio Financial Services"),
    ("JSWSTEEL", "JSW Steel"),
    ("KOTAKBANK", "Kotak Mahindra Bank"),
    ("LT", "Larsen and Toubro"),
    ("M&M", "Mahindra and Mahindra"),
    ("MARUTI", "Maruti Suzuki"),
    ("NESTLEIND", "Nestle India"),
    ("NTPC", "NTPC"),
    ("ONGC", "Oil and Natural Gas Corporation"),
    ("POWERGRID", "Power Grid Corporation"),
    ("RELIANCE", "Reliance Industries"),
    ("SBILIFE", "SBI Life Insurance"),
    ("SHRIRAMFIN", "Shriram Finance"),
    ("SBIN", "State Bank of India"),
    ("SUNPHARMA", "Sun Pharmaceutical"),
    ("TCS", "Tata Consultancy Services"),
    ("TATACONSUM", "Tata Consumer Products"),
    ("TATAMOTORS", "Tata Motors"),
    ("TATASTEEL", "Tata Steel"),
    ("TECHM", "Tech Mahindra"),
    ("TITAN", "Titan Company"),
    ("TRENT", "Trent"),
    ("ULTRACEMCO", "UltraTech Cement"),
    ("WIPRO", "Wipro"),
]

US_POPULAR_COMPANIES = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("GOOGL", "Alphabet"),
    ("AMZN", "Amazon"),
    ("META", "Meta Platforms"),
    ("TSLA", "Tesla"),
    ("BRK-B", "Berkshire Hathaway"),
    ("JPM", "JPMorgan Chase"),
    ("V", "Visa"),
    ("WMT", "Walmart"),
    ("UNH", "UnitedHealth Group"),
    ("MA", "Mastercard"),
    ("PG", "Procter and Gamble"),
    ("HD", "Home Depot"),
    ("JNJ", "Johnson and Johnson"),
    ("ABBV", "AbbVie"),
    ("KO", "Coca-Cola"),
    ("COST", "Costco Wholesale"),
    ("PEP", "PepsiCo"),
    ("NFLX", "Netflix"),
    ("CRM", "Salesforce"),
    ("AVGO", "Broadcom"),
    ("AMD", "Advanced Micro Devices"),
    ("INTC", "Intel"),
]

UK_POPULAR_COMPANIES = [
    ("SHEL", "Shell"),
    ("AZN", "AstraZeneca"),
    ("HSBA", "HSBC Holdings"),
    ("ULVR", "Unilever"),
    ("BP", "BP"),
    ("RIO", "Rio Tinto"),
    ("GSK", "GSK"),
    ("DGE", "Diageo"),
    ("LSEG", "London Stock Exchange"),
    ("REL", "RELX"),
    ("BATS", "British American Tobacco"),
    ("NG", "National Grid"),
    ("VOD", "Vodafone Group"),
    ("LLOY", "Lloyds Banking Group"),
    ("BARC", "Barclays"),
]

DE_POPULAR_COMPANIES = [
    ("SAP", "SAP SE"),
    ("SIE", "Siemens"),
    ("ALV", "Allianz"),
    ("DTE", "Deutsche Telekom"),
    ("BAS", "BASF"),
    ("MBG", "Mercedes-Benz Group"),
    ("BMW", "BMW"),
    ("MUV2", "Munich Re"),
    ("ADS", "Adidas"),
    ("DB1", "Deutsche Boerse"),
    ("IFX", "Infineon Technologies"),
    ("HEN3", "Henkel"),
    ("VOW3", "Volkswagen"),
    ("EOAN", "E.ON"),
    ("RWE", "RWE"),
]

JP_POPULAR_COMPANIES = [
    ("7203", "Toyota Motor"),
    ("6758", "Sony Group"),
    ("9984", "SoftBank Group"),
    ("8306", "Mitsubishi UFJ Financial"),
    ("6861", "Keyence"),
    ("9432", "Nippon Telegraph and Telephone"),
    ("6501", "Hitachi"),
    ("4502", "Takeda Pharmaceutical"),
    ("6902", "Denso"),
    ("7751", "Canon"),
    ("7267", "Honda Motor"),
    ("8035", "Tokyo Electron"),
    ("6954", "Fanuc"),
    ("4063", "Shin-Etsu Chemical"),
    ("6098", "Recruit Holdings"),
]

HK_POPULAR_COMPANIES = [
    ("0700", "Tencent Holdings"),
    ("9988", "Alibaba Group"),
    ("1299", "AIA Group"),
    ("0005", "HSBC Holdings"),
    ("2318", "Ping An Insurance"),
    ("0941", "China Mobile"),
    ("3690", "Meituan"),
    ("1810", "Xiaomi"),
    ("0388", "Hong Kong Exchanges"),
    ("2269", "WuXi Biologics"),
    ("1398", "ICBC"),
    ("0883", "CNOOC"),
    ("3988", "Bank of China"),
    ("2628", "China Life Insurance"),
    ("0001", "CK Hutchison"),
]

AU_POPULAR_COMPANIES = [
    ("BHP", "BHP Group"),
    ("CBA", "Commonwealth Bank"),
    ("CSL", "CSL Limited"),
    ("NAB", "National Australia Bank"),
    ("WBC", "Westpac Banking"),
    ("ANZ", "ANZ Group"),
    ("MQG", "Macquarie Group"),
    ("WES", "Wesfarmers"),
    ("WDS", "Woodside Energy"),
    ("TLS", "Telstra"),
    ("RIO", "Rio Tinto (AU)"),
    ("FMG", "Fortescue Metals"),
    ("ALL", "Aristocrat Leisure"),
    ("COL", "Coles Group"),
    ("STO", "Santos"),
]

CA_POPULAR_COMPANIES = [
    ("RY", "Royal Bank of Canada"),
    ("TD", "Toronto-Dominion Bank"),
    ("ENB", "Enbridge"),
    ("CNR", "Canadian National Railway"),
    ("BN", "Brookfield Corporation"),
    ("BMO", "Bank of Montreal"),
    ("CP", "Canadian Pacific Kansas City"),
    ("SU", "Suncor Energy"),
    ("TRI", "Thomson Reuters"),
    ("MFC", "Manulife Financial"),
    ("ATD", "Alimentation Couche-Tard"),
    ("CSU", "Constellation Software"),
    ("ABX", "Barrick Gold"),
    ("CNQ", "Canadian Natural Resources"),
    ("BCE", "BCE Inc"),
]

# Map universe keys to (records_list, suffix)
_COUNTRY_UNIVERSE_MAP = {
    "UK popular stocks": (UK_POPULAR_COMPANIES, ".L"),
    "DE popular stocks": (DE_POPULAR_COMPANIES, ".DE"),
    "JP popular stocks": (JP_POPULAR_COMPANIES, ".T"),
    "HK popular stocks": (HK_POPULAR_COMPANIES, ".HK"),
    "AU popular stocks": (AU_POPULAR_COMPANIES, ".AX"),
    "CA popular stocks": (CA_POPULAR_COMPANIES, ".TO"),
}


def _records_to_frame(records: list[tuple[str, str]], suffix: str = "") -> pd.DataFrame:
    rows = []
    for symbol, company in records:
        rows.append({"Symbol": symbol, "Company": company, "Ticker": f"{symbol}{suffix}"})
    return pd.DataFrame(rows)


def nifty_50_universe() -> pd.DataFrame:
    """Return a NIFTY 50 fallback universe."""
    return _records_to_frame(NIFTY_50_COMPANIES, ".NS")


def us_popular_universe() -> pd.DataFrame:
    """Return a small US demo universe."""
    return _records_to_frame(US_POPULAR_COMPANIES)


def fetch_nse_equity_universe(timeout: int = 8) -> pd.DataFrame:
    """Fetch NSE's listed equity symbols and convert them to Yahoo Finance tickers."""
    try:
        import requests

        response = requests.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            headers={
                "User-Agent": "Mozilla/5.0 AI-Financial-Assistant",
                "Accept": "text/csv,application/csv,text/plain,*/*",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
    except Exception:
        return nifty_50_universe()

    frame.columns = [str(column).strip().upper() for column in frame.columns]
    if "SYMBOL" not in frame.columns:
        return nifty_50_universe()

    if "SERIES" in frame.columns:
        frame = frame[frame["SERIES"].astype(str).str.upper().eq("EQ")]

    company_column = "NAME OF COMPANY" if "NAME OF COMPANY" in frame.columns else "SYMBOL"
    output = pd.DataFrame(
        {
            "Symbol": frame["SYMBOL"].astype(str).str.strip().str.upper(),
            "Company": frame[company_column].astype(str).str.strip(),
        }
    )
    output = output[output["Symbol"].ne("")]
    output["Ticker"] = output["Symbol"] + ".NS"
    return output.drop_duplicates("Ticker").sort_values("Symbol").reset_index(drop=True)


def custom_universe(tickers: list[str]) -> pd.DataFrame:
    """Create a display universe from user-entered tickers."""
    rows = []
    for ticker in tickers:
        symbol = ticker.split(".")[0]
        rows.append({"Symbol": symbol, "Company": symbol, "Ticker": ticker})
    return pd.DataFrame(rows)


def get_company_universe(name: str, custom_tickers: list[str] | None = None) -> pd.DataFrame:
    """Resolve a named universe into a dataframe with Symbol, Company, and Ticker columns."""
    if name == "NSE all listed equities":
        return fetch_nse_equity_universe()
    if name == "US popular stocks":
        return us_popular_universe()
    if name == "NIFTY 50":
        return nifty_50_universe()
    if name == "Custom tickers":
        return custom_universe(custom_tickers or [])

    # Country-specific popular stock universes
    if name in _COUNTRY_UNIVERSE_MAP:
        records, suffix = _COUNTRY_UNIVERSE_MAP[name]
        return _records_to_frame(records, suffix)

    # Fallback
    return nifty_50_universe()
