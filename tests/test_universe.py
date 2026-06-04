from src.universe import custom_universe, nifty_50_universe


def test_nifty_50_universe_contains_yahoo_tickers():
    universe = nifty_50_universe()

    assert len(universe) == 50
    assert {"Symbol", "Company", "Ticker"}.issubset(universe.columns)
    assert universe["Ticker"].str.endswith(".NS").all()


def test_custom_universe_preserves_tickers():
    universe = custom_universe(["RELIANCE.NS", "AAPL"])

    assert universe["Ticker"].tolist() == ["RELIANCE.NS", "AAPL"]

