import pytest
import pandas as pd
import numpy as np
import datetime
from unittest.mock import patch, MagicMock
from src.option_chain.chain_analytics import OptionChainAnalytics
from src.pricing.black_scholes import OptionType

class MockOptionChain:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame):
        self.calls = calls
        self.puts = puts

@pytest.fixture
def mock_ticker_instance():
    ticker = MagicMock()
    
    # Mock spot history
    hist_df = pd.DataFrame({"Close": [100.0]})
    ticker.history.return_value = hist_df
    
    # Generate dynamic expiration dates
    today = datetime.date.today()
    exp1 = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    exp2 = (today + datetime.timedelta(days=14)).strftime("%Y-%m-%d")
    
    ticker.options = (exp1, exp2)
    
    # Mock option chain
    calls = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0],
        "bid": [7.0, 3.0, 1.0],
        "ask": [7.2, 3.2, 1.2],
        "lastPrice": [7.1, 3.1, 1.1],
        "volume": [100, 500, 200],
        "openInterest": [1000, 5000, 2000],
        "impliedVolatility": [0.25, 0.20, 0.22]
    })
    
    puts = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0],
        "bid": [1.0, 3.0, 7.0],
        "ask": [1.2, 3.2, 7.2],
        "lastPrice": [1.1, 3.1, 7.1],
        "volume": [300, 400, 100],
        "openInterest": [3000, 4000, 1000],
        "impliedVolatility": [0.24, 0.21, 0.23]
    })
    
    ticker.option_chain.return_value = MockOptionChain(calls, puts)
    return ticker

@patch("yfinance.Ticker")
def test_option_chain_analytics(mock_ticker_class, mock_ticker_instance):
    mock_ticker_class.return_value = mock_ticker_instance

    analytics = OptionChainAnalytics(symbol="AAPL", risk_free_rate=0.05)
    
    spot = analytics.get_current_spot()
    assert spot == 100.0

    expirations = analytics.get_expirations()
    assert len(expirations) == 2

    # Analyze chain for first expiration
    calls, puts = analytics.analyze_chain(expirations[0])

    assert "computed_iv" in calls.columns
    assert "delta" in calls.columns
    assert "gamma" in calls.columns
    assert "vol_oi_ratio" in calls.columns
    assert "spread_pct" in calls.columns

    # Verify open interest aggregation
    oi_summary = analytics.get_open_interest_summary(calls, puts)
    assert oi_summary["total_call_oi"] == 8000
    assert oi_summary["total_put_oi"] == 8000
    assert oi_summary["max_call_oi_strike"] == 100.0
    assert oi_summary["max_put_oi_strike"] == 100.0

    # Verify volume aggregation
    vol_summary = analytics.get_volume_summary(calls, puts)
    assert vol_summary["total_call_volume"] == 800
    assert vol_summary["total_put_volume"] == 800
    assert vol_summary["max_call_vol_strike"] == 100.0
    assert vol_summary["max_put_vol_strike"] == 100.0

    # Rank by liquidity
    ranked_calls = analytics.rank_contracts(calls, by="liquidity")
    assert len(ranked_calls) == 3
    # Lowest spread % should be first
    assert ranked_calls.iloc[0]["strike"] == 95.0

    # Verify new Bid IV, Ask IV, and vol spread columns
    assert "bid_iv" in calls.columns
    assert "ask_iv" in calls.columns
    assert "vol_spread" in calls.columns
    # Vol spread should be non-negative (ask_iv >= bid_iv)
    for idx, row in calls.iterrows():
        if not np.isnan(row["vol_spread"]):
            assert row["vol_spread"] >= -1e-8

def test_option_chain_filtering():
    analytics = OptionChainAnalytics(symbol="AAPL", risk_free_rate=0.05, dividend_yield=0.0)
    
    # Mock data frame with:
    # 1. Normal contract (valid)
    # 2. High spread % contract
    # 3. Low volume contract
    # 4. Lower arbitrage bound violator (Call priced at 1.0 when spot is 100 and strike is 80 -> minimum value is ~20)
    # 5. Upper arbitrage bound violator (Call priced at 110 when spot is 100)
    df = pd.DataFrame({
        "strike": [100.0, 100.0, 100.0, 80.0, 100.0],
        "bid": [4.9, 2.0, 4.9, 0.9, 109.9],
        "ask": [5.1, 8.0, 5.1, 1.1, 110.1],
        "mid_price": [5.0, 5.0, 5.0, 1.0, 110.0],
        "lastPrice": [5.0, 5.0, 5.0, 1.0, 110.0],
        "volume": [100, 100, 2, 100, 100],
        "openInterest": [1000, 1000, 1000, 1000, 1000],
        "spread_pct": [4.0, 120.0, 4.0, 20.0, 0.18]
    })
    
    spot = 100.0
    T = 0.5 # 6 months
    
    # Test volume filter
    filtered_vol = analytics.filter_chain(df, OptionType.CALL, spot, T, max_spread_pct=None, min_volume=10, filter_arbitrage=False)
    assert len(filtered_vol) == 4 # removes index 2 which has volume 2
    
    # Test spread filter
    filtered_spread = analytics.filter_chain(df, OptionType.CALL, spot, T, max_spread_pct=50.0, min_volume=0, filter_arbitrage=False)
    assert len(filtered_spread) == 4 # removes index 1 which has spread_pct 120.0
    
    # Test arbitrage bounds filter
    filtered_arb = analytics.filter_chain(df, OptionType.CALL, spot, T, max_spread_pct=None, min_volume=0, filter_arbitrage=True)
    # Index 3 violates lower bound (1.0 < spot - strike * e^{-rT} = 100 - 80*e^{-0.025} approx 22)
    # Index 4 violates upper bound (110.0 > spot = 100)
    # Indices 0, 1, 2 should remain
    assert len(filtered_arb) == 3
    assert 80.0 not in filtered_arb["strike"].values
    assert 110.0 not in filtered_arb["mid_price"].values

