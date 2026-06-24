import pytest
import numpy as np
import pandas as pd
from src.volatility.historical_vol import HistoricalVolatility

def test_historical_volatility_calculation():
    # Construct a synthetic price series with constant daily return of 1%
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    prices = [100.0 * (1.01 ** i) for i in range(100)]
    price_series = pd.Series(prices, index=dates)

    hv = HistoricalVolatility(price_series)
    returns = hv.calculate_returns()
    
    # First value should be NaN
    assert pd.isna(returns.iloc[0])
    
    # Log return should be ln(1.01) ~ 0.00995
    assert np.isclose(returns.iloc[1], np.log(1.01))

    # Rolling vol with window 10 should be 0 because daily returns are constant (standard deviation is 0)
    vol = hv.rolling_volatility(window=10)
    assert np.isclose(vol.dropna().iloc[0], 0.0, atol=1e-8)

def test_volatility_regimes():
    # Construct a price series with a low volatility section and a high volatility section
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=120, freq="D")
    
    # Low volatility returns
    ret_low = np.random.normal(0.0001, 0.005, 60)
    # High volatility returns
    ret_high = np.random.normal(0.0005, 0.03, 60)
    
    all_returns = np.concatenate([ret_low, ret_high])
    prices = 100.0 * np.exp(np.cumsum(all_returns))
    price_series = pd.Series(prices, index=dates)

    hv = HistoricalVolatility(price_series)
    regimes = hv.detect_regimes(window=10)
    
    # First 10 elements should be NaN (1 from diff, 9 from rolling window)
    assert regimes.isna().sum() == 10
    
    # The early part should mostly be classified as Low/Medium
    early_regimes = regimes.iloc[9:50].dropna()
    assert (early_regimes != "High").all()

    # The later part should mostly be classified as High
    late_regimes = regimes.iloc[90:].dropna()
    assert "High" in late_regimes.values

def test_historical_vol_input_validation():
    with pytest.raises(TypeError):
        HistoricalVolatility([100, 101, 102]) # type: ignore
    with pytest.raises(ValueError):
        HistoricalVolatility(pd.Series([], dtype=float))
    with pytest.raises(ValueError):
        HistoricalVolatility(pd.Series([-100.0, 100.0]))
