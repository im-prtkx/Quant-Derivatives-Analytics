import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.volatility.vol_smile import VolatilitySmile
from src.pricing.black_scholes import BlackScholes, OptionType

def test_volatility_smile_calculation():
    spot = 100.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    dividend_yield = 0.01
    
    smile_engine = VolatilitySmile(
        spot=spot,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield
    )

    # Let's generate market prices for a range of strikes using a constant volatility
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    true_vol = 0.20
    
    market_prices = []
    for strike in strikes:
        bs = BlackScholes(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            volatility=true_vol,
            dividend_yield=dividend_yield
        )
        market_prices.append(bs.call_price())

    smile_df = smile_engine.calculate_smile(strikes, market_prices, OptionType.CALL)
    
    assert list(smile_df.columns) == ["strike", "market_price", "implied_volatility"]
    assert len(smile_df) == 5
    # The calculated implied volatilities should match true_vol
    for iv in smile_df["implied_volatility"]:
        assert np.isclose(iv, true_vol, atol=1e-4)

def test_volatility_smile_mismatch_lengths():
    smile_engine = VolatilitySmile(spot=100.0, time_to_maturity=1.0, risk_free_rate=0.05)
    with pytest.raises(ValueError):
        smile_engine.calculate_smile([90, 100], [10.0], OptionType.CALL)

def test_volatility_smile_nan_fallback():
    smile_engine = VolatilitySmile(spot=100.0, time_to_maturity=1.0, risk_free_rate=0.05)
    # market price of 0.01 for a call with strike 50 is out of bounds (too cheap, violates lower bound)
    smile_df = smile_engine.calculate_smile([50.0], [0.01], OptionType.CALL)
    assert pd.isna(smile_df["implied_volatility"].iloc[0])

def test_volatility_smile_plotting():
    smile_engine = VolatilitySmile(spot=100.0, time_to_maturity=1.0, risk_free_rate=0.05)
    df = pd.DataFrame({
        "strike": [90, 100, 110],
        "market_price": [15.0, 8.0, 3.0],
        "implied_volatility": [0.22, 0.20, 0.21]
    })
    fig = smile_engine.plot_smile(df, OptionType.CALL)
    assert isinstance(fig, go.Figure)

def test_sabr_volatility_formula():
    from src.volatility.vol_smile import sabr_volatility
    # Test ATM boundary case (F == K)
    vol_atm = sabr_volatility(F=100.0, K=100.0, T=1.0, alpha=0.2, beta=0.5, rho=-0.4, nu=0.4)
    assert vol_atm > 0.0
    
    # Test non-ATM case
    vol_otm = sabr_volatility(F=100.0, K=110.0, T=1.0, alpha=0.2, beta=0.5, rho=-0.4, nu=0.4)
    assert vol_otm > 0.0

def test_sabr_calibration():
    from src.volatility.vol_smile import sabr_volatility
    spot = 100.0
    T = 0.5
    r = 0.05
    q = 0.01
    beta = 0.5
    
    smile_engine = VolatilitySmile(
        spot=spot,
        time_to_maturity=T,
        risk_free_rate=r,
        dividend_yield=q
    )
    
    # True parameters
    true_alpha = 0.18
    true_rho = -0.3
    true_nu = 0.35
    
    F = spot * np.exp((r - q) * T)
    strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    
    # Generate implied vols using SABR formula
    mkt_vols = np.array([
        sabr_volatility(F, k, T, true_alpha, beta, true_rho, true_nu)
        for k in strikes
    ])
    
    # Calibrate parameters
    calib_alpha, calib_rho, calib_nu = smile_engine.calibrate_sabr(strikes, mkt_vols, beta=beta)
    
    # Check that they are close to the true parameters (within reasonable tolerance)
    assert np.isclose(calib_alpha, true_alpha, atol=1e-2)
    assert np.isclose(calib_rho, true_rho, atol=1e-1)
    # Vol of vol (nu) has a slightly wider tolerance depending on the optimization surface
    assert np.isclose(calib_nu, true_nu, atol=1e-1)
    
    # Test plotting with SABR parameters
    df = pd.DataFrame({
        "strike": strikes,
        "market_price": [10.0]*len(strikes), # dummy values
        "implied_volatility": mkt_vols
    })
    fig = smile_engine.plot_smile(df, OptionType.CALL, sabr_params=(calib_alpha, calib_rho, calib_nu), beta=beta)
    assert isinstance(fig, go.Figure)

