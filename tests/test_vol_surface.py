import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.volatility.vol_surface import VolatilitySurface
from src.pricing.black_scholes import BlackScholes, OptionType

def test_volatility_surface_build():
    spot = 100.0
    r = 0.05
    q = 0.01
    
    surface_engine = VolatilitySurface(spot=spot, risk_free_rate=r, dividend_yield=q)
    
    # Create sample grid data
    strikes = [90, 100, 110]
    maturities = [0.25, 0.5]
    
    rows = []
    true_vol = 0.25
    for k in strikes:
        for t in maturities:
            bs = BlackScholes(spot=spot, strike=k, time_to_maturity=t, risk_free_rate=r, volatility=true_vol, dividend_yield=q)
            rows.append({
                "strike": k,
                "maturity": t,
                "market_price": bs.call_price(),
                "option_type": OptionType.CALL
            })
            
    df_input = pd.DataFrame(rows)
    df_output = surface_engine.build_surface(df_input)
    
    assert "implied_volatility" in df_output.columns
    assert len(df_output) == 6
    for iv in df_output["implied_volatility"]:
        assert np.isclose(iv, true_vol, atol=1e-4)

def test_volatility_surface_plotting():
    surface_engine = VolatilitySurface(spot=100.0, risk_free_rate=0.05)
    
    # 6 points is enough for interpolation
    df = pd.DataFrame({
        "strike": [90, 100, 110, 90, 100, 110],
        "maturity": [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        "implied_volatility": [0.26, 0.24, 0.25, 0.28, 0.25, 0.26]
    })
    
    fig = surface_engine.plot_surface_3d(df)
    assert isinstance(fig, go.Figure)

def test_volatility_surface_invalid_inputs():
    surface_engine = VolatilitySurface(spot=100.0, risk_free_rate=0.05)
    df = pd.DataFrame({"invalid_col": [1, 2]})
    with pytest.raises(ValueError):
        surface_engine.build_surface(df)
        
    df_too_few = pd.DataFrame({
        "strike": [90, 100],
        "maturity": [0.25, 0.25],
        "implied_volatility": [0.25, 0.25]
    })
    with pytest.raises(ValueError):
        surface_engine.plot_surface_3d(df_too_few)
