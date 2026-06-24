import pytest
import numpy as np
import pandas as pd
from src.portfolio.portfolio import PortfolioRiskEngine, StockPosition, OptionPosition
from src.pricing.black_scholes import OptionType, BlackScholes
from src.greeks.greeks import Greeks

def test_stock_position_greeks():
    pos = StockPosition(quantity=150)
    greeks = pos.get_greeks(spot_price=100.0)
    assert greeks["delta"] == 150
    assert greeks["gamma"] == 0.0
    assert greeks["vega"] == 0.0
    assert greeks["theta"] == 0.0
    assert greeks["rho"] == 0.0

def test_option_position_greeks():
    pos = OptionPosition(
        option_type=OptionType.CALL,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        quantity=50
    )
    greeks = pos.get_greeks(spot_price=100.0)
    
    # Check that it scales correctly
    bs = BlackScholes(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.20)
    g = Greeks(bs)
    
    assert np.isclose(greeks["delta"], g.delta_call() * 50)
    assert np.isclose(greeks["gamma"], g.gamma() * 50)

def test_portfolio_greek_aggregation():
    spot = 100.0
    pos1 = StockPosition(quantity=100)
    pos2 = OptionPosition(
        option_type=OptionType.PUT,
        strike=100.0,
        time_to_maturity=0.5,
        risk_free_rate=0.05,
        volatility=0.20,
        quantity=-10 # Short Put
    )
    
    engine = PortfolioRiskEngine(spot=spot, positions=[pos1, pos2])
    agg = engine.aggregate_greeks()
    
    bs = BlackScholes(spot=spot, strike=100.0, time_to_maturity=0.5, risk_free_rate=0.05, volatility=0.20)
    g = Greeks(bs)
    
    # Portfolio delta = 100 * 1 (stock) + (-10) * put_delta
    expected_delta = 100.0 + (-10.0) * g.delta_put()
    assert np.isclose(agg["delta"], expected_delta)
    
    # Portfolio gamma = 0 (stock) + (-10) * gamma
    expected_gamma = -10.0 * g.gamma()
    assert np.isclose(agg["gamma"], expected_gamma)

def test_portfolio_stress_test():
    spot = 100.0
    pos = StockPosition(quantity=100)
    engine = PortfolioRiskEngine(spot=spot, positions=[pos])
    
    spot_shifts = [-0.10, 0.0, 0.10]
    vol_shifts = [-0.05, 0.0, 0.05]
    
    df = engine.stress_test(spot_shifts, vol_shifts)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    # With only stock, volatility shifts have no impact.
    # -10% spot shift should lead to -10% portfolio value change.
    assert np.isclose(df.loc["+0.0% Vol", "-10.0% Spot"], -10.0)
    assert np.isclose(df.loc["+0.0% Vol", "+10.0% Spot"], 10.0)

def test_portfolio_input_validation():
    with pytest.raises(ValueError):
        PortfolioRiskEngine(spot=-1, positions=[])

def test_portfolio_var_es_calculations():
    spot = 100.0
    # Stock position: 100 shares
    pos = StockPosition(quantity=100)
    engine = PortfolioRiskEngine(spot=spot, positions=[pos])
    
    vol = 0.20
    conf = 0.95
    hp = 1
    
    # Calculate VaR/ES
    dn_var, dn_es = engine.calculate_delta_normal_var_es(volatility=vol, confidence_level=conf, holding_period=hp)
    dg_var, dg_es = engine.calculate_delta_gamma_var_es(volatility=vol, confidence_level=conf, holding_period=hp)
    mc_var, mc_es = engine.calculate_monte_carlo_var_es(volatility=vol, confidence_level=conf, holding_period=hp, num_simulations=5000, random_seed=42)
    
    # For a stock-only portfolio, Gamma is 0, so Delta-Normal and Delta-Gamma must be identical
    assert np.isclose(dn_var, dg_var)
    assert np.isclose(dn_es, dg_es)
    
    # Check that ES >= VaR
    assert dn_es >= dn_var
    assert dg_es >= dg_var
    assert mc_es >= mc_var
    
    # Check manual calculation:
    # sigma_hp = 0.20 * sqrt(1/252) = 0.0125988
    # sigma_V = 100 * 100 * 0.0125988 = 125.988
    # z = norm.ppf(0.95) = 1.64485
    # expected_var = 1.64485 * 125.988 = 207.23
    assert np.isclose(dn_var, 207.23, atol=0.02)
    
    # Monte Carlo should converge close to linear analytical risk for a purely linear portfolio
    assert np.isclose(mc_var, dn_var, rtol=0.05)
    assert np.isclose(mc_es, dn_es, rtol=0.05)

def test_portfolio_var_es_with_options():
    spot = 100.0
    # Long call option
    pos_opt = OptionPosition(
        option_type=OptionType.CALL,
        strike=100.0,
        time_to_maturity=0.25,
        risk_free_rate=0.05,
        volatility=0.20,
        quantity=100
    )
    engine = PortfolioRiskEngine(spot=spot, positions=[pos_opt])
    
    vol = 0.20
    conf = 0.95
    hp = 5
    
    dn_var, dn_es = engine.calculate_delta_normal_var_es(volatility=vol, confidence_level=conf, holding_period=hp)
    dg_var, dg_es = engine.calculate_delta_gamma_var_es(volatility=vol, confidence_level=conf, holding_period=hp)
    mc_var, mc_es = engine.calculate_monte_carlo_var_es(volatility=vol, confidence_level=conf, holding_period=hp, num_simulations=2000, random_seed=42)
    
    # Since option is non-linear, Delta-Normal (linear) and Delta-Gamma (quadratic) should differ
    # Long call has positive gamma, which reduces downside tail risk (reduces losses at extremes)
    # Therefore, Delta-Gamma VaR should be lower than Delta-Normal VaR
    assert dg_var < dn_var
    assert dg_es < dn_es
    
    # Expected Shortfall must exceed Value at Risk
    assert dn_es >= dn_var
    assert dg_es >= dg_var
    assert mc_es >= mc_var

