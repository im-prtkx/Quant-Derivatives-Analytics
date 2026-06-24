import pytest
import math
from src.pricing.monte_carlo import MonteCarloEngine
from src.pricing.black_scholes import BlackScholes, OptionType

def test_monte_carlo_path_shape():
    engine = MonteCarloEngine(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2
    )
    paths = engine.simulate_paths(num_paths=100, num_steps=50, antithetic=True)
    assert paths.shape == (100, 51)
    # Check that initial spot prices are correct
    assert math.isclose(paths[0, 0], 100.0)

def test_monte_carlo_pricing_convergence():
    spot = 100.0
    strike = 105.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    volatility = 0.25
    dividend_yield = 0.01

    bs = BlackScholes(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield
    )
    
    mc = MonteCarloEngine(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield
    )
    
    # Run with a large number of paths to achieve convergence
    mc_price, std_err, (ci_l, ci_u) = mc.price(
        OptionType.CALL,
        num_paths=100000,
        num_steps=100,
        antithetic=True,
        seed=42
    )
    
    bs_price = bs.call_price()
    
    # Price should be close and inside the confidence interval
    assert math.isclose(mc_price, bs_price, abs_tol=0.1)
    assert ci_l <= bs_price <= ci_u

def test_monte_carlo_antithetic_variance_reduction():
    spot = 100.0
    strike = 100.0
    time_to_maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.3

    mc = MonteCarloEngine(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )

    # Compute standard errors for both cases
    _, std_err_anti, _ = mc.price(OptionType.PUT, num_paths=2000, antithetic=True, seed=42)
    _, std_err_standard, _ = mc.price(OptionType.PUT, num_paths=2000, antithetic=False, seed=42)
    
    # Antithetic should have a lower standard error
    assert std_err_anti < std_err_standard

def test_monte_carlo_input_validation():
    with pytest.raises(ValueError):
        MonteCarloEngine(spot=-10, strike=100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)
    with pytest.raises(ValueError):
        MonteCarloEngine(spot=100, strike=-100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)

def test_monte_carlo_convergence_method():
    mc = MonteCarloEngine(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)
    data = mc.compare_convergence(OptionType.CALL, max_paths=10000, path_increment=3000, seed=42)
    assert len(data["paths"]) == 4
    assert data["bs_prices"][0] > 0

def test_monte_carlo_control_variates():
    """Verify that Control Variates reduces standard error compared to standard Monte Carlo."""
    spot = 100.0
    strike = 100.0
    time_to_maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.2
    
    mc = MonteCarloEngine(spot, strike, time_to_maturity, risk_free_rate, volatility)
    
    _, std_err_cv, _ = mc.price(OptionType.CALL, num_paths=2000, variance_reduction="control_variates", seed=42)
    _, std_err_std, _ = mc.price(OptionType.CALL, num_paths=2000, variance_reduction="none", antithetic=False, seed=42)
    
    assert std_err_cv < std_err_std

def test_monte_carlo_path_schemes():
    """Verify different path simulation schemes (Euler and Milstein) produce expected shapes."""
    mc = MonteCarloEngine(spot=100.0, strike=100.0, time_to_maturity=0.5, risk_free_rate=0.05, volatility=0.25)
    
    paths_euler = mc.simulate_paths(num_paths=50, num_steps=20, scheme="euler", seed=42)
    paths_milstein = mc.simulate_paths(num_paths=50, num_steps=20, scheme="milstein", seed=42)
    
    assert paths_euler.shape == (50, 21)
    assert paths_milstein.shape == (50, 21)

def test_heston_simulation_and_pricing():
    """Verify Heston simulation and pricing function outputs."""
    mc = MonteCarloEngine(spot=100.0, strike=100.0, time_to_maturity=0.5, risk_free_rate=0.05, volatility=0.2)
    
    S, v = mc.simulate_heston(num_paths=30, num_steps=10, seed=42)
    assert S.shape == (30, 11)
    assert v.shape == (30, 11)
    
    price, std_err = mc.price_heston(OptionType.CALL, num_paths=1000, num_steps=20, seed=42)
    assert price > 0.0
    assert std_err > 0.0
