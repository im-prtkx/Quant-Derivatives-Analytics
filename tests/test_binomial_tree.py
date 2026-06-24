import pytest
import math
from src.pricing.binomial_tree import BinomialTree
from src.pricing.black_scholes import BlackScholes, OptionType

def test_binomial_tree_european_vs_black_scholes():
    """Verify that CRR European prices converge close to Black-Scholes for a high step count."""
    spot = 100.0
    strike = 100.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    volatility = 0.2
    dividend_yield = 0.02
    
    bs = BlackScholes(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield
    )
    
    # European Call
    tree_call = BinomialTree(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        steps=200
    )
    
    assert math.isclose(tree_call.price(OptionType.CALL, american=False), bs.call_price(), abs_tol=1e-2)
    assert math.isclose(tree_call.price(OptionType.PUT, american=False), bs.put_price(), abs_tol=1e-2)

def test_binomial_tree_american_vs_european():
    """Verify that American option is at least as valuable as European option, and strictly more for Put."""
    spot = 90.0
    strike = 100.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    volatility = 0.3
    
    tree = BinomialTree(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        steps=100
    )
    
    eur_put = tree.price(OptionType.PUT, american=False)
    am_put = tree.price(OptionType.PUT, american=True)
    
    assert am_put >= eur_put
    # Under standard conditions (high interest rate and OTM/ITM put), American Put early exercise has value
    assert am_put > eur_put

def test_binomial_tree_input_validation():
    with pytest.raises(ValueError):
        BinomialTree(spot=-10, strike=100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)
    with pytest.raises(ValueError):
        BinomialTree(spot=100, strike=-100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2)
    with pytest.raises(ValueError):
        BinomialTree(spot=100, strike=100, time_to_maturity=-1.0, risk_free_rate=0.05, volatility=0.2)
    with pytest.raises(ValueError):
        BinomialTree(spot=100, strike=100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=-0.2)
    with pytest.raises(ValueError):
        BinomialTree(spot=100, strike=100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2, steps=0)

def test_binomial_tree_get_trees():
    tree = BinomialTree(spot=100, strike=100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.2, steps=3)
    stock_tree, option_tree = tree.get_trees(max_display_steps=3)
    assert len(stock_tree) == 4
    assert len(option_tree) == 4
    assert len(stock_tree[0]) == 1
    assert len(stock_tree[3]) == 4

def test_binomial_tree_convergence():
    tree = BinomialTree(spot=100, strike=100, time_to_maturity=0.5, risk_free_rate=0.05, volatility=0.2, steps=50)
    data = tree.compare_convergence(OptionType.CALL, max_steps=50, step_increment=10)
    assert "steps" in data
    assert "binomial_prices" in data
    assert len(data["steps"]) == 5

def test_leisen_reimer_convergence():
    """Verify Leisen-Reimer convergence against Black-Scholes."""
    spot = 100.0
    strike = 100.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    volatility = 0.2
    
    bs = BlackScholes(spot, strike, time_to_maturity, risk_free_rate, volatility)
    
    # Leisen-Reimer converges very rapidly even with small odd steps
    tree = BinomialTree(spot, strike, time_to_maturity, risk_free_rate, volatility, steps=21, model_type="lr")
    price = tree.price(OptionType.CALL, american=False)
    
    assert math.isclose(price, bs.call_price(), abs_tol=1e-3)

def test_early_exercise_boundary():
    """Verify American Put early exercise boundary values."""
    spot = 100.0
    strike = 100.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    volatility = 0.2
    
    tree = BinomialTree(spot, strike, time_to_maturity, risk_free_rate, volatility, steps=10)
    boundary = tree.get_early_exercise_boundary(OptionType.PUT)
    
    assert len(boundary) == 10
    t0, s_critical = boundary[0]
    assert t0 == 0.0
    if s_critical is not None:
        assert s_critical < strike
