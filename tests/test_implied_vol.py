import pytest
import math
from src.volatility.implied_vol import ImpliedVolatilitySolver
from src.pricing.black_scholes import BlackScholes, OptionType

def test_implied_vol_solver_call_put():
    spot = 100.0
    strike = 102.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    volatility = 0.22
    dividend_yield = 0.01

    # Get true Black-Scholes price
    bs = BlackScholes(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield
    )
    call_price = bs.call_price()
    put_price = bs.put_price()

    # Calculate implied volatility
    iv_call = ImpliedVolatilitySolver.calculate_iv(
        market_price=call_price,
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        option_type=OptionType.CALL,
        dividend_yield=dividend_yield
    )

    iv_put = ImpliedVolatilitySolver.calculate_iv(
        market_price=put_price,
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        option_type=OptionType.PUT,
        dividend_yield=dividend_yield
    )

    assert math.isclose(iv_call, volatility, abs_tol=1e-4)
    assert math.isclose(iv_put, volatility, abs_tol=1e-4)

def test_implied_vol_solver_bisection_fallback():
    # An option deep out of the money where vega is practically zero
    # Newton-Raphson should fail or hit small vega, triggering bisection.
    spot = 100.0
    strike = 170.0 # Out of the money, small Vega
    time_to_maturity = 0.1
    risk_free_rate = 0.05
    volatility = 0.40

    bs = BlackScholes(spot=spot, strike=strike, time_to_maturity=time_to_maturity, risk_free_rate=risk_free_rate, volatility=volatility)
    call_price = bs.call_price()

    iv = ImpliedVolatilitySolver.calculate_iv(
        market_price=call_price,
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        option_type=OptionType.CALL,
        tolerance=1e-5
    )
    assert math.isclose(iv, volatility, abs_tol=1e-3)

def test_implied_vol_solver_out_of_bounds():
    # Market price below lower bound (arbitrage)
    with pytest.raises(ValueError):
        ImpliedVolatilitySolver.calculate_iv(
            market_price=1.0, # Too cheap
            spot=100.0,
            strike=80.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )

    # Market price above upper bound
    with pytest.raises(ValueError):
        ImpliedVolatilitySolver.calculate_iv(
            market_price=120.0, # Exceeds spot price
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )

def test_implied_vol_solver_halley():
    spot = 100.0
    strike = 100.0
    time_to_maturity = 0.5
    risk_free_rate = 0.05
    volatility = 0.25

    bs = BlackScholes(spot, strike, time_to_maturity, risk_free_rate, volatility)
    call_price = bs.call_price()

    iv = ImpliedVolatilitySolver.calculate_iv(
        market_price=call_price,
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        option_type=OptionType.CALL,
        method="halley"
    )
    assert math.isclose(iv, volatility, abs_tol=1e-4)
