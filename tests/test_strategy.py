import pytest
import numpy as np
import plotly.graph_objects as go
from src.strategies.strategy import (
    LongCall, LongPut, CoveredCall, ProtectivePut,
    BullCallSpread, BearPutSpread, IronCondor,
    LongStraddle, LongStrangle
)

def test_long_call_payoff_and_metrics():
    strategy = LongCall(strike=100.0, premium=5.0)
    spots = np.array([90.0, 100.0, 110.0])
    
    payoffs = strategy.payoff(spots)
    profits = strategy.profit(spots)
    metrics = strategy.get_metrics()
    
    assert np.allclose(payoffs, [0.0, 0.0, 10.0])
    assert np.allclose(profits, [-5.0, -5.0, 5.0])
    assert metrics["max_profit"] == float("inf")
    assert metrics["max_loss"] == 5.0
    assert metrics["breakevens"] == [105.0]

def test_covered_call():
    strategy = CoveredCall(stock_purchase_price=100.0, strike=105.0, premium=3.0)
    spots = np.array([90.0, 105.0, 110.0])
    
    profits = strategy.profit(spots)
    metrics = strategy.get_metrics()
    
    # S_T = 90: Stock loses 10, option premium is +3 -> profit = -7
    # S_T = 105: Stock gains 5, option premium is +3 -> profit = 8
    # S_T = 110: Stock gains 10, option gains are capped at 5 + 3 -> profit = 8
    assert np.allclose(profits, [-7.0, 8.0, 8.0])
    assert metrics["max_profit"] == 8.0
    assert metrics["max_loss"] == 97.0
    assert metrics["breakevens"] == [97.0]

def test_bull_call_spread_validation():
    with pytest.raises(ValueError):
        BullCallSpread(strike_long=100, premium_long=5, strike_short=95, premium_short=2)

def test_iron_condor():
    strategy = IronCondor(
        strike_long_put=90.0, premium_long_put=1.0,
        strike_short_put=95.0, premium_short_put=2.5,
        strike_short_call=105.0, premium_short_call=3.0,
        strike_long_call=110.0, premium_long_call=1.5
    )
    # Net credit = (2.5 + 3.0) - (1.0 + 1.5) = 3.0
    spots = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    profits = strategy.profit(spots)
    metrics = strategy.get_metrics()
    
    # S_T = 100: inside range -> profit = net credit = 3.0
    assert np.isclose(profits[2], 3.0)
    assert metrics["max_profit"] == 3.0
    # width of wings = 5. Max loss = 5 - 3 = 2.0
    assert metrics["max_loss"] == 2.0
    assert metrics["breakevens"] == [92.0, 108.0]

def test_strategy_plot():
    strategy = LongStraddle(strike=100.0, premium_call=4.0, premium_put=3.0)
    spots = np.linspace(80, 120, 50)
    fig = strategy.plot_payoff(spots)
    assert isinstance(fig, go.Figure)
