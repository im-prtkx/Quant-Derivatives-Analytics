import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.backtesting.backtester import OptionBacktester

@pytest.fixture
def synthetic_prices():
    # Construct 100 days of daily stock prices increasing by 0.1% daily
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    prices = [100.0 * (1.001 ** i) for i in range(100)]
    return pd.Series(prices, index=dates)

def test_covered_call_backtest(synthetic_prices):
    backtester = OptionBacktester(synthetic_prices, risk_free_rate=0.05)
    res = backtester.backtest_covered_call(holding_period=10, otm_pct=0.03, initial_capital=10000.0)
    
    assert res["strategy"] == "Covered Call"
    assert "cagr" in res
    assert "sharpe" in res
    assert "max_drawdown" in res
    assert "win_rate" in res
    assert isinstance(res["equity_curve"], pd.Series)
    assert len(res["equity_curve"]) > 1
    assert res["win_rate"] >= 0.0 and res["win_rate"] <= 1.0
    assert res["max_drawdown"] >= 0.0

def test_straddle_backtest(synthetic_prices):
    backtester = OptionBacktester(synthetic_prices, risk_free_rate=0.05)
    res = backtester.backtest_straddle(holding_period=10, allocation_fraction=0.05, initial_capital=10000.0)
    
    assert res["strategy"] == "Long Straddle"
    assert "cagr" in res
    assert isinstance(res["equity_curve"], pd.Series)
    assert len(res["equity_curve"]) > 1

def test_iron_condor_backtest(synthetic_prices):
    backtester = OptionBacktester(synthetic_prices, risk_free_rate=0.05)
    res = backtester.backtest_iron_condor(
        holding_period=10,
        otm_short=0.02,
        otm_long=0.05,
        allocation_fraction=0.10,
        initial_capital=10000.0
    )
    
    assert res["strategy"] == "Iron Condor"
    assert "cagr" in res
    assert isinstance(res["equity_curve"], pd.Series)
    assert len(res["equity_curve"]) > 1

def test_iron_condor_validation(synthetic_prices):
    backtester = OptionBacktester(synthetic_prices)
    with pytest.raises(ValueError):
        # OTM long cannot be less than short
        backtester.backtest_iron_condor(otm_short=0.05, otm_long=0.02)

def test_backtester_plotting(synthetic_prices):
    backtester = OptionBacktester(synthetic_prices)
    res1 = backtester.backtest_covered_call(holding_period=10)
    res2 = backtester.backtest_straddle(holding_period=10)
    
    fig = OptionBacktester.plot_equity_curves([res1, res2])
    assert isinstance(fig, go.Figure)
