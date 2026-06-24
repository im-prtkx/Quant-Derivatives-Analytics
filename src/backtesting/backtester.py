import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.pricing.black_scholes import BlackScholes, OptionType
from src.volatility.historical_vol import HistoricalVolatility

class OptionBacktester:
    """
    Simulates option strategies historically using daily stock price data
    and synthetic option pricing.
    """

    def __init__(self, prices: pd.Series, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        if prices.empty:
            raise ValueError("Prices series cannot be empty.")
        self.prices = prices.sort_index()
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        
        # Calculate rolling volatility to use as the option pricing parameter
        hv = HistoricalVolatility(self.prices)
        self.rolling_vol = hv.rolling_volatility(window=20).fillna(0.20) # Fallback to 20% vol if not enough data

    def backtest_covered_call(
        self,
        holding_period: int = 21,
        otm_pct: float = 0.05,
        initial_capital: float = 100000.0
    ) -> dict:
        """
        Backtest a rolling Covered Call strategy.
        At step t, buy stock and sell an OTM Call (strike = spot * (1 + otm_pct)).
        Hold for holding_period days, then settle and roll.
        """
        N = len(self.prices)
        equity = [initial_capital]
        dates = [self.prices.index[0]]
        returns = []

        # Iterate in steps of holding_period
        for i in range(0, N - holding_period, holding_period):
            s_entry = float(self.prices.iloc[i])
            s_exit = float(self.prices.iloc[i + holding_period])
            vol = float(self.rolling_vol.iloc[i])
            
            strike = s_entry * (1.0 + otm_pct)
            T = holding_period / 252.0 # Time to maturity in years

            try:
                # Price the short call option at entry
                bs = BlackScholes(
                    spot=s_entry,
                    strike=strike,
                    time_to_maturity=T,
                    risk_free_rate=self.risk_free_rate,
                    volatility=vol,
                    dividend_yield=self.dividend_yield
                )
                call_premium = bs.call_price()
            except Exception:
                call_premium = 0.0

            # Covered Call return:
            # Net Capital Outlay = s_entry - call_premium
            # Settle Value = s_exit - max(s_exit - strike, 0.0)
            net_outlay = s_entry - call_premium
            settle_val = s_exit - max(s_exit - strike, 0.0)
            
            period_return = (settle_val / net_outlay) - 1.0
            returns.append(period_return)
            
            # Update equity
            new_equity = equity[-1] * (1.0 + period_return)
            equity.append(new_equity)
            dates.append(self.prices.index[i + holding_period])

        return self._compile_results("Covered Call", dates, equity, returns, initial_capital, holding_period)

    def backtest_straddle(
        self,
        holding_period: int = 21,
        allocation_fraction: float = 0.10,
        initial_capital: float = 100000.0
    ) -> dict:
        """
        Backtest a rolling Long Straddle strategy.
        At step t, allocate allocation_fraction of portfolio to buy ATM Call + Put.
        Hold for holding_period days, settle and roll.
        """
        N = len(self.prices)
        equity = [initial_capital]
        dates = [self.prices.index[0]]
        returns = []

        for i in range(0, N - holding_period, holding_period):
            s_entry = float(self.prices.iloc[i])
            s_exit = float(self.prices.iloc[i + holding_period])
            vol = float(self.rolling_vol.iloc[i])
            
            strike = s_entry
            T = holding_period / 252.0

            try:
                bs = BlackScholes(
                    spot=s_entry,
                    strike=strike,
                    time_to_maturity=T,
                    risk_free_rate=self.risk_free_rate,
                    volatility=vol,
                    dividend_yield=self.dividend_yield
                )
                premium = bs.call_price() + bs.put_price()
            except Exception:
                premium = 0.0

            if premium <= 0.0:
                period_return = 0.0
            else:
                # Straddle payoff at expiration: |s_exit - strike|
                payoff = abs(s_exit - strike)
                option_return = (payoff / premium) - 1.0
                period_return = allocation_fraction * option_return
            
            returns.append(period_return)
            new_equity = equity[-1] * (1.0 + period_return)
            equity.append(new_equity)
            dates.append(self.prices.index[i + holding_period])

        return self._compile_results("Long Straddle", dates, equity, returns, initial_capital, holding_period)

    def backtest_iron_condor(
        self,
        holding_period: int = 21,
        otm_short: float = 0.05,
        otm_long: float = 0.10,
        allocation_fraction: float = 0.20,
        initial_capital: float = 100000.0
    ) -> dict:
        """
        Backtest a rolling Iron Condor strategy.
        Strikes:
          Long Put K1 = Spot * (1 - otm_long)
          Short Put K2 = Spot * (1 - otm_short)
          Short Call K3 = Spot * (1 + otm_short)
          Long Call K4 = Spot * (1 + otm_long)
        """
        if otm_long <= otm_short:
            raise ValueError("otm_long must be strictly greater than otm_short.")

        N = len(self.prices)
        equity = [initial_capital]
        dates = [self.prices.index[0]]
        returns = []

        for i in range(0, N - holding_period, holding_period):
            s_entry = float(self.prices.iloc[i])
            s_exit = float(self.prices.iloc[i + holding_period])
            vol = float(self.rolling_vol.iloc[i])
            
            k1 = s_entry * (1.0 - otm_long)
            k2 = s_entry * (1.0 - otm_short)
            k3 = s_entry * (1.0 + otm_short)
            k4 = s_entry * (1.0 + otm_long)
            T = holding_period / 252.0

            try:
                # Sell condor (collect credit)
                bs_k1 = BlackScholes(s_entry, k1, T, self.risk_free_rate, vol, self.dividend_yield)
                bs_k2 = BlackScholes(s_entry, k2, T, self.risk_free_rate, vol, self.dividend_yield)
                bs_k3 = BlackScholes(s_entry, k3, T, self.risk_free_rate, vol, self.dividend_yield)
                bs_k4 = BlackScholes(s_entry, k4, T, self.risk_free_rate, vol, self.dividend_yield)
                
                credit = (bs_k2.put_price() + bs_k3.call_price()) - (bs_k1.put_price() + bs_k4.call_price())
            except Exception:
                credit = 0.0

            # Width of wings
            width = max(k2 - k1, k4 - k3)
            # Max loss (Margin Required)
            margin = width - credit

            if margin <= 0.0:
                period_return = 0.0
            else:
                # Settle option values at exit
                payoff_long_put = max(k1 - s_exit, 0.0)
                payoff_short_put = -max(k2 - s_exit, 0.0)
                payoff_short_call = -max(s_exit - k3, 0.0)
                payoff_long_call = max(s_exit - k4, 0.0)
                
                payoff = payoff_long_put + payoff_short_put + payoff_short_call + payoff_long_call
                
                # Net profit of trade is payoff + credit
                trade_return = (payoff + credit) / margin
                period_return = allocation_fraction * trade_return

            returns.append(period_return)
            new_equity = equity[-1] * (1.0 + period_return)
            equity.append(new_equity)
            dates.append(self.prices.index[i + holding_period])

        return self._compile_results("Iron Condor", dates, equity, returns, initial_capital, holding_period)

    def _compile_results(
        self,
        strategy_name: str,
        dates: list,
        equity: list[float],
        returns: list[float],
        initial_capital: float,
        holding_period: int
    ) -> dict:
        """Helper to compute portfolio performance metrics and pack results."""
        if not returns:
            return {
                "strategy": strategy_name,
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "equity_curve": pd.Series([initial_capital], index=[self.prices.index[0]])
            }

        equity_series = pd.Series(equity, index=dates)
        
        # Calculate CAGR
        total_days = (dates[-1] - dates[0]).days
        years = total_days / 365.25
        if years > 0:
            cagr = (equity[-1] / initial_capital) ** (1.0 / years) - 1.0
        else:
            cagr = 0.0

        # Calculate Sharpe Ratio
        # annualization factor = sqrt(252 / holding_period)
        ann_factor = math.sqrt(252.0 / holding_period)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        
        if std_ret > 0.0:
            sharpe = ann_factor * (mean_ret / std_ret)
        else:
            sharpe = 0.0

        # Calculate Max Drawdown
        peaks = equity_series.cummax()
        drawdowns = (peaks - equity_series) / peaks
        max_dd = float(drawdowns.max())

        # Calculate Win Rate
        wins = sum(1 for r in returns if r > 0.0)
        win_rate = wins / len(returns)

        return {
            "strategy": strategy_name,
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "equity_curve": equity_series
        }

    @staticmethod
    def plot_equity_curves(backtest_results: list[dict]) -> go.Figure:
        """
        Plot the equity curve of one or more backtest result dicts.
        """
        fig = go.Figure()

        for res in backtest_results:
            curve = res["equity_curve"]
            fig.add_trace(go.Scatter(
                x=curve.index,
                y=curve.values,
                mode="lines",
                name=f"{res['strategy']} (CAGR: {res['cagr']*100:.1f}%)"
            ))

        fig.update_layout(
            title=dict(
                text="Equity Curve Performance Comparison",
                font=dict(size=18, family="Inter")
            ),
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)")
        )

        return fig
