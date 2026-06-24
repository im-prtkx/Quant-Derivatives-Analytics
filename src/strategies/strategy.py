from abc import ABC, abstractmethod
import numpy as np
import plotly.graph_objects as go

class OptionStrategy(ABC):
    """
    Abstract base class representing an Option Strategy.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        """
        Calculate strategy payoff (gross value of position at expiration)
        for a grid of terminal spot prices.
        """
        pass

    @abstractmethod
    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        """
        Calculate net profit (payoff minus net debit, or payoff plus net credit)
        for a grid of terminal spot prices.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, float | list[float] | str]:
        """
        Calculate key metrics of the strategy: max profit, max loss, and breakeven points.
        """
        pass

    def plot_payoff(self, spot_grid: np.ndarray) -> go.Figure:
        """
        Generate an interactive Plotly diagram of the payoff and profit.
        """
        payoffs = self.payoff(spot_grid)
        profits = self.profit(spot_grid)
        metrics = self.get_metrics()
        
        fig = go.Figure()

        # Add Profit Trace
        fig.add_trace(go.Scatter(
            x=spot_grid,
            y=profits,
            mode="lines",
            name="Profit / Loss",
            line=dict(color="#00CC96", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 204, 150, 0.1)"
        ))

        # Add Payoff Trace
        fig.add_trace(go.Scatter(
            x=spot_grid,
            y=payoffs,
            mode="lines",
            name="Payoff at Expiration",
            line=dict(color="#636EFA", width=1.5, dash="dash")
        ))

        # Add horizontal line at zero profit
        fig.add_hline(y=0.0, line_color="rgba(255, 255, 255, 0.5)", line_width=1)

        # Add vertical line(s) for breakevens
        breakevens = metrics.get("breakevens")
        if isinstance(breakevens, list):
            for i, be in enumerate(breakevens):
                if spot_grid.min() <= be <= spot_grid.max():
                    fig.add_vline(
                        x=be,
                        line_dash="dot",
                        line_color="#EF553B",
                        annotation_text=f"Breakeven {i+1 if len(breakevens)>1 else ''} ({be:.2f})",
                        annotation_position="bottom right"
                    )

        # Create title description
        max_p = metrics["max_profit"]
        max_l = metrics["max_loss"]
        max_p_str = f"${max_p:.2f}" if isinstance(max_p, (int, float)) else str(max_p)
        max_l_str = f"${max_l:.2f}" if isinstance(max_l, (int, float)) else str(max_l)
        
        title_text = (
            f"Option Strategy Payoff: {self.name}<br>"
            f"<sup>Max Profit: {max_p_str} | Max Loss: {max_l_str}</sup>"
        )

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=18, family="Inter")),
            xaxis_title="Terminal Spot Price S_T",
            yaxis_title="Value ($)",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=80, b=40),
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)")
        )

        return fig


class LongCall(OptionStrategy):
    def __init__(self, strike: float, premium: float):
        super().__init__("Long Call")
        self.strike = float(strike)
        self.premium = float(premium)

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        return np.maximum(spot_grid - self.strike, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.premium

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": float("inf"),
            "max_loss": self.premium,
            "breakevens": [self.strike + self.premium]
        }


class LongPut(OptionStrategy):
    def __init__(self, strike: float, premium: float):
        super().__init__("Long Put")
        self.strike = float(strike)
        self.premium = float(premium)

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - spot_grid, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.premium

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": max(self.strike - self.premium, 0.0),
            "max_loss": self.premium,
            "breakevens": [max(self.strike - self.premium, 0.0)]
        }


class CoveredCall(OptionStrategy):
    """
    Covered Call = Long 1 Stock Position + Short 1 Call Option
    """
    def __init__(self, stock_purchase_price: float, strike: float, premium: float):
        super().__init__("Covered Call")
        self.stock_purchase_price = float(stock_purchase_price)
        self.strike = float(strike)
        self.premium = float(premium)

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        # Stock payoff is S_T, Call short payoff is -max(S_T - K, 0)
        return spot_grid - np.maximum(spot_grid - self.strike, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.stock_purchase_price + self.premium

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": self.strike - self.stock_purchase_price + self.premium,
            "max_loss": self.stock_purchase_price - self.premium,
            "breakevens": [self.stock_purchase_price - self.premium]
        }


class ProtectivePut(OptionStrategy):
    """
    Protective Put = Long 1 Stock Position + Long 1 Put Option
    """
    def __init__(self, stock_purchase_price: float, strike: float, premium: float):
        super().__init__("Protective Put")
        self.stock_purchase_price = float(stock_purchase_price)
        self.strike = float(strike)
        self.premium = float(premium)

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        # Stock payoff is S_T, Put long payoff is max(K - S_T, 0)
        return spot_grid + np.maximum(self.strike - spot_grid, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.stock_purchase_price - self.premium

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": float("inf"),
            "max_loss": self.stock_purchase_price + self.premium - self.strike,
            "breakevens": [self.stock_purchase_price + self.premium]
        }


class BullCallSpread(OptionStrategy):
    """
    Bull Call Spread = Long lower strike Call (K1) + Short higher strike Call (K2)
    """
    def __init__(self, strike_long: float, premium_long: float, strike_short: float, premium_short: float):
        super().__init__("Bull Call Spread")
        if strike_short <= strike_long:
            raise ValueError("Short strike must be strictly greater than long strike.")
        self.strike_long = float(strike_long)
        self.premium_long = float(premium_long)
        self.strike_short = float(strike_short)
        self.premium_short = float(premium_short)
        self.net_debit = self.premium_long - self.premium_short

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        return np.maximum(spot_grid - self.strike_long, 0.0) - np.maximum(spot_grid - self.strike_short, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.net_debit

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": (self.strike_short - self.strike_long) - self.net_debit,
            "max_loss": self.net_debit,
            "breakevens": [self.strike_long + self.net_debit]
        }


class BearPutSpread(OptionStrategy):
    """
    Bear Put Spread = Long higher strike Put (K2) + Short lower strike Put (K1)
    """
    def __init__(self, strike_long: float, premium_long: float, strike_short: float, premium_short: float):
        super().__init__("Bear Put Spread")
        if strike_long <= strike_short:
            raise ValueError("Long strike must be strictly greater than short strike.")
        self.strike_long = float(strike_long)
        self.premium_long = float(premium_long)
        self.strike_short = float(strike_short)
        self.premium_short = float(premium_short)
        self.net_debit = self.premium_long - self.premium_short

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike_long - spot_grid, 0.0) - np.maximum(self.strike_short - spot_grid, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.net_debit

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": (self.strike_long - self.strike_short) - self.net_debit,
            "max_loss": self.net_debit,
            "breakevens": [self.strike_long - self.net_debit]
        }


class IronCondor(OptionStrategy):
    """
    Iron Condor = Buy OTM Put (K1) + Sell OTM Put (K2) + Sell OTM Call (K3) + Buy OTM Call (K4)
    Strikes: K1 < K2 < K3 < K4
    """
    def __init__(
        self,
        strike_long_put: float, premium_long_put: float,
        strike_short_put: float, premium_short_put: float,
        strike_short_call: float, premium_short_call: float,
        strike_long_call: float, premium_long_call: float
    ):
        super().__init__("Iron Condor")
        if not (strike_long_put < strike_short_put < strike_short_call < strike_long_call):
            raise ValueError("Strikes must satisfy: Long Put (K1) < Short Put (K2) < Short Call (K3) < Long Call (K4).")

        self.k1 = float(strike_long_put)
        self.p1 = float(premium_long_put)
        self.k2 = float(strike_short_put)
        self.p2 = float(premium_short_put)
        self.k3 = float(strike_short_call)
        self.c3 = float(premium_short_call)
        self.k4 = float(strike_long_call)
        self.c4 = float(premium_long_call)

        # Net Credit collected
        self.net_credit = (self.p2 + self.c3) - (self.p1 + self.c4)

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        long_put_val = np.maximum(self.k1 - spot_grid, 0.0)
        short_put_val = -np.maximum(self.k2 - spot_grid, 0.0)
        short_call_val = -np.maximum(spot_grid - self.k3, 0.0)
        long_call_val = np.maximum(spot_grid - self.k4, 0.0)
        
        return long_put_val + short_put_val + short_call_val + long_call_val

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) + self.net_credit

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        # Max profit is net credit collected
        # Max loss is width of wings minus net credit
        width_put_side = self.k2 - self.k1
        width_call_side = self.k4 - self.k3
        max_loss = max(width_put_side, width_call_side) - self.net_credit
        
        return {
            "max_profit": self.net_credit,
            "max_loss": max_loss,
            "breakevens": [
                self.k2 - self.net_credit, # Lower breakeven
                self.k3 + self.net_credit  # Upper breakeven
            ]
        }


class LongStraddle(OptionStrategy):
    """
    Long Straddle = Long ATM Call (K) + Long ATM Put (K)
    Same strike and maturity.
    """
    def __init__(self, strike: float, premium_call: float, premium_put: float):
        super().__init__("Long Straddle")
        self.strike = float(strike)
        self.premium_call = float(premium_call)
        self.premium_put = float(premium_put)
        self.net_debit = self.premium_call + self.premium_put

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        return np.maximum(spot_grid - self.strike, 0.0) + np.maximum(self.strike - spot_grid, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.net_debit

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": float("inf"),
            "max_loss": self.net_debit,
            "breakevens": [
                max(self.strike - self.net_debit, 0.0),
                self.strike + self.net_debit
            ]
        }


class LongStrangle(OptionStrategy):
    """
    Long Strangle = Long OTM Put (K1) + Long OTM Call (K2)
    Strikes K1 < K2
    """
    def __init__(self, strike_put: float, premium_put: float, strike_call: float, premium_call: float):
        super().__init__("Long Strangle")
        if strike_call <= strike_put:
            raise ValueError("Call strike must be strictly greater than Put strike.")
        self.strike_put = float(strike_put)
        self.premium_put = float(premium_put)
        self.strike_call = float(strike_call)
        self.premium_call = float(premium_call)
        self.net_debit = self.premium_put + self.premium_call

    def payoff(self, spot_grid: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike_put - spot_grid, 0.0) + np.maximum(spot_grid - self.strike_call, 0.0)

    def profit(self, spot_grid: np.ndarray) -> np.ndarray:
        return self.payoff(spot_grid) - self.net_debit

    def get_metrics(self) -> dict[str, float | list[float] | str]:
        return {
            "max_profit": float("inf"),
            "max_loss": self.net_debit,
            "breakevens": [
                max(self.strike_put - self.net_debit, 0.0),
                self.strike_call + self.net_debit
            ]
        }
