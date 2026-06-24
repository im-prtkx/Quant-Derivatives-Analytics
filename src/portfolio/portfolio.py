import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import scipy.stats as stats
import scipy.integrate as integrate
from src.pricing.black_scholes import OptionType, BlackScholes
from src.greeks.greeks import Greeks as AnalyticalGreeks

class Position(ABC):
    """
    Abstract base class representing a portfolio position.
    """

    def __init__(self, quantity: float):
        self.quantity = float(quantity)

    @abstractmethod
    def value(self, spot_price: float, vol_shift: float = 0.0, spot_shift_pct: float = 0.0) -> float:
        """Calculate the absolute market value of the position under shifts."""
        pass

    @abstractmethod
    def get_greeks(self, spot_price: float) -> dict[str, float]:
        """Calculate and return the individual position Greeks (scaled by quantity)."""
        pass


class StockPosition(Position):
    """
    Represents a position in the underlying asset.
    """

    def __init__(self, quantity: float):
        super().__init__(quantity)

    def value(self, spot_price: float, vol_shift: float = 0.0, spot_shift_pct: float = 0.0) -> float:
        shifted_spot = spot_price * (1.0 + spot_shift_pct)
        return shifted_spot * self.quantity

    def get_greeks(self, spot_price: float) -> dict[str, float]:
        # Delta = 1.0 per share (d Value / d Spot = quantity)
        # Other Greeks are 0.0
        return {
            "delta": self.quantity,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }


class OptionPosition(Position):
    """
    Represents an option position in the portfolio.
    """

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        quantity: float,
        dividend_yield: float = 0.0
    ):
        super().__init__(quantity)
        self.option_type = option_type
        self.strike = float(strike)
        self.time_to_maturity = float(time_to_maturity)
        self.risk_free_rate = float(risk_free_rate)
        self.volatility = float(volatility)
        self.dividend_yield = float(dividend_yield)

    def _get_model(self, spot_price: float, vol_shift: float = 0.0, spot_shift_pct: float = 0.0) -> BlackScholes:
        shifted_spot = spot_price * (1.0 + spot_shift_pct)
        shifted_vol = max(1e-4, self.volatility + vol_shift) # Volatility cannot be <= 0
        return BlackScholes(
            spot=shifted_spot,
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            risk_free_rate=self.risk_free_rate,
            volatility=shifted_vol,
            dividend_yield=self.dividend_yield
        )

    def value(self, spot_price: float, vol_shift: float = 0.0, spot_shift_pct: float = 0.0) -> float:
        model = self._get_model(spot_price, vol_shift, spot_shift_pct)
        price = model.price(self.option_type)
        return price * self.quantity

    def get_greeks(self, spot_price: float) -> dict[str, float]:
        model = self._get_model(spot_price)
        greeks_engine = AnalyticalGreeks(model)
        
        return {
            "delta": greeks_engine.delta(self.option_type) * self.quantity,
            "gamma": greeks_engine.gamma() * self.quantity,
            "vega": greeks_engine.vega() * self.quantity,
            "theta": greeks_engine.theta(self.option_type) * self.quantity,
            "rho": greeks_engine.rho(self.option_type) * self.quantity
        }


class PortfolioRiskEngine:
    """
    Aggregates and monitors risk metrics (Greeks and stress test scenarios)
    for a portfolio of Stock and Option positions.
    """

    def __init__(self, spot: float, positions: list[Position]):
        if spot <= 0.0:
            raise ValueError("Base spot price must be strictly positive.")
        
        self.spot = float(spot)
        self.positions = positions

    def aggregate_greeks(self) -> dict[str, float]:
        """
        Aggregate Delta, Gamma, Vega, Theta, and Rho across all positions.

        Returns:
            dict: Summed Greeks.
        """
        portfolio_greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }

        for pos in self.positions:
            pos_greeks = pos.get_greeks(self.spot)
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += pos_greeks[greek]

        return portfolio_greeks

    def base_value(self) -> float:
        """Calculate current total portfolio value."""
        return sum(pos.value(self.spot) for pos in self.positions)

    def stress_test(
        self,
        spot_pct_shifts: list[float] | np.ndarray,
        vol_abs_shifts: list[float] | np.ndarray
    ) -> pd.DataFrame:
        """
        Perform a 2D scenario analysis varying underlying spot price shifts (%)
        and volatility shifts (absolute addition, e.g. +0.05 for +5% vol).

        Returns:
            pd.DataFrame: Matrix of portfolio value change (%) for each scenario.
        """
        base_val = self.base_value()
        
        # Avoid division by zero if portfolio is net zero value
        if abs(base_val) < 1e-8:
            base_val_div = 1.0
        else:
            base_val_div = base_val

        results = []
        for vol_shift in vol_abs_shifts:
            row_results = {}
            for spot_shift in spot_pct_shifts:
                # Value of portfolio under shift
                shifted_val = sum(
                    pos.value(self.spot, vol_shift=vol_shift, spot_shift_pct=spot_shift)
                    for pos in self.positions
                )
                # Percent return of portfolio
                pct_change = (shifted_val - base_val) / base_val_div * 100.0
                # Use spot shift % as column header
                col_name = f"{spot_shift * 100.0:+.1f}% Spot"
                row_results[col_name] = pct_change
            
            # Row index name
            row_name = f"{vol_shift * 100.0:+.1f}% Vol"
            row_results["Vol Shift"] = row_name
            results.append(row_results)

        df = pd.DataFrame(results)
        df.set_index("Vol Shift", inplace=True)
        return df

    def calculate_delta_normal_var_es(
        self,
        volatility: float,
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> tuple[float, float]:
        """
        Calculate portfolio Value at Risk (VaR) and Expected Shortfall (ES)
        using the Delta-Normal (linear) method.

        Args:
            volatility (float): Annualized volatility of the underlying asset.
            confidence_level (float): Confidence level for risk (default 0.95).
            holding_period (int): Holding period in business days (default 1).

        Returns:
            tuple: (VaR, Expected Shortfall) as positive dollar values.
        """
        greeks = self.aggregate_greeks()
        delta = greeks["delta"]
        
        # Scale volatility to the holding period: sigma_hp = vol * sqrt(H / 252)
        sigma_hp = volatility * np.sqrt(holding_period / 252.0)
        
        # Standard deviation of portfolio value change (dollar term)
        sigma_V = abs(delta) * self.spot * sigma_hp
        
        if sigma_V < 1e-12:
            return 0.0, 0.0
            
        z = stats.norm.ppf(confidence_level)
        
        var = z * sigma_V
        es = sigma_V * (stats.norm.pdf(z) / (1.0 - confidence_level))
        
        return float(var), float(es)

    def calculate_delta_gamma_var_es(
        self,
        volatility: float,
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> tuple[float, float]:
        """
        Calculate portfolio Value at Risk (VaR) and Expected Shortfall (ES)
        using the Delta-Gamma Cornish-Fisher expansion method.

        Args:
            volatility (float): Annualized volatility of the underlying asset.
            confidence_level (float): Confidence level for risk (default 0.95).
            holding_period (int): Holding period in business days (default 1).

        Returns:
            tuple: (VaR, Expected Shortfall) as positive dollar values.
        """
        greeks = self.aggregate_greeks()
        delta = greeks["delta"]
        gamma = greeks["gamma"]
        
        # Scale volatility
        sigma_hp = volatility * np.sqrt(holding_period / 252.0)
        
        # a and b parameters for delta-gamma portfolio return:
        # dV = a * dS + b * dS^2, where dS = dS/S0 ~ N(0, sigma_hp^2)
        a = delta * self.spot
        b = 0.5 * gamma * (self.spot ** 2)
        
        # Loss L = -dV = -a*dS - b*dS^2
        # Moments of L:
        mu_L = -b * (sigma_hp ** 2)
        variance_L = (a ** 2) * (sigma_hp ** 2) + 2.0 * (b ** 2) * (sigma_hp ** 4)
        sigma_L = np.sqrt(variance_L)
        
        if sigma_L < 1e-12:
            return float(mu_L), float(mu_L)
            
        # Third central moment of L
        mu3_L = -6.0 * (a ** 2) * b * (sigma_hp ** 4) - 8.0 * (b ** 3) * (sigma_hp ** 6)
        skew_L = mu3_L / (sigma_L ** 3)
        
        # Fourth central moment of L
        mu4_L = 3.0 * (a ** 4) * (sigma_hp ** 4) + 60.0 * (a ** 2) * (b ** 2) * (sigma_hp ** 6) + 60.0 * (b ** 4) * (sigma_hp ** 8)
        kurt_L = (mu4_L / (sigma_L ** 4)) - 3.0
        
        # Cap moments to avoid extreme behaviors outside validity envelope of Cornish-Fisher
        skew_L = np.clip(skew_L, -2.0, 2.0)
        kurt_L = np.clip(kurt_L, -1.0, 5.0)
        
        # Cornish-Fisher quantile
        z = stats.norm.ppf(confidence_level)
        w = z + (z**2 - 1.0) * skew_L / 6.0 + (z**3 - 3.0*z) * kurt_L / 24.0 - (2.0*z**3 - 5.0*z) * (skew_L**2) / 36.0
        
        var = mu_L + w * sigma_L
        
        # Numerical integration of the Cornish-Fisher quantile over standard normal density to compute Expected Shortfall
        # ES = mu_L + sigma_L * (1 / (1 - alpha)) * \int_{z_alpha}^{\infty} w(z) * phi(z) dz
        z_alpha = stats.norm.ppf(confidence_level)
        
        def integrand(z):
            wz = z + (z**2 - 1.0) * skew_L / 6.0 + (z**3 - 3.0*z) * kurt_L / 24.0 - (2.0*z**3 - 5.0*z) * (skew_L**2) / 36.0
            return wz * stats.norm.pdf(z)
            
        val, _ = integrate.quad(integrand, z_alpha, np.inf)
        es_factor = val / (1.0 - confidence_level)
        es = mu_L + es_factor * sigma_L
        
        # Fallback guard
        if es < var:
            es = var
            
        return float(var), float(es)

    def calculate_monte_carlo_var_es(
        self,
        volatility: float,
        confidence_level: float = 0.95,
        holding_period: int = 1,
        num_simulations: int = 5000,
        random_seed: int | None = None
    ) -> tuple[float, float]:
        """
        Calculate portfolio Value at Risk (VaR) and Expected Shortfall (ES)
        using Monte Carlo Simulation (Full Revaluation).

        Args:
            volatility (float): Annualized volatility of the underlying asset.
            confidence_level (float): Confidence level for risk (default 0.95).
            holding_period (int): Holding period in business days (default 1).
            num_simulations (int): Number of simulated scenarios (default 5000).
            random_seed (int): Optional random seed for reproducible results.

        Returns:
            tuple: (VaR, Expected Shortfall) as positive dollar values.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        base_val = self.base_value()
        
        # Scale volatility
        sigma_hp = volatility * np.sqrt(holding_period / 252.0)
        returns = np.random.normal(0.0, sigma_hp, num_simulations)
        
        shifted_vals = np.zeros(num_simulations)
        for i, r in enumerate(returns):
            shifted_vals[i] = sum(
                pos.value(self.spot, vol_shift=0.0, spot_shift_pct=r)
                for pos in self.positions
            )
            
        losses = base_val - shifted_vals
        losses_sorted = np.sort(losses)
        
        # Quantile index
        idx = int(np.floor(confidence_level * num_simulations))
        idx = min(idx, num_simulations - 1)
        
        var = losses_sorted[idx]
        
        # ES is tail expectation
        tail_losses = losses_sorted[idx:]
        es = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        # Fallback guard
        if es < var:
            es = var
            
        return float(var), float(es)
