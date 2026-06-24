import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
import math
from scipy.optimize import minimize
from src.pricing.black_scholes import OptionType
from src.volatility.implied_vol import ImpliedVolatilitySolver

logger = logging.getLogger(__name__)

def sabr_volatility(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Calculate the implied volatility using Hagan's SABR model formula.

    Args:
        F (float): Forward asset price.
        K (float): Strike price.
        T (float): Time to maturity.
        alpha (float): Initial volatility parameter.
        beta (float): Exponent parameter (e.g. 0.5 for square-root process, 1.0 for log-normal).
        rho (float): Correlation between asset and volatility.
        nu (float): Volatility of volatility.

    Returns:
        float: Implied volatility.
    """
    # Guard bounds
    alpha = max(1e-6, alpha)
    nu = max(1e-6, nu)
    rho = max(-0.9999, min(0.9999, rho))

    if abs(F - K) < 1e-8:
        # ATM formula
        term1 = alpha / (F ** (1.0 - beta))
        term2 = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2) / (F ** (2.0 - 2.0 * beta))
            + (0.25 * rho * beta * nu * alpha) / (F ** (1.0 - beta))
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
        ) * T
        return term1 * term2

    # Non-ATM formula
    log_FK = math.log(F / K)
    FK_beta = (F * K) ** ((1.0 - beta) / 2.0)
    
    # Pre-factor denominator
    denom_pre = FK_beta * (
        1.0 
        + ((1.0 - beta) ** 2 / 24.0) * (log_FK ** 2) 
        + ((1.0 - beta) ** 4 / 1920.0) * (log_FK ** 4)
    )
    
    z = (nu / alpha) * FK_beta * log_FK
    
    # Calculate x(z)
    sqrt_val = 1.0 - 2.0 * rho * z + z ** 2
    if sqrt_val < 0.0:
        sqrt_val = 0.0
    
    # Guard domain of log
    log_arg = (math.sqrt(sqrt_val) + z - rho) / (1.0 - rho)
    if log_arg <= 1e-12:
        log_arg = 1e-12
    x_z = math.log(log_arg)
    
    # ATM correction term
    correction = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2) / ((F * K) ** (1.0 - beta))
        + (0.25 * rho * beta * nu * alpha) / FK_beta
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
    ) * T
    
    if abs(x_z) < 1e-8:
        # If x_z is close to zero, z/x_z -> 1
        ratio = 1.0
    else:
        ratio = z / x_z

    vol = (alpha / denom_pre) * ratio * correction
    return vol


class VolatilitySmile:
    """
    Analytics engine to compute the implied volatility smile,
    and calibrate model parameters (such as the SABR model) to market data.
    """

    def __init__(
        self,
        spot: float,
        time_to_maturity: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ):
        if spot <= 0.0 or time_to_maturity <= 0.0:
            raise ValueError("Spot price and time to maturity must be strictly positive.")

        self.spot = float(spot)
        self.time_to_maturity = float(time_to_maturity)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)

    def calculate_smile(
        self,
        strikes: list[float] | np.ndarray,
        market_prices: list[float] | np.ndarray,
        option_type: OptionType
    ) -> pd.DataFrame:
        """
        Calculate implied volatility for each strike.
        """
        if len(strikes) != len(market_prices):
            raise ValueError("Strikes and market prices must have the same length.")

        ivs = []
        for strike, price in zip(strikes, market_prices):
            try:
                iv = ImpliedVolatilitySolver.calculate_iv(
                    market_price=price,
                    spot=self.spot,
                    strike=strike,
                    time_to_maturity=self.time_to_maturity,
                    risk_free_rate=self.risk_free_rate,
                    option_type=option_type,
                    dividend_yield=self.dividend_yield
                )
                ivs.append(iv)
            except Exception as e:
                logger.warning(f"Failed to calculate IV for strike {strike} with price {price}: {e}")
                ivs.append(np.nan)

        df = pd.DataFrame({
            "strike": strikes,
            "market_price": market_prices,
            "implied_volatility": ivs
        })
        return df

    def calibrate_sabr(
        self,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        beta: float = 0.5
    ) -> tuple[float, float, float]:
        """
        Calibrate the SABR model to a set of market implied volatilities.
        Under risk-neutral measure, Forward F = S * e^{(r - q)*T}.

        Args:
            strikes (np.ndarray): Array of strikes.
            market_vols (np.ndarray): Array of implied volatilities (as decimals).
            beta (float): Fixed beta parameter (defaults to 0.5).

        Returns:
            tuple: Calibrated (alpha, rho, nu).
        """
        # Filter out NaNs
        mask = ~np.isnan(market_vols) & (market_vols > 0.0)
        k_fit = np.array(strikes)[mask]
        v_fit = np.array(market_vols)[mask]

        if len(k_fit) < 3:
            raise ValueError("Need at least 3 valid implied volatilities for SABR calibration.")

        # Forward price
        F = self.spot * math.exp((self.risk_free_rate - self.dividend_yield) * self.time_to_maturity)
        T = self.time_to_maturity

        # Initial guesses
        avg_vol = np.mean(v_fit)
        initial_alpha = avg_vol * (F ** (1.0 - beta))
        
        # Initial guess: alpha, rho, nu
        x0 = [initial_alpha, -0.4, 0.4]
        
        # Parameter bounds
        bounds = [(1e-4, 5.0), (-0.99, 0.99), (1e-4, 5.0)]

        def objective(params):
            alpha, rho, nu = params
            sse = 0.0
            for k, mkt_v in zip(k_fit, v_fit):
                try:
                    fitted_vol = sabr_volatility(F, k, T, alpha, beta, rho, nu)
                    sse += (fitted_vol - mkt_v) ** 2
                except Exception:
                    sse += 1e6
            return sse

        res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        return float(res.x[0]), float(res.x[1]), float(res.x[2])

    def plot_smile(
        self,
        smile_df: pd.DataFrame,
        option_type: OptionType,
        sabr_params: tuple[float, float, float] | None = None,
        beta: float = 0.5
    ) -> go.Figure:
        """
        Generate an interactive Plotly plot of the volatility smile.
        Optionally plots the fitted SABR volatility smile.
        """
        plot_df = smile_df.dropna(subset=["implied_volatility"])

        fig = go.Figure()

        # Add Market Scatter trace
        fig.add_trace(go.Scatter(
            x=plot_df["strike"],
            y=plot_df["implied_volatility"] * 100.0,
            mode="markers",
            name="Market Implied Vol",
            marker=dict(size=10, color="#EF553B", symbol="diamond")
        ))

        # Add SABR fitted smile if parameters are provided
        if sabr_params is not None:
            alpha, rho, nu = sabr_params
            F = self.spot * math.exp((self.risk_free_rate - self.dividend_yield) * self.time_to_maturity)
            T = self.time_to_maturity
            
            # Create a smooth strike grid for plotting SABR curve
            fit_strikes = np.linspace(smile_df["strike"].min(), smile_df["strike"].max(), 100)
            fit_vols = []
            for k in fit_strikes:
                try:
                    vol = sabr_volatility(F, k, T, alpha, beta, rho, nu)
                    fit_vols.append(vol * 100.0)
                except Exception:
                    fit_vols.append(np.nan)
            
            fig.add_trace(go.Scatter(
                x=fit_strikes,
                y=fit_vols,
                mode="lines",
                name=f"SABR Fit (α={alpha:.3f}, ρ={rho:.3f}, ν={nu:.3f})",
                line=dict(color="#00CC96", width=2.5)
            ))
        else:
            # Spline line if no SABR curve
            fig.add_trace(go.Scatter(
                x=plot_df["strike"],
                y=plot_df["implied_volatility"] * 100.0,
                mode="lines",
                name="Spline Interpolation",
                line=dict(shape="spline", color="#636EFA", width=2)
            ))

        # Spot Price vline
        fig.add_vline(
            x=self.spot,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Spot ({self.spot})",
            annotation_position="top right"
        )

        fig.update_layout(
            title=dict(
                text=f"Volatility Smile & SABR Calibration (T = {self.time_to_maturity:.2f} yrs)",
                font=dict(size=18, family="Inter")
            ),
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)")
        )

        return fig
