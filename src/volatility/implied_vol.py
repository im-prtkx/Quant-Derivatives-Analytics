import math
import logging
from src.pricing.black_scholes import BlackScholes, OptionType
from src.greeks.greeks import Greeks

logger = logging.getLogger(__name__)

class ImpliedVolatilitySolver:
    """
    Solver for implied volatility of European options using Newton-Raphson
    or Halley's method, with a Bisection method fallback.
    """

    @staticmethod
    def calculate_iv(
        market_price: float,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
        initial_guess: float = 0.20,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        method: str = "newton"
    ) -> float:
        """
        Solve for implied volatility using the Newton-Raphson or Halley's method,
        falling back to Bisection if divergence is detected.

        Args:
            market_price (float): Observed option market price.
            spot (float): Spot price of the underlying asset.
            strike (float): Strike price of the option.
            time_to_maturity (float): Time to maturity in years.
            risk_free_rate (float): Annualized risk-free interest rate.
            option_type (OptionType): OptionType.CALL or OptionType.PUT.
            dividend_yield (float): Annualized dividend yield.
            initial_guess (float): Initial volatility guess (defaults to 0.20).
            max_iter (int): Maximum number of iterations (defaults to 100).
            tolerance (float): Convergence tolerance for option price error.
            method (str): 'newton' (second-order) or 'halley' (third-order, using Volga).

        Returns:
            float: Implied volatility (as a decimal, e.g. 0.25 for 25%).
        """
        # Validate core inputs
        if market_price < 0.0:
            raise ValueError(f"Market price must be non-negative. Got: {market_price}")
        if spot < 0.0 or strike <= 0.0 or time_to_maturity < 0.0:
            raise ValueError("Invalid spot, strike, or time to maturity parameters.")

        # Zero time to maturity edge case
        if time_to_maturity == 0.0:
            intrinsic = max(spot - strike, 0.0) if option_type == OptionType.CALL else max(strike - spot, 0.0)
            if math.isclose(market_price, intrinsic, abs_tol=1e-4):
                return 0.0
            else:
                raise ValueError(
                    f"Option is expired. Market price ({market_price}) must equal intrinsic value ({intrinsic})."
                )

        # Calculate mathematical bounds to check for arbitrage / invalid prices
        discount_r = math.exp(-risk_free_rate * time_to_maturity)
        discount_q = math.exp(-dividend_yield * time_to_maturity)
        
        if option_type == OptionType.CALL:
            lower_bound = max(spot * discount_q - strike * discount_r, 0.0)
            upper_bound = spot * discount_q
        else:
            lower_bound = max(strike * discount_r - spot * discount_q, 0.0)
            upper_bound = strike * discount_r

        # Tolerance adjusted bounds checks
        if market_price < lower_bound - 1e-9 or market_price > upper_bound + 1e-9:
            raise ValueError(
                f"Market price {market_price:.4f} is outside arbitrage bounds for a "
                f"{option_type.value} option. Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]"
            )

        # Handle extreme boundary conditions
        if market_price <= lower_bound + 1e-9:
            return 1e-6
        if market_price >= upper_bound - 1e-9:
            return 5.0 # Return a very high volatility

        # Try analytical search (Newton-Raphson or Halley)
        sigma = initial_guess
        solver_name = method.lower()
        
        for idx in range(max_iter):
            try:
                bs = BlackScholes(
                    spot=spot,
                    strike=strike,
                    time_to_maturity=time_to_maturity,
                    risk_free_rate=risk_free_rate,
                    volatility=sigma,
                    dividend_yield=dividend_yield
                )
                price = bs.price(option_type)
                
                greeks = Greeks(bs)
                vega = greeks.vega()

                price_error = price - market_price
                if abs(price_error) < tolerance:
                    return sigma

                # If vega is too small, step will blow up. Switch to bisection.
                if vega < 1e-8:
                    break

                if solver_name == "halley":
                    volga = greeks.volga()
                    denom = 2.0 * vega * vega - price_error * volga
                    if abs(denom) < 1e-12:
                        break
                    diff = (2.0 * price_error * vega) / denom
                else:
                    diff = price_error / vega
                    
                sigma_new = sigma - diff

                # If sigma goes out of reasonable bounds, switch to bisection.
                if sigma_new <= 1e-6 or sigma_new > 5.0:
                    break

                sigma = sigma_new
            except Exception as e:
                logger.debug(f"{solver_name.capitalize()} error in iteration {idx}: {e}")
                break

        # Fallback to Bisection method
        logger.debug(f"{solver_name.capitalize()} failed to converge or went out of bounds. Falling back to Bisection.")
        low = 1e-6
        high = 5.0
        
        for idx in range(max_iter):
            mid = 0.5 * (low + high)
            try:
                bs_mid = BlackScholes(
                    spot=spot,
                    strike=strike,
                    time_to_maturity=time_to_maturity,
                    risk_free_rate=risk_free_rate,
                    volatility=mid,
                    dividend_yield=dividend_yield
                )
                price_mid = bs_mid.price(option_type)
            except Exception as e:
                logger.error(f"Bisection evaluation failed at vol {mid}: {e}")
                return mid

            price_error = price_mid - market_price
            
            if abs(price_error) < tolerance or (high - low) < tolerance:
                return mid

            if price_mid < market_price:
                low = mid
            else:
                high = mid

        return 0.5 * (low + high)
