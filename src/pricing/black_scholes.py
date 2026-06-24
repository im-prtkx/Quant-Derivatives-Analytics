import math
from enum import Enum
from scipy.stats import norm

class OptionType(str, Enum):
    """Enumeration representing the option type: CALL or PUT."""
    CALL = "call"
    PUT = "put"

class BlackScholes:
    """
    An analytical pricing model for European call and put options based on the
    Black-Scholes-Merton (1973) formula.

    This class supports options pricing on dividend-paying underlying assets.

    Attributes:
        spot (float): Current price of the underlying asset (S >= 0)
        strike (float): Strike price of the option (K > 0)
        time_to_maturity (float): Time to maturity in years (T >= 0)
        risk_free_rate (float): Annualized continuously compounded risk-free rate (r)
        volatility (float): Annualized asset volatility (sigma > 0)
        dividend_yield (float): Annualized continuously compounded dividend yield (q, defaults to 0.0)
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0
    ):
        self._validate_inputs(spot, strike, time_to_maturity, volatility)
        
        self.spot = float(spot)
        self.strike = float(strike)
        self.time_to_maturity = float(time_to_maturity)
        self.risk_free_rate = float(risk_free_rate)
        self.volatility = float(volatility)
        self.dividend_yield = float(dividend_yield)

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float
    ) -> None:
        """Validates option parameters, raising ValueError if any constraint is violated."""
        if spot < 0.0:
            raise ValueError(f"Spot price must be non-negative. Got: {spot}")
        if strike <= 0.0:
            raise ValueError(f"Strike price must be strictly positive. Got: {strike}")
        if time_to_maturity < 0.0:
            raise ValueError(f"Time to maturity must be non-negative. Got: {time_to_maturity}")
        if volatility <= 0.0:
            raise ValueError(f"Volatility must be strictly positive. Got: {volatility}")

    @property
    def d1(self) -> float:
        """
        Calculate the d1 parameter of the Black-Scholes formula.

        d1 = [ln(S/K) + (r - q + 0.5 * sigma^2) * T] / [sigma * sqrt(T)]

        Returns:
            float: The value of d1.

        Raises:
            ValueError: If time_to_maturity is 0 (d1 is undefined).
        """
        if self.time_to_maturity == 0.0:
            raise ValueError("d1 is undefined when time_to_maturity is 0.")
        if self.spot == 0.0:
            return -float("inf")
        
        numerator = (
            math.log(self.spot / self.strike)
            + (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility ** 2)
            * self.time_to_maturity
        )
        denominator = self.volatility * math.sqrt(self.time_to_maturity)
        return numerator / denominator

    @property
    def d2(self) -> float:
        """
        Calculate the d2 parameter of the Black-Scholes formula.

        d2 = d1 - sigma * sqrt(T)

        Returns:
            float: The value of d2.

        Raises:
            ValueError: If time_to_maturity is 0 (d2 is undefined).
        """
        if self.time_to_maturity == 0.0:
            raise ValueError("d2 is undefined when time_to_maturity is 0.")
        return self.d1 - self.volatility * math.sqrt(self.time_to_maturity)

    def call_price(self) -> float:
        """
        Calculate the Black-Scholes analytical price of a European Call option.

        C = S * e^(-q * T) * N(d1) - K * e^(-r * T) * N(d2)

        Returns:
            float: The European Call option price.
        """
        if self.time_to_maturity == 0.0:
            return max(self.spot - self.strike, 0.0)
        if self.spot == 0.0:
            return 0.0

        d1_val = self.d1
        d2_val = self.d2

        discounted_spot = self.spot * math.exp(-self.dividend_yield * self.time_to_maturity)
        discounted_strike = self.strike * math.exp(-self.risk_free_rate * self.time_to_maturity)

        return discounted_spot * norm.cdf(d1_val) - discounted_strike * norm.cdf(d2_val)

    def put_price(self) -> float:
        """
        Calculate the Black-Scholes analytical price of a European Put option.

        P = K * e^(-r * T) * N(-d2) - S * e^(-q * T) * N(-d1)

        Returns:
            float: The European Put option price.
        """
        if self.time_to_maturity == 0.0:
            return max(self.strike - self.spot, 0.0)
        if self.spot == 0.0:
            return self.strike * math.exp(-self.risk_free_rate * self.time_to_maturity)

        d1_val = self.d1
        d2_val = self.d2

        discounted_spot = self.spot * math.exp(-self.dividend_yield * self.time_to_maturity)
        discounted_strike = self.strike * math.exp(-self.risk_free_rate * self.time_to_maturity)

        return discounted_strike * norm.cdf(-d2_val) - discounted_spot * norm.cdf(-d1_val)

    def price(self, option_type: OptionType) -> float:
        """
        Calculate the option price based on the specified option type.

        Args:
            option_type (OptionType): OptionType.CALL or OptionType.PUT.

        Returns:
            float: The calculated option price.
        """
        if option_type == OptionType.CALL:
            return self.call_price()
        elif option_type == OptionType.PUT:
            return self.put_price()
        else:
            raise ValueError(f"Invalid option type: {option_type}")
