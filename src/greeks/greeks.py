import math
from scipy.stats import norm
from src.pricing.black_scholes import BlackScholes, OptionType

class Greeks:
    """
    An analytical engine to calculate Option Greeks (Delta, Gamma, Vega, Theta, Rho)
    for European Call and Put options under the Black-Scholes-Merton model.

    This engine supports underlying assets paying a continuous dividend yield.

    Attributes:
        model (BlackScholes): An instance of the BlackScholes pricing model.
        T_THRESHOLD (float): Threshold for time to maturity to apply analytical limits (T -> 0).
    """

    T_THRESHOLD: float = 1e-12

    def __init__(self, model: BlackScholes):
        """
        Initialize the Greeks engine with a BlackScholes model instance.

        Args:
            model (BlackScholes): The Black-Scholes pricing model instance.
        """
        self.model = model

    def _is_near_expiry(self) -> bool:
        """Determines if the option is extremely close to or at expiration."""
        return self.model.time_to_maturity < self.T_THRESHOLD

    def delta_call(self) -> float:
        """
        Calculate Delta of a European Call option.

        Delta represents the rate of change of the option price with respect to
        changes in the underlying asset's price.

        Delta_call = e^(-q * T) * N(d1)

        Returns:
            float: Call Delta.
        """
        T = self.model.time_to_maturity
        q = self.model.dividend_yield
        S = self.model.spot
        K = self.model.strike

        if self._is_near_expiry():
            if S > K:
                return math.exp(-q * T)
            elif S < K:
                return 0.0
            else:
                return 0.5 * math.exp(-q * T)

        return math.exp(-q * T) * norm.cdf(self.model.d1)

    def delta_put(self) -> float:
        """
        Calculate Delta of a European Put option.

        Delta represents the rate of change of the option price with respect to
        changes in the underlying asset's price.

        Delta_put = -e^(-q * T) * N(-d1)

        Returns:
            float: Put Delta.
        """
        T = self.model.time_to_maturity
        q = self.model.dividend_yield
        S = self.model.spot
        K = self.model.strike

        if self._is_near_expiry():
            if S > K:
                return 0.0
            elif S < K:
                return -math.exp(-q * T)
            else:
                return -0.5 * math.exp(-q * T)

        return -math.exp(-q * T) * norm.cdf(-self.model.d1)

    def delta(self, option_type: OptionType) -> float:
        """
        Calculate Delta based on the option type.

        Args:
            option_type (OptionType): OptionType.CALL or OptionType.PUT.

        Returns:
            float: Option Delta.
        """
        if option_type == OptionType.CALL:
            return self.delta_call()
        elif option_type == OptionType.PUT:
            return self.delta_put()
        else:
            raise ValueError(f"Invalid option type: {option_type}")

    def gamma(self) -> float:
        """
        Calculate Gamma of a European Call or Put option.

        Gamma represents the rate of change in Delta with respect to changes in
        the underlying asset's price. Gamma is identical for calls and puts.

        Gamma = [e^(-q * T) * n(d1)] / [S * sigma * sqrt(T)]

        Returns:
            float: Gamma.
        """
        T = self.model.time_to_maturity
        S = self.model.spot
        sigma = self.model.volatility
        q = self.model.dividend_yield
        K = self.model.strike

        if self._is_near_expiry():
            if S == K:
                return float("inf")
            else:
                return 0.0

        if S == 0.0:
            return 0.0

        d1_val = self.model.d1
        denominator = S * sigma * math.sqrt(T)
        return math.exp(-q * T) * norm.pdf(d1_val) / denominator

    def vega(self) -> float:
        """
        Calculate Vega of a European Call or Put option.

        Vega represents the rate of change of the option price with respect to
        changes in the underlying asset's volatility. Vega is identical for calls and puts.

        Vega = S * e^(-q * T) * sqrt(T) * n(d1)

        Returns:
            float: Vega (sensitivity per unit change in volatility).
        """
        T = self.model.time_to_maturity
        S = self.model.spot
        q = self.model.dividend_yield

        if self._is_near_expiry() or S == 0.0:
            return 0.0

        d1_val = self.model.d1
        return S * math.exp(-q * T) * math.sqrt(T) * norm.pdf(d1_val)

    def theta_call(self, annualized: bool = True) -> float:
        """
        Calculate Theta of a European Call option.

        Theta represents the rate of change of the option price with respect to
        the passage of time (time decay).

        Returns:
            float: Call Theta (negative under standard conditions).
        """
        T = self.model.time_to_maturity
        S = self.model.spot
        K = self.model.strike
        r = self.model.risk_free_rate
        q = self.model.dividend_yield
        sigma = self.model.volatility

        if self._is_near_expiry():
            if S > K:
                val = q * S * math.exp(-q * T) - r * K * math.exp(-r * T)
            elif S < K:
                val = 0.0
            else:
                val = -float("inf")
            return val / 365.0 if not annualized else val

        d1_val = self.model.d1
        d2_val = self.model.d2

        term1 = -(S * math.exp(-q * T) * norm.pdf(d1_val) * sigma) / (2.0 * math.sqrt(T))
        term2 = q * S * math.exp(-q * T) * norm.cdf(d1_val)
        term3 = -r * K * math.exp(-r * T) * norm.cdf(d2_val)
        
        val = term1 + term2 + term3
        return val / 365.0 if not annualized else val

    def theta_put(self, annualized: bool = True) -> float:
        """
        Calculate Theta of a European Put option.

        Theta represents the rate of change of the option price with respect to
        the passage of time (time decay).

        Returns:
            float: Put Theta.
        """
        T = self.model.time_to_maturity
        S = self.model.spot
        K = self.model.strike
        r = self.model.risk_free_rate
        q = self.model.dividend_yield
        sigma = self.model.volatility

        if self._is_near_expiry():
            if S < K:
                val = -q * S * math.exp(-q * T) + r * K * math.exp(-r * T)
            elif S > K:
                val = 0.0
            else:
                val = -float("inf")
            return val / 365.0 if not annualized else val

        d1_val = self.model.d1
        d2_val = self.model.d2

        term1 = -(S * math.exp(-q * T) * norm.pdf(d1_val) * sigma) / (2.0 * math.sqrt(T))
        term2 = -q * S * math.exp(-q * T) * norm.cdf(-d1_val)
        term3 = r * K * math.exp(-r * T) * norm.cdf(-d2_val)

        val = term1 + term2 + term3
        return val / 365.0 if not annualized else val

    def theta(self, option_type: OptionType, annualized: bool = True) -> float:
        """
        Calculate Theta based on the option type.

        Args:
            option_type (OptionType): OptionType.CALL or OptionType.PUT.
            annualized (bool): If True, returns annualized Theta. Otherwise, returns daily Theta (divided by 365).

        Returns:
            float: Option Theta.
        """
        if option_type == OptionType.CALL:
            return self.theta_call(annualized)
        elif option_type == OptionType.PUT:
            return self.theta_put(annualized)
        else:
            raise ValueError(f"Invalid option type: {option_type}")

    def rho_call(self) -> float:
        """
        Calculate Rho of a European Call option.

        Rho represents the rate of change of the option price with respect to
        changes in the risk-free interest rate.

        Rho_call = K * T * e^(-r * T) * N(d2)

        Returns:
            float: Call Rho.
        """
        T = self.model.time_to_maturity
        K = self.model.strike
        r = self.model.risk_free_rate

        if self._is_near_expiry():
            return 0.0

        return K * T * math.exp(-r * T) * norm.cdf(self.model.d2)

    def rho_put(self) -> float:
        """
        Calculate Rho of a European Put option.

        Rho represents the rate of change of the option price with respect to
        changes in the risk-free interest rate.

        Rho_put = -K * T * e^(-r * T) * N(-d2)

        Returns:
            float: Put Rho.
        """
        T = self.model.time_to_maturity
        K = self.model.strike
        r = self.model.risk_free_rate

        if self._is_near_expiry():
            return 0.0

        return -K * T * math.exp(-r * T) * norm.cdf(-self.model.d2)

    def rho(self, option_type: OptionType) -> float:
        """
        Calculate Rho based on the option type.

        Args:
            option_type (OptionType): OptionType.CALL or OptionType.PUT.

        Returns:
            float: Option Rho.
        """
        if option_type == OptionType.CALL:
            return self.rho_call()
        elif option_type == OptionType.PUT:
            return self.rho_put()
        else:
            raise ValueError(f"Invalid option type: {option_type}")

    def vanna(self) -> float:
        """
        Calculate Vanna of a European Call or Put option.
        Vanna represents the sensitivity of Delta with respect to Volatility
        (d Delta / d Vol) or sensitivity of Vega to Spot (d Vega / d Spot).
        """
        if self._is_near_expiry():
            return 0.0
        T = self.model.time_to_maturity
        q = self.model.dividend_yield
        sigma = self.model.volatility
        d1_val = self.model.d1
        d2_val = self.model.d2
        return -math.exp(-q * T) * norm.pdf(d1_val) * d2_val / sigma

    def volga(self) -> float:
        """
        Calculate Volga (Vomma) of a European Call or Put option.
        Volga represents the sensitivity of Vega with respect to Volatility (d Vega / d Vol).
        """
        if self._is_near_expiry():
            return 0.0
        sigma = self.model.volatility
        d1_val = self.model.d1
        d2_val = self.model.d2
        vega_val = self.vega()
        return vega_val * d1_val * d2_val / sigma

    def charm_call(self) -> float:
        """Calculate Charm of a European Call option (Delta decay per unit of time)."""
        if self._is_near_expiry():
            return 0.0
        T = self.model.time_to_maturity
        q = self.model.dividend_yield
        r = self.model.risk_free_rate
        sigma = self.model.volatility
        d1_val = self.model.d1
        d2_val = self.model.d2
        
        term1 = q * math.exp(-q * T) * norm.cdf(d1_val)
        term2 = math.exp(-q * T) * norm.pdf(d1_val) * ((r - q) / (sigma * math.sqrt(T)) - d2_val / (2.0 * T))
        return term1 - term2

    def charm_put(self) -> float:
        """Calculate Charm of a European Put option (Delta decay per unit of time)."""
        if self._is_near_expiry():
            return 0.0
        T = self.model.time_to_maturity
        q = self.model.dividend_yield
        r = self.model.risk_free_rate
        sigma = self.model.volatility
        d1_val = self.model.d1
        d2_val = self.model.d2
        
        term1 = -q * math.exp(-q * T) * norm.cdf(-d1_val)
        term2 = math.exp(-q * T) * norm.pdf(d1_val) * ((r - q) / (sigma * math.sqrt(T)) - d2_val / (2.0 * T))
        return term1 - term2

    def charm(self, option_type: OptionType) -> float:
        """
        Calculate Charm based on the option type.
        """
        if option_type == OptionType.CALL:
            return self.charm_call()
        elif option_type == OptionType.PUT:
            return self.charm_put()
        else:
            raise ValueError(f"Invalid option type: {option_type}")

    def speed(self) -> float:
        """
        Calculate Speed of a European Call or Put option.
        Speed represents the rate of change of Gamma with respect to Spot (d Gamma / d Spot).
        """
        if self._is_near_expiry() or self.model.spot == 0.0:
            return 0.0
        S = self.model.spot
        sigma = self.model.volatility
        T = self.model.time_to_maturity
        d1_val = self.model.d1
        gamma_val = self.gamma()
        return -gamma_val / S * (d1_val / (sigma * math.sqrt(T)) + 1.0)

    def color(self) -> float:
        """
        Calculate Color (Gamma decay) of a European Call or Put option.
        Color represents the rate of change of Gamma with respect to Time (d Gamma / d Time).
        """
        if self._is_near_expiry() or self.model.spot == 0.0:
            return 0.0
        S = self.model.spot
        T = self.model.time_to_maturity
        r = self.model.risk_free_rate
        q = self.model.dividend_yield
        sigma = self.model.volatility
        d1_val = self.model.d1
        d2_val = self.model.d2
        
        numerator = math.exp(-q * T) * norm.pdf(d1_val)
        denominator = 2.0 * S * sigma * T * math.sqrt(T)
        
        term1 = 2.0 * q * T
        term2 = 1.0
        term3 = d1_val * (2.0 * (r - q) * T - d2_val * sigma * math.sqrt(T)) / (2.0 * sigma * math.sqrt(T))
        
        return -numerator / denominator * (term1 + term2 + term3)


if __name__ == "__main__":
    # Example usage block
    model = BlackScholes(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20
    )
    greeks = Greeks(model)

    print("Call Delta:", greeks.delta_call())
    print("Gamma:", greeks.gamma())
    print("Vega:", greeks.vega())
    print("Call Theta (Annualized):", greeks.theta_call())
    print("Call Theta (Daily):", greeks.theta_call(annualized=False))
    print("Call Rho:", greeks.rho_call())
    print("Vanna:", greeks.vanna())
    print("Volga:", greeks.volga())
    print("Call Charm:", greeks.charm_call())
    print("Speed:", greeks.speed())
    print("Color:", greeks.color())
