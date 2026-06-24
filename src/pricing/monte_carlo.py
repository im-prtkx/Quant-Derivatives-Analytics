import math
import logging
import numpy as np
from src.pricing.black_scholes import BlackScholes, OptionType

logger = logging.getLogger(__name__)

class MonteCarloEngine:
    """
    Pricing engine for European options using Monte Carlo simulation of
    Geometric Brownian Motion (GBM) paths and Heston Stochastic Volatility.

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

    def simulate_paths(
        self,
        num_paths: int = 10000,
        num_steps: int = 100,
        antithetic: bool = True,
        scheme: str = "exact",
        seed: int | None = None
    ) -> np.ndarray:
        """
        Simulate asset paths under Geometric Brownian Motion.

        Args:
            num_paths (int): Number of paths to simulate.
            num_steps (int): Number of time steps.
            antithetic (bool): If True, use antithetic variates for variance reduction.
            scheme (str): 'exact' (analytical log-normal), 'euler' (Euler-Maruyama),
                          or 'milstein' (Milstein).
            seed (int): Optional random seed.

        Returns:
            np.ndarray: Array of shape (num_paths, num_steps + 1) containing simulated stock paths.
        """
        if num_steps < 1:
            raise ValueError(f"Number of steps must be at least 1. Got: {num_steps}")
        if num_paths < 2:
            raise ValueError(f"Number of paths must be at least 2. Got: {num_paths}")
        scheme = scheme.lower()
        if scheme not in ["exact", "euler", "milstein"]:
            raise ValueError(f"Invalid path scheme: {scheme}. Must be 'exact', 'euler', or 'milstein'.")

        if seed is not None:
            np.random.seed(seed)

        dt = self.time_to_maturity / num_steps
        mu = self.risk_free_rate - self.dividend_yield
        sigma = self.volatility

        # Generate normal draws
        if antithetic:
            half_paths = num_paths // 2
            Z = np.random.normal(0.0, 1.0, (half_paths, num_steps))
            Z_full = np.vstack([Z, -Z])
            if num_paths % 2 != 0:
                Z_extra = np.random.normal(0.0, 1.0, (1, num_steps))
                Z_full = np.vstack([Z_full, Z_extra])
        else:
            Z_full = np.random.normal(0.0, 1.0, (num_paths, num_steps))

        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot

        if scheme == "exact":
            # Analytical log-normal accumulation
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * math.sqrt(dt)
            increments = drift + diffusion * Z_full
            log_returns = np.cumsum(increments, axis=1)
            paths[:, 1:] = self.spot * np.exp(log_returns)
        elif scheme == "euler":
            # Euler-Maruyama simulation loop
            for t in range(num_steps):
                S_t = paths[:, t]
                S_next = S_t + mu * S_t * dt + sigma * S_t * math.sqrt(dt) * Z_full[:, t]
                paths[:, t + 1] = np.maximum(S_next, 0.0)
        elif scheme == "milstein":
            # Milstein simulation loop
            for t in range(num_steps):
                S_t = paths[:, t]
                z = Z_full[:, t]
                S_next = (
                    S_t
                    + mu * S_t * dt
                    + sigma * S_t * math.sqrt(dt) * z
                    + 0.5 * (sigma ** 2) * S_t * (z ** 2 - 1.0) * dt
                )
                paths[:, t + 1] = np.maximum(S_next, 0.0)

        return paths

    def simulate_heston(
        self,
        num_paths: int = 10000,
        num_steps: int = 100,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate joint asset and variance paths under the Heston Stochastic Volatility Model.
        Uses full truncation to handle negative variance.

        Returns:
            tuple: (stock_paths, variance_paths) of shape (num_paths, num_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = self.time_to_maturity / num_steps
        mu = self.risk_free_rate - self.dividend_yield

        # Correlated Brownian motions
        Z1 = np.random.normal(0.0, 1.0, (num_paths, num_steps))
        Z2 = np.random.normal(0.0, 1.0, (num_paths, num_steps))
        
        # Correlated random variables
        Z_S = Z1
        Z_v = rho * Z1 + math.sqrt(1.0 - rho ** 2) * Z2

        S = np.zeros((num_paths, num_steps + 1))
        v = np.zeros((num_paths, num_steps + 1))
        
        S[:, 0] = self.spot
        v[:, 0] = v0

        for t in range(num_steps):
            S_t = S[:, t]
            v_t = v[:, t]
            
            # Full truncation: use max(v_t, 0) in drift and diffusion coefficients
            v_t_plus = np.maximum(v_t, 0.0)
            
            # Asset update
            S[:, t + 1] = S_t * np.exp((mu - 0.5 * v_t_plus) * dt + np.sqrt(v_t_plus * dt) * Z_S[:, t])
            
            # Variance update (Euler scheme with full truncation)
            v[:, t + 1] = v_t + kappa * (theta - v_t_plus) * dt + xi * np.sqrt(v_t_plus * dt) * Z_v[:, t]

        return S, v

    def price(
        self,
        option_type: OptionType,
        num_paths: int = 50000,
        num_steps: int = 100,
        antithetic: bool = True,
        variance_reduction: str = "none",
        scheme: str = "exact",
        seed: int | None = None
    ) -> tuple[float, float, tuple[float, float]]:
        """
        Calculate option price, standard error, and 95% confidence interval.
        Supports Antithetic Variates and Control Variates.

        Args:
            option_type (OptionType): OptionType.CALL or OptionType.PUT.
            num_paths (int): Number of paths to simulate.
            num_steps (int): Number of time steps.
            antithetic (bool): If True, use antithetic variates (ignored if control variates is chosen).
            variance_reduction (str): 'none', 'antithetic', or 'control_variates'.
            scheme (str): 'exact', 'euler', or 'milstein'.
            seed (int): Optional random seed.

        Returns:
            tuple: (price, standard_error, (ci_lower, ci_upper))
        """
        if self.time_to_maturity == 0.0:
            payoff = max(self.spot - self.strike, 0.0) if option_type == OptionType.CALL else max(self.strike - self.spot, 0.0)
            return payoff, 0.0, (payoff, payoff)

        vr = variance_reduction.lower()
        use_anti = antithetic or (vr == "antithetic")
        if vr == "control_variates":
            # For control variates, we simulate standard paths (no antithetic structure required for simplicity)
            use_anti = False

        paths = self.simulate_paths(
            num_paths=num_paths,
            num_steps=num_steps,
            antithetic=use_anti,
            scheme=scheme,
            seed=seed
        )
        S_T = paths[:, -1]

        # Base Payoffs
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - self.strike, 0.0)
        else:
            payoffs = np.maximum(self.strike - S_T, 0.0)

        discount = math.exp(-self.risk_free_rate * self.time_to_maturity)
        discounted_payoffs = payoffs * discount

        if vr == "control_variates":
            # Control Variate: Under risk-neutral measure, E[S_T * e^(-r T)] = S_0 * e^(-q T)
            # We define X = S_T * e^(-r T) as our control variate
            X = S_T * discount
            E_X = self.spot * math.exp(-self.dividend_yield * self.time_to_maturity)
            
            # Covariance and Variance estimates
            cov_matrix = np.cov(discounted_payoffs, X)
            cov_Y_X = cov_matrix[0, 1]
            var_X = cov_matrix[1, 1]
            
            if var_X > 1e-12:
                beta = cov_Y_X / var_X
            else:
                beta = 0.0
                
            # Adjusted payoffs
            adjusted_payoffs = discounted_payoffs - beta * (X - E_X)
            price = float(np.mean(adjusted_payoffs))
            std_err = float(np.std(adjusted_payoffs, ddof=1) / math.sqrt(num_paths))
        else:
            price = float(np.mean(discounted_payoffs))
            if use_anti and num_paths % 2 == 0:
                half_paths = num_paths // 2
                paired = 0.5 * (discounted_payoffs[:half_paths] + discounted_payoffs[half_paths:])
                std_err = float(np.std(paired, ddof=1) / math.sqrt(half_paths))
            else:
                std_err = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(num_paths))

        ci_lower = price - 1.96 * std_err
        ci_upper = price + 1.96 * std_err

        return price, std_err, (ci_lower, ci_upper)

    def price_heston(
        self,
        option_type: OptionType,
        num_paths: int = 20000,
        num_steps: int = 100,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        seed: int | None = None
    ) -> tuple[float, float]:
        """
        Price a European option under the Heston Stochastic Volatility Model.

        Returns:
            tuple: (price, standard_error)
        """
        if self.time_to_maturity == 0.0:
            payoff = max(self.spot - self.strike, 0.0) if option_type == OptionType.CALL else max(self.strike - self.spot, 0.0)
            return payoff, 0.0

        S, v = self.simulate_heston(
            num_paths=num_paths,
            num_steps=num_steps,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            seed=seed
        )
        S_T = S[:, -1]

        if option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - self.strike, 0.0)
        else:
            payoffs = np.maximum(self.strike - S_T, 0.0)

        discount = math.exp(-self.risk_free_rate * self.time_to_maturity)
        discounted_payoffs = payoffs * discount

        price = float(np.mean(discounted_payoffs))
        std_err = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(num_paths))

        return price, std_err

    def compare_convergence(
        self,
        option_type: OptionType,
        max_paths: int = 100000,
        path_increment: int = 5000,
        antithetic: bool = True,
        variance_reduction: str = "none",
        seed: int | None = 42
    ) -> dict[str, list]:
        """
        Compare pricing convergence of Monte Carlo against the analytical Black-Scholes price.
        """
        bs_model = BlackScholes(
            spot=self.spot,
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield
        )
        bs_price = bs_model.price(option_type)

        paths_list = list(range(1000, max_paths + 1, path_increment))
        prices = []
        std_errs = []
        ci_lowers = []
        ci_uppers = []

        for M in paths_list:
            p, se, (ci_l, ci_u) = self.price(
                option_type=option_type,
                num_paths=M,
                num_steps=50,
                antithetic=antithetic,
                variance_reduction=variance_reduction,
                seed=seed
            )
            prices.append(p)
            std_errs.append(se)
            ci_lowers.append(ci_l)
            ci_uppers.append(ci_u)

        return {
            "paths": paths_list,
            "simulated_prices": prices,
            "bs_prices": [bs_price] * len(paths_list),
            "std_errors": std_errs,
            "ci_lowers": ci_lowers,
            "ci_uppers": ci_uppers
        }
