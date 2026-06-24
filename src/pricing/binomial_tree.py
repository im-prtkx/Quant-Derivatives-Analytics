import math
import logging
import numpy as np
from src.pricing.black_scholes import BlackScholes, OptionType

logger = logging.getLogger(__name__)

class BinomialTree:
    """
    Pricing engine for European and American options using the Cox-Ross-Rubinstein (CRR)
    or Leisen-Reimer (LR) binomial tree models.

    Attributes:
        spot (float): Current price of the underlying asset (S >= 0)
        strike (float): Strike price of the option (K > 0)
        time_to_maturity (float): Time to maturity in years (T >= 0)
        risk_free_rate (float): Annualized continuously compounded risk-free rate (r)
        volatility (float): Annualized asset volatility (sigma > 0)
        dividend_yield (float): Annualized continuously compounded dividend yield (q, defaults to 0.0)
        steps (int): Number of steps in the binomial tree (N >= 1, defaults to 100)
        model_type (str): 'crr' for Cox-Ross-Rubinstein, 'lr' for Leisen-Reimer.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
        steps: int = 100,
        model_type: str = "crr"
    ):
        self._validate_inputs(spot, strike, time_to_maturity, volatility, steps)

        self.spot = float(spot)
        self.strike = float(strike)
        self.time_to_maturity = float(time_to_maturity)
        self.risk_free_rate = float(risk_free_rate)
        self.volatility = float(volatility)
        self.dividend_yield = float(dividend_yield)
        self.steps = int(steps)
        self.model_type = model_type.lower()
        if self.model_type not in ["crr", "lr"]:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be 'crr' or 'lr'.")

    def _validate_inputs(
        self,
        spot: float,
        strike: float,
        time_to_maturity: float,
        volatility: float,
        steps: int
    ) -> None:
        """Validates pricing parameters, raising ValueError if any constraint is violated."""
        if spot < 0.0:
            raise ValueError(f"Spot price must be non-negative. Got: {spot}")
        if strike <= 0.0:
            raise ValueError(f"Strike price must be strictly positive. Got: {strike}")
        if time_to_maturity < 0.0:
            raise ValueError(f"Time to maturity must be non-negative. Got: {time_to_maturity}")
        if volatility <= 0.0:
            raise ValueError(f"Volatility must be strictly positive. Got: {volatility}")
        if steps < 1:
            raise ValueError(f"Number of steps must be at least 1. Got: {steps}")

    def price(self, option_type: OptionType, american: bool = False) -> float:
        """
        Calculate the option price using the selected binomial tree model (CRR or LR).

        Args:
            option_type (OptionType): OptionType.CALL or OptionType.PUT.
            american (bool): True for American style, False for European (default).

        Returns:
            float: Option price.
        """
        if self.time_to_maturity == 0.0:
            if option_type == OptionType.CALL:
                return max(self.spot - self.strike, 0.0)
            else:
                return max(self.strike - self.spot, 0.0)

        N = self.steps
        if self.model_type == "lr":
            # Force odd steps to ensure symmetry around strike at expiry
            N = self.steps if self.steps % 2 != 0 else self.steps + 1

        dt = self.time_to_maturity / N
        discount = math.exp(-self.risk_free_rate * dt)
        growth = math.exp((self.risk_free_rate - self.dividend_yield) * dt)

        if self.model_type == "lr":
            bs = BlackScholes(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=self.time_to_maturity,
                risk_free_rate=self.risk_free_rate,
                volatility=self.volatility,
                dividend_yield=self.dividend_yield
            )
            try:
                d1_val = bs.d1
                d2_val = bs.d2
            except ValueError:
                # Handle S=0 boundary
                if option_type == OptionType.CALL:
                    return 0.0
                else:
                    return self.strike * math.exp(-self.risk_free_rate * self.time_to_maturity)

            def peizer_pratt(x: float, N_steps: int) -> float:
                if abs(x) < 1e-12:
                    return 0.5
                val = (x / (N_steps + 1.0/3.0 + 0.1 / (N_steps + 1.0))) ** 2 * (N_steps + 1.0/6.0)
                return 0.5 + np.sign(x) * math.sqrt(0.25 - 0.25 * math.exp(-val))

            p_prime = peizer_pratt(d1_val, N)
            p = peizer_pratt(d2_val, N)
            
            p_prime = max(1e-12, min(1.0 - 1e-12, p_prime))
            p = max(1e-12, min(1.0 - 1e-12, p))

            u = growth * p_prime / p
            d = (growth - p * u) / (1.0 - p)
        else:
            # Cox-Ross-Rubinstein
            u = math.exp(self.volatility * math.sqrt(dt))
            d = 1.0 / u
            p = (growth - d) / (u - d)

        p = max(0.0, min(1.0, p))

        # Vectorized backward induction
        j_range = np.arange(N + 1)
        S = self.spot * (u ** j_range) * (d ** (N - j_range))

        if option_type == OptionType.CALL:
            V = np.maximum(S - self.strike, 0.0)
        else:
            V = np.maximum(self.strike - S, 0.0)

        for i in range(N - 1, -1, -1):
            V_cont = discount * (p * V[1:] + (1.0 - p) * V[:-1])
            if american:
                j_range_i = np.arange(i + 1)
                S_i = self.spot * (u ** j_range_i) * (d ** (i - j_range_i))
                if option_type == OptionType.CALL:
                    payoff = np.maximum(S_i - self.strike, 0.0)
                else:
                    payoff = np.maximum(self.strike - S_i, 0.0)
                V = np.maximum(payoff, V_cont)
            else:
                V = V_cont

        return float(V[0])

    def get_trees(self, max_display_steps: int = 5, option_type: OptionType = OptionType.CALL) -> tuple[list[list[float]], list[list[float]]]:
        """
        Generates stock and option price trees for visualization purposes.
        """
        steps = min(self.steps, max_display_steps)
        if self.model_type == "lr":
            steps = steps if steps % 2 != 0 else steps + 1

        dt = self.time_to_maturity / steps
        discount = math.exp(-self.risk_free_rate * dt)
        growth = math.exp((self.risk_free_rate - self.dividend_yield) * dt)

        if self.model_type == "lr":
            bs = BlackScholes(self.spot, self.strike, self.time_to_maturity, self.risk_free_rate, self.volatility, self.dividend_yield)
            try:
                d1_val = bs.d1
                d2_val = bs.d2
            except ValueError:
                d1_val = d2_val = 0.0
            
            def peizer_pratt(x: float, N_steps: int) -> float:
                if abs(x) < 1e-12:
                    return 0.5
                val = (x / (N_steps + 1.0/3.0 + 0.1 / (N_steps + 1.0))) ** 2 * (N_steps + 1.0/6.0)
                return 0.5 + np.sign(x) * math.sqrt(0.25 - 0.25 * math.exp(-val))

            p_prime = peizer_pratt(d1_val, steps)
            p = peizer_pratt(d2_val, steps)
            p_prime = max(1e-12, min(1.0 - 1e-12, p_prime))
            p = max(1e-12, min(1.0 - 1e-12, p))

            u = growth * p_prime / p
            d = (growth - p * u) / (1.0 - p)
        else:
            u = math.exp(self.volatility * math.sqrt(dt))
            d = 1.0 / u
            p = (growth - d) / (u - d)

        p = max(0.0, min(1.0, p))

        # Build Stock Tree
        stock_tree = []
        for i in range(steps + 1):
            step_stock = []
            for j in range(i + 1):
                step_stock.append(self.spot * (u ** j) * (d ** (i - j)))
            stock_tree.append(step_stock)

        # Build Option Tree backwards
        option_tree = [[] for _ in range(steps + 1)]
        
        # Maturity payoffs
        for j in range(steps + 1):
            S = stock_tree[steps][j]
            if option_type == OptionType.CALL:
                val = max(S - self.strike, 0.0)
            else:
                val = max(self.strike - S, 0.0)
            option_tree[steps].append(val)

        # Backward steps
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                V_cont = discount * (p * option_tree[i+1][j+1] + (1.0 - p) * option_tree[i+1][j])
                option_tree[i].append(V_cont)

        return stock_tree, option_tree

    def get_early_exercise_boundary(self, option_type: OptionType) -> list[tuple[float, float | None]]:
        """
        Calculates the American early exercise boundary (critical stock prices) at each step.
        Returns a list of tuples: (time, critical_price) or (time, None) if early exercise is not optimal.
        """
        if self.time_to_maturity == 0.0:
            return [(0.0, self.strike)]

        N = self.steps
        if self.model_type == "lr":
            N = self.steps if self.steps % 2 != 0 else self.steps + 1

        dt = self.time_to_maturity / N
        discount = math.exp(-self.risk_free_rate * dt)
        growth = math.exp((self.risk_free_rate - self.dividend_yield) * dt)

        if self.model_type == "lr":
            bs = BlackScholes(self.spot, self.strike, self.time_to_maturity, self.risk_free_rate, self.volatility, self.dividend_yield)
            try:
                d1_val = bs.d1
                d2_val = bs.d2
            except ValueError:
                d1_val = d2_val = 0.0
            
            def peizer_pratt(x: float, N_steps: int) -> float:
                if abs(x) < 1e-12:
                    return 0.5
                val = (x / (N_steps + 1.0/3.0 + 0.1 / (N_steps + 1.0))) ** 2 * (N_steps + 1.0/6.0)
                return 0.5 + np.sign(x) * math.sqrt(0.25 - 0.25 * math.exp(-val))

            p_prime = peizer_pratt(d1_val, N)
            p = peizer_pratt(d2_val, N)
            p_prime = max(1e-12, min(1.0 - 1e-12, p_prime))
            p = max(1e-12, min(1.0 - 1e-12, p))

            u = growth * p_prime / p
            d = (growth - p * u) / (1.0 - p)
        else:
            u = math.exp(self.volatility * math.sqrt(dt))
            d = 1.0 / u
            p = (growth - d) / (u - d)

        p = max(0.0, min(1.0, p))

        # Generate stock price tree
        stock_tree = []
        for i in range(N + 1):
            j_range = np.arange(i + 1)
            S_i = self.spot * (u ** j_range) * (d ** (i - j_range))
            stock_tree.append(S_i)

        # Generate option value tree backward
        option_tree = [None] * (N + 1)
        S_T = stock_tree[N]
        if option_type == OptionType.CALL:
            option_tree[N] = np.maximum(S_T - self.strike, 0.0)
        else:
            option_tree[N] = np.maximum(self.strike - S_T, 0.0)

        for i in range(N - 1, -1, -1):
            V_cont = discount * (p * option_tree[i+1][1:] + (1.0 - p) * option_tree[i+1][:-1])
            S_i = stock_tree[i]
            if option_type == OptionType.CALL:
                payoff = np.maximum(S_i - self.strike, 0.0)
            else:
                payoff = np.maximum(self.strike - S_i, 0.0)
            option_tree[i] = np.maximum(payoff, V_cont)

        # Find early exercise boundary
        boundary = []
        for i in range(N):
            t = i * dt
            S_i = stock_tree[i]
            V_cont_i = discount * (p * option_tree[i+1][1:] + (1.0 - p) * option_tree[i+1][:-1])
            
            critical_price = None
            if option_type == OptionType.PUT:
                # Put early exercise occurs for low spot prices
                exercise_indices = np.where(self.strike - S_i > 0.0)[0]
                for j in exercise_indices:
                    payoff = self.strike - S_i[j]
                    if payoff > V_cont_i[j] + 1e-9:
                        critical_price = S_i[j]
            else:
                # Call early exercise occurs for high spot prices (under q > 0)
                exercise_indices = np.where(S_i - self.strike > 0.0)[0]
                for j in reversed(exercise_indices):
                    payoff = S_i[j] - self.strike
                    if payoff > V_cont_i[j] + 1e-9:
                        critical_price = S_i[j]

            boundary.append((t, critical_price))
            
        return boundary

    def compare_convergence(
        self,
        option_type: OptionType,
        american: bool = False,
        max_steps: int = 150,
        step_increment: int = 5
    ) -> dict[str, list]:
        """
        Compares binomial tree pricing convergence against the analytical Black-Scholes price.
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

        steps_list = list(range(1, max_steps + 1, step_increment))
        binomial_prices = []
        errors = []

        for N in steps_list:
            tree = BinomialTree(
                spot=self.spot,
                strike=self.strike,
                time_to_maturity=self.time_to_maturity,
                risk_free_rate=self.risk_free_rate,
                volatility=self.volatility,
                dividend_yield=self.dividend_yield,
                steps=N,
                model_type=self.model_type
            )
            bin_price = tree.price(option_type, american=american)
            binomial_prices.append(bin_price)
            errors.append(abs(bin_price - bs_price))

        return {
            "steps": steps_list,
            "binomial_prices": binomial_prices,
            "bs_prices": [bs_price] * len(steps_list),
            "errors": errors
        }
