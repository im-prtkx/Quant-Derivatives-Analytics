import math
import unittest
from src.pricing.black_scholes import BlackScholes, OptionType

class TestBlackScholes(unittest.TestCase):
    """Unit tests for the BlackScholes pricing class."""

    def setUp(self):
        # Base parameters for tests
        self.spot = 100.0
        self.strike = 100.0
        self.time_to_maturity = 1.0
        self.risk_free_rate = 0.05
        self.volatility = 0.20
        self.dividend_yield = 0.00

    def test_atm_option_pricing_without_dividends(self):
        """Test At-The-Money option pricing without dividends against known analytical values."""
        bs = BlackScholes(
            spot=self.spot,
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield
        )
        
        # d1 and d2 mathematical values
        # d1 = (ln(1) + (0.05 + 0.02) * 1) / (0.2 * 1) = 0.07 / 0.20 = 0.35
        # d2 = 0.35 - 0.20 = 0.15
        self.assertAlmostEqual(bs.d1, 0.35, places=6)
        self.assertAlmostEqual(bs.d2, 0.15, places=6)

        # Expected analytical prices
        expected_call = 10.450585
        expected_put = 5.573526
        
        self.assertAlmostEqual(bs.call_price(), expected_call, places=5)
        self.assertAlmostEqual(bs.put_price(), expected_put, places=5)
        
        # Test generic price method
        self.assertAlmostEqual(bs.price(OptionType.CALL), expected_call, places=5)
        self.assertAlmostEqual(bs.price(OptionType.PUT), expected_put, places=5)
        self.assertAlmostEqual(bs.price("call"), expected_call, places=5)
        self.assertAlmostEqual(bs.price("put"), expected_put, places=5)

    def test_option_pricing_with_dividends(self):
        """Test option pricing on dividend-paying assets against known analytical benchmarks."""
        # Benchmark: Spot=100, Strike=95, T=0.5, r=0.08, sigma=0.30, q=0.03
        bs = BlackScholes(
            spot=100.0,
            strike=95.0,
            time_to_maturity=0.5,
            risk_free_rate=0.08,
            volatility=0.30,
            dividend_yield=0.03
        )
        
        expected_call = 12.144378
        expected_put = 4.908181
        
        self.assertAlmostEqual(bs.call_price(), expected_call, places=5)
        self.assertAlmostEqual(bs.put_price(), expected_put, places=5)

    def test_zero_time_to_maturity(self):
        """Test pricing at expiration (T = 0) returns intrinsic option values."""
        # ITM Call, OTM Put
        bs_itm_call = BlackScholes(spot=110.0, strike=100.0, time_to_maturity=0.0, risk_free_rate=0.05, volatility=0.20)
        self.assertEqual(bs_itm_call.call_price(), 10.0)
        self.assertEqual(bs_itm_call.put_price(), 0.0)

        # OTM Call, ITM Put
        bs_itm_put = BlackScholes(spot=90.0, strike=100.0, time_to_maturity=0.0, risk_free_rate=0.05, volatility=0.20)
        self.assertEqual(bs_itm_put.call_price(), 0.0)
        self.assertEqual(bs_itm_put.put_price(), 10.0)
        
        # Accessing d1/d2 at T=0 should raise ValueError
        with self.assertRaises(ValueError):
            _ = bs_itm_call.d1
        with self.assertRaises(ValueError):
            _ = bs_itm_call.d2

    def test_zero_spot_price(self):
        """Test boundary condition where spot price is 0."""
        bs = BlackScholes(spot=0.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.20)
        
        # If asset price is 0, Call option is worthless, and Put option price equals discounted strike
        expected_put = 100.0 * math.exp(-0.05 * 1.0)
        self.assertEqual(bs.call_price(), 0.0)
        self.assertAlmostEqual(bs.put_price(), expected_put, places=6)

    def test_input_validation(self):
        """Verify that constructor enforces parameter bounds and raises ValueError."""
        # Spot cannot be negative
        with self.assertRaises(ValueError):
            BlackScholes(spot=-1.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.20)
            
        # Strike must be strictly positive
        with self.assertRaises(ValueError):
            BlackScholes(spot=100.0, strike=0.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.20)
        with self.assertRaises(ValueError):
            BlackScholes(spot=100.0, strike=-10.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.20)
            
        # Time to maturity cannot be negative
        with self.assertRaises(ValueError):
            BlackScholes(spot=100.0, strike=100.0, time_to_maturity=-0.5, risk_free_rate=0.05, volatility=0.20)
            
        # Volatility must be strictly positive
        with self.assertRaises(ValueError):
            BlackScholes(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.0)
        with self.assertRaises(ValueError):
            BlackScholes(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=-0.1)

    def test_put_call_parity(self):
        """Verify that put-call parity holds: C - P = S * e^(-q * T) - K * e^(-r * T)."""
        spots = [80.0, 100.0, 120.0]
        strikes = [90.0, 100.0, 110.0]
        times = [0.1, 0.5, 2.0]
        rates = [-0.01, 0.0, 0.05]  # Supports negative interest rates
        vols = [0.10, 0.30, 0.50]
        dividends = [0.0, 0.02, 0.05]

        for s in spots:
            for k in strikes:
                for t in times:
                    for r in rates:
                        for vol in vols:
                            for q in dividends:
                                bs = BlackScholes(
                                    spot=s,
                                    strike=k,
                                    time_to_maturity=t,
                                    risk_free_rate=r,
                                    volatility=vol,
                                    dividend_yield=q
                                )
                                call = bs.call_price()
                                put = bs.put_price()
                                
                                lhs = call - put
                                rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
                                
                                self.assertAlmostEqual(lhs, rhs, places=6)

if __name__ == '__main__':
    unittest.main()
