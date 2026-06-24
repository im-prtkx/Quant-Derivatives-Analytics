import math
import unittest
from src.pricing.black_scholes import BlackScholes, OptionType
from src.greeks.greeks import Greeks

class TestGreeks(unittest.TestCase):
    """Unit tests for the analytical Greeks calculations."""

    def setUp(self):
        # Benchmark parameters: Spot=100, Strike=100, T=1.0, r=0.05, sigma=0.20, q=0.0
        self.spot = 100.0
        self.strike = 100.0
        self.time_to_maturity = 1.0
        self.risk_free_rate = 0.05
        self.volatility = 0.20
        self.dividend_yield = 0.00
        
        self.model = BlackScholes(
            spot=self.spot,
            strike=self.strike,
            time_to_maturity=self.time_to_maturity,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield
        )
        self.greeks = Greeks(self.model)

    def test_analytical_benchmark_values(self):
        """Test Greeks against exact mathematical values from benchmark parameters."""
        # Exact values
        expected_c_delta = 0.63683065
        expected_p_delta = -0.36316935
        expected_gamma = 0.01876202
        expected_vega = 37.52403469
        expected_c_theta = -6.41402755
        expected_p_theta = -1.65788042
        expected_c_rho = 53.23248155
        expected_p_rho = -41.89046090

        self.assertAlmostEqual(self.greeks.delta_call(), expected_c_delta, places=6)
        self.assertAlmostEqual(self.greeks.delta_put(), expected_p_delta, places=6)
        self.assertAlmostEqual(self.greeks.gamma(), expected_gamma, places=6)
        self.assertAlmostEqual(self.greeks.vega(), expected_vega, places=6)
        self.assertAlmostEqual(self.greeks.theta_call(), expected_c_theta, places=6)
        self.assertAlmostEqual(self.greeks.theta_put(), expected_p_theta, places=6)
        self.assertAlmostEqual(self.greeks.rho_call(), expected_c_rho, places=6)
        self.assertAlmostEqual(self.greeks.rho_put(), expected_p_rho, places=6)
        
        # Test generic option type methods
        self.assertAlmostEqual(self.greeks.delta(OptionType.CALL), expected_c_delta, places=6)
        self.assertAlmostEqual(self.greeks.delta(OptionType.PUT), expected_p_delta, places=6)
        self.assertAlmostEqual(self.greeks.theta(OptionType.CALL), expected_c_theta, places=6)
        self.assertAlmostEqual(self.greeks.theta(OptionType.PUT), expected_p_theta, places=6)
        self.assertAlmostEqual(self.greeks.rho(OptionType.CALL), expected_c_rho, places=6)
        self.assertAlmostEqual(self.greeks.rho(OptionType.PUT), expected_p_rho, places=6)

    def test_greek_bounds_and_signs(self):
        """Verify Delta ranges, and check that Gamma and Vega are strictly positive."""
        spots = [80.0, 100.0, 120.0]
        strikes = [90.0, 100.0, 110.0]
        dividends = [0.0, 0.03, 0.07]
        
        for s in spots:
            for k in strikes:
                for q in dividends:
                    model = BlackScholes(spot=s, strike=k, time_to_maturity=0.5, risk_free_rate=0.05, volatility=0.25, dividend_yield=q)
                    g = Greeks(model)
                    
                    df = math.exp(-q * 0.5)
                    
                    # Delta bounds: Call in [0, e^(-qT)], Put in [-e^(-qT), 0]
                    self.assertTrue(0.0 <= g.delta_call() <= df)
                    self.assertTrue(-df <= g.delta_put() <= 0.0)
                    
                    # Gamma & Vega must be positive
                    self.assertTrue(g.gamma() > 0.0)
                    self.assertTrue(g.vega() > 0.0)

    def test_put_call_greek_relationships(self):
        """Verify standard Put-Call Greek relationships hold under various conditions."""
        spots = [90.0, 100.0, 110.0]
        strikes = [90.0, 100.0, 110.0]
        times = [0.25, 1.0, 2.0]
        rates = [0.01, 0.05, 0.08]
        vols = [0.15, 0.25, 0.40]
        dividends = [0.0, 0.02, 0.05]

        for s in spots:
            for k in strikes:
                for t in times:
                    for r in rates:
                        for vol in vols:
                            for q in dividends:
                                model = BlackScholes(spot=s, strike=k, time_to_maturity=t, risk_free_rate=r, volatility=vol, dividend_yield=q)
                                g = Greeks(model)
                                
                                # 1. Delta: Call Delta - Put Delta = e^(-q * T)
                                self.assertAlmostEqual(g.delta_call() - g.delta_put(), math.exp(-q * t), places=6)
                                
                                # 2. Rho: Call Rho - Put Rho = K * T * e^(-r * T)
                                self.assertAlmostEqual(g.rho_call() - g.rho_put(), k * t * math.exp(-r * t), places=6)
                                
                                # 3. Theta: Call Theta - Put Theta = q * S * e^(-q * T) - r * K * e^(-r * T)
                                expected_diff = q * s * math.exp(-q * t) - r * k * math.exp(-r * t)
                                self.assertAlmostEqual(g.theta_call() - g.theta_put(), expected_diff, places=6)

    def test_near_expiry_edge_cases(self):
        """Test Greeks behavior when time_to_maturity approaches zero."""
        # Deep ITM Call (S = 110, K = 100) -> Delta=1, Gamma=0, Vega=0, Theta=q*S - r*K, Rho=0
        model_itm_call = BlackScholes(spot=110.0, strike=100.0, time_to_maturity=1e-13, risk_free_rate=0.05, volatility=0.20, dividend_yield=0.02)
        g_itm_call = Greeks(model_itm_call)
        
        self.assertAlmostEqual(g_itm_call.delta_call(), math.exp(-0.02 * 1e-13), places=10)
        self.assertAlmostEqual(g_itm_call.delta_put(), 0.0, places=10)
        self.assertEqual(g_itm_call.gamma(), 0.0)
        self.assertEqual(g_itm_call.vega(), 0.0)
        self.assertAlmostEqual(g_itm_call.theta_call(), 0.02 * 110.0 - 0.05 * 100.0, places=5)
        self.assertEqual(g_itm_call.rho_call(), 0.0)

        # Deep OTM Call (S = 90, K = 100) -> Delta=0, Gamma=0, Vega=0, Theta=0, Rho=0
        model_otm_call = BlackScholes(spot=90.0, strike=100.0, time_to_maturity=1e-13, risk_free_rate=0.05, volatility=0.20, dividend_yield=0.02)
        g_otm_call = Greeks(model_otm_call)
        
        self.assertEqual(g_otm_call.delta_call(), 0.0)
        self.assertAlmostEqual(g_otm_call.delta_put(), -math.exp(-0.02 * 1e-13), places=10)
        self.assertEqual(g_otm_call.gamma(), 0.0)
        self.assertEqual(g_otm_call.vega(), 0.0)
        self.assertEqual(g_otm_call.theta_call(), 0.0)
        self.assertEqual(g_itm_call.rho_call(), 0.0)

        # ATM Option (S = 100, K = 100) -> Delta=0.5, Gamma=inf, Theta=-inf
        model_atm = BlackScholes(spot=100.0, strike=100.0, time_to_maturity=1e-13, risk_free_rate=0.05, volatility=0.20, dividend_yield=0.00)
        g_atm = Greeks(model_atm)
        
        self.assertAlmostEqual(g_atm.delta_call(), 0.5, places=5)
        self.assertAlmostEqual(g_atm.delta_put(), -0.5, places=5)
        self.assertEqual(g_atm.gamma(), float('inf'))
        self.assertEqual(g_atm.theta_call(), -float('inf'))
        self.assertEqual(g_atm.theta_put(), -float('inf'))

    def test_very_low_volatility_and_deep_options(self):
        """Test numerical stability under very low volatility and deep ITM/OTM conditions."""
        # Volatility = 1e-5 (extremely low but positive)
        model_low_vol = BlackScholes(spot=100.0, strike=95.0, time_to_maturity=0.5, risk_free_rate=0.06, volatility=1e-5, dividend_yield=0.00)
        g_low_vol = Greeks(model_low_vol)
        
        # S=100 > K=95, so Delta call should be ~1
        self.assertAlmostEqual(g_low_vol.delta_call(), 1.0, places=6)
        self.assertAlmostEqual(g_low_vol.delta_put(), 0.0, places=6)
        self.assertAlmostEqual(g_low_vol.gamma(), 0.0, places=6)
        self.assertAlmostEqual(g_low_vol.vega(), 0.0, places=6)
        self.assertAlmostEqual(g_low_vol.theta_call(), -0.06 * 95.0 * math.exp(-0.06 * 0.5), places=4)
        
        # Deep OTM (S=100, K=500) -> Delta ~ 0, Gamma ~ 0, Vega ~ 0
        model_deep_otm = BlackScholes(spot=100.0, strike=500.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.20, dividend_yield=0.0)
        g_deep_otm = Greeks(model_deep_otm)
        self.assertAlmostEqual(g_deep_otm.delta_call(), 0.0, places=6)
        self.assertAlmostEqual(g_deep_otm.gamma(), 0.0, places=6)
        self.assertAlmostEqual(g_deep_otm.vega(), 0.0, places=6)

    def test_higher_order_greeks(self):
        """Test calculation of higher-order Greeks (Vanna, Volga, Charm, Speed, Color)."""
        model = BlackScholes(spot=100.0, strike=100.0, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0.20)
        g = Greeks(model)
        
        self.assertNotEqual(g.vanna(), 0.0)
        self.assertNotEqual(g.volga(), 0.0)
        self.assertNotEqual(g.charm_call(), 0.0)
        self.assertNotEqual(g.charm_put(), 0.0)
        self.assertNotEqual(g.speed(), 0.0)
        self.assertNotEqual(g.color(), 0.0)

if __name__ == '__main__':
    unittest.main()
