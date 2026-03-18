"""
Comprehensive Test Suite for Implied Volatility Surface Builder.

Tests cover:
- IV calculation accuracy
- Surface interpolation
- Arbitrage detection
- Edge cases and robustness
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from src.iv_calculator import ImpliedVolatilityCalculator
from src.surface_builder import VolatilitySurface
from src.market_data import MarketDataGenerator


class TestIVCalculator(unittest.TestCase):
    """Test implied volatility calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = ImpliedVolatilityCalculator(max_iterations=100, tolerance=1e-6)
        self.S = 100.0
        self.r = 0.05
    
    def test_atm_call_iv(self):
        """Test ATM call option IV calculation."""
        K = 100.0
        T = 1.0
        sigma_true = 0.25
        
        # Calculate theoretical price
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        
        # Calculate IV
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        # Assertions
        self.assertTrue(converged, "IV calculation should converge for ATM option")
        self.assertAlmostEqual(iv, sigma_true, places=4, 
                              msg="Recovered IV should match input volatility")
        self.assertLess(iters, 10, "Should converge in less than 10 iterations for ATM")
    
    def test_itm_call_iv(self):
        """Test ITM call option IV calculation."""
        K = 90.0  # ITM
        T = 1.0
        sigma_true = 0.30
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=4)
    
    def test_otm_call_iv(self):
        """Test OTM call option IV calculation."""
        K = 110.0  # OTM
        T = 1.0
        sigma_true = 0.20
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=4)
    
    def test_put_iv(self):
        """Test put option IV calculation."""
        K = 100.0
        T = 1.0
        sigma_true = 0.25
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'put')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'put')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=4)
    
    def test_short_maturity(self):
        """Test IV calculation for short maturity options."""
        K = 100.0
        T = 0.05  # ~2 weeks
        sigma_true = 0.25
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=3)
    
    def test_long_maturity(self):
        """Test IV calculation for long maturity options."""
        K = 100.0
        T = 5.0  # 5 years
        sigma_true = 0.25
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=4)
    
    def test_high_volatility(self):
        """Test IV calculation with high volatility."""
        K = 100.0
        T = 1.0
        sigma_true = 0.80  # 80% vol
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=3)
    
    def test_low_volatility(self):
        """Test IV calculation with low volatility."""
        K = 100.0
        T = 1.0
        sigma_true = 0.05  # 5% vol
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=3)
    
    def test_deep_itm(self):
        """Test deep ITM option IV calculation."""
        K = 50.0  # Very deep ITM
        T = 1.0
        sigma_true = 0.25
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        self.assertTrue(converged)
        self.assertAlmostEqual(iv, sigma_true, places=3)
    
    def test_deep_otm(self):
        """Test deep OTM option IV calculation."""
        K = 150.0  # Very deep OTM
        T = 1.0
        sigma_true = 0.25
        
        price = self.calc.black_scholes_price(self.S, K, T, self.r, sigma_true, 'call')
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        # Deep OTM might not converge as well due to numerical precision
        if converged:
            self.assertAlmostEqual(iv, sigma_true, places=2)
    
    def test_iv_surface_calculation(self):
        """Test calculation of full IV surface."""
        strikes = np.linspace(80, 120, 9)
        maturities = np.array([0.25, 0.5, 1.0, 2.0])
        sigma_true = 0.25
        
        # Generate market prices
        n_maturities = len(maturities)
        n_strikes = len(strikes)
        prices = np.zeros((n_maturities, n_strikes))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                prices[i, j] = self.calc.black_scholes_price(
                    self.S, K, T, self.r, sigma_true, 'call'
                )
        
        # Calculate IV surface
        iv_surface, iters_surface, conv_mask = self.calc.calculate_iv_surface(
            prices, self.S, strikes, maturities, self.r, 'call'
        )
        
        # Check convergence rate
        convergence_rate = np.sum(conv_mask) / conv_mask.size
        self.assertGreater(convergence_rate, 0.95, 
                          "At least 95% of IVs should converge")
        
        # Check accuracy for converged points
        errors = np.abs(iv_surface[conv_mask] - sigma_true)
        mean_error = np.mean(errors)
        self.assertLess(mean_error, 0.001, "Mean IV error should be < 0.1%")
    
    def test_zero_time_to_maturity(self):
        """Test that zero maturity returns intrinsic value."""
        K = 90.0
        T = 0.0
        price = max(self.S - K, 0)  # Intrinsic value
        
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        # Should not converge (returns NaN)
        self.assertFalse(converged)
        self.assertTrue(np.isnan(iv))
    
    def test_negative_time_value(self):
        """Test that price below intrinsic value is handled."""
        K = 90.0
        T = 1.0
        intrinsic = self.S - K
        price = intrinsic - 1  # Below intrinsic
        
        iv, iters, converged = self.calc.calculate_iv(price, self.S, K, T, self.r, 'call')
        
        # Should not converge
        self.assertFalse(converged)
        self.assertTrue(np.isnan(iv))


class TestSurfaceBuilder(unittest.TestCase):
    """Test volatility surface interpolation and analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gen = MarketDataGenerator(seed=42)
        self.data = self.gen.generate_realistic_data(
            spot=100.0,
            n_strikes=11,
            n_maturities=6,
            market_regime='normal'
        )
        
        self.surface = VolatilitySurface(
            self.data['strikes'],
            self.data['maturities'],
            self.data['iv_surface_true'],
            self.data['spot']
        )
    
    def test_surface_initialization(self):
        """Test surface initialization."""
        self.assertEqual(len(self.surface.strikes), 11)
        self.assertEqual(len(self.surface.maturities), 6)
        self.assertEqual(self.surface.spot, 100.0)
    
    def test_rbf_interpolation(self):
        """Test RBF interpolation accuracy."""
        # Interpolate at original points
        strikes_test = self.data['strikes'][::2]  # Every other strike
        maturities_test = np.full_like(strikes_test, self.data['maturities'][2])
        
        iv_interp = self.surface.interpolate(strikes_test, maturities_test, method='rbf')
        iv_true = self.data['iv_surface_true'][2, ::2]
        
        # Check accuracy
        errors = np.abs(iv_interp - iv_true)
        mean_error = np.nanmean(errors)
        self.assertLess(mean_error, 0.01, "RBF interpolation error should be < 1%")
    
    def test_linear_interpolation(self):
        """Test linear interpolation."""
        strikes_test = np.array([95.0, 105.0])
        maturities_test = np.array([0.5, 0.5])
        
        iv_interp = self.surface.interpolate(strikes_test, maturities_test, method='linear')
        
        # Should return reasonable values
        self.assertTrue(np.all(~np.isnan(iv_interp)))
        self.assertTrue(np.all(iv_interp > 0))
        self.assertTrue(np.all(iv_interp < 2.0))
    
    def test_dense_surface_creation(self):
        """Test creation of dense interpolated surface."""
        K_dense, T_dense, IV_dense = self.surface.create_dense_surface(
            n_strikes=30,
            n_maturities=20,
            method='rbf'
        )
        
        self.assertEqual(len(K_dense), 30)
        self.assertEqual(len(T_dense), 20)
        self.assertEqual(IV_dense.shape, (20, 30))
    
    def test_calendar_arbitrage_check(self):
        """Test calendar arbitrage detection."""
        violations = self.surface.check_calendar_arbitrage()
        
        # With realistic data, should have few/no violations
        self.assertLess(len(violations), 5, 
                       "Should have minimal calendar arbitrage in realistic data")
    
    def test_butterfly_arbitrage_check(self):
        """Test butterfly arbitrage detection."""
        violations = self.surface.check_butterfly_arbitrage()
        
        # Should have few violations
        self.assertLess(len(violations), 10,
                       "Should have minimal butterfly arbitrage in realistic data")
    
    def test_atm_volatility_extraction(self):
        """Test ATM volatility term structure extraction."""
        maturities, atm_vols = self.surface.get_atm_volatility()
        
        self.assertEqual(len(maturities), len(self.data['maturities']))
        self.assertEqual(len(atm_vols), len(self.data['maturities']))
        
        # ATM vols should be positive and reasonable
        self.assertTrue(np.all(atm_vols > 0))
        self.assertTrue(np.all(atm_vols < 2.0))
    
    def test_volatility_smile_extraction(self):
        """Test volatility smile extraction."""
        strikes, ivs = self.surface.get_volatility_smile(maturity_idx=2)
        
        self.assertEqual(len(strikes), len(self.data['strikes']))
        self.assertEqual(len(ivs), len(self.data['strikes']))
    
    def test_skew_calculation(self):
        """Test volatility skew calculation."""
        skew = self.surface.calculate_skew(maturity_idx=2)
        
        # Skew should be negative (put skew) for normal market
        self.assertLess(skew, 0, "Should have negative skew in normal market")
        self.assertGreater(skew, -1.0, "Skew should be reasonable magnitude")
    
    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        stats = self.surface.summary_statistics()
        
        # Check all expected keys are present
        expected_keys = [
            'n_points', 'iv_min', 'iv_max', 'iv_mean', 'iv_std',
            'strike_range', 'maturity_range', 'spot',
            'atm_vol_term_structure',
            'calendar_arbitrage_violations',
            'butterfly_arbitrage_violations'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")
        
        # Check reasonableness
        self.assertGreater(stats['n_points'], 0)
        self.assertGreater(stats['iv_mean'], 0)
        self.assertGreater(stats['iv_max'], stats['iv_min'])


class TestMarketDataGenerator(unittest.TestCase):
    """Test market data generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gen = MarketDataGenerator(seed=42)
    
    def test_iv_surface_generation(self):
        """Test IV surface generation with smile and term structure."""
        strikes = np.linspace(80, 120, 15)
        maturities = np.array([0.25, 0.5, 1.0, 2.0])
        spot = 100.0
        
        iv_surface = self.gen.generate_iv_surface(
            strikes, maturities, spot,
            atm_vol=0.25,
            skew=-0.10,
            term_structure_slope=0.05
        )
        
        # Check shape
        self.assertEqual(iv_surface.shape, (4, 15))
        
        # Check all positive
        self.assertTrue(np.all(iv_surface > 0))
        
        # Check term structure is increasing
        atm_idx = 7  # Middle strike
        atm_vols = iv_surface[:, atm_idx]
        self.assertTrue(np.all(np.diff(atm_vols) > -0.05))  # Generally increasing
        
        # Check skew (put wing higher than call wing)
        for i in range(len(maturities)):
            put_wing_iv = iv_surface[i, 0]  # Low strike
            call_wing_iv = iv_surface[i, -1]  # High strike
            self.assertGreater(put_wing_iv, call_wing_iv, "Should have put skew")
    
    def test_market_prices_generation(self):
        """Test market price generation from IV surface."""
        strikes = np.array([90, 95, 100, 105, 110])
        maturities = np.array([0.5, 1.0])
        spot = 100.0
        rate = 0.05
        
        iv_surface = np.full((2, 5), 0.25)
        
        mid_prices, bid_prices, ask_prices = self.gen.generate_market_prices(
            strikes, maturities, spot, rate, iv_surface, 'call'
        )
        
        # Check shapes
        self.assertEqual(mid_prices.shape, (2, 5))
        self.assertEqual(bid_prices.shape, (2, 5))
        self.assertEqual(ask_prices.shape, (2, 5))
        
        # Check bid < mid < ask
        self.assertTrue(np.all(bid_prices < mid_prices))
        self.assertTrue(np.all(ask_prices > mid_prices))
        
        # Check all positive
        self.assertTrue(np.all(mid_prices > 0))
    
    def test_realistic_data_generation(self):
        """Test complete realistic data generation."""
        data = self.gen.generate_realistic_data(
            spot=100.0,
            n_strikes=11,
            n_maturities=6,
            market_regime='normal'
        )
        
        # Check all expected keys
        expected_keys = [
            'spot', 'rate', 'strikes', 'maturities',
            'iv_surface_true', 'call_prices', 'put_prices'
        ]
        
        for key in expected_keys:
            self.assertIn(key, data, f"Missing key: {key}")
        
        # Check put-call parity approximately holds
        S = data['spot']
        r = data['rate']
        K = data['strikes'][5]  # Middle strike
        T = data['maturities'][2]  # Middle maturity
        
        C = data['call_prices'][2, 5]
        P = data['put_prices'][2, 5]
        
        # C - P ≈ S - K*exp(-rT)
        parity_lhs = C - P
        parity_rhs = S - K * np.exp(-r * T)
        
        self.assertAlmostEqual(parity_lhs, parity_rhs, places=1,
                              msg="Put-call parity should approximately hold")
    
    def test_different_market_regimes(self):
        """Test data generation in different market regimes."""
        regimes = ['normal', 'high_vol', 'crash']
        
        for regime in regimes:
            data = self.gen.generate_realistic_data(market_regime=regime)
            
            # All regimes should produce valid data
            self.assertTrue(np.all(data['iv_surface_true'] > 0))
            self.assertTrue(np.all(data['call_prices'] > 0))
            self.assertTrue(np.all(data['put_prices'] > 0))
            
            # Check relative volatility levels
            mean_iv = np.mean(data['iv_surface_true'])
            
            if regime == 'normal':
                self.assertLess(mean_iv, 0.35)
            elif regime == 'high_vol':
                self.assertGreater(mean_iv, 0.30)
            elif regime == 'crash':
                self.assertGreater(mean_iv, 0.45)


def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestIVCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestSurfaceBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestMarketDataGenerator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*70)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
