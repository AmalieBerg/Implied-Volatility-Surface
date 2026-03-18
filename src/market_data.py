"""
Market Data Generator for Testing.

This module generates realistic synthetic market option prices for testing
the IV surface builder, including volatility smiles and term structures.
"""

import numpy as np
from typing import Tuple, Optional
from src.iv_calculator import ImpliedVolatilityCalculator


class MarketDataGenerator:
    """
    Generate synthetic market data with realistic volatility patterns.
    
    Features:
    - Volatility smile (skew) modeling
    - Term structure patterns
    - Configurable noise levels
    - Support for different market regimes
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.iv_calc = ImpliedVolatilityCalculator()
    
    def generate_iv_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        spot: float,
        atm_vol: float = 0.25,
        skew: float = -0.1,
        term_structure_slope: float = 0.05,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """
        Generate implied volatility surface with smile and term structure.
        
        IV model:
        σ(K, T) = σ_ATM(T) + skew * log(K/S) + noise
        σ_ATM(T) = atm_vol + term_structure_slope * sqrt(T)
        
        Args:
            strikes: Strike prices
            maturities: Times to maturity
            spot: Spot price
            atm_vol: ATM volatility level
            skew: Volatility skew parameter (negative = put skew)
            term_structure_slope: Term structure slope
            noise_level: Random noise standard deviation
            
        Returns:
            IV surface [maturities x strikes]
        """
        n_maturities = len(maturities)
        n_strikes = len(strikes)
        
        iv_surface = np.zeros((n_maturities, n_strikes))
        
        for i, T in enumerate(maturities):
            # ATM vol with term structure
            atm_vol_T = atm_vol + term_structure_slope * np.sqrt(T)
            
            for j, K in enumerate(strikes):
                # Log-moneyness
                log_moneyness = np.log(K / spot)
                
                # Volatility smile
                iv = atm_vol_T + skew * log_moneyness
                
                # Add curvature (smile effect)
                iv += 0.5 * abs(skew) * log_moneyness**2
                
                # Add noise
                noise = np.random.normal(0, noise_level)
                iv += noise
                
                # Ensure positive
                iv = max(iv, 0.05)
                
                iv_surface[i, j] = iv
        
        return iv_surface
    
    def generate_market_prices(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        spot: float,
        rate: float,
        iv_surface: np.ndarray,
        option_type: str = 'call',
        bid_ask_spread: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate market option prices from IV surface.
        
        Args:
            strikes: Strike prices
            maturities: Times to maturity
            spot: Spot price
            rate: Risk-free rate
            iv_surface: Implied volatility surface
            option_type: 'call' or 'put'
            bid_ask_spread: Bid-ask spread as fraction of mid price
            
        Returns:
            Tuple of (mid_prices, bid_prices, ask_prices)
        """
        n_maturities = len(maturities)
        n_strikes = len(strikes)
        
        mid_prices = np.zeros((n_maturities, n_strikes))
        bid_prices = np.zeros((n_maturities, n_strikes))
        ask_prices = np.zeros((n_maturities, n_strikes))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                sigma = iv_surface[i, j]
                
                # Calculate theoretical price
                mid = self.iv_calc.black_scholes_price(
                    spot, K, T, rate, sigma, option_type
                )
                
                # Add bid-ask spread
                spread = mid * bid_ask_spread
                
                mid_prices[i, j] = mid
                bid_prices[i, j] = mid - spread / 2
                ask_prices[i, j] = mid + spread / 2
        
        return mid_prices, bid_prices, ask_prices
    
    def generate_realistic_data(
        self,
        spot: float = 100.0,
        rate: float = 0.05,
        n_strikes: int = 15,
        n_maturities: int = 8,
        market_regime: str = 'normal'
    ) -> dict:
        """
        Generate complete realistic market dataset.
        
        Args:
            spot: Spot price
            rate: Risk-free rate
            n_strikes: Number of strike prices
            n_maturities: Number of maturities
            market_regime: 'normal', 'high_vol', or 'crash'
            
        Returns:
            Dictionary with market data
        """
        # Define market regime parameters
        regime_params = {
            'normal': {
                'atm_vol': 0.20,
                'skew': -0.10,
                'term_slope': 0.03,
                'noise': 0.005
            },
            'high_vol': {
                'atm_vol': 0.40,
                'skew': -0.15,
                'term_slope': -0.05,  # Inverted term structure
                'noise': 0.015
            },
            'crash': {
                'atm_vol': 0.60,
                'skew': -0.30,  # Steep put skew
                'term_slope': -0.10,
                'noise': 0.025
            }
        }
        
        params = regime_params.get(market_regime, regime_params['normal'])
        
        # Generate strike grid (centered around spot)
        strike_range = spot * 0.4  # ±40% from spot
        strikes = np.linspace(spot - strike_range, spot + strike_range, n_strikes)
        
        # Generate maturity grid
        maturities = np.linspace(0.1, 2.0, n_maturities)  # 1 month to 2 years
        
        # Generate IV surface
        iv_surface = self.generate_iv_surface(
            strikes,
            maturities,
            spot,
            atm_vol=params['atm_vol'],
            skew=params['skew'],
            term_structure_slope=params['term_slope'],
            noise_level=params['noise']
        )
        
        # Generate option prices
        call_prices, call_bids, call_asks = self.generate_market_prices(
            strikes, maturities, spot, rate, iv_surface, 'call'
        )
        
        put_prices, put_bids, put_asks = self.generate_market_prices(
            strikes, maturities, spot, rate, iv_surface, 'put'
        )
        
        return {
            'spot': spot,
            'rate': rate,
            'strikes': strikes,
            'maturities': maturities,
            'iv_surface_true': iv_surface,
            'call_prices': call_prices,
            'call_bids': call_bids,
            'call_asks': call_asks,
            'put_prices': put_prices,
            'put_bids': put_bids,
            'put_asks': put_asks,
            'market_regime': market_regime
        }
    
    def add_sparse_data(
        self,
        data: dict,
        missing_fraction: float = 0.2
    ) -> dict:
        """
        Make data sparse by randomly removing some market quotes.
        
        Args:
            data: Market data dictionary
            missing_fraction: Fraction of data to remove
            
        Returns:
            Modified data dictionary with missing values
        """
        n_maturities, n_strikes = data['call_prices'].shape
        n_total = n_maturities * n_strikes
        n_missing = int(n_total * missing_fraction)
        
        # Randomly select indices to remove
        all_indices = np.arange(n_total)
        missing_indices = np.random.choice(all_indices, n_missing, replace=False)
        
        # Convert to 2D indices
        missing_i = missing_indices // n_strikes
        missing_j = missing_indices % n_strikes
        
        # Set to NaN
        data['call_prices'][missing_i, missing_j] = np.nan
        data['put_prices'][missing_i, missing_j] = np.nan
        
        return data
