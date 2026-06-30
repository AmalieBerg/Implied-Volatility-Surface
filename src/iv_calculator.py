"""
Implied Volatility Calculator using Newton-Raphson method.

This module provides efficient implied volatility calculation from market option prices
using Newton's method with Vega as the derivative. Includes robust convergence handling
and bounds checking.
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple


class ImpliedVolatilityCalculator:
    """
    Calculate implied volatility from option prices using Newton-Raphson method.
    
    Features:
    - Fast convergence (typically 3-5 iterations)
    - Robust error handling for edge cases
    - Support for both calls and puts
    - Configurable tolerance and iteration limits
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize the IV calculator.
        
        Args:
            max_iterations: Maximum number of Newton iterations
            tolerance: Convergence tolerance for price difference
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate option Vega (sensitivity to volatility).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Vega value
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return vega
    
    def calculate_iv(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        initial_guess: Optional[float] = None
    ) -> Tuple[float, int, bool]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market option price
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            initial_guess: Starting volatility guess (default: 0.3)
            
        Returns:
            Tuple of (implied_vol, iterations, converged)
        """
        # Handle edge cases
        if T <= 0:
            return np.nan, 0, False
        
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        if market_price <= intrinsic:
            return np.nan, 0, False
        
        # Initial guess using Brenner-Subrahmanyam approximation
        if initial_guess is None:
            initial_guess = np.sqrt(2 * np.pi / T) * (market_price / S)
            initial_guess = np.clip(initial_guess, 0.01, 5.0)
        
        sigma = initial_guess
        
        for iteration in range(self.max_iterations):
            # Calculate price and vega at current sigma
            price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            vega_value = self.vega(S, K, T, r, sigma)
            
            # Check convergence
            price_diff = price - market_price
            if abs(price_diff) < self.tolerance:
                return sigma, iteration + 1, True
            
            # Vega check to avoid division by zero
            if vega_value < 1e-10:
                return np.nan, iteration + 1, False
            
            # Newton-Raphson update
            sigma_new = sigma - price_diff / vega_value
            
            # Ensure sigma stays positive and reasonable
            sigma_new = np.clip(sigma_new, 0.001, 5.0)
            
            # Check for stagnation
            if abs(sigma_new - sigma) < 1e-10:
                return sigma_new, iteration + 1, False
            
            sigma = sigma_new
        
        # Max iterations reached
        return sigma, self.max_iterations, False
    
    def calculate_iv_surface(
        self,
        market_prices: np.ndarray,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        option_type: str = 'call'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate IV surface for a grid of strikes and maturities.
        
        Args:
            market_prices: 2D array of market prices [maturities x strikes]
            S: Spot price
            strikes: Array of strike prices
            maturities: Array of times to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            
        Returns:
            Tuple of (iv_surface, iterations_surface, convergence_mask)
        """
        n_maturities = len(maturities)
        n_strikes = len(strikes)
        
        iv_surface = np.zeros((n_maturities, n_strikes))
        iterations_surface = np.zeros((n_maturities, n_strikes), dtype=int)
        convergence_mask = np.zeros((n_maturities, n_strikes), dtype=bool)
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                market_price = market_prices[i, j]
                
                if np.isnan(market_price) or market_price <= 0:
                    iv_surface[i, j] = np.nan
                    continue
                
                iv, iters, converged = self.calculate_iv(
                    market_price, S, K, T, r, option_type
                )
                
                iv_surface[i, j] = iv
                iterations_surface[i, j] = iters
                convergence_mask[i, j] = converged
        
        return iv_surface, iterations_surface, convergence_mask


def intrinsic_value(S: float, K: float, option_type: str = 'call') -> float:
    """Calculate intrinsic value of an option."""
    if option_type == 'call':
        return max(S - K, 0)
    else:
        return max(K - S, 0)


def time_value(option_price: float, S: float, K: float, option_type: str = 'call') -> float:
    """Calculate time value of an option."""
    return option_price - intrinsic_value(S, K, option_type)
