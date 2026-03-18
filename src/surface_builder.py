"""
Volatility Surface Interpolation and Arbitrage Detection.

This module provides methods for interpolating implied volatility surfaces
and checking for arbitrage violations (calendar spreads, butterfly spreads).
"""

import numpy as np
from scipy.interpolate import RBFInterpolator, griddata
from typing import Tuple, List, Dict, Optional


class VolatilitySurface:
    """
    Volatility surface with interpolation and arbitrage checking.
    
    Features:
    - Multiple interpolation methods (RBF, linear, cubic)
    - Calendar spread arbitrage detection
    - Butterfly spread arbitrage detection
    - Surface smoothing and extrapolation
    """
    
    def __init__(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        iv_surface: np.ndarray,
        spot: float
    ):
        """
        Initialize volatility surface.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of times to maturity
            iv_surface: 2D array of implied volatilities [maturities x strikes]
            spot: Current spot price
        """
        self.strikes = np.array(strikes)
        self.maturities = np.array(maturities)
        self.iv_surface = np.array(iv_surface)
        self.spot = spot
        
        # Create moneyness grid
        self.moneyness = self.strikes / spot
        
        # Store valid data points (non-NaN)
        self.valid_mask = ~np.isnan(iv_surface)
        
        # Create flattened arrays for interpolation
        self._setup_interpolation_data()
    
    def _setup_interpolation_data(self):
        """Prepare data for interpolation."""
        # Create mesh grid
        T_grid, K_grid = np.meshgrid(self.maturities, self.strikes, indexing='ij')
        M_grid = K_grid / self.spot  # Moneyness grid
        
        # Flatten and filter valid points
        self.T_flat = T_grid[self.valid_mask]
        self.K_flat = K_grid[self.valid_mask]
        self.M_flat = M_grid[self.valid_mask]
        self.IV_flat = self.iv_surface[self.valid_mask]
        
        # Stack for interpolation (using log-moneyness and sqrt-time)
        self.X_interp = np.column_stack([
            np.log(self.M_flat),
            np.sqrt(self.T_flat)
        ])
    
    def interpolate(
        self,
        strikes_query: np.ndarray,
        maturities_query: np.ndarray,
        method: str = 'rbf'
    ) -> np.ndarray:
        """
        Interpolate IV surface at query points.
        
        Args:
            strikes_query: Query strike prices
            maturities_query: Query maturities
            method: 'rbf', 'linear', or 'cubic'
            
        Returns:
            Interpolated IV values
        """
        if len(self.IV_flat) < 4:
            raise ValueError("Insufficient valid data points for interpolation")
        
        # Create query points in transformed space
        moneyness_query = strikes_query / self.spot
        X_query = np.column_stack([
            np.log(moneyness_query),
            np.sqrt(maturities_query)
        ])
        
        if method == 'rbf':
            # Radial Basis Function interpolation (smooth)
            rbf = RBFInterpolator(
                self.X_interp,
                self.IV_flat,
                kernel='thin_plate_spline',
                smoothing=0.001
            )
            iv_interp = rbf(X_query)
        else:
            # Griddata interpolation
            iv_interp = griddata(
                self.X_interp,
                self.IV_flat,
                X_query,
                method=method,
                fill_value=np.nan
            )
        
        return iv_interp
    
    def create_dense_surface(
        self,
        n_strikes: int = 50,
        n_maturities: int = 50,
        method: str = 'rbf'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a dense interpolated surface for visualization.
        
        Args:
            n_strikes: Number of strike points
            n_maturities: Number of maturity points
            method: Interpolation method
            
        Returns:
            Tuple of (strikes_dense, maturities_dense, iv_dense)
        """
        # Create dense grids
        strike_min = self.strikes.min() * 0.8
        strike_max = self.strikes.max() * 1.2
        strikes_dense = np.linspace(strike_min, strike_max, n_strikes)
        
        mat_min = max(self.maturities.min() * 0.5, 0.01)
        mat_max = self.maturities.max() * 1.2
        maturities_dense = np.linspace(mat_min, mat_max, n_maturities)
        
        # Create mesh
        T_dense, K_dense = np.meshgrid(maturities_dense, strikes_dense, indexing='ij')
        
        # Interpolate at all points
        iv_dense = np.zeros_like(T_dense)
        for i in range(n_maturities):
            for j in range(n_strikes):
                try:
                    iv_dense[i, j] = self.interpolate(
                        np.array([K_dense[i, j]]),
                        np.array([T_dense[i, j]]),
                        method=method
                    )[0]
                except:
                    iv_dense[i, j] = np.nan
        
        return strikes_dense, maturities_dense, iv_dense
    
    def check_calendar_arbitrage(self, tolerance: float = 1e-6) -> List[Dict]:
        """
        Check for calendar spread arbitrage violations.
        
        Total variance should be increasing with maturity:
        σ²(T₁) * T₁ < σ²(T₂) * T₂ for T₁ < T₂
        
        Args:
            tolerance: Tolerance for violations
            
        Returns:
            List of violations with details
        """
        violations = []
        
        # For each strike
        for j, K in enumerate(self.strikes):
            # Get IVs at this strike across maturities
            ivs = self.iv_surface[:, j]
            
            # Check consecutive maturities
            for i in range(len(self.maturities) - 1):
                if np.isnan(ivs[i]) or np.isnan(ivs[i + 1]):
                    continue
                
                T1, T2 = self.maturities[i], self.maturities[i + 1]
                var1 = ivs[i]**2 * T1
                var2 = ivs[i + 1]**2 * T2
                
                if var2 < var1 - tolerance:
                    violations.append({
                        'type': 'calendar',
                        'strike': K,
                        'T1': T1,
                        'T2': T2,
                        'IV1': ivs[i],
                        'IV2': ivs[i + 1],
                        'total_var1': var1,
                        'total_var2': var2,
                        'violation': var1 - var2
                    })
        
        return violations
    
    def check_butterfly_arbitrage(self, tolerance: float = 1e-6) -> List[Dict]:
        """
        Check for butterfly spread arbitrage violations.
        
        The call price should be convex in strike:
        C(K₁) + C(K₃) ≥ 2*C(K₂) for K₁ < K₂ < K₃
        
        This translates to constraints on the IV surface.
        
        Args:
            tolerance: Tolerance for violations
            
        Returns:
            List of violations with details
        """
        violations = []
        
        # For each maturity
        for i, T in enumerate(self.maturities):
            ivs = self.iv_surface[i, :]
            
            # Check consecutive strikes (butterfly)
            for j in range(1, len(self.strikes) - 1):
                if np.isnan(ivs[j-1]) or np.isnan(ivs[j]) or np.isnan(ivs[j+1]):
                    continue
                
                K1, K2, K3 = self.strikes[j-1:j+2]
                
                # Simplified check: local convexity of IV
                # More rigorous would involve actual option prices
                iv_convexity = ivs[j-1] + ivs[j+1] - 2 * ivs[j]
                
                if iv_convexity < -tolerance:
                    violations.append({
                        'type': 'butterfly',
                        'maturity': T,
                        'K1': K1,
                        'K2': K2,
                        'K3': K3,
                        'IV1': ivs[j-1],
                        'IV2': ivs[j],
                        'IV3': ivs[j+1],
                        'convexity': iv_convexity
                    })
        
        return violations
    
    def get_atm_volatility(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ATM volatility term structure.
        
        Returns:
            Tuple of (maturities, atm_vols)
        """
        atm_vols = np.zeros(len(self.maturities))
        
        for i, T in enumerate(self.maturities):
            # Find closest strike to ATM
            atm_idx = np.argmin(np.abs(self.strikes - self.spot))
            atm_vols[i] = self.iv_surface[i, atm_idx]
        
        return self.maturities, atm_vols
    
    def get_volatility_smile(self, maturity_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract volatility smile for a specific maturity.
        
        Args:
            maturity_idx: Index of the maturity
            
        Returns:
            Tuple of (strikes, implied_vols)
        """
        return self.strikes, self.iv_surface[maturity_idx, :]
    
    def calculate_skew(self, maturity_idx: int) -> float:
        """
        Calculate volatility skew at a specific maturity.
        
        Skew = (IV_90% - IV_110%) / (110% - 90%)
        
        Args:
            maturity_idx: Index of the maturity
            
        Returns:
            Skew measure
        """
        # Define strikes as percentage of spot
        K_low = self.spot * 0.9
        K_high = self.spot * 1.1
        
        # Interpolate IVs at these strikes
        T = self.maturities[maturity_idx]
        iv_low = self.interpolate(np.array([K_low]), np.array([T]))[0]
        iv_high = self.interpolate(np.array([K_high]), np.array([T]))[0]
        
        skew = (iv_low - iv_high) / 0.2  # Normalized by strike range
        
        return skew
    
    def summary_statistics(self) -> Dict:
        """
        Calculate summary statistics for the surface.
        
        Returns:
            Dictionary of statistics
        """
        valid_ivs = self.iv_surface[self.valid_mask]
        
        stats = {
            'n_points': len(valid_ivs),
            'iv_min': np.min(valid_ivs),
            'iv_max': np.max(valid_ivs),
            'iv_mean': np.mean(valid_ivs),
            'iv_std': np.std(valid_ivs),
            'strike_range': (self.strikes.min(), self.strikes.max()),
            'maturity_range': (self.maturities.min(), self.maturities.max()),
            'spot': self.spot
        }
        
        # ATM vol term structure
        mats, atm_vols = self.get_atm_volatility()
        stats['atm_vol_term_structure'] = list(zip(mats, atm_vols))
        
        # Arbitrage checks
        calendar_violations = self.check_calendar_arbitrage()
        butterfly_violations = self.check_butterfly_arbitrage()
        
        stats['calendar_arbitrage_violations'] = len(calendar_violations)
        stats['butterfly_arbitrage_violations'] = len(butterfly_violations)
        
        return stats
