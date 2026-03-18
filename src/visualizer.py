"""
Volatility Surface Visualization Tools.

This module provides comprehensive visualization capabilities for implied
volatility surfaces including 3D plots, heatmaps, term structures, and smiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class SurfaceVisualizer:
    """
    Comprehensive visualization toolkit for volatility surfaces.
    
    Features:
    - Interactive 3D surface plots
    - 2D heatmaps with contours
    - ATM term structure plots
    - Volatility smile plots
    - Convergence diagnostics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_surface_3d(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        iv_surface: np.ndarray,
        spot: float,
        title: str = "Implied Volatility Surface",
        save_path: Optional[str] = None
    ):
        """
        Create 3D surface plot of implied volatility.
        
        Args:
            strikes: Strike prices
            maturities: Times to maturity
            iv_surface: IV surface [maturities x strikes]
            spot: Spot price
            title: Plot title
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh
        T_grid, K_grid = np.meshgrid(maturities, strikes, indexing='ij')
        
        # Plot surface
        surf = ax.plot_surface(
            K_grid,
            T_grid,
            iv_surface * 100,  # Convert to percentage
            cmap=cm.viridis,
            alpha=0.8,
            edgecolor='none',
            antialiased=True
        )
        
        # Mark ATM line
        atm_line = np.full_like(maturities, spot)
        ax.plot(
            atm_line,
            maturities,
            [np.nan] * len(maturities),
            'r--',
            linewidth=2,
            label='ATM'
        )
        
        # Labels and formatting
        ax.set_xlabel('Strike Price', fontsize=10, labelpad=10)
        ax.set_ylabel('Time to Maturity (years)', fontsize=10, labelpad=10)
        ax.set_zlabel('Implied Volatility (%)', fontsize=10, labelpad=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_heatmap(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        iv_surface: np.ndarray,
        spot: float,
        title: str = "Implied Volatility Heatmap",
        save_path: Optional[str] = None
    ):
        """
        Create 2D heatmap of implied volatility with contours.
        
        Args:
            strikes: Strike prices
            maturities: Times to maturity
            iv_surface: IV surface [maturities x strikes]
            spot: Spot price
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create mesh
        T_grid, K_grid = np.meshgrid(maturities, strikes, indexing='ij')
        
        # Plot heatmap
        im = ax.contourf(
            K_grid,
            T_grid,
            iv_surface * 100,
            levels=20,
            cmap='viridis',
            alpha=0.8
        )
        
        # Add contour lines
        contours = ax.contour(
            K_grid,
            T_grid,
            iv_surface * 100,
            levels=10,
            colors='white',
            alpha=0.3,
            linewidths=0.5
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f%%')
        
        # Mark ATM line
        ax.axvline(spot, color='red', linestyle='--', linewidth=2, label='ATM')
        
        # Labels
        ax.set_xlabel('Strike Price', fontsize=12)
        ax.set_ylabel('Time to Maturity (years)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Implied Volatility (%)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_atm_term_structure(
        self,
        maturities: np.ndarray,
        atm_vols: np.ndarray,
        title: str = "ATM Volatility Term Structure",
        save_path: Optional[str] = None
    ):
        """
        Plot ATM volatility term structure.
        
        Args:
            maturities: Times to maturity
            atm_vols: ATM implied volatilities
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Plot term structure
        ax.plot(
            maturities,
            atm_vols * 100,
            'bo-',
            linewidth=2,
            markersize=8,
            label='ATM Volatility'
        )
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Time to Maturity (years)', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_volatility_smile(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        spot: float,
        maturity: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot volatility smile for a specific maturity.
        
        Args:
            strikes: Strike prices
            ivs: Implied volatilities
            spot: Spot price
            maturity: Time to maturity
            title: Plot title
            save_path: Path to save figure
        """
        if title is None:
            title = f"Volatility Smile (T = {maturity:.2f} years)"
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Calculate moneyness
        moneyness = strikes / spot
        
        # Plot smile
        ax.plot(
            moneyness,
            ivs * 100,
            'go-',
            linewidth=2,
            markersize=8,
            label='Implied Volatility'
        )
        
        # Mark ATM
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='ATM')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Moneyness (K/S)', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_multiple_smiles(
        self,
        strikes: np.ndarray,
        iv_surface: np.ndarray,
        maturities: np.ndarray,
        spot: float,
        maturity_indices: Optional[List[int]] = None,
        title: str = "Volatility Smiles Across Maturities",
        save_path: Optional[str] = None
    ):
        """
        Plot multiple volatility smiles for comparison.
        
        Args:
            strikes: Strike prices
            iv_surface: IV surface [maturities x strikes]
            maturities: Times to maturity
            spot: Spot price
            maturity_indices: Specific maturities to plot (None = all)
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if maturity_indices is None:
            maturity_indices = range(len(maturities))
        
        # Calculate moneyness
        moneyness = strikes / spot
        
        # Plot each smile
        colors = plt.cm.viridis(np.linspace(0, 1, len(maturity_indices)))
        
        for idx, color in zip(maturity_indices, colors):
            T = maturities[idx]
            ivs = iv_surface[idx, :]
            
            ax.plot(
                moneyness,
                ivs * 100,
                'o-',
                color=color,
                linewidth=2,
                markersize=6,
                label=f'T = {T:.2f}y',
                alpha=0.8
            )
        
        # Mark ATM
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='ATM')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Moneyness (K/S)', fontsize=12)
        ax.set_ylabel('Implied Volatility (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', ncol=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence_analysis(
        self,
        iterations_surface: np.ndarray,
        convergence_mask: np.ndarray,
        title: str = "IV Calculation Convergence Analysis",
        save_path: Optional[str] = None
    ):
        """
        Visualize convergence statistics for IV calculations.
        
        Args:
            iterations_surface: Number of iterations per point
            convergence_mask: Boolean mask of converged points
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # Flatten valid data
        iterations_flat = iterations_surface[convergence_mask]
        
        # Plot 1: Iteration histogram
        axes[0].hist(
            iterations_flat,
            bins=range(1, int(iterations_flat.max()) + 2),
            edgecolor='black',
            alpha=0.7,
            color='steelblue'
        )
        axes[0].set_xlabel('Number of Iterations', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Convergence Speed Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_iters = np.mean(iterations_flat)
        median_iters = np.median(iterations_flat)
        axes[0].axvline(mean_iters, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iters:.1f}')
        axes[0].axvline(median_iters, color='green', linestyle='--', linewidth=2, label=f'Median: {median_iters:.0f}')
        axes[0].legend()
        
        # Plot 2: Convergence rate
        convergence_rate = np.sum(convergence_mask) / convergence_mask.size * 100
        non_convergence_rate = 100 - convergence_rate
        
        axes[1].bar(
            ['Converged', 'Not Converged'],
            [convergence_rate, non_convergence_rate],
            color=['green', 'red'],
            alpha=0.7,
            edgecolor='black'
        )
        axes[1].set_ylabel('Percentage (%)', fontsize=11)
        axes[1].set_title('Overall Convergence Rate', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 105])
        
        # Add percentage labels
        for i, (label, value) in enumerate([('Converged', convergence_rate), 
                                            ('Not Converged', non_convergence_rate)]):
            axes[1].text(i, value + 2, f'{value:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
