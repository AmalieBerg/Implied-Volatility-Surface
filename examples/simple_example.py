"""
Simple Example: Calculate and visualize an implied volatility surface.

This is the quickest way to get started with the IV surface builder.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import ImpliedVolatilityCalculator, VolatilitySurface, MarketDataGenerator, SurfaceVisualizer


def main():
    print("Simple IV Surface Builder Example\n")
    
    # 1. Generate market data
    print("1. Generating market data...")
    gen = MarketDataGenerator(seed=42)
    data = gen.generate_realistic_data(
        spot=100.0,
        n_strikes=11,
        n_maturities=6,
        market_regime='normal'
    )
    
    # 2. Calculate implied volatilities
    print("2. Calculating implied volatilities...")
    calc = ImpliedVolatilityCalculator()
    iv_surface, iters, converged = calc.calculate_iv_surface(
        data['call_prices'],
        data['spot'],
        data['strikes'],
        data['maturities'],
        data['rate'],
        'call'
    )
    
    convergence_rate = converged.sum() / converged.size * 100
    print(f"   Convergence rate: {convergence_rate:.1f}%")
    
    # 3. Build surface
    print("3. Building volatility surface...")
    surface = VolatilitySurface(
        data['strikes'],
        data['maturities'],
        iv_surface,
        data['spot']
    )
    
    # 4. Check arbitrage
    print("4. Checking for arbitrage...")
    calendar_violations = surface.check_calendar_arbitrage()
    butterfly_violations = surface.check_butterfly_arbitrage()
    print(f"   Calendar violations: {len(calendar_violations)}")
    print(f"   Butterfly violations: {len(butterfly_violations)}")
    
    # 5. Display ATM term structure
    print("\n5. ATM Volatility Term Structure:")
    print("   Maturity | ATM Vol")
    print("   " + "-"*25)
    maturities, atm_vols = surface.get_atm_volatility()
    for T, vol in zip(maturities, atm_vols):
        print(f"   {T:7.2f}y | {vol*100:6.2f}%")
    
    # 6. Visualize
    print("\n6. Creating visualizations...")
    viz = SurfaceVisualizer()
    
    viz.plot_surface_3d(
        data['strikes'],
        data['maturities'],
        iv_surface,
        data['spot'],
        title="Implied Volatility Surface"
    )
    
    viz.plot_volatility_smile(
        data['strikes'],
        iv_surface[2, :],
        data['spot'],
        data['maturities'][2],
        title=f"Volatility Smile (T={data['maturities'][2]:.2f}y)"
    )
    
    print("\n✓ Example complete!")


if __name__ == '__main__':
    main()
