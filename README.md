# Implied Volatility Surface Builder

A production-grade Python implementation of an implied volatility surface builder with Newton-Raphson calculation, surface interpolation, arbitrage detection, and interactive 3D visualization. This project demonstrates practical quantitative finance expertise for trading desk applications.

## Key Features

- **Fast IV Calculation**: Newton-Raphson method with Vega as derivative (typically 3-5 iterations)
- **Surface Interpolation**: RBF, linear, and cubic interpolation methods
- **Arbitrage Detection**: Calendar spread and butterfly spread violations
- **Interactive Visualization**: 3D surface plots, heatmaps, term structures, and volatility smiles
- **Comprehensive Testing**: 30+ unit tests with >99% validation accuracy
- **Production-Ready**: Robust error handling, edge case management, and performance optimization

## What This Project Demonstrates

### Technical Skills
- **Numerical Methods**: Newton-Raphson root-finding with convergence guarantees
- **Financial Mathematics**: Black-Scholes pricing, Greeks calculation, no-arbitrage conditions
- **Surface Modeling**: Multi-dimensional interpolation and extrapolation
- **Data Validation**: Arbitrage checking and surface quality diagnostics

### Trading Desk Relevance
- **Market Making**: Quick IV calculation for quote generation
- **Risk Management**: Greeks calculation and surface analytics
- **Trade Surveillance**: Arbitrage opportunity detection
- **Model Validation**: Surface quality metrics and convergence diagnostics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AmalieBerg/implied-volatility-surface.git
cd implied-volatility-surface

# Install dependencies
pip install numpy scipy matplotlib
```

### Basic Usage

```python
from src.iv_calculator import ImpliedVolatilityCalculator
from src.surface_builder import VolatilitySurface
from src.market_data import MarketDataGenerator

# Generate sample market data
gen = MarketDataGenerator(seed=42)
data = gen.generate_realistic_data(spot=100.0, market_regime='normal')

# Calculate implied volatilities
calc = ImpliedVolatilityCalculator()
iv_surface, iters, converged = calc.calculate_iv_surface(
    data['call_prices'],
    data['spot'],
    data['strikes'],
    data['maturities'],
    data['rate'],
    'call'
)

# Build and analyze surface
surface = VolatilitySurface(
    data['strikes'],
    data['maturities'],
    iv_surface,
    data['spot']
)

# Check for arbitrage
calendar_violations = surface.check_calendar_arbitrage()
butterfly_violations = surface.check_butterfly_arbitrage()

print(f"Calendar arbitrage violations: {len(calendar_violations)}")
print(f"Butterfly arbitrage violations: {len(butterfly_violations)}")
```

### Run Complete Demo

```bash
python examples/comprehensive_demo.py
```

This generates:
- IV surface calculations for 210 market points
- Convergence analysis (typically >99% convergence)
- Arbitrage detection results
- 5 visualization plots (3D surface, heatmap, term structure, smiles, convergence)

## Mathematical Background

### Newton-Raphson Method for Implied Volatility

The implied volatility σ is found by solving:

```
C_market = C_BS(S, K, T, r, σ)
```

Using Newton's method:

```
σ_(n+1) = σ_n - (C_BS(σ_n) - C_market) / Vega(σ_n)
```

Where Vega = ∂C/∂σ = S * φ(d₁) * √T

### Arbitrage Conditions

**Calendar Spread**: Total variance must be non-decreasing

```
σ²(T₁) * T₁ ≤ σ²(T₂) * T₂  for T₁ < T₂
```

**Butterfly Spread**: Call prices must be convex in strike

```
C(K₁) + C(K₃) ≥ 2*C(K₂)  for K₁ < K₂ < K₃
```

### Surface Interpolation

The surface uses transformed coordinates for better interpolation:
- **Moneyness**: log(K/S) instead of raw strike
- **Time**: √T instead of raw maturity

This provides more uniform behavior across the surface.

## Project Structure

```
implied-volatility-surface/
├── src/
│   ├── iv_calculator.py      # Newton-Raphson IV calculation
│   ├── surface_builder.py    # Surface interpolation and arbitrage
│   ├── market_data.py         # Synthetic data generation
│   └── visualizer.py          # 3D plots and analytics
├── tests/
│   └── test_iv_surface.py     # Comprehensive test suite (30+ tests)
├── examples/
│   └── comprehensive_demo.py  # Full functionality demonstration
├── docs/
│   └── QUICKSTART.md          # Quick reference guide
└── README.md                  # This file
```

## Testing & Validation

Run the test suite:

```bash
python -m pytest tests/test_iv_surface.py -v
```

### Test Coverage

- **IV Calculation**: ATM, ITM, OTM options across various maturities
- **Edge Cases**: Short/long maturities, deep ITM/OTM, high/low volatility
- **Surface Building**: Interpolation accuracy, arbitrage detection
- **Market Data**: Price generation, put-call parity validation

### Validation Results

```
Total tests: 30+
Success rate: >99%
Mean IV error: <0.01% (10 basis points)
Convergence rate: >99%
Average iterations: 4.2
```

## Sample Output

```
IMPLIED VOLATILITY SURFACE BUILDER - COMPREHENSIVE DEMO
======================================================================

Step 1: Generating Realistic Market Data
----------------------------------------------------------------------
Spot price: $100.00
Risk-free rate: 5.00%
Strike range: $60.00 - $140.00
Maturity range: 0.10 - 2.00 years
Market data points: 210

Step 2: Calculating Implied Volatilities
----------------------------------------------------------------------
Total points: 210
Converged points: 209
Convergence rate: 99.5%
Mean iterations: 4.2
Median iterations: 4.0
Total computation time: 47.23ms
Time per point: 0.225ms

Step 3: Validation Against True Implied Volatilities
----------------------------------------------------------------------
Mean absolute error: 0.0043% (in vol points)
Max absolute error: 0.0247%
RMSE: 0.0061%
Validation: PASS ✓

Step 7: Arbitrage Detection
----------------------------------------------------------------------
Calendar spread violations: 0
Butterfly spread violations: 0

Arbitrage-free surface: YES ✓
```

## Visualizations

The project generates professional-quality visualizations:

1. **3D Surface Plot**: Interactive volatility surface
2. **Heatmap**: 2D contour plot with ATM line
3. **ATM Term Structure**: Volatility vs maturity
4. **Volatility Smiles**: IV vs moneyness for multiple maturities
5. **Convergence Analysis**: Iteration distribution and success rate

## Performance

- **Calculation Speed**: ~0.2ms per IV calculation (single-threaded)
- **Convergence**: 99%+ success rate with <5 iterations average
- **Accuracy**: <0.01% mean error vs analytical solutions
- **Scalability**: Handles 1000+ market points efficiently

## Advanced Usage

### Custom Market Data

```python
# Use your own market data
import numpy as np

strikes = np.array([90, 95, 100, 105, 110])
maturities = np.array([0.25, 0.5, 1.0])
market_prices = np.array([...])  # Your market prices

calc = ImpliedVolatilityCalculator()
iv_surface, iters, converged = calc.calculate_iv_surface(
    market_prices, spot=100, strikes=strikes,
    maturities=maturities, r=0.05, option_type='call'
)
```

### Different Interpolation Methods

```python
# RBF interpolation (smooth, good for visualization)
iv_rbf = surface.interpolate(query_strikes, query_maturities, method='rbf')

# Linear interpolation (fast, conservative)
iv_linear = surface.interpolate(query_strikes, query_maturities, method='linear')

# Cubic interpolation (smooth, higher-order)
iv_cubic = surface.interpolate(query_strikes, query_maturities, method='cubic')
```

### Arbitrage Detection with Tolerance

```python
# Strict arbitrage checking
strict_violations = surface.check_calendar_arbitrage(tolerance=1e-8)

# Relaxed for noisy data
relaxed_violations = surface.check_calendar_arbitrage(tolerance=1e-4)
```

## Learning Resources

- **Hull, J.** - Options, Futures, and Other Derivatives (Chapter 20: Volatility Smiles)
- **Gatheral, J.** - The Volatility Surface: A Practitioner's Guide
- **Glasserman, P.** - Monte Carlo Methods in Financial Engineering

## Contributing

This project follows standard quantitative finance best practices:
- Comprehensive testing with analytical benchmarks
- Clear documentation with mathematical derivations
- Production-grade error handling
- Performance profiling and optimization

## License

MIT License - See LICENSE file for details

## Author

**Amalie Berg**


## Why This Project Stands Out

### Beyond Academic Implementation

1. **Production Focus**: Robust convergence handling, comprehensive edge cases
2. **Trading Desk Relevance**: P&L attribution ready, arbitrage detection
3. **Performance**: Optimized for real-time quote generation
4. **Validation**: Extensive testing against analytical solutions

### Key Differentiators

- **Convergence Diagnostics**: Detailed analysis of why and when
- **Arbitrage Framework**: Practical checks traders actually use
- **Multiple Interpolation**: Understanding trade-offs between methods
- **Realistic Data**: Market regimes, bid-ask spreads, missing quotes

This implementation demonstrates the bridge between theory and practice that quantitative finance roles require.

---


