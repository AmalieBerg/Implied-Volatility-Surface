# Quick Start Guide - Implied Volatility Surface Builder

## Installation

```bash
pip install numpy scipy matplotlib
```

## 5-Minute Tutorial

### 1. Calculate Single Implied Volatility

```python
from src.iv_calculator import ImpliedVolatilityCalculator

# Initialize calculator
calc = ImpliedVolatilityCalculator()

# Calculate IV from market price
iv, iterations, converged = calc.calculate_iv(
    market_price=10.45,  # Observed option price
    S=100,               # Spot price
    K=100,               # Strike price
    T=1.0,               # Time to maturity (years)
    r=0.05,              # Risk-free rate
    option_type='call'
)

print(f"Implied Volatility: {iv*100:.2f}%")
print(f"Converged in {iterations} iterations")
```

### 2. Build Complete IV Surface

```python
from src.market_data import MarketDataGenerator

# Generate realistic market data
gen = MarketDataGenerator(seed=42)
data = gen.generate_realistic_data(
    spot=100.0,
    n_strikes=15,
    n_maturities=8,
    market_regime='normal'  # or 'high_vol', 'crash'
)

# Calculate IV surface
iv_surface, iters, converged = calc.calculate_iv_surface(
    data['call_prices'],
    data['spot'],
    data['strikes'],
    data['maturities'],
    data['rate'],
    'call'
)

print(f"Convergence rate: {converged.sum()/converged.size*100:.1f}%")
```

### 3. Analyze Surface

```python
from src.surface_builder import VolatilitySurface

# Build surface
surface = VolatilitySurface(
    data['strikes'],
    data['maturities'],
    iv_surface,
    data['spot']
)

# Get ATM term structure
maturities, atm_vols = surface.get_atm_volatility()

# Check for arbitrage
calendar_violations = surface.check_calendar_arbitrage()
butterfly_violations = surface.check_butterfly_arbitrage()

print(f"Arbitrage violations: {len(calendar_violations)} calendar, {len(butterfly_violations)} butterfly")
```

### 4. Interpolate Surface

```python
import numpy as np

# Query specific points
query_strikes = np.array([95, 100, 105])
query_maturities = np.array([0.5, 0.5, 0.5])

iv_interpolated = surface.interpolate(
    query_strikes,
    query_maturities,
    method='rbf'  # or 'linear', 'cubic'
)

for K, iv in zip(query_strikes, iv_interpolated):
    print(f"K={K}: IV={iv*100:.2f}%")
```

### 5. Visualize Results

```python
from src.visualizer import SurfaceVisualizer

viz = SurfaceVisualizer()

# 3D surface plot
viz.plot_surface_3d(
    data['strikes'],
    data['maturities'],
    iv_surface,
    data['spot']
)

# Volatility smile
viz.plot_volatility_smile(
    data['strikes'],
    iv_surface[2, :],  # Select maturity index
    data['spot'],
    data['maturities'][2]
)
```

## Common Use Cases

### Use Case 1: Price Validation

```python
# Check if market price is arbitrage-free
market_price = 10.50
theoretical_iv, _, _ = calc.calculate_iv(market_price, S, K, T, r, 'call')

# Compare with surface
surface_iv = surface.interpolate(np.array([K]), np.array([T]))[0]

if abs(theoretical_iv - surface_iv) > 0.02:  # 2% threshold
    print("WARNING: Price inconsistent with surface")
```

### Use Case 2: Quote Generation

```python
# Generate bid-ask quotes for a new strike
new_strike = 102.5
new_maturity = 0.75

iv = surface.interpolate(np.array([new_strike]), np.array([new_maturity]))[0]

# Calculate prices
mid_price = calc.black_scholes_price(S, new_strike, new_maturity, r, iv, 'call')
spread = mid_price * 0.01  # 1% bid-ask spread

bid = mid_price - spread/2
ask = mid_price + spread/2

print(f"Quote: {bid:.2f} / {ask:.2f}")
```

### Use Case 3: Greeks from Surface

```python
# Calculate Vega at specific point
K = 100
T = 1.0
iv = surface.interpolate(np.array([K]), np.array([T]))[0]

vega = calc.vega(S, K, T, r, iv)
print(f"Vega: {vega:.4f}")
```

## Running Tests

```bash
# Run all tests
python tests/test_iv_surface.py

# Run specific test class
python -m unittest tests.test_iv_surface.TestIVCalculator

# Run with verbose output
python tests/test_iv_surface.py -v
```

## Complete Demo

```bash
# Run comprehensive demonstration
python examples/comprehensive_demo.py
```

This will:
- Generate 210 market data points
- Calculate all implied volatilities
- Validate against true IVs
- Check for arbitrage
- Create 5 visualization plots
- Display performance metrics

## Performance Tips

1. **Use initial guesses**: Provide good starting points for faster convergence
2. **Batch processing**: Calculate full surface at once instead of point-by-point
3. **Cache results**: Store interpolated surfaces for repeated queries
4. **Sparse data**: Use fewer points for real-time applications

## Common Issues

### Issue: Slow convergence for deep OTM options
**Solution**: Use higher tolerance or better initial guess

```python
calc = ImpliedVolatilityCalculator(tolerance=1e-5)  # Relaxed
```

### Issue: Interpolation artifacts
**Solution**: Use RBF with smoothing

```python
iv = surface.interpolate(strikes, maturities, method='rbf')
```

### Issue: Many arbitrage violations
**Solution**: Check data quality or relax tolerance

```python
violations = surface.check_calendar_arbitrage(tolerance=1e-4)
```

## Next Steps

1. Try different market regimes: `'normal'`, `'high_vol'`, `'crash'`
2. Experiment with interpolation methods
3. Add your own market data
4. Integrate with trading strategies
5. Extend to exotic options

## Resources

- Full documentation: `README.md`
- Test examples: `tests/test_iv_surface.py`
- Complete demo: `examples/comprehensive_demo.py`

---

**Questions?** Check the main README or examine the test suite for more examples.
