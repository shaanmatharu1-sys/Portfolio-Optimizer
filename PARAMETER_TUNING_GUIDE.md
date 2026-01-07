# Quick Start: Return Estimation Tuning

## What Changed?

Your portfolio returns were too conservative (2%) because the model was over-shrinking return estimates. This has been fixed with:

1. **Better defaults** in `src/risk.py`
2. **Interactive sliders** in the Streamlit app
3. **Visible return estimates** in the Analysis tab

## How to Use

### In the Streamlit App

1. **Open the sidebar** (click ☰ if on mobile)
2. **Scroll to "Return Estimation Tuning"**
3. **Adjust the sliders:**

   - **Return Shrinkage** (0 = historic, 1 = zero)
     - Drag LEFT for higher returns (trusts historical data more)
     - Drag RIGHT for lower returns (more conservative)
     - Default: 0.10 (recommended for tech stocks)
   
   - **Max Return Bound (±%)**
     - Drag LEFT for tighter constraints (-20% to +20%)
     - Drag RIGHT for looser constraints (-100% to +100%)
     - Default: 1.0 (±100%, good for volatile stocks)

4. **Click "Run Optimization"** to recalculate portfolio

5. **Check the "Analysis" tab** to see:
   - Expected return for each stock
   - How the sliders affect forecasts
   - Current vs target portfolio weights

### Suggested Settings by Portfolio Type

#### Aggressive Growth (Tech/High-Growth)
```
Return Shrinkage: 0.05
Max Return Bound: 1.5 (±150%)
→ Expects 40-50% returns from top performers
```

#### Balanced (Mix of Growth & Value)
```
Return Shrinkage: 0.10
Max Return Bound: 1.0 (±100%)
→ Expects 20-35% returns from growth stocks
```

#### Conservative (Large-Cap Blue Chips)
```
Return Shrinkage: 0.20
Max Return Bound: 0.50 (±50%)
→ Expects 8-15% returns from dividend stocks
```

#### Ultra-Conservative (Fixed Income/Bonds)
```
Return Shrinkage: 0.30
Max Return Bound: 0.20 (±20%)
→ Expects 2-5% returns from stable assets
```

## Example Results

Using your sample portfolio (AAPL, MSFT, GOOGL, AMZN, TSLA):

### Before (Old Settings)
```
AAPL  → 13.7%
MSFT  → 14.2%
GOOGL → 23.8%
AMZN  → 19.6%
TSLA  → 30.5%
─────────────
AVG   → 20.4%  ← Too low for tech stocks
```

### After (New Settings)
```
AAPL  → 24.6%
MSFT  → 25.6%
GOOGL → 42.9%
AMZN  → 35.3%
TSLA  → 54.9%
─────────────
AVG   → 36.6%  ← More realistic
```

## Understanding the Parameters

### Return Shrinkage Explained

**Shrinkage** is a technique to prevent overfitting. It blends historical returns with a neutral estimate:

- **Low shrinkage (0.05)**: "Trust the data" → Higher return forecasts
- **Medium shrinkage (0.10-0.20)**: "Cautiously optimistic" → Balanced forecasts
- **High shrinkage (0.50+)**: "Be very conservative" → Lower return forecasts

### Max Return Bound Explained

**Clipping** prevents extreme return forecasts:

- **Tight bound (±20%)**: Max annual return is 20%, Min is -20%
- **Loose bound (±100%)**: Max annual return is 100%, Min is -100%

## Troubleshooting

**Returns still seem too low?**
- Decrease Return Shrinkage (move slider left)
- Increase Max Return Bound (move slider right)

**Returns seem too high to be realistic?**
- Increase Return Shrinkage (move slider right)
- Decrease Max Return Bound (move slider left)

**Not sure what to use?**
- Start with defaults (0.10 shrinkage, 1.0 bound)
- Run optimization
- Check Analysis tab to see if returns match your expectations
- Adjust from there

## Technical Details

### Files Changed
- `src/risk.py` - New function defaults
- `app.py` - Added sliders + return display

### Backward Compatibility
Old code still works. To use old parameters:

```python
from src.risk import estimate_mu_sigma
mu, sigma, assets = estimate_mu_sigma(
    returns,
    mu_shrink_beta=0.50,  # Old: 50% shrinkage
    mu_clip=0.50          # Old: ±50% bound
)
```

## Questions?

The shrinkage parameters affect **expected return forecasts only**. Volatility and correlations use separate `cov_shrink_alpha` parameter (unchanged at 0.15).
