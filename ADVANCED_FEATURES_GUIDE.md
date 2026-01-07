"""
INTEGRATION GUIDE: Advanced Portfolio Optimizer Features
=========================================================

This document explains how to use the new advanced features added to the Portfolio Optimizer.

## NEW MODULES

### 1. VOLATILITY FORECASTING (volatility_forecasting.py)
   - GARCH(1,1) models for dynamic vol
   - Forward-looking covariance estimation
   - Superior to historical volatility

   Usage:
   ```python
   from src.volatility_forecasting import forecast_mu_sigma_forward_looking
   
   mu, sigma, assets = forecast_mu_sigma_forward_looking(
       returns=daily_returns_df,
       use_garch=True,
       trading_days=252
   )
   ```

### 2. ADVANCED OPTIMIZATION (optimization_objectives.py)
   - Sharpe ratio maximization (not just min variance)
   - Risk parity allocation
   - Efficient frontier generation
   
   Usage:
   ```python
   from src.optimization_objectives import optimize_sharpe_ratio, efficient_frontier
   
   result = optimize_sharpe_ratio(df, mu, sigma, fund_cfg)
   frontier = efficient_frontier(df, mu, sigma, fund_cfg, n_points=20)
   ```

### 3. BLACK-LITTERMAN MODEL (black_litterman.py)
   - Blend market expectations with analyst views
   - Reduces over-concentration
   - Produces well-diversified portfolios
   
   Usage:
   ```python
   from src.black_litterman import BlackLittermanModel, create_relative_views
   
   bl = BlackLittermanModel(market_weights, cov_matrix, risk_aversion=2.5)
   P, q, Omega = create_relative_views(
       tickers=['AAPL', 'MSFT', 'GOOGL'],
       outperformers=['AAPL', 'MSFT'],
       underperformers=['GOOGL'],
       expected_alpha=0.02
   )
   bl.add_view(P, q, Omega)
   mu_posterior = bl.posterior_returns()
   sigma_posterior = bl.posterior_covariance()
   ```

### 4. ADVANCED ML ALPHA (ml_alpha.py)
   - Rich feature engineering (momentum, volatility, correlation)
   - Ridge regression with regularization
   - Feature importance analysis
   - Ensemble methods combining multiple alpha sources
   
   Usage:
   ```python
   from src.ml_alpha import MLAlphaModel, ensemble_alpha_forecast
   
   ml_model = MLAlphaModel(returns_df)
   ml_model.fit(ridge_alpha=10.0, verbose=True)
   predictions = ml_model.predict_next_returns()
   importance = ml_model.get_feature_importance(top_n=10)
   
   # Ensemble multiple alpha sources
   blended_alpha = ensemble_alpha_forecast(
       returns, 
       fundamental_alpha=fundamentals,
       technical_alpha=technicals,
       ml_alpha=predictions,
       weights=[0.3, 0.3, 0.4]
   )
   ```

### 5. FACTOR MODEL (factor_model.py)
   - Decompose portfolio risk into factors
   - Attribution analysis
   - Factor exposure monitoring
   
   Usage:
   ```python
   from src.factor_model import FactorModel
   
   fm = FactorModel(returns_df)
   fm.fit()
   risk_decomp = fm.decompose_risk(weights)
   loadings = fm.get_factor_loadings()
   attribution = fm.factor_attribution(weights, period_returns)
   ```

### 6. TRANSACTION COSTS (transaction_costs.py)
   - Market impact modeling
   - Execution cost analysis (VWAP, TWAP, MOO)
   - Rebalance optimization accounting for friction
   
   Usage:
   ```python
   from src.transaction_costs import TransactionCostModel, optimize_with_transaction_costs
   
   cost_model = TransactionCostModel(
       commission_bps=5.0,
       spread_bps=2.0,
       market_impact_coef=0.1
   )
   costs = cost_model.compute_total_costs(current_w, target_w, aum)
   
   adjusted = optimize_with_transaction_costs(
       target_w, current_w, aum, cost_model,
       rebalance_threshold_bps=50.0
   )
   ```

### 7. REGIME DETECTION (regime_detection.py)
   - Automatic market regime identification (Bull, Bear, Normal, Crash)
   - Regime-adaptive positioning
   - Defensive before downturns, offensive in recoveries
   
   Usage:
   ```python
   from src.regime_detection import RegimeDetector, RegimeAdaptivePortfolio
   
   detector = RegimeDetector(returns_df, lookback=60)
   regime_df = detector.detect_regimes_heuristic()  # or detect_regimes_gmm(n_regimes=3)
   current_regime = detector.get_current_regime()
   
   adapter = RegimeAdaptivePortfolio()
   allocation = adapter.get_regime_allocation(current_regime)
   adjusted_weights = adapter.adjust_weights_for_regime(base_w, regime, asset_classes)
   ```

### 8. STRESS TESTING (stress_testing.py)
   - Historical crisis scenarios (2008, 2020, etc.)
   - Custom hypothetical scenarios
   - Sensitivity analysis
   - Correlation breakdown scenarios
   
   Usage:
   ```python
   from src.stress_testing import (
       HistoricalStressScenarios,
       HypotheticalStressScenarios,
       SensitivityAnalysis
   )
   
   # Apply 2008 crisis
   impact = HistoricalStressScenarios.apply_scenario(
       portfolio_value, weights, asset_classes, '2008_financial_crisis'
   )
   
   # Parallel yield curve shift
   scenario = HypotheticalStressScenarios.parallel_yield_curve_shift(
       weights, asset_classes, yield_shift_bps=100
   )
   
   # Sensitivity to vol changes
   sensitivity = SensitivityAnalysis.volatility_sensitivity(mu, sigma, weights)
   ```

## RECOMMENDED WORKFLOW

1. **Data Preparation**
   - Use existing pipeline.py to validate and fetch data

2. **Risk Modeling (Choose One or Combine)**
   - Option A: Historical volatility (existing risk.py)
   - Option B: GARCH forecasting (NEW - volatility_forecasting.py)
   - Option C: Factor model decomposition (NEW - factor_model.py)

3. **Return Estimation (Choose One or Combine)**
   - Option A: Historical mean returns
   - Option B: ML-based predictions (NEW - ml_alpha.py)
   - Option C: Black-Litterman views (NEW - black_litterman.py)
   - Option D: Ensemble of A+B+C (NEW - ml_alpha.py ensemble_alpha_forecast)

4. **Optimization (Choose One)**
   - Traditional: Minimum variance
   - NEW: Sharpe ratio maximization
   - NEW: Risk parity
   - Apply transaction costs adjustment (NEW)

5. **Risk Management**
   - NEW: Detect market regime
   - NEW: Adjust allocation based on regime
   - NEW: Run stress tests

6. **Analysis**
   - NEW: Factor attribution
   - NEW: Risk decomposition
   - Existing: Backtest performance

## EXAMPLE: UPGRADED PIPELINE

```python
import pandas as pd
import numpy as np
from src.data import fetch_price_data, compute_log_returns
from src.validate import validate_holdings
from src.volatility_forecasting import forecast_mu_sigma_forward_looking
from src.ml_alpha import MLAlphaModel, ensemble_alpha_forecast
from src.black_litterman import BlackLittermanModel, create_relative_views
from src.optimization_objectives import optimize_sharpe_ratio
from src.transaction_costs import TransactionCostModel, optimize_with_transaction_costs
from src.regime_detection import RegimeDetector, RegimeAdaptivePortfolio
from src.stress_testing import HistoricalStressScenarios
from src.factor_model import FactorModel

# Load data
holdings_df, _ = validate_holdings('holdings.csv', config)
prices = fetch_price_data(holdings_df['ticker'].tolist(), start_date, end_date)
returns = compute_log_returns(prices)

# ADVANCED: Forward-looking volatility
mu_garch, sigma_garch, assets = forecast_mu_sigma_forward_looking(
    returns, use_garch=True
)

# ADVANCED: ML alpha predictions
ml_model = MLAlphaModel(returns)
ml_model.fit()
ml_alpha = ml_model.predict_next_returns()

# ADVANCED: Black-Litterman blending
market_caps = holdings_df['market_cap_usd'].values
market_weights = market_caps / market_caps.sum()

bl = BlackLittermanModel(market_weights, sigma_garch, risk_aversion=2.5)
# Add view: Tech will outperform
P, q, Omega = create_relative_views(
    assets, 
    outperformers=['AAPL', 'MSFT'],
    underperformers=['XOM'],
    expected_alpha=0.03
)
bl.add_view(P, q, Omega)
mu_posterior = bl.posterior_returns()

# ADVANCED: Optimize Sharpe ratio (not just min variance)
result = optimize_sharpe_ratio(holdings_df, mu_posterior, sigma_garch, config)

# ADVANCED: Account for transaction costs
cost_model = TransactionCostModel(commission_bps=5, spread_bps=2)
adjusted = optimize_with_transaction_costs(
    result['weights'], 
    current_weights, 
    portfolio_value,
    cost_model,
    rebalance_threshold_bps=50
)

# ADVANCED: Detect regime and adjust
detector = RegimeDetector(returns, lookback=60)
regime_df = detector.detect_regimes_heuristic()
regime = detector.get_current_regime()

adapter = RegimeAdaptivePortfolio()
regime_adjusted_weights = adapter.adjust_weights_for_regime(
    adjusted['adjusted_weights'],
    regime,
    holdings_df['asset_class'].tolist()
)

# ADVANCED: Stress test
stress_2008 = HistoricalStressScenarios.apply_scenario(
    portfolio_value, regime_adjusted_weights, asset_classes, '2008_financial_crisis'
)

# ADVANCED: Factor attribution
fm = FactorModel(returns)
fm.fit()
risk_decomp = fm.decompose_risk(regime_adjusted_weights)

print(f\"Regime: {regime}\")
print(f\"Expected Return: {np.sum(mu_posterior * regime_adjusted_weights):.2%}\")
print(f\"Stress (2008): {stress_2008['loss_pct']:.1f}%\")
print(f\"\\nRisk Decomposition:\\n{risk_decomp}\")
```

## KEY IMPROVEMENTS OVER BASELINE

| Feature | Baseline | Enhanced |
|---------|----------|----------|
| Vol forecasting | Historical | GARCH (captures regime changes) |
| Optimization | Min variance | Sharpe, Risk Parity, EF |
| Return estimation | Simple mean | ML ensemble + BL |
| Risk modeling | Simple cov matrix | Factor decomposition |
| Trading costs | Ignored | Market impact + execution |
| Market regimes | None | Automatic detection |
| Stress testing | Walk-forward only | 8 historical crises + custom |
| Attribution | None | Factor and regime based |

## TIPS & BEST PRACTICES

1. **GARCH is overkill for small portfolios** - use if you have >2 years daily data
2. **Black-Litterman needs careful view specification** - start with 1-2 views
3. **ML models need sufficient data** - minimum 2-3 years daily returns
4. **Rebalance threshold should be 50-100 bps** - too low causes excessive trading
5. **Regime detection works best with 60+ day lookback** - shorter windows are noisy
6. **Stress test at least quarterly** - market regimes change
7. **Factor models useful for >10 assets** - diversification benefit visible
8. **Transaction costs reduce after-cost returns by 5-10% annually** - don't ignore

## NEXT STEPS FOR EVEN MORE POWER

- Add mean-reverting Kalman filter for return dynamics
- Implement optimal execution algorithms (POV, TWAP, VWAP)
- Add ESG constraints and impact measurement
- Build Monte Carlo scenario engine for VaR/CVaR
- Add derivatives (options) for tail hedging
- Implement robo-advisor logic for client segmentation
"""
