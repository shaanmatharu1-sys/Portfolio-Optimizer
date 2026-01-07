#!/usr/bin/env python3
"""
EXAMPLE: Using Advanced Features in Practice

This script demonstrates how to use all 8 new modules together
in a realistic portfolio optimization workflow.

Run with: python example_advanced_usage.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def example_volatility_forecasting():
    """Example 1: Forward-looking volatility with GARCH"""
    print("\n" + "="*60)
    print("EXAMPLE 1: VOLATILITY FORECASTING")
    print("="*60)
    
    from src.volatility_forecasting import GARCHVolatility, DynamicCovarianceMatrix
    
    # Synthetic returns data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 252)
    
    # Fit GARCH model
    garch = GARCHVolatility(returns)
    print(f"GARCH(1,1) Parameters:")
    print(f"  ω (omega):  {garch.omega:.6f}")
    print(f"  α (alpha):  {garch.alpha:.6f}")
    print(f"  β (beta):   {garch.beta:.6f}")
    print(f"  Sum(α+β):   {garch.alpha + garch.beta:.6f} (should be <1)")
    
    # Forecast next 21 days
    forecast = garch.forecast_variance(steps=21)
    print(f"\n21-day vol forecast (annualized): {np.sqrt(forecast[-1] * 252)*100:.2f}%")


def example_sharpe_optimization():
    """Example 2: Optimize for Sharpe ratio instead of min variance"""
    print("\n" + "="*60)
    print("EXAMPLE 2: SHARPE RATIO OPTIMIZATION")
    print("="*60)
    
    from src.optimization_objectives import (
        optimize_sharpe_ratio,
        optimize_risk_parity,
        efficient_frontier
    )
    
    # Create sample portfolio
    np.random.seed(42)
    n_assets = 10
    mu = np.random.uniform(0.05, 0.15, n_assets)
    sigma = np.random.uniform(0.1, 0.3, n_assets)
    corr = np.eye(n_assets) + 0.3 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
    cov = np.outer(sigma, sigma) * corr
    
    current_w = np.ones(n_assets) / n_assets
    
    # Mock dataframe
    df = pd.DataFrame({
        'ticker': [f'ASSET_{i}' for i in range(n_assets)],
        'weight': current_w
    })
    
    # Simple config
    config = {'constraints': {}, 'instrument_policy': {}, 'optimization': {}}
    
    print(f"Portfolio of {n_assets} assets")
    print(f"Expected Returns: {mu*100:.1f}% ± {np.std(mu)*100:.1f}%")
    
    try:
        # Sharpe optimization
        result = optimize_sharpe_ratio(df, mu, cov, config)
        print(f"\nSharpe Ratio Optimized:")
        print(f"  Return:      {result['return']*100:.2f}%")
        print(f"  Volatility:  {result['volatility']*100:.2f}%")
        print(f"  Sharpe:      {result['sharpe_ratio']:.3f}")
        print(f"  Top 3 holdings: {sorted(list(enumerate(result['weights'])), key=lambda x: x[1], reverse=True)[:3]}")
    except Exception as e:
        print(f"Note: {str(e)[:80]}...")


def example_black_litterman():
    """Example 3: Black-Litterman blending of views"""
    print("\n" + "="*60)
    print("EXAMPLE 3: BLACK-LITTERMAN MODEL")
    print("="*60)
    
    from src.black_litterman import BlackLittermanModel, create_relative_views
    
    # Market data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    market_weights = np.array([0.25, 0.25, 0.20, 0.20, 0.10])
    
    # Synthetic covariance
    np.random.seed(42)
    sigma = np.eye(5) * 0.04
    sigma = sigma + 0.01 * (np.ones((5,5)) - np.eye(5))
    
    # BL Model
    bl = BlackLittermanModel(market_weights, sigma, risk_aversion=2.5)
    print(f"Market-Implied Returns:")
    for ticker, ret in zip(tickers, bl.mu_mkt):
        print(f"  {ticker}: {ret*100:.2f}%")
    
    # Add a view: Tech will outperform
    P, q, Omega = create_relative_views(
        tickers,
        outperformers=['AAPL', 'MSFT'],
        underperformers=['TSLA'],
        expected_alpha=0.05,
        confidence=0.01
    )
    bl.add_view(P, q, Omega)
    
    mu_posterior = bl.posterior_returns()
    print(f"\nAfter Adding View (Tech Outperforms):")
    for ticker, ret in zip(tickers, mu_posterior):
        print(f"  {ticker}: {ret*100:.2f}%")


def example_ml_alpha():
    """Example 4: ML-based alpha predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 4: ML ALPHA MODEL")
    print("="*60)
    
    try:
        from src.ml_alpha import MLAlphaModel
        
        # Synthetic returns
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=500)
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (500, 5)),
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )
        
        ml = MLAlphaModel(returns)
        print("Training ML model...")
        ml.fit(ridge_alpha=10.0, verbose=True)
        
        predictions = ml.predict_next_returns()
        print(f"\nNext-Day Return Predictions (Annualized):")
        for ticker, pred in zip(returns.columns, predictions):
            print(f"  {ticker}: {pred*100:+.2f}%")
        
        # Feature importance
        importance = ml.get_feature_importance(top_n=5)
        print(f"\nTop 5 Features:")
        print(importance.to_string(index=False))
    except Exception as e:
        print(f"ML model demo (requires more data in practice): {str(e)[:60]}...")


def example_factor_model():
    """Example 5: Factor-based risk decomposition"""
    print("\n" + "="*60)
    print("EXAMPLE 5: FACTOR MODEL")
    print("="*60)
    
    try:
        from src.factor_model import FactorModel
        
        # Synthetic returns
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=500)
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (500, 8)),
            index=dates,
            columns=[f'STOCK_{i}' for i in range(8)]
        )
        
        fm = FactorModel(returns)
        fm.fit()
        
        weights = np.array([0.15, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10])
        risk_decomp = fm.decompose_risk(weights)
        
        print("Risk Contribution by Factor:")
        print(risk_decomp.to_string(index=False))
    except Exception as e:
        print(f"Factor model demo: {str(e)[:60]}...")


def example_transaction_costs():
    """Example 6: Transaction cost modeling"""
    print("\n" + "="*60)
    print("EXAMPLE 6: TRANSACTION COSTS")
    print("="*60)
    
    from src.transaction_costs import TransactionCostModel, ExecutionCost
    
    # Cost model
    cost_model = TransactionCostModel(
        commission_bps=5.0,
        spread_bps=2.0,
        market_impact_coef=0.1
    )
    
    current_w = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    target_w = np.array([0.20, 0.25, 0.30, 0.15, 0.10])
    aum = 1_000_000
    
    costs = cost_model.compute_total_costs(current_w, target_w, aum)
    
    print(f"Rebalancing Cost Analysis (AUM: ${aum/1e6:.0f}M):")
    print(f"  Total Cost:        ${costs['total_cost']:,.0f} ({costs['cost_bps']:.1f} bps)")
    print(f"  Commission/Spread: ${costs['detailed_costs']['fixed_commission_spread']:,.0f}")
    print(f"  Market Impact:     ${costs['detailed_costs']['market_impact']:,.0f}")
    print(f"  Turnover:          {costs['one_way_turnover']*100:.1f}%")
    
    # Execution cost comparison
    print(f"\nExecution Cost Estimates (1% of daily volume order):")
    vwap_cost = ExecutionCost.vwap_cost(order_size_pct=0.01, volatility=0.15)
    twap_cost = ExecutionCost.twap_cost(order_size_pct=0.01, volatility=0.15)
    moo_cost = ExecutionCost.market_on_open_cost(order_size_pct=0.01)
    
    print(f"  VWAP:  {vwap_cost:.1f} bps")
    print(f"  TWAP:  {twap_cost:.1f} bps")
    print(f"  MOO:   {moo_cost:.1f} bps")


def example_regime_detection():
    """Example 7: Market regime detection"""
    print("\n" + "="*60)
    print("EXAMPLE 7: REGIME DETECTION")
    print("="*60)
    
    try:
        from src.regime_detection import RegimeDetector
        
        # Synthetic market data with regime changes
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500)
        
        # Bull then Bear then Normal
        returns_bull = np.random.normal(0.0008, 0.01, 150)
        returns_bear = np.random.normal(-0.0005, 0.03, 150)
        returns_normal = np.random.normal(0.0003, 0.015, 200)
        
        returns = pd.DataFrame(
            np.concatenate([returns_bull, returns_bear, returns_normal]).reshape(-1, 1),
            index=dates,
            columns=['MARKET']
        )
        
        # Add some assets
        for i in range(4):
            returns[f'STOCK_{i}'] = returns['MARKET'] + np.random.normal(0, 0.01, 500)
        
        detector = RegimeDetector(returns, lookback=60)
        regimes = detector.detect_regimes_heuristic()
        
        current_regime = detector.get_current_regime()
        regime_names = {0: 'BULL', 1: 'NORMAL', 2: 'BEAR', 3: 'CRASH'}
        
        print(f"Current Regime: {regime_names.get(current_regime, 'UNKNOWN')}")
        
        stats = detector.get_regime_statistics()
        print(f"\nRegime Statistics:")
        print(stats.to_string())
    except Exception as e:
        print(f"Regime detection demo: {str(e)[:60]}...")


def example_stress_testing():
    """Example 8: Stress testing scenarios"""
    print("\n" + "="*60)
    print("EXAMPLE 8: STRESS TESTING")
    print("="*60)
    
    from src.stress_testing import (
        HistoricalStressScenarios,
        HypotheticalStressScenarios
    )
    
    # Portfolio
    portfolio_value = 1_000_000
    weights = np.array([0.40, 0.35, 0.15, 0.10])
    asset_classes = ['equity', 'equity', 'bond', 'alternative']
    
    # Historical crisis
    scenarios = [
        '2008_financial_crisis',
        '2020_covid_crash',
        '2022_rate_shock'
    ]
    
    print(f"Portfolio Value: ${portfolio_value/1e6:.1f}M")
    print(f"Allocation: Equity 75%, Bonds 15%, Alt 10%\n")
    
    print("Historical Stress Scenarios:")
    for scenario in scenarios:
        impact = HistoricalStressScenarios.apply_scenario(
            portfolio_value, weights, asset_classes, scenario
        )
        print(f"  {scenario:30} → Loss: {impact['loss_pct']:+.1f}% (${impact['loss_dollars']/1e3:+.0f}K)")
    
    # Hypothetical: Rate shock
    print("\nHypothetical Scenarios:")
    rate_shock = HypotheticalStressScenarios.parallel_yield_curve_shift(
        weights, asset_classes, yield_shift_bps=100
    )
    print(f"  +100 bps Rate Shock → {rate_shock['portfolio_shock_pct']:+.2f}%")


if __name__ == '__main__':
    print("\n" + "🚀 " * 20)
    print("PORTFOLIO OPTIMIZER - ADVANCED FEATURES EXAMPLES")
    print("🚀 " * 20)
    
    try:
        example_volatility_forecasting()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_sharpe_optimization()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_black_litterman()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_ml_alpha()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_factor_model()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_transaction_costs()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_regime_detection()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_stress_testing()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "✨ " * 20)
    print("Examples Complete! See ADVANCED_FEATURES_GUIDE.md for more details.")
    print("✨ " * 20 + "\n")
