"""
Comprehensive stress testing and scenario analysis framework.
Tests portfolio resilience to market shocks, tail events, and adverse scenarios.
Includes: historical shocks, hypothetical scenarios, sensitivity analysis, correlation breakdown.
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class HistoricalStressScenarios:
    """
    Apply historical market crisis scenarios to portfolio.
    
    Includes: 2008 Financial Crisis, 2011 Debt Crisis, 2020 COVID crash,
    2022 Rate Shock, Black Monday 1987, etc.
    """
    
    # Define major historical crises
    CRISIS_SCENARIOS = {
        '2008_financial_crisis': {
            'name': '2008 Financial Crisis',
            'equity_shock': -0.37,
            'bond_shock': -0.05,
            'credit_shock': -0.25,
            'volatility_multiplier': 3.5,
            'correlation_increase': 0.3
        },
        '2011_debt_crisis': {
            'name': '2011 European Debt Crisis',
            'equity_shock': -0.20,
            'bond_shock': 0.05,
            'credit_shock': -0.15,
            'volatility_multiplier': 2.0,
            'correlation_increase': 0.2
        },
        '2020_covid_crash': {
            'name': '2020 COVID-19 Crash',
            'equity_shock': -0.34,
            'bond_shock': 0.10,
            'credit_shock': -0.20,
            'volatility_multiplier': 4.0,
            'correlation_increase': 0.4
        },
        '2022_rate_shock': {
            'name': '2022 Interest Rate Shock',
            'equity_shock': -0.18,
            'bond_shock': -0.16,
            'credit_shock': -0.12,
            'volatility_multiplier': 1.8,
            'correlation_increase': 0.15
        },
        'black_monday_1987': {
            'name': 'Black Monday 1987',
            'equity_shock': -0.22,
            'bond_shock': 0.02,
            'credit_shock': -0.15,
            'volatility_multiplier': 5.0,
            'correlation_increase': 0.5
        },
        'vix_spike': {
            'name': 'VIX Spike (300%+)',
            'equity_shock': -0.15,
            'bond_shock': 0.02,
            'credit_shock': -0.10,
            'volatility_multiplier': 2.5,
            'correlation_increase': 0.25
        }
    }
    
    @staticmethod
    def apply_scenario(
        portfolio_value: float,
        weights: np.ndarray,
        asset_classes: list,
        scenario_name: str
    ) -> dict:
        """
        Apply historical crisis scenario to portfolio.
        
        Args:
            portfolio_value: Current portfolio value
            weights: Portfolio weights
            asset_classes: Asset class for each holding
            scenario_name: Key from CRISIS_SCENARIOS
        
        Returns:
            dict with portfolio impact
        """
        if scenario_name not in HistoricalStressScenarios.CRISIS_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = HistoricalStressScenarios.CRISIS_SCENARIOS[scenario_name]
        
        # Map asset classes to shocks
        shocks = np.zeros_like(weights, dtype=float)
        for i, ac in enumerate(asset_classes):
            if 'equity' in ac.lower() or 'stock' in ac.lower():
                shocks[i] = scenario['equity_shock']
            elif 'bond' in ac.lower() or 'fixed' in ac.lower():
                shocks[i] = scenario['bond_shock']
            elif 'credit' in ac.lower() or 'high_yield' in ac.lower():
                shocks[i] = scenario['credit_shock']
            else:
                shocks[i] = scenario['equity_shock'] * 0.5  # Mild shock for others
        
        # Portfolio shock
        portfolio_shock = np.sum(weights * shocks)
        new_value = portfolio_value * (1 + portfolio_shock)
        loss_dollars = portfolio_value - new_value
        loss_pct = -portfolio_shock * 100
        
        return {
            'scenario': scenario['name'],
            'portfolio_shock_pct': portfolio_shock * 100,
            'new_portfolio_value': new_value,
            'loss_dollars': loss_dollars,
            'loss_pct': loss_pct,
            'asset_shocks': shocks * 100,
            'volatility_multiplier': scenario['volatility_multiplier'],
            'correlation_increase': scenario['correlation_increase']
        }


class HypotheticalStressScenarios:
    """
    Create custom hypothetical stress scenarios.
    Examples: parallel yield curve shift, sector rotation, commodity spike, etc.
    """
    
    @staticmethod
    def parallel_yield_curve_shift(
        weights: np.ndarray,
        asset_classes: list,
        yield_shift_bps: float = 100
    ) -> dict:
        """
        Scenario: Parallel shift in yield curve (e.g., rates up 100 bps).
        
        Impact:
        - Bonds: negative (price declines with rate increase)
        - Equities: mixed (lower valuation, but growth concerns)
        """
        duration = 5  # Average duration of bond holdings
        
        shocks = np.zeros_like(weights, dtype=float)
        for i, ac in enumerate(asset_classes):
            if 'bond' in ac.lower() or 'fixed' in ac.lower():
                # Bond price decline ≈ -duration * yield_change
                shocks[i] = -(duration / 100) * yield_shift_bps / 10000
            elif 'equity' in ac.lower():
                # Equities down less, higher rates compress multiples
                shocks[i] = -(yield_shift_bps / 10000) * 2
            else:
                shocks[i] = 0
        
        portfolio_shock = np.sum(weights * shocks)
        
        return {
            'scenario': f'Parallel Yield Curve Shift: +{yield_shift_bps} bps',
            'portfolio_shock_pct': portfolio_shock * 100,
            'asset_shocks': shocks * 100
        }
    
    @staticmethod
    def sector_rotation(
        weights: np.ndarray,
        sectors: list,
        winners: list,
        losers: list,
        magnitude: float = 0.20
    ) -> dict:
        """
        Scenario: Rotation from one sector to another.
        Winners gain, losers decline.
        
        Example: Defensive to Cyclical (recession recovery)
        """
        shocks = np.zeros_like(weights, dtype=float)
        
        for i, sector in enumerate(sectors):
            if sector in winners:
                shocks[i] = magnitude / len(winners)
            elif sector in losers:
                shocks[i] = -magnitude / len(losers)
        
        portfolio_shock = np.sum(weights * shocks)
        
        return {
            'scenario': f'Sector Rotation: {winners} vs {losers}',
            'portfolio_shock_pct': portfolio_shock * 100,
            'asset_shocks': shocks * 100
        }
    
    @staticmethod
    def correlation_breakdown(
        sigma: np.ndarray,
        weights: np.ndarray,
        correlation_multiplier: float = 0.5
    ) -> np.ndarray:
        """
        Scenario: Diversification benefits evaporate (correlations increase).
        
        During crises, correlations spike toward 1.0.
        This scenarios compresses the covariance matrix accordingly.
        """
        # Extract correlations and volatilities
        corr = np.corrcoef(sigma)
        vols = np.sqrt(np.diag(sigma))
        
        # Increase correlations (move toward 1.0)
        corr_stressed = corr.copy()
        off_diag = ~np.eye(len(corr), dtype=bool)
        corr_stressed[off_diag] = corr[off_diag] + correlation_multiplier * (1 - corr[off_diag])
        
        # Reconstruct covariance
        D = np.diag(vols)
        sigma_stressed = D @ corr_stressed @ D
        
        # Portfolio volatility increase
        vol_base = np.sqrt(weights @ sigma @ weights)
        vol_stressed = np.sqrt(weights @ sigma_stressed @ weights)
        
        return {
            'correlation_matrix_stressed': corr_stressed,
            'covariance_stressed': sigma_stressed,
            'base_vol': vol_base,
            'stressed_vol': vol_stressed,
            'vol_increase_pct': (vol_stressed / vol_base - 1) * 100
        }


class SensitivityAnalysis:
    """
    Sensitivity analysis: how does portfolio perform with changes to key inputs?
    
    Useful for:
    - Understanding key risk drivers
    - Identifying hidden assumptions
    - Stress testing assumptions
    """
    
    @staticmethod
    def volatility_sensitivity(
        mu: np.ndarray,
        sigma: np.ndarray,
        weights: np.ndarray,
        vol_shifts: list = None
    ) -> pd.DataFrame:
        """
        What if volatility changes by X%?
        
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            weights: Portfolio weights
            vol_shifts: List of volatility shifts (e.g., [-0.2, -0.1, 0, 0.1, 0.2])
        
        Returns:
            DataFrame with Sharpe ratio under each vol scenario
        """
        if vol_shifts is None:
            vol_shifts = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
        
        results = []
        
        for shift in vol_shifts:
            # Shift volatilities
            sigma_stressed = sigma.copy()
            diag_vol = np.sqrt(np.diag(sigma))
            sigma_stressed = sigma_stressed * (1 + shift)
            
            port_return = np.sum(weights * mu)
            port_vol = np.sqrt(weights @ sigma_stressed @ weights)
            sharpe = (port_return - 0.02) / max(port_vol, 1e-10)
            
            results.append({
                'vol_shift_pct': shift * 100,
                'portfolio_return': port_return * 100,
                'portfolio_volatility': port_vol * 100,
                'sharpe_ratio': sharpe
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def correlation_sensitivity(
        sigma: np.ndarray,
        weights: np.ndarray,
        correlation_shifts: list = None
    ) -> pd.DataFrame:
        """
        What if correlations increase/decrease?
        
        Shows benefit of diversification under stress.
        """
        if correlation_shifts is None:
            correlation_shifts = [-0.2, -0.1, 0, 0.1, 0.2, 0.3]
        
        results = []
        vols_base = np.sqrt(np.diag(sigma))
        
        for shift in correlation_shifts:
            corr = np.corrcoef(sigma)
            corr_stressed = corr.copy()
            
            # Shift correlations
            off_diag = ~np.eye(len(corr), dtype=bool)
            corr_stressed[off_diag] = corr[off_diag] + shift
            # Clip to valid range
            corr_stressed = np.clip(corr_stressed, -1, 1)
            
            # Reconstruct covariance
            D = np.diag(vols_base)
            sigma_stressed = D @ corr_stressed @ D
            
            port_vol = np.sqrt(weights @ sigma_stressed @ weights)
            
            results.append({
                'correlation_shift': shift,
                'portfolio_volatility': port_vol * 100,
                'diversification_benefit': (np.sum(weights * vols_base) - port_vol) * 100
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def return_assumption_sensitivity(
        mu: np.ndarray,
        sigma: np.ndarray,
        weights: np.ndarray,
        return_shifts: list = None
    ) -> pd.DataFrame:
        """
        What if expected returns are different than assumed?
        
        Helps understand robustness to return estimation errors.
        """
        if return_shifts is None:
            return_shifts = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]
        
        results = []
        port_vol = np.sqrt(weights @ sigma @ weights)
        
        for shift in return_shifts:
            mu_stressed = mu + shift
            port_return = np.sum(weights * mu_stressed)
            sharpe = (port_return - 0.02) / max(port_vol, 1e-10)
            
            results.append({
                'return_shift_pct': shift * 100,
                'portfolio_return': port_return * 100,
                'sharpe_ratio': sharpe
            })
        
        return pd.DataFrame(results)
