"""
Multi-factor risk model for decomposing portfolio risk into systematic sources.
Includes: market, momentum, value, quality, low-volatility factors.
Enables better understanding and control of factor exposures.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class FactorModel:
    """
    Linear factor model: R = alpha + b1*F1 + b2*F2 + ... + epsilon
    
    Factors can be:
    - Market (beta to broad index)
    - Size (small vs large cap)
    - Value (high book-to-market)
    - Momentum (recent winners)
    - Quality (profitability, low debt)
    - Low-Vol (low volatility)
    """
    
    def __init__(self, returns: pd.DataFrame, factors: pd.DataFrame = None):
        """
        Args:
            returns: Asset returns DataFrame (T x n_assets)
            factors: Factor returns DataFrame (T x n_factors), optional
                    If None, will auto-construct common factors
        """
        self.returns = returns.copy()
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        
        if factors is None:
            self.factors = self._construct_factors(returns)
        else:
            self.factors = factors.copy()
        
        self.factor_names = list(self.factors.columns)
        self.n_factors = len(self.factor_names)
        
        self.alphas = None
        self.betas = None
        self.residuals = None
        self.factor_vols = None
    
    def _construct_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Construct common factors from returns data.
        
        Factors created:
        - Market: 1/N portfolio return (broad market)
        - Momentum: Difference between top 30% and bottom 30% performers (21-day)
        - Value: Mean-reversion factor
        - Quality: Inverse of volatility
        - Low-Vol: Inverse of rolling volatility
        """
        factors_dict = {}
        
        # Market factor: equal-weight return
        factors_dict['Market'] = returns.mean(axis=1)
        
        # Momentum: top 30% vs bottom 30%
        for i in range(21, len(returns)):
            window = returns.iloc[i-21:i]
            cum_rets = (1 + window).prod() - 1
            
            threshold_high = cum_rets.quantile(0.7)
            threshold_low = cum_rets.quantile(0.3)
            
            winners = (cum_rets >= threshold_high).astype(float)
            losers = (cum_rets <= threshold_low).astype(float)
            
            momentum = winners.sum() - losers.sum()
            if (winners.sum() + losers.sum()) > 0:
                momentum = momentum / (winners.sum() + losers.sum())
        
        # Simplify: 21-day momentum factor
        mom_factor = []
        for i in range(21, len(returns)):
            window_rets = returns.iloc[i-21:i]
            cumulative = (1 + window_rets).prod() - 1
            avg_perf = cumulative.mean()
            mom_factor.append(avg_perf)
        
        mom_factor = pd.Series(mom_factor, index=returns.index[21:], name='Momentum')
        factors_dict['Momentum'] = pd.Series(0.0, index=returns.index)
        factors_dict['Momentum'].iloc[21:] = mom_factor.values
        
        # Value: mean-reversion based on recent underperformance
        value_factor = []
        for i in range(21, len(returns)):
            window_rets = returns.iloc[i-21:i]
            lagged_perf = window_rets.mean()
            next_mean = lagged_perf.mean()
            value_signal = -(lagged_perf - next_mean).mean()
            value_factor.append(value_signal)
        
        value_factor = pd.Series(0.0, index=returns.index)
        if len(value_factor.iloc[21:]) == len(value_factor):
            pass
        else:
            value_factor.iloc[21:len(value_factor)] = value_factor.iloc[:len(value_factor)-21].values
        factors_dict['Value'] = value_factor
        
        # Quality: inverse volatility (stability reward)
        quality_factor = -returns.rolling(21).std().mean(axis=1)
        factors_dict['Quality'] = quality_factor
        
        # Low-Vol: inverse of volatility
        factors_dict['LowVol'] = -returns.rolling(21).std().mean(axis=1)
        
        return pd.DataFrame(factors_dict)
    
    def fit(self):
        """Fit factor model to returns using cross-sectional regression."""
        # Align data
        common_index = self.returns.index.intersection(self.factors.index)
        X = self.factors.loc[common_index].values  # (T x n_factors)
        Y = self.returns.loc[common_index].values  # (T x n_assets)
        
        # Add intercept for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Fit: Y = X_with_const @ beta (betas are columns)
        self.betas_with_alpha = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]
        
        self.alphas = self.betas_with_alpha[0, :].copy()  # Intercepts
        self.betas = self.betas_with_alpha[1:, :].copy()  # Factor loadings
        
        # Residuals
        Y_fitted = X_with_const @ self.betas_with_alpha
        self.residuals = Y - Y_fitted
        self.residual_vol = np.std(self.residuals, axis=0)
        
        # Factor volatility
        self.factor_vols = np.std(X, axis=0)
    
    def decompose_risk(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Decompose portfolio risk into factor contributions.
        
        Args:
            weights: Portfolio weights (n_assets,)
        
        Returns:
            DataFrame showing risk contribution by factor
        """
        if self.betas is None:
            self.fit()
        
        w = np.asarray(weights, dtype=float).reshape(-1)
        
        # Portfolio betas
        portfolio_betas = self.betas @ w  # (n_factors,)
        
        # Factor covariance
        X = self.factors.iloc[self.returns.index.intersection(self.factors.index)].values
        factor_cov = np.cov(X.T)  # (n_factors x n_factors)
        
        # Risk contribution: each factor's variance contribution to portfolio
        risk_contrib = portfolio_betas**2 * np.diag(factor_cov)
        
        # Idiosyncratic risk
        idio_risk = (w**2) @ (self.residual_vol**2)
        
        return pd.DataFrame({
            "factor": self.factor_names + ["Idiosyncratic"],
            "risk_contribution": np.concatenate([risk_contrib, [idio_risk]]),
            "pct_of_total": np.concatenate([
                risk_contrib / (np.sum(risk_contrib) + idio_risk),
                [idio_risk / (np.sum(risk_contrib) + idio_risk)]
            ])
        })
    
    def get_factor_loadings(self, assets: list = None) -> pd.DataFrame:
        """
        Get factor loadings (betas) for specified assets.
        
        Args:
            assets: List of asset names, or None for all
        
        Returns:
            DataFrame of loadings
        """
        if self.betas is None:
            self.fit()
        
        if assets is None:
            assets = self.assets
        
        indices = [self.assets.index(a) for a in assets if a in self.assets]
        
        return pd.DataFrame(
            self.betas[:, indices].T,
            columns=self.factor_names,
            index=[self.assets[i] for i in indices]
        )
    
    def factor_attribution(self, weights: np.ndarray, period_returns: np.ndarray) -> dict:
        """
        Attribution analysis: decompose return into factor contributions.
        
        Args:
            weights: Portfolio weights
            period_returns: Portfolio returns over period
        
        Returns:
            dict with contribution of each factor to return
        """
        if self.betas is None:
            self.fit()
        
        w = np.asarray(weights, dtype=float).reshape(-1)
        portfolio_betas = self.betas @ w
        
        # Factor returns during period
        X = self.factors.iloc[self.returns.index.intersection(self.factors.index)].values
        avg_factor_rets = np.mean(X, axis=0)
        
        # Contribution = beta * factor_return
        contributions = portfolio_betas * avg_factor_rets
        alpha_contrib = self.alphas @ w
        
        attribution = {}
        for factor, contrib in zip(self.factor_names, contributions):
            attribution[factor] = contrib
        attribution['Alpha'] = alpha_contrib
        
        return attribution


def create_custom_factors(
    returns: pd.DataFrame,
    market_index: pd.Series = None,
    dividend_yields: dict = None,
    earnings_growth: dict = None
) -> pd.DataFrame:
    """
    Create custom factors from market data.
    
    Args:
        returns: Asset returns
        market_index: Market index returns (for market factor)
        dividend_yields: Dict mapping asset -> dividend yield
        earnings_growth: Dict mapping asset -> EPS growth rate
    
    Returns:
        DataFrame of custom factors
    """
    factors = {}
    
    # Market factor
    if market_index is not None:
        factors['Market'] = market_index.copy()
    else:
        factors['Market'] = returns.mean(axis=1)
    
    # Value factor: high dividend yield
    if dividend_yields:
        div_factor = []
        for date in returns.index:
            div_vec = np.array([
                dividend_yields.get(asset, 0.02)
                for asset in returns.columns
            ])
            div_factor.append(np.mean(div_vec))
        factors['Dividend_Yield'] = pd.Series(div_factor, index=returns.index)
    
    # Growth factor: EPS growth
    if earnings_growth:
        growth_factor = []
        for date in returns.index:
            growth_vec = np.array([
                earnings_growth.get(asset, 0.05)
                for asset in returns.columns
            ])
            growth_factor.append(np.mean(growth_vec))
        factors['EPS_Growth'] = pd.Series(growth_factor, index=returns.index)
    
    return pd.DataFrame(factors)
