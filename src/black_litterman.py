"""
Black-Litterman model for superior return estimation.
Blends market-implied returns (from equilibrium CAPM) with analyst views.
Solves under-diversification problems from pure mean-variance optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BlackLittermanModel:
    """
    Black-Litterman framework for combining market data with views.
    
    Key features:
    - Extracts market expectations from market cap weights
    - Incorporates analyst/systematic views with confidence levels
    - Produces well-diversified return estimates
    - Particularly useful when returns are hard to estimate
    """
    
    def __init__(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 2.5,
        risk_free_rate: float = 0.02,
        tau: float = 0.05
    ):
        """
        Args:
            market_weights: Market cap weights (1D array, sums to 1)
            cov_matrix: Covariance matrix of returns
            risk_aversion: Risk aversion parameter (typical 2-4)
            risk_free_rate: Risk-free rate for implied returns
            tau: Scalar uncertainty in market estimates (0.01 to 0.1)
        """
        self.w_mkt = np.asarray(market_weights, dtype=float).reshape(-1)
        self.sigma = np.asarray(cov_matrix, dtype=float)
        self.delta = risk_aversion
        self.rf = risk_free_rate
        self.tau = tau
        self.n = len(self.w_mkt)
        
        # Compute market-implied returns (equilibrium returns)
        self.mu_mkt = self._equilibrium_returns()
    
    def _equilibrium_returns(self) -> np.ndarray:
        """
        Extract market-implied (equilibrium) returns from market weights.
        Based on CAPM: E[R] = rf + delta * Sigma * w_mkt
        """
        return self.rf + self.delta * (self.sigma @ self.w_mkt)
    
    def add_view(self, view_matrix: np.ndarray, view_returns: np.ndarray, view_uncertainty: np.ndarray):
        """
        Add views on returns.
        
        Args:
            view_matrix: P matrix (n_views x n_assets), each row is a view
                        e.g., [[1, -1, 0]] means asset 1 outperforms asset 2
            view_returns: Expected returns under each view (1D, length n_views)
            view_uncertainty: Confidence in each view, expressed as variance
                             (1D array, higher = less confident)
        """
        self.P = np.asarray(view_matrix, dtype=float)
        self.q = np.asarray(view_returns, dtype=float).reshape(-1)
        self.Omega = np.diag(np.asarray(view_uncertainty, dtype=float))
    
    def posterior_returns(self) -> np.ndarray:
        """
        Compute posterior (blended) expected returns.
        
        Formula:
        E[R | views] = E[R_mkt] + tau * Sigma * P^T * (P * tau * Sigma * P^T + Omega)^-1 * (q - P * E[R_mkt])
        """
        if not hasattr(self, 'P'):
            # No views added, return market implied
            return self.mu_mkt.copy()
        
        # Covariance of views
        tau_sigma = self.tau * self.sigma
        cov_views = self.P @ tau_sigma @ self.P.T + self.Omega
        
        # View surprise
        surprise = self.q - (self.P @ self.mu_mkt)
        
        # Posterior returns
        delta_mu = tau_sigma @ self.P.T @ np.linalg.solve(cov_views, surprise)
        mu_posterior = self.mu_mkt + delta_mu
        
        return mu_posterior
    
    def posterior_covariance(self) -> np.ndarray:
        """
        Compute posterior covariance (typically slightly reduced from prior).
        """
        if not hasattr(self, 'P'):
            return self.sigma.copy()
        
        tau_sigma = self.tau * self.sigma
        cov_views = self.P @ tau_sigma @ self.P.T + self.Omega
        
        delta_sigma = tau_sigma @ self.P.T @ np.linalg.solve(cov_views, self.P) @ tau_sigma
        sigma_posterior = self.sigma - delta_sigma
        
        return 0.5 * (sigma_posterior + sigma_posterior.T)


def create_relative_views(
    tickers: list,
    outperformers: list,
    underperformers: list,
    expected_alpha: float = 0.02,
    confidence: float = 0.01
) -> tuple:
    """
    Helper to create relative-strength views (asset A will outperform asset B).
    
    Args:
        tickers: List of ticker symbols
        outperformers: List of tickers expected to outperform
        underperformers: List of tickers expected to underperform
        expected_alpha: Expected outperformance (e.g., 0.02 = 2% annual)
        confidence: Uncertainty in the view (higher = less confident)
    
    Returns:
        (P, q, Omega) ready for add_view()
    """
    n = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    
    P = np.zeros((1, n))
    for t in outperformers:
        P[0, ticker_idx[t]] = 1.0 / len(outperformers)
    for t in underperformers:
        P[0, ticker_idx[t]] = -1.0 / len(underperformers)
    
    q = np.array([expected_alpha])
    Omega = np.array([[confidence]])
    
    return P, q, Omega


def create_absolute_views(
    tickers: list,
    asset_returns: dict,
    confidence_by_asset: dict = None
) -> tuple:
    """
    Helper to create absolute return views (this asset will return X).
    
    Args:
        tickers: List of ticker symbols
        asset_returns: Dict mapping ticker -> expected annual return
        confidence_by_asset: Dict mapping ticker -> uncertainty (optional)
    
    Returns:
        (P, q, Omega) ready for add_view()
    """
    n = len(tickers)
    n_views = len(asset_returns)
    
    P = np.zeros((n_views, n))
    q = np.zeros(n_views)
    Omega_diag = np.zeros(n_views)
    
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    
    for view_idx, (ticker, ret) in enumerate(asset_returns.items()):
        P[view_idx, ticker_idx[ticker]] = 1.0
        q[view_idx] = ret
        
        if confidence_by_asset and ticker in confidence_by_asset:
            Omega_diag[view_idx] = confidence_by_asset[ticker]
        else:
            Omega_diag[view_idx] = 0.01
    
    Omega = np.diag(Omega_diag)
    
    return P, q, Omega
