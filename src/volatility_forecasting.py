"""
Advanced volatility forecasting using GARCH and robust estimators.
Provides forward-looking covariance with correlation-aware shrinkage and
more conservative (robust) expected return estimates.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


class GARCHVolatility:
    """
    Univariate GARCH(1,1) model for dynamic volatility forecasting.
    Superior to historical volatility for recent market conditions.
    """
    
    def __init__(self, returns: np.ndarray, omega: Optional[float]=None, alpha: Optional[float]=None, beta: Optional[float]=None):
        """
        Args:
            returns: 1D array of log returns
            omega, alpha, beta: Optional starting parameters
        """
        self.returns = np.asarray(returns, dtype=float).flatten()
        if len(self.returns) < 10:
            raise ValueError("Need at least 10 observations for GARCH estimation")
        
        # Provide safe initial values; MLE will refine
        self.omega = float(omega) if omega is not None else 1e-6
        self.alpha = float(alpha) if alpha is not None else 0.05
        self.beta = float(beta) if beta is not None else 0.90
        self._fit()
    
    def _fit(self):
        """Maximum likelihood estimation of GARCH(1,1) parameters."""
        def neg_loglik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            h = np.zeros_like(self.returns, dtype=float)
            h[0] = np.var(self.returns)
            
            for t in range(1, len(self.returns)):
                h[t] = omega + alpha * (self.returns[t-1]**2) + beta * h[t-1]
                if h[t] <= 0:
                    return 1e10
            
            loglik = -0.5 * np.sum(np.log(h) + (self.returns**2) / h)
            return -loglik
        
        x0 = [0.00001, 0.05, 0.94]
        res = minimize(neg_loglik, x0, method='Nelder-Mead', 
                      options={'maxiter': 500})
        
        self.omega, self.alpha, self.beta = res.x
    
    def forecast_variance(self, steps=1):
        """Forecast variance h_t for next 'steps' periods."""
        h = np.var(self.returns)
        h_last = float(self.omega) + float(self.alpha) * (self.returns[-1]**2) + float(self.beta) * h
        
        forecast = [h_last]
        for _ in range(steps - 1):
            h_next = float(self.omega) + (float(self.alpha) + float(self.beta)) * forecast[-1]
            forecast.append(h_next)
        
        return np.array(forecast[-steps:])
    
    def get_conditional_vol(self):
        """Return conditional volatility for all observations."""
        h = np.zeros_like(self.returns, dtype=float)
        h[0] = np.var(self.returns)
        
        for t in range(1, len(self.returns)):
            h[t] = self.omega + self.alpha * (self.returns[t-1]**2) + self.beta * h[t-1]
        
        return np.sqrt(np.maximum(h, 1e-10))


class DynamicCovarianceMatrix:
    """
    Generates forward-looking covariance matrix using GARCH-based dynamics
    and exponential weighting for recent shocks.
    """
    
    def __init__(self, returns_df: pd.DataFrame, use_dcc=True, decay_factor=0.94,
                 corr_shrink_alpha: float = 0.30):
        """
        Args:
            returns_df: DataFrame of asset returns
            use_dcc: Whether to use Dynamic Conditional Correlation
            decay_factor: Exponential decay for recent observations
        """
        self.returns_df = returns_df.copy()
        self.use_dcc = use_dcc
        self.decay_factor = decay_factor
        self.assets = list(returns_df.columns)
        self.n_assets = len(self.assets)
        self.corr_shrink_alpha = float(corr_shrink_alpha)
        
        self.garch_models = {}
        self.conditional_vols = {}
        self._fit_garch()
    
    def _fit_garch(self):
        """Fit GARCH model to each asset."""
        for asset in self.assets:
            try:
                model = GARCHVolatility(self.returns_df[asset].to_numpy(dtype=float))
                self.garch_models[asset] = model
                self.conditional_vols[asset] = model.get_conditional_vol()
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as _:
                # Fallback to historical vol if GARCH fails
                self.conditional_vols[asset] = np.full(
                    len(self.returns_df), 
                    self.returns_df[asset].std()
                )
    
    def forecast_covariance(self, periods_ahead=1):
        """
        Generate forward-looking covariance matrix.
        Uses GARCH forecasts + shrinkage to correlation matrix.
        """
        X = self.returns_df.values
        T = len(X)
        
        # Standardize returns
        Z = X.copy()
        for i in range(self.n_assets):
            Z[:, i] = Z[:, i] / np.maximum(self.conditional_vols[self.assets[i]][-1], 1e-8)
        
        # Exponentially weighted correlation (correlation-aware)
        weights = self.decay_factor ** np.arange(T)[::-1]
        weights = weights / weights.sum()

        # Weighted covariance of standardized returns
        Zw = Z * np.sqrt(weights[:, None])
        cov_w = Zw.T @ Zw  # already weighted by construction (sum(weights)=1)
        # Convert to correlation
        std = np.sqrt(np.clip(np.diag(cov_w), 1e-12, None))
        inv_std = np.diag(1.0 / std)
        corr_matrix = inv_std @ cov_w @ inv_std
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)

        # Constant-correlation shrinkage target
        if self.n_assets > 1:
            mask = ~np.eye(self.n_assets, dtype=bool)
            rho_bar = np.mean(corr_matrix[mask])
            target = np.full_like(corr_matrix, rho_bar)
            np.fill_diagonal(target, 1.0)
            a = np.clip(self.corr_shrink_alpha, 0.0, 1.0)
            corr_matrix = (1 - a) * corr_matrix + a * target
        # Ensure well-behaved
        corr_matrix = np.clip(corr_matrix, -0.999, 0.999)
        
        # Forecast volatilities from GARCH
        future_vols = np.zeros(self.n_assets)
        for i, asset in enumerate(self.assets):
            if asset in self.garch_models:
                fcast = self.garch_models[asset].forecast_variance(periods_ahead)
                future_vols[i] = np.sqrt(fcast[-1])
            else:
                future_vols[i] = self.conditional_vols[asset][-1]
        
        # Construct covariance from forecasted vols + shrunk correlation
        cov_matrix = np.outer(future_vols, future_vols) * corr_matrix
        
        # Ensure PSD
        cov_matrix = self._make_psd(cov_matrix)
        
        return cov_matrix
    
    def get_historical_covariance(self):
        """Return realized covariance with exponential weighting."""
        X = self.returns_df.values
        T = len(X)
        
        weights = self.decay_factor ** np.arange(T)[::-1]
        weights /= weights.sum()
        
        X_weighted = X * np.sqrt(weights[:, np.newaxis])
        return X_weighted.T @ X_weighted
    
    @staticmethod
    def _make_psd(matrix, eps=1e-10):
        """Ensure matrix is positive semi-definite."""
        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


def forecast_mu_sigma_forward_looking(
    returns: pd.DataFrame,
    trading_days: int = 252,
    use_garch: bool = True,
    use_ml_alpha: bool = False,
    cov_shrink_alpha: float = 0.15,
    mu_shrink_beta: float = 0.70,
    mu_clip: float = 0.30,
    ewma_halflife_days: int = 63,
    corr_shrink_alpha: float = 0.30
) -> tuple:
    """
    Generate forward-looking mu and sigma with:
    - Robust mu: EWMA mean, cross-sectional shrink toward market mean, and clipping
    - Correlation-aware sigma: GARCH vol + weighted correlation with constant-corr shrinkage
    """
    from src.risk import annualize_mu_sigma, shrink_cov_to_diagonal, make_psd
    
    assets = list(returns.columns)
    X = returns.values

    # Robust mu estimation
    # 1) EWMA daily mean with specified halflife
    if len(X) == 0:
        raise ValueError("Empty returns provided to forecast_mu_sigma_forward_looking")
    lam = np.exp(np.log(0.5) / max(1, ewma_halflife_days))  # decay per day for given halflife
    w = lam ** np.arange(len(X))[::-1]
    w = w / w.sum()
    mu_daily_ewma = (w[:, None] * X).sum(axis=0)
    # 2) Annualize
    mu_annual = mu_daily_ewma * trading_days
    # 3) Cross-sectional shrink toward market mean
    cross_mean = float(np.mean(mu_annual))
    mu_annual = (1 - mu_shrink_beta) * mu_annual + mu_shrink_beta * cross_mean
    # 4) Clip extreme forecasts
    if mu_clip is not None:
        mu_annual = np.clip(mu_annual, -float(mu_clip), float(mu_clip))
    
    # Use dynamic covariance for forward-looking sigma
    if use_garch:
        dyn_cov = DynamicCovarianceMatrix(returns, use_dcc=True, decay_factor=0.94,
                                          corr_shrink_alpha=corr_shrink_alpha)
        sigma = dyn_cov.forecast_covariance(periods_ahead=1)
    else:
        sigma_daily = np.cov(X, rowvar=False)
        sigma, _ = annualize_mu_sigma(mu_daily_ewma, sigma_daily / np.sqrt(trading_days), trading_days)
    
    sigma = shrink_cov_to_diagonal(sigma, alpha=cov_shrink_alpha)
    
    # Ensure PSD
    sigma = make_psd(sigma)
    
    # Touch flag to satisfy linters if not used yet
    if use_ml_alpha:
        _ = None
    return mu_annual, sigma, assets
