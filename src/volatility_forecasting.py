"""
Advanced volatility forecasting using GARCH and regime-switching models.
Provides forward-looking covariance matrix estimates for superior optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class GARCHVolatility:
    """
    Univariate GARCH(1,1) model for dynamic volatility forecasting.
    Superior to historical volatility for recent market conditions.
    """
    
    def __init__(self, returns: np.ndarray, omega=None, alpha=None, beta=None):
        """
        Args:
            returns: 1D array of log returns
            omega, alpha, beta: Optional starting parameters
        """
        self.returns = np.asarray(returns, dtype=float).flatten()
        if len(self.returns) < 10:
            raise ValueError("Need at least 10 observations for GARCH estimation")
        
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
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
        h_last = self.omega + self.alpha * (self.returns[-1]**2) + self.beta * h
        
        forecast = [h_last]
        for _ in range(steps - 1):
            h_next = self.omega + (self.alpha + self.beta) * forecast[-1]
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
    
    def __init__(self, returns_df: pd.DataFrame, use_dcc=True, decay_factor=0.94):
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
        
        self.garch_models = {}
        self.conditional_vols = {}
        self._fit_garch()
    
    def _fit_garch(self):
        """Fit GARCH model to each asset."""
        for asset in self.assets:
            try:
                model = GARCHVolatility(self.returns_df[asset].values)
                self.garch_models[asset] = model
                self.conditional_vols[asset] = model.get_conditional_vol()
            except Exception:
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
        
        # Exponentially weighted correlation
        weights = self.decay_factor ** np.arange(T)[::-1]
        weights /= weights.sum()
        
        corr_matrix = np.corrcoef(Z.T, ddof=0)
        corr_matrix = np.nan_to_num(corr_matrix)
        corr_matrix = 0.5 * (corr_matrix + np.eye(self.n_assets))  # Shrink to diagonal
        
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
    mu_shrink_beta: float = 0.50,
    mu_clip: float = 0.50
) -> tuple:
    """
    Generate forward-looking mu and sigma with GARCH vol forecasting.
    
    Returns:
        (mu, sigma, assets) where sigma is forward-looking from GARCH
    """
    from src.risk import annualize_mu_sigma, shrink_cov_to_diagonal, make_psd
    
    assets = list(returns.columns)
    X = returns.values
    
    mu_daily = X.mean(axis=0)
    mu, _ = annualize_mu_sigma(mu_daily, np.std(X, axis=0), trading_days)
    
    # Use dynamic covariance for forward-looking sigma
    if use_garch:
        dyn_cov = DynamicCovarianceMatrix(returns)
        sigma = dyn_cov.forecast_covariance(periods_ahead=1)
    else:
        sigma_daily = np.cov(X, rowvar=False)
        sigma, _ = annualize_mu_sigma(mu_daily, sigma_daily / np.sqrt(trading_days), trading_days)
    
    sigma = shrink_cov_to_diagonal(sigma, alpha=cov_shrink_alpha)
    
    if True:  # Always ensure PSD
        sigma = make_psd(sigma)
    
    # Apply shrinkage to returns
    mu = (1 - mu_shrink_beta) * mu
    if mu_clip is not None:
        mu = np.clip(mu, -mu_clip, mu_clip)
    
    return mu, sigma, assets
