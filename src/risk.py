import numpy as np
import pandas as pd

def annualize_mu_sigma(mu_daily, sigma_daily, trading_days=252):
    mu = mu_daily * trading_days
    sigma = sigma_daily * trading_days
    return mu, sigma

def shrink_cov_to_diagonal(sigma, alpha=0.15):
    if alpha <= 0:
        return sigma
    if alpha >= 1:
        return np.diag(np.diag(sigma))
    diag = np.diag(np.diag(sigma))
    return (1 - alpha) * sigma + alpha * diag

def make_psd(sigma, eps=1e-10):
    sigma = 0.5 * (sigma + sigma.T)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals_clipped = np.maximum(eigvals, eps)
    sigma_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    return 0.5 * (sigma_psd + sigma_psd.T)

def estimate_mu_sigma(returns, trading_days=252, cov_shrink_alpha=0.15, ensure_psd=True, mu_shrink_beta=0.10, mu_clip: float | None = 1.0):
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    if returns.isna().any().any():
        raise ValueError("Returns DataFrame contains NaN values.")
    
    assets = list(returns.columns)
    X = returns.values
    
    # Handle low-variance stocks by adding small regularization
    mu_daily = X.mean(axis=0)
    sigma_daily = np.cov(X, rowvar=False)
    
    # Ensure sigma_daily is 2D (handles single asset case)
    if sigma_daily.ndim == 0:
        sigma_daily = np.array([[sigma_daily]])
    elif sigma_daily.ndim == 1:
        sigma_daily = np.diag(sigma_daily)
    
    mu, sigma = annualize_mu_sigma(mu_daily, sigma_daily, trading_days=trading_days)
    
    # Apply stronger shrinkage for better numerical stability
    # Use adaptive shrinkage based on condition number
    cond_num = np.linalg.cond(sigma)
    if cond_num > 1000:  # Ill-conditioned matrix
        adaptive_shrink = min(0.5, cov_shrink_alpha * 3)  # Increase shrinkage
    else:
        adaptive_shrink = cov_shrink_alpha
    
    sigma = shrink_cov_to_diagonal(sigma, alpha=adaptive_shrink)
    
    # Add small ridge regularization for numerical stability
    sigma = sigma + 1e-6 * np.eye(len(sigma))
    
    if ensure_psd:
        sigma = make_psd(sigma)
    
    mu = (1 - mu_shrink_beta) * mu
    if mu_clip is not None:
        mu = np.clip(mu, -mu_clip, mu_clip)
    
    return mu, sigma, assets

def portfolio_vol(sigma, w):
    return float(np.sqrt(w @ sigma @ w))

def portfolio_return(mu, w):
    return float(mu @ w)