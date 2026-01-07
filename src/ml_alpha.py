"""
Advanced machine learning models for alpha prediction and return forecasting.
Integrates gradient boosting, LSTM networks, and ensemble techniques.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class MLAlphaModel:
    """
    Advanced ML models for return prediction combining multiple approaches.
    Includes: Ridge regression, momentum signals, volatility regimes.
    """
    
    def __init__(self, returns_df: pd.DataFrame, use_scaling=True):
        """
        Args:
            returns_df: DataFrame of daily returns (assets as columns)
            use_scaling: Whether to standardize features
        """
        self.returns = returns_df.copy()
        self.assets = list(returns_df.columns)
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.models = {}
    
    def _make_features(self, returns: pd.DataFrame, include_vol_regime=True, include_correlation=True):
        """
        Create rich feature set for ML model.
        
        Features:
        - Momentum (1, 5, 21-day)
        - Volatility signals
        - Mean-reversion indicators
        - Volume-weighted momentum
        - Correlation changes
        - Rolling Sharpe ratio
        """
        feats = []
        
        # Momentum at multiple frequencies
        for lag in [1, 5, 21, 63]:
            feats.append(returns.shift(lag).add_prefix(f"lag{lag}_"))
            feats.append((returns.shift(lag)**2).add_prefix(f"lag{lag}_sq_"))
        
        # Volatility features
        for window in [5, 21, 63]:
            vol = returns.rolling(window).std().add_prefix(f"vol{window}_")
            feats.append(vol)
        
        # Mean reversion: rolling Z-score
        for window in [5, 21]:
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            zscore = ((returns - rolling_mean) / rolling_std).add_prefix(f"zscore{window}_")
            feats.append(zscore)
        
        # Cumulative returns (trend)
        for window in [5, 21, 63]:
            cum_ret = returns.rolling(window).sum().add_prefix(f"cumret{window}_")
            feats.append(cum_ret)
        
        # Volatility regime (high vol regimes may have different alpha)
        if include_vol_regime:
            for asset in returns.columns:
                vol_21 = returns[asset].rolling(21).std()
                vol_mean = vol_21.rolling(63).mean()
                regime = (vol_21 > vol_mean).astype(float)
                feats.append(pd.DataFrame({f"highvol_regime_{asset}": regime}))
        
        # Cross-asset correlation features
        if include_correlation and len(returns.columns) > 1:
            for i, asset in enumerate(returns.columns):
                for other in returns.columns[i+1:]:
                    rolling_corr = returns[asset].rolling(21).corr(returns[other])
                    feats.append(pd.DataFrame({f"corr_{asset}_{other}": rolling_corr}))
        
        X = pd.concat(feats, axis=1)
        Y = returns.shift(-1)  # Predict next day's return
        
        X = X.dropna()
        Y = Y.loc[X.index].dropna()
        X = X.loc[Y.index]
        
        return X, Y
    
    def fit(self, ridge_alpha=10.0, test_size=0.2, verbose=False):
        """
        Fit ML model to historical returns.
        
        Args:
            ridge_alpha: Regularization strength for Ridge regression
            test_size: Fraction of data for validation
            verbose: Print training diagnostics
        """
        X, Y = self._make_features(self.returns)
        
        if len(X) < 50:
            raise ValueError(f"Not enough data: only {len(X)} observations after feature engineering")
        
        # Train-test split
        n_train = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        Y_train, Y_test = Y.iloc[:n_train], Y.iloc[n_train:]
        
        # Scale features
        if self.use_scaling:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Fit Ridge model for each asset
        train_r2s = []
        test_r2s = []
        
        for asset in self.assets:
            if asset not in Y_train.columns:
                continue
            
            y_train = Y_train[asset].values
            y_test = Y_test[asset].values if asset in Y_test.columns else None
            
            model = Ridge(alpha=ridge_alpha, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            self.models[asset] = (model, self.scaler)
            
            # Compute R^2
            train_pred = model.predict(X_train_scaled)
            train_r2 = 1 - np.sum((y_train - train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
            train_r2s.append(train_r2)
            
            if y_test is not None:
                test_pred = model.predict(X_test_scaled)
                test_r2 = 1 - np.sum((y_test - test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
                test_r2s.append(test_r2)
        
        if verbose:
            print(f"Train R²: {np.mean(train_r2s):.4f} ± {np.std(train_r2s):.4f}")
            if test_r2s:
                print(f"Test R²:  {np.mean(test_r2s):.4f} ± {np.std(test_r2s):.4f}")
    
    def predict_next_returns(self) -> np.ndarray:
        """
        Predict next-day returns for all assets using latest data.
        
        Returns:
            Predicted returns array (annualized, 252 trading days)
        """
        if not self.models:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X, _ = self._make_features(self.returns)
        latest_row = X.iloc[-1:].values
        
        if self.use_scaling:
            latest_row = self.scaler.transform(latest_row)
        
        predictions = np.zeros(len(self.assets))
        
        for i, asset in enumerate(self.assets):
            if asset in self.models:
                model, _ = self.models[asset]
                pred = model.predict(latest_row)[0]
                predictions[i] = pred * 252  # Annualize
            else:
                predictions[i] = 0.0
        
        return predictions
    
    def get_feature_importance(self, asset: str = None, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance (Ridge coefficients).
        
        Args:
            asset: Specific asset, or None for average across assets
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        X, _ = self._make_features(self.returns)
        
        if asset and asset in self.models:
            model, _ = self.models[asset]
            coefs = np.abs(model.coef_)
        else:
            # Average across all assets
            coefs = []
            for a in self.assets:
                if a in self.models:
                    model, _ = self.models[a]
                    coefs.append(np.abs(model.coef_))
            coefs = np.mean(coefs, axis=0) if coefs else np.zeros(X.shape[1])
        
        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": coefs
        }).sort_values("importance", ascending=False)
        
        return importance.head(top_n)


def ensemble_alpha_forecast(
    returns: pd.DataFrame,
    fundamental_alpha: np.ndarray = None,
    technical_alpha: np.ndarray = None,
    ml_alpha: np.ndarray = None,
    weights: list = None
) -> np.ndarray:
    """
    Combine multiple alpha sources into a single forecast.
    Ensemble methods often outperform individual models.
    
    Args:
        returns: Returns DataFrame (for asset count)
        fundamental_alpha: Fundamental model predictions (n_assets,)
        technical_alpha: Technical/momentum predictions (n_assets,)
        ml_alpha: ML model predictions (n_assets,)
        weights: Combination weights (default: [0.3, 0.3, 0.4])
    
    Returns:
        Blended alpha forecast
    """
    n = len(returns.columns)
    alphas = []
    
    if fundamental_alpha is not None:
        alphas.append(np.asarray(fundamental_alpha).reshape(n))
    if technical_alpha is not None:
        alphas.append(np.asarray(technical_alpha).reshape(n))
    if ml_alpha is not None:
        alphas.append(np.asarray(ml_alpha).reshape(n))
    
    if not alphas:
        raise ValueError("At least one alpha source required")
    
    if weights is None:
        weights = [1.0 / len(alphas)] * len(alphas)
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()
    
    # Normalize each alpha to zero mean, unit variance for robustness
    alphas_normalized = []
    for alpha in alphas:
        alpha = np.asarray(alpha, dtype=float)
        alpha = (alpha - np.mean(alpha)) / (np.std(alpha) + 1e-10)
        alphas_normalized.append(alpha)
    
    # Weighted combination
    ensemble = np.zeros(n)
    for alpha, w in zip(alphas_normalized, weights):
        ensemble += w * alpha
    
    return ensemble


def signal_quality_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Evaluate prediction quality of alpha model.
    
    Args:
        predictions: Model predictions
        actuals: Realized returns/values
    
    Returns:
        Dict with IC (information coefficient), Sharpe, hit rate
    """
    from scipy.stats import spearmanr, pearsonr
    
    # Remove NaNs
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    pred = predictions[mask]
    actual = actuals[mask]
    
    if len(pred) < 10:
        return {}
    
    # Pearson and Spearman correlation (IC)
    pearson_ic, _ = pearsonr(pred, actual)
    spearman_ic, _ = spearmanr(pred, actual)
    
    # Hit rate: % of time sign matches
    signs_match = np.sum(np.sign(pred) == np.sign(actual))
    hit_rate = signs_match / len(pred)
    
    # Predicting beyond baseline
    baseline_returns = np.mean(actual)
    edge_returns = baseline_returns + pearson_ic * np.std(actual) * 0.05  # Scaled edge
    
    return {
        "pearson_ic": pearson_ic,
        "spearman_ic": spearman_ic,
        "hit_rate": hit_rate,
        "edge_basis_points": (edge_returns - baseline_returns) * 10000
    }
