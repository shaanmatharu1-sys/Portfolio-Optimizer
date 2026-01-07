"""
Market regime detection and regime-adaptive portfolio management.
Identifies bull/bear/sideways/high-vol regimes and adjusts strategy accordingly.
Enables defensive positioning before downturns and offensive positioning in recoveries.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import zscore


class RegimeDetector:
    """
    Detects market regimes using multiple indicators:
    - Volatility regime (VIX-like)
    - Trend regime (momentum)
    - Dispersion regime (market correlation)
    - Drawdown regime (severity of decline)
    """
    
    def __init__(self, returns: pd.DataFrame, lookback: int = 60):
        """
        Args:
            returns: Asset returns DataFrame (T x n_assets)
            lookback: Window for regime calculation
        """
        self.returns = returns.copy()
        self.lookback = lookback
        self.regimes = None
        self.regime_probs = None
    
    def compute_regime_indicators(self) -> pd.DataFrame:
        """
        Compute raw regime indicators.
        
        Returns:
            DataFrame with regime indicators
        """
        indicators = {}
        T = len(self.returns)
        
        # 1. Volatility regime: rolling volatility of market
        market_ret = self.returns.mean(axis=1)
        vol_rolling = market_ret.rolling(self.lookback).std()
        vol_recent = vol_rolling.iloc[-1] if len(vol_rolling) > 0 else market_ret.std()
        vol_median = vol_rolling.median()
        indicators['volatility_zscore'] = (vol_rolling - vol_rolling.mean()) / vol_rolling.std()
        
        # 2. Trend regime: positive or negative momentum
        market_cum_ret = (1 + market_ret).rolling(self.lookback).apply(lambda x: (x.prod() - 1))
        indicators['momentum_sign'] = market_cum_ret.apply(np.sign)
        
        # 3. Market breadth: % of assets with positive returns
        daily_pos_count = (self.returns > 0).sum(axis=1)
        breadth_pct = daily_pos_count / self.returns.shape[1]
        indicators['breadth'] = breadth_pct
        
        # 4. Correlation regime: average correlation
        corr_rolling = self.returns.rolling(self.lookback).corr()
        avg_corr = []
        for i in range(len(self.returns)):
            if i >= self.lookback:
                corr_matrix = corr_rolling.loc[self.returns.index[i]].values
                # Average off-diagonal correlation
                avg_c = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                avg_corr.append(avg_c)
            else:
                avg_corr.append(np.nan)
        
        indicators['correlation'] = pd.Series(avg_corr, index=self.returns.index)
        
        # 5. Drawdown regime: current drawdown from peak
        cum_returns = (1 + market_ret).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns / running_max) - 1
        indicators['drawdown'] = drawdown
        
        # 6. Skewness regime: tail risk
        skew_rolling = self.returns.mean(axis=1).rolling(self.lookback).skew()
        indicators['skewness'] = skew_rolling
        
        return pd.DataFrame(indicators)
    
    def detect_regimes_gmm(self, n_regimes: int = 3) -> pd.DataFrame:
        """
        Detect regimes using Gaussian Mixture Model clustering.
        
        Typical 3 regimes:
        - Bull: low vol, positive trend, low correlation
        - Bear: high vol, negative trend, high correlation
        - Normal: moderate vol, sideways
        
        Args:
            n_regimes: Number of regimes to identify
        
        Returns:
            DataFrame with regime labels and probabilities
        """
        indicators = self.compute_regime_indicators()
        
        # Normalize indicators
        X = indicators.fillna(0).values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
        labels = gmm.fit_predict(X)
        probs = gmm.predict_proba(X)
        
        regime_df = pd.DataFrame({
            'regime': labels,
            'regime_prob': probs.max(axis=1),
            **{f'prob_regime_{i}': probs[:, i] for i in range(n_regimes)}
        }, index=self.returns.index)
        
        self.regimes = regime_df
        self.regime_probs = probs
        
        return regime_df
    
    def detect_regimes_heuristic(self) -> pd.DataFrame:
        """
        Simple heuristic regime detection based on vol and trend.
        
        Regimes:
        - 0: Bull (low vol, positive momentum)
        - 1: Normal (medium vol, sideways)
        - 2: Bear (high vol, negative momentum)
        - 3: Crash (very high vol, sharp drawdown)
        """
        market_ret = self.returns.mean(axis=1)
        vol_rolling = market_ret.rolling(self.lookback).std()
        momentum = (1 + market_ret).rolling(self.lookback).apply(lambda x: x.prod() - 1)
        
        # Normalize to z-scores
        vol_z = (vol_rolling - vol_rolling.mean()) / vol_rolling.std()
        mom_z = (momentum - momentum.mean()) / momentum.std()
        
        regimes = np.zeros(len(self.returns), dtype=int)
        
        for i in range(len(self.returns)):
            if pd.isna(vol_z.iloc[i]) or pd.isna(mom_z.iloc[i]):
                regimes[i] = 1  # Default to normal
                continue
            
            v = vol_z.iloc[i]
            m = mom_z.iloc[i]
            
            if v > 1.0:  # High volatility
                if m < -0.5:
                    regimes[i] = 3  # Crash
                else:
                    regimes[i] = 2  # Bear
            elif v < -0.5:  # Low volatility
                if m > 0.5:
                    regimes[i] = 0  # Bull
                else:
                    regimes[i] = 1  # Normal
            else:
                regimes[i] = 1  # Normal
        
        regime_df = pd.DataFrame({
            'regime': regimes,
            'vol_zscore': vol_z,
            'momentum_zscore': mom_z
        }, index=self.returns.index)
        
        self.regimes = regime_df
        
        return regime_df
    
    def get_current_regime(self) -> int:
        """Get the current regime (most recent observation)."""
        if self.regimes is None:
            raise ValueError("Run detect_regimes first")
        
        return int(self.regimes['regime'].iloc[-1])
    
    def get_regime_statistics(self) -> pd.DataFrame:
        """
        Show performance and characteristics by regime.
        
        Returns:
            DataFrame with regime statistics
        """
        if self.regimes is None:
            raise ValueError("Run detect_regimes first")
        
        market_ret = self.returns.mean(axis=1)
        combined = pd.concat([self.regimes[['regime']], market_ret], axis=1)
        combined.columns = ['regime', 'market_ret']
        
        stats = combined.groupby('regime')['market_ret'].agg([
            'count',
            ('avg_ret', np.mean),
            ('std_ret', np.std),
            ('sharpe', lambda x: np.mean(x) / np.std(x) * np.sqrt(252)),
            ('max_drawdown', lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())
        ])
        
        return stats


class RegimeAdaptivePortfolio:
    """
    Adjusts portfolio allocation based on detected market regime.
    
    Example mappings:
    - Bull regime: Higher equity allocation, momentum tilts
    - Normal regime: Balanced allocation
    - Bear regime: Defensive positioning, quality/low-vol focus
    - Crash regime: Maximum defensive, high cash
    """
    
    def __init__(self, regime_allocation_map: dict = None):
        """
        Args:
            regime_allocation_map: Dict mapping regime -> portfolio config
                Example:
                {
                    0: {'equity': 0.80, 'bonds': 0.20},  # Bull
                    1: {'equity': 0.60, 'bonds': 0.40},  # Normal
                    2: {'equity': 0.30, 'bonds': 0.70},  # Bear
                    3: {'equity': 0.10, 'bonds': 0.90}   # Crash
                }
        """
        if regime_allocation_map is None:
            self.allocation_map = {
                0: {'equity': 0.80, 'bonds': 0.20, 'alternatives': 0.00},  # Bull
                1: {'equity': 0.60, 'bonds': 0.30, 'alternatives': 0.10},  # Normal
                2: {'equity': 0.30, 'bonds': 0.60, 'alternatives': 0.10},  # Bear
                3: {'equity': 0.10, 'bonds': 0.80, 'alternatives': 0.10}   # Crash
            }
        else:
            self.allocation_map = regime_allocation_map
    
    def get_regime_allocation(self, regime: int) -> dict:
        """Get asset allocation for specified regime."""
        return self.allocation_map.get(regime, self.allocation_map.get(1, {}))
    
    def adjust_weights_for_regime(
        self,
        base_weights: np.ndarray,
        regime: int,
        asset_classes: list
    ) -> np.ndarray:
        """
        Adjust portfolio weights based on regime.
        
        Args:
            base_weights: Base portfolio weights
            regime: Current regime (0-3)
            asset_classes: Asset class for each weight (e.g., ['equity', 'equity', 'bonds'])
        
        Returns:
            Adjusted weights
        """
        allocation = self.get_regime_allocation(regime)
        
        # Group weights by asset class
        adjusted_w = base_weights.copy()
        
        for asset_class, target_alloc in allocation.items():
            indices = [i for i, ac in enumerate(asset_classes) if ac == asset_class]
            
            if len(indices) > 0:
                # Proportionally adjust weights within each class
                current_class_w = adjusted_w[indices].sum()
                if current_class_w > 0:
                    # Scale to target allocation
                    scale = target_alloc / current_class_w
                    adjusted_w[indices] = adjusted_w[indices] * scale
        
        # Renormalize
        adjusted_w = adjusted_w / adjusted_w.sum()
        
        return adjusted_w
