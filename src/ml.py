import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def make_features(returns, lags=(1, 5, 21)):
    feats = []
    for lag in lags:
        feats.append(returns.shift(lag).add_prefix(f"lag{lag}_"))
    mom_21 = returns.rolling(21).sum().add_prefix("mom21_")
    vol_21 = returns.rolling(21).std().add_prefix("vol21_")
    X = pd.concat(feats + [mom_21, vol_21], axis=1)
    Y = returns.shift(-1)
    X = X.dropna()
    Y = Y.loc[X.index].dropna()
    X = X.loc[Y.index]
    return X, Y


def fit_predict_mu_ml(returns, ridge_alpha=10.0):
    X, Y = make_features(returns)
    assets = list(returns.columns)
    mu_d = np.zeros(len(assets), dtype=float)
    Xv = X.values
    for i, a in enumerate(assets):
        y = Y[a].values
        m = Ridge(alpha=ridge_alpha)
        m.fit(Xv, y)
        mu_d[i] = float(m.predict(Xv[-1:].reshape(1, -1))[0])
    return mu_d * 252.0
