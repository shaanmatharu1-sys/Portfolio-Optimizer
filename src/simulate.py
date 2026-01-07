import numpy as np
import pandas as pd


def mc_paths_parametric(mu, sigma, w, start_value, trading_days, n_sims, seed=42):
    rng = np.random.default_rng(seed)
    mu_d = mu / 252.0
    sigma_d = sigma / 252.0
    sigma_d = _make_psd(sigma_d)
    L = np.linalg.cholesky(sigma_d)
    z = rng.standard_normal((n_sims, trading_days, len(w)))
    asset_rets = mu_d + z @ L.T
    port_rets = asset_rets @ w
    values = np.empty((n_sims, trading_days + 1))
    values[:, 0] = start_value
    values[:, 1:] = start_value * np.exp(np.cumsum(port_rets, axis=1))
    cols = [f"day_{i}" for i in range(trading_days + 1)]
    return pd.DataFrame(values, columns=cols)


def mc_paths_bootstrap(portfolio_log_returns, start_value, trading_days, n_sims, seed=42):
    rng = np.random.default_rng(seed)
    r = np.asarray(portfolio_log_returns, dtype=float)
    idx = rng.integers(0, len(r), size=(n_sims, trading_days))
    sims = r[idx]
    values = np.empty((n_sims, trading_days + 1))
    values[:, 0] = start_value
    values[:, 1:] = start_value * np.exp(np.cumsum(sims, axis=1))
    cols = [f"day_{i}" for i in range(trading_days + 1)]
    return pd.DataFrame(values, columns=cols)


def summarize_paths(paths, quantiles=(0.1, 0.5, 0.9)):
    q = np.quantile(paths.values, q=quantiles, axis=0)
    out = pd.DataFrame(q.T, columns=[f"q{int(100*x)}" for x in quantiles])
    out["day"] = np.arange(len(out))
    return out


def _make_psd(a, eps=1e-10):
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, eps)
    b = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (b + b.T)
