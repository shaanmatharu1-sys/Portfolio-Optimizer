import numpy as np
import pandas as pd

from src.risk import estimate_mu_sigma
from src.optimize import optimize_portfolio


def walk_forward_backtest(holdings_df, returns, fund_cfg, lookback=252, rebalance_every=21):
    df = holdings_df.copy().reset_index(drop=True)
    tickers = df["ticker"].tolist()
    rets = returns[tickers].copy()

    w = df["weight"].astype(float).values
    w = w / w.sum()

    port_logrets = []
    dates = []
    weights_hist = []

    rm = fund_cfg.get("risk_model", {})
    shrink = float(rm.get("shrinkage_alpha", 0.15))
    ensure_psd = bool(rm.get("ensure_psd", True))

    for t in range(lookback, len(rets)):
        if (t - lookback) % rebalance_every == 0:
            window = rets.iloc[t - lookback:t]
            mu, sigma, _ = estimate_mu_sigma(
                window,
                cov_shrink_alpha=shrink,
                ensure_psd=ensure_psd,
                mu_shrink_beta=0.50,
                mu_clip=0.50
            )
            df["weight"] = w
            res = optimize_portfolio(df, mu, sigma, fund_cfg)
            w = res["weights"]

        r_t = float(rets.iloc[t].values @ w)
        port_logrets.append(r_t)
        dates.append(rets.index[t])
        weights_hist.append(w.copy())

    curve = pd.Series(np.exp(np.cumsum(port_logrets)), index=pd.Index(dates, name="date"), name="equity_mult")
    weights_hist = pd.DataFrame(weights_hist, index=pd.Index(dates, name="date"), columns=tickers)

    diagnostics = {
        "realized_ann_return": float(np.mean(port_logrets) * 252),
        "realized_ann_vol": float(np.std(port_logrets) * np.sqrt(252)),
        "max_drawdown": float(_max_drawdown(curve.values)),
        "avg_turnover": float(_avg_turnover(weights_hist.values))
    }

    return curve.to_frame(), weights_hist, diagnostics


def _max_drawdown(equity_mult):
    peak = np.maximum.accumulate(equity_mult)
    dd = (equity_mult / peak) - 1.0
    return float(dd.min())


def _avg_turnover(W):
    if len(W) < 2:
        return 0.0
    return float(np.mean(np.sum(np.abs(W[1:] - W[:-1]), axis=1)))
