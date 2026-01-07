import numpy as np
import pandas as pd

def portfolio_return(mu, w):
    return float(mu @ w)

def portfolio_vol(sigma, w):
    return float(np.sqrt(w @ sigma @ w))

def portfolio_sharpe(mu, sigma, w, rf=0.02):
    vol = portfolio_vol(sigma, w)
    if vol <= 0:
        return np.nan
    return (portfolio_return(mu, w) - rf) / vol

def one_way_turnover(w0, w1):
    return float(np.sum(np.abs(w1 - w0)))

def portfolio_yield(dividend_yield, w):
    return float(dividend_yield @ w)

def weighted_avg_log_market_cap(market_cap, w):
    m = np.asarray(market_cap, dtype=float)
    m = np.maximum(m, 1.0)
    return float(np.exp(np.sum(w * np.log(m))))

def exposures_by_group(df, w, group_col):
    g = df[group_col].fillna('Unknown').astype(str).str.strip()
    out = (
        pd.DataFrame({"group": g, "weight": w})
        .groupby("group", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "weight"})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    return out

def trade_list(df, w0, w1, min_trade_threshold=0.0025):
    out = pd.DataFrame({
        "ticker": df["ticker"].values,
        "current_w": w0,
        "target_w": w1
    })
    out["delta"] = out["target_w"] - out["current_w"]
    out["abs_delta"] = out["delta"].abs()
    out = out.sort_values("abs_delta", ascending=False)

    trades = out[out["abs_delta"] >= float(min_trade_threshold)].copy()
    trades["action"] = np.where(trades["delta"] > 0, "BUY", "SELL")
    trades = trades.reset_index(drop=True)
    return trades

def risk_contributions(sigma: np.ndarray, w: np.ndarray):
    w = np.asarray(w, dtype=float).reshape(-1)
    sig_w = sigma @ w
    total_var = float(w @ sig_w)
    if total_var <= 0:
        return pd.DataFrame({"pct_contrib_var": np.zeros_like(w)})

    contrib_var = w * sig_w
    pct = contrib_var / total_var
    return pd.DataFrame({"pct_contrib_var": pct})

def compliance_report(df: pd.DataFrame, w: np.ndarray, fund_cfg: dict) -> pd.DataFrame:
    checks = []

    cons = fund_cfg.get("constraints", {})
    opt = fund_cfg.get("optimization", {})
    rt = opt.get("return_target", {})

    wmax = cons.get("max_single_security_weight", None)
    if wmax is not None:
        observed = float(np.max(w))
        checks.append({
            "check": "max_single_security_weight",
            "limit": float(wmax),
            "observed": observed,
            "status": "PASS" if observed <= float(wmax) + 1e-12 else "FAIL"
        })

    sector_caps = cons.get("sector_caps", {})
    if sector_caps.get("enabled", False):
        max_sector = float(sector_caps.get("max_sector_weight", 1.0))
        sec = exposures_by_group(df, w, "sector")
        worst = float(sec["weight"].max()) if len(sec) else 0.0
        checks.append({
            "check": "max_sector_weight",
            "limit": max_sector,
            "observed": worst,
            "status": "PASS" if worst <= max_sector + 1e-12 else "FAIL"
        })

    income = cons.get("income_constraints", {})
    if income.get("enabled", False):
        y = pd.to_numeric(df["dividend_yield"], errors="coerce").fillna(0.0).values
        port_y = portfolio_yield(y, w)
        y_min = float(income.get("min_portfolio_dividend_yield", 0.0))
        checks.append({
            "check": "min_portfolio_dividend_yield",
            "limit": y_min,
            "observed": port_y,
            "status": "PASS" if port_y >= y_min - 1e-12 else "FAIL"
        })

        max_zero = income.get("max_weight_in_zero_yield_assets", None)
        if max_zero is not None:
            zero_w = float(np.sum(w[y <= 0.0]))
            checks.append({
                "check": "max_weight_in_zero_yield_assets",
                "limit": float(max_zero),
                "observed": zero_w,
                "status": "PASS" if zero_w <= float(max_zero) + 1e-12 else "FAIL"
            })

    mcap = cons.get("market_cap_constraints", {})
    if mcap.get("enabled", False):
        enf = mcap.get("enforcement", {})
        cap_limit = float(enf.get("cap_limit_usd", 2_000_000_000))
        m = pd.to_numeric(df["market_cap_usd"], errors="coerce").fillna(np.nan).values
        wavg_log = weighted_avg_log_market_cap(m, w)
        checks.append({
            "check": "weighted_avg_market_cap_usd",
            "limit": cap_limit,
            "observed": wavg_log,
            "status": "PASS" if wavg_log <= cap_limit + 1e-12 else "FAIL"
        })

        drift = enf.get("allow_drift_bucket_weight", None)
        if drift is not None:
            above = float(np.sum(w[m > cap_limit]))
            checks.append({
                "check": "weight_above_cap_limit",
                "limit": float(drift),
                "observed": above,
                "status": "PASS" if above <= float(drift) + 1e-12 else "FAIL"
            })

    if rt and rt.get("target_return", None) is not None:
        checks.append({
            "check": "return_target_configured",
            "limit": float(rt["target_return"]),
            "observed": np.nan,
            "status": "INFO"
        })

    return pd.DataFrame(checks)

def build_analytics(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    fund_cfg: dict
) -> dict:
    rf = float(fund_cfg.get("risk_model", {}).get("risk_free_rate_annual", 0.02))
    min_trade = float(fund_cfg.get("optimization", {}).get("turnover_control", {}).get("min_trade_threshold_weight", 0.0025))

    y = pd.to_numeric(df.get("dividend_yield", pd.Series([0]*len(df))), errors="coerce").fillna(0.0).values
    m = pd.to_numeric(df.get("market_cap_usd", pd.Series([np.nan]*len(df))), errors="coerce").fillna(np.nan).values

    stats_current = {
        "exp_return": portfolio_return(mu, w0),
        "vol": portfolio_vol(sigma, w0),
        "sharpe": portfolio_sharpe(mu, sigma, w0, rf=rf),
        "dividend_yield": portfolio_yield(y, w0) if np.isfinite(y).all() else np.nan,
        "wavg_market_cap_usd": weighted_avg_log_market_cap(m, w0) if np.isfinite(m).all() else np.nan
    }
    stats_target = {
        "exp_return": portfolio_return(mu, w1),
        "vol": portfolio_vol(sigma, w1),
        "sharpe": portfolio_sharpe(mu, sigma, w1, rf=rf),
        "dividend_yield": portfolio_yield(y, w1) if np.isfinite(y).all() else np.nan,
        "wavg_market_cap_usd": weighted_avg_log_market_cap(m, w1) if np.isfinite(m).all() else np.nan,
        "one_way_turnover": one_way_turnover(w0, w1)
    }

    sector_current = exposures_by_group(df, w0, "sector") if "sector" in df.columns else None
    sector_target = exposures_by_group(df, w1, "sector") if "sector" in df.columns else None
    asset_current = exposures_by_group(df, w0, "asset_class") if "asset_class" in df.columns else None
    asset_target = exposures_by_group(df, w1, "asset_class") if "asset_class" in df.columns else None

    trades = trade_list(df, w0, w1, min_trade_threshold=min_trade)

    rc = risk_contributions(sigma, w1)
    rc["ticker"] = df["ticker"].values
    rc = rc.sort_values("pct_contrib_var", ascending=False).reset_index(drop=True)

    compliance = compliance_report(df, w1, fund_cfg)

    return {
        "stats_current": stats_current,
        "stats_target": stats_target,
        "trades": trades,
        "sector_exposure_current": sector_current,
        "sector_exposure_target": sector_target,
        "asset_class_exposure_current": asset_current,
        "asset_class_exposure_target": asset_target,
        "risk_contributions_target": rc,
        "compliance_target": compliance
    }