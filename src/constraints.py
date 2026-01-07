import numpy as np
import pandas as pd
import cvxpy as cp


def build_base_constraints(w, fund_cfg: dict):
    cons = [cp.sum(w) == 1]

    ip = fund_cfg.get("instrument_policy", {})
    if ip.get("long_only", True):
        cons.append(w >= 0)

    c = fund_cfg.get("constraints", {})
    wmin = c.get("min_single_security_weight", None)
    wmax = c.get("max_single_security_weight", None)

    if wmin is not None:
        cons.append(w >= float(wmin))
    if wmax is not None:
        cons.append(w <= float(wmax))

    return cons


def build_turnover_controls(w, w0: np.ndarray, fund_cfg: dict):
    opt = fund_cfg.get("optimization", {})
    tc = opt.get("turnover_control", {})

    turnover_expr = cp.sum(cp.abs(w - w0))
    cons = []

    max_turn = tc.get("max_one_way_turnover", None)
    if max_turn is not None:
        cons.append(turnover_expr <= float(max_turn))

    gamma = float(tc.get("turnover_penalty_l1", 0.0))
    use_pen = bool(tc.get("use_turnover_penalty", True))
    penalty = gamma * cp.norm1(w - w0) if (use_pen and gamma > 0) else None

    return cons, turnover_expr, penalty


def build_sector_caps(w, df: pd.DataFrame, fund_cfg: dict):
    c = fund_cfg.get("constraints", {})
    sc = c.get("sector_caps", {})
    if not sc.get("enabled", False):
        return []

    max_sector = float(sc.get("max_sector_weight", 1.0))
    sectors = df["sector"].fillna("Unknown").astype(str).str.strip().values

    cons = []
    for s in np.unique(sectors):
        idx = np.where(sectors == s)[0]
        cons.append(cp.sum(w[idx]) <= max_sector)
    return cons


def build_income_constraints(w, df: pd.DataFrame, fund_cfg: dict):
    c = fund_cfg.get("constraints", {})
    inc = c.get("income_constraints", {})
    if not inc.get("enabled", False):
        return []

    y = pd.to_numeric(df["dividend_yield"], errors="coerce").fillna(0.0).values
    min_y = float(inc.get("min_portfolio_dividend_yield", 0.0))

    cons = [y @ w >= min_y]

    max_zero = inc.get("max_weight_in_zero_yield_assets", None)
    if max_zero is not None:
        zero_idx = np.where(y <= 0.0)[0]
        if len(zero_idx) > 0:
            cons.append(cp.sum(w[zero_idx]) <= float(max_zero))

    return cons

def build_smallcap_constraints(w, df: pd.DataFrame, fund_cfg: dict):
    c = fund_cfg.get("constraints", {})
    mc = c.get("market_cap_constraints", {})
    if not mc.get("enabled", False):
        return []

    enf = mc.get("enforcement", {})
    method = enf.get("method", "weighted_avg_log_market_cap")
    cap_limit = float(enf.get("cap_limit_usd", 2_000_000_000))
    drift = enf.get("allow_drift_bucket_weight", None)

    ac = df["asset_class"].astype(str).str.strip().values
    eq_idx = np.where(ac == "Equity")[0]
    if len(eq_idx) == 0:
        return []

    # Check if market_cap_usd column exists
    if "market_cap_usd" not in df.columns:
        return []  # Skip constraint if column doesn't exist
    
    m = pd.to_numeric(df["market_cap_usd"], errors="coerce").values.astype(float)
    
    # Skip constraint if market cap data is missing
    if np.isnan(m[eq_idx]).any():
        return []  # Return empty constraints instead of raising error

    cons = []
    if method == "weighted_avg_log_market_cap":
        logm = np.log(np.maximum(m[eq_idx], 1.0))
        cons.append(logm @ w[eq_idx] <= np.log(cap_limit))
    elif method == "hard_exclude_above_cap":
        above = eq_idx[m[eq_idx] > cap_limit]
        if len(above) > 0:
            cons.append(w[above] == 0)
    else:
        raise ValueError(f"Unknown small-cap enforcement method: {method}")

    if drift is not None and method != "weighted_avg_log_market_cap":
        above = eq_idx[m[eq_idx] > cap_limit]
        if len(above) > 0:
            cons.append(cp.sum(w[above]) <= float(drift))

    return cons



def build_return_target_constraint(w, mu: np.ndarray, fund_cfg: dict):
    opt = fund_cfg.get("optimization", {})
    mode = opt.get("mode", "min_vol_subject_to_return")
    rt = opt.get("return_target", {})
    target = rt.get("target_return", None)

    if mode == "min_vol_subject_to_return" and target is not None:
        return [mu @ w >= float(target)]
    return []


def build_all_constraints(df: pd.DataFrame, w, w0: np.ndarray, mu: np.ndarray, fund_cfg: dict):
    """
    IMPORTANT:
      The second argument MUST be the cvxpy Variable (w).
      If this signature differs, you'll get the 'AAPL' string conversion error.
    """
    constraints = []
    penalty_terms = []

    constraints += build_base_constraints(w, fund_cfg)

    c_turn, _, turn_penalty = build_turnover_controls(w, w0, fund_cfg)
    constraints += c_turn
    if turn_penalty is not None:
        penalty_terms.append(turn_penalty)

    constraints += build_sector_caps(w, df, fund_cfg)
    constraints += build_income_constraints(w, df, fund_cfg)
    constraints += build_smallcap_constraints(w, df, fund_cfg)
    constraints += build_return_target_constraint(w, mu, fund_cfg)

    return constraints, penalty_terms, None
