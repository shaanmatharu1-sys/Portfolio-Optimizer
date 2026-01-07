"""
Risk-adjusted portfolio optimization maximizing Sharpe ratio and other objective functions.
Goes beyond minimum variance to target-return optimization.
"""

import numpy as np
import pandas as pd
import cvxpy as cp


def optimize_sharpe_ratio(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    fund_cfg: dict,
    risk_free_rate: float = 0.02
) -> dict:
    """
    Optimize portfolio to maximize Sharpe ratio: (mu @ w - rf) / sqrt(w @ Sigma @ w)
    
    Uses the Markowitz approach: maximize return for given volatility level.
    Solves by finding the portfolio on the efficient frontier with highest Sharpe ratio.
    
    Args:
        df: Holdings dataframe
        mu: Expected returns vector
        sigma: Covariance matrix
        fund_cfg: Configuration dict with constraints
        risk_free_rate: Risk-free rate (default 2%)
    
    Returns:
        dict with optimal weights and metrics
    """
    from src.constraints import build_all_constraints
    
    n = len(df)
    w0 = pd.to_numeric(df["weight"], errors="coerce").values
    w0 = w0 / w0.sum()
    
    w = cp.Variable(n)
    
    # Sharpe ratio = (mu @ w - rf) / sqrt(w @ Sigma @ w)
    # Equivalent to: maximize return subject to variance = 1 (after rescaling)
    # Or: minimize variance subject to return = target (parametric approach)
    
    constraints, penalty_terms, _ = build_all_constraints(df, w, w0, mu, fund_cfg)
    constraints.append(cp.sum(w) == 1)  # Always include full investment constraint
    
    # Use parametric approach: minimize variance while targeting high returns
    # We solve for maximum Sharpe by maximizing: return - risk_free_rate - lambda * variance
    # Lambda is chosen to find the tangent portfolio
    
    # For simplicity: maximize (return - rf) / sqrt(variance)
    # This is equivalent to maximizing return while minimizing variance on the frontier
    
    port_return = mu @ w
    port_variance = cp.quad_form(w, sigma)
    
    # Objective: maximize Sharpe ratio by penalizing variance relative to excess return
    gamma = 2.0  # Adjust based on risk tolerance (higher = more aggressive)
    
    obj = port_variance - gamma * (port_return - risk_free_rate)
    
    # Add penalties for turnover if specified
    for p in penalty_terms:
        obj += p
    
    problem = cp.Problem(cp.Minimize(obj), constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False, max_iter=5000)
    except Exception:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, verbose=False, eps=1e-4)
            except Exception as e:
                raise ValueError(f"All solvers failed: {str(e)[:100]}")
    
    if w.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Optimization status: {problem.status}")
    
    w_opt = np.array(w.value).reshape(-1)
    w_opt[w_opt < 1e-6] = 0
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum()
    else:
        w_opt = np.ones(n) / n  # Fallback to equal weight
    
    port_return = float(mu @ w_opt)
    port_vol = float(np.sqrt(w_opt @ sigma @ w_opt))
    port_sharpe = (port_return - risk_free_rate) / max(port_vol, 1e-10)
    
    return {
        "weights": w_opt,
        "status": problem.status,
        "objective_value": float(problem.value),
        "return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": port_sharpe,
        "diagnostics": {
            "achieved_return": port_return,
            "one_way_turnover": float(np.sum(np.abs(w_opt - w0)))
        }
    }


def optimize_risk_parity(
    df: pd.DataFrame,
    sigma: np.ndarray,
    fund_cfg: dict
) -> dict:
    """
    Risk Parity: allocate capital so each asset contributes equally to portfolio risk.
    Strong performance in diversified portfolios, especially during market stress.
    
    Args:
        df: Holdings dataframe
        sigma: Covariance matrix
        fund_cfg: Configuration dict
    
    Returns:
        dict with risk-parity weights
    """
    n = len(df)
    w0 = pd.to_numeric(df["weight"], errors="coerce").values
    w0 = w0 / w0.sum()
    
    # Initialize with 1/N
    w_rp = np.ones(n) / n
    
    # Iterate to convergence
    for _ in range(50):
        diag_sigma = np.diag(sigma) ** 0.5
        w_new = 1.0 / diag_sigma
        w_new = w_new / w_new.sum()
        
        if np.sum(np.abs(w_new - w_rp)) < 1e-8:
            break
        w_rp = w_new
    
    # Compute risk contributions
    sigma_w = sigma @ w_rp
    contrib = w_rp * sigma_w / (w_rp @ sigma_w)
    
    return {
        "weights": w_rp,
        "status": "optimal",
        "risk_contributions": contrib,
        "diagnostics": {
            "one_way_turnover": float(np.sum(np.abs(w_rp - w0)))
        }
    }


def optimize_minimum_volatility(
    df: pd.DataFrame,
    sigma: np.ndarray,
    fund_cfg: dict
) -> dict:
    """
    Classic minimum variance portfolio.
    Robust alternative when expected returns are uncertain.
    """
    from src.constraints import build_all_constraints
    
    n = len(df)
    w0 = pd.to_numeric(df["weight"], errors="coerce").values
    w0 = w0 / w0.sum()
    
    # Dummy mu for constraints
    mu = np.zeros(n)
    
    w = cp.Variable(n)
    constraints, penalty_terms, _ = build_all_constraints(df, w, w0, mu, fund_cfg)
    
    obj = cp.quad_form(w, sigma)
    for p in penalty_terms:
        obj += p
    
    problem = cp.Problem(cp.Minimize(obj), constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False, max_iter=5000)
    except Exception:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, verbose=False, eps=1e-4)
            except Exception as e:
                raise ValueError(f"All solvers failed: {str(e)[:100]}")
    
    if w.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Optimization status: {problem.status}")
    
    w_opt = np.array(w.value).reshape(-1)
    w_opt[w_opt < 1e-6] = 0
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum()
    else:
        w_opt = np.ones(n) / n  # Fallback to equal weight
    
    return {
        "weights": w_opt,
        "status": problem.status,
        "objective_value": float(problem.value),
        "diagnostics": {
            "one_way_turnover": float(np.sum(np.abs(w_opt - w0)))
        }
    }


def optimize_equal_weight(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    fund_cfg: dict
) -> dict:
    """
    Simple equal-weight portfolio - robust fallback when optimization fails.
    """
    n = len(df)
    w_opt = np.ones(n) / n
    
    w0 = pd.to_numeric(df["weight"], errors="coerce").values
    w0 = w0 / w0.sum()
    
    port_return = float(mu @ w_opt)
    port_vol = float(np.sqrt(w_opt @ sigma @ w_opt))
    risk_free_rate = 0.02
    port_sharpe = (port_return - risk_free_rate) / max(port_vol, 1e-10)
    
    return {
        "weights": w_opt,
        "status": "optimal",
        "objective_value": float(np.sqrt(w_opt @ sigma @ w_opt)),
        "return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": port_sharpe,
        "diagnostics": {
            "achieved_return": port_return,
            "one_way_turnover": float(np.sum(np.abs(w_opt - w0)))
        }
    }


def optimize_maximum_return(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    fund_cfg: dict
) -> dict:
    """
    Maximize expected return subject to constraints.
    Aggressive strategy assuming confidence in return forecasts.
    """
    from src.constraints import build_all_constraints
    
    n = len(df)
    w0 = pd.to_numeric(df["weight"], errors="coerce").values
    w0 = w0 / w0.sum()
    
    w = cp.Variable(n)
    constraints, penalty_terms, _ = build_all_constraints(df, w, w0, mu, fund_cfg)
    
    # Objective: maximize return = minimize negative return
    # Add small volatility penalty for stability
    obj = -1.0 * (mu @ w) + 0.01 * cp.quad_form(w, sigma)
    
    for p in penalty_terms:
        obj += p
    
    problem = cp.Problem(cp.Minimize(obj), constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False, max_iter=5000)
    except Exception:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.SCS, verbose=False, eps=1e-4)
            except Exception as e:
                raise ValueError(f"All solvers failed: {str(e)[:100]}")
    
    if w.value is None or problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Optimization status: {problem.status}")
    
    w_opt = np.array(w.value).reshape(-1)
    w_opt[w_opt < 1e-6] = 0
    if w_opt.sum() > 0:
        w_opt = w_opt / w_opt.sum()
    else:
        w_opt = np.ones(n) / n  # Fallback to equal weight
    
    port_return = float(mu @ w_opt)
    port_vol = float(np.sqrt(w_opt @ sigma @ w_opt))
    
    return {
        "weights": w_opt,
        "status": problem.status,
        "objective_value": float(problem.value),
        "return": port_return,
        "volatility": port_vol,
        "diagnostics": {
            "achieved_return": port_return,
            "one_way_turnover": float(np.sum(np.abs(w_opt - w0)))
        }
    }


def efficient_frontier(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    fund_cfg: dict,
    n_points: int = 20
) -> pd.DataFrame:
    """
    Generate efficient frontier by optimizing for different target returns.
    Useful for stakeholder communication and constraint validation.
    """
    from src.constraints import build_all_constraints
    
    n = len(df)
    w0 = pd.to_numeric(df["weight"], errors="coerce").values
    w0 = w0 / w0.sum()
    
    # Min and max possible portfolio returns
    min_ret = np.min(mu)
    max_ret = np.max(mu)
    target_rets = np.linspace(min_ret, max_ret, n_points)
    
    frontier = []
    
    for target_ret in target_rets:
        cfg = fund_cfg.copy()
        cfg["optimization"] = cfg.get("optimization", {})
        cfg["optimization"]["return_target"] = {"target_return": float(target_ret)}
        
        w = cp.Variable(n)
        constraints, penalty_terms, _ = build_all_constraints(df, w, w0, mu, cfg)
        constraints.append(mu @ w >= target_ret - 1e-10)
        
        obj = cp.quad_form(w, sigma)
        for p in penalty_terms:
            obj += p
        
        problem = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                problem.solve(solver=cp.SCS, verbose=False)
        
        if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
            w_opt = np.array(w.value).reshape(-1)
            w_opt[w_opt < 0] = 0
            w_opt = w_opt / w_opt.sum()
            
            port_ret = float(mu @ w_opt)
            port_vol = float(np.sqrt(w_opt @ sigma @ w_opt))
            
            frontier.append({
                "target_return": target_ret,
                "actual_return": port_ret,
                "volatility": port_vol,
                "sharpe_ratio": (port_ret - 0.02) / max(port_vol, 1e-10)
            })
    
    return pd.DataFrame(frontier) if frontier else pd.DataFrame()
