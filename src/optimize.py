import numpy as np
import pandas as pd
import cvxpy as cp

from src.constraints import build_all_constraints


def optimize_portfolio(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, fund_cfg: dict):
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("No holdings to optimize.")

    fund_cfg.setdefault("_runtime_warnings", [])

    w0 = pd.to_numeric(df["weight"], errors="coerce").values
    if np.isnan(w0).any():
        raise ValueError("Current weights contain NaNs.")
    w0 = w0 / w0.sum()

    c = fund_cfg.get("constraints", {})
    wmax = c.get("max_single_security_weight", None)
    if wmax is not None:
        wmax = float(wmax)
        if wmax * n < 1.0:
            c["max_single_security_weight"] = 1.0 / n
            fund_cfg["_runtime_warnings"].append(f"Adjusted max_single_security_weight to {c['max_single_security_weight']:.6f} (wmax*n < 1).")

    wmin = c.get("min_single_security_weight", None)
    if wmin is not None:
        wmin = float(wmin)
        if wmin * n > 1.0:
            c["min_single_security_weight"] = 0.0
            fund_cfg["_runtime_warnings"].append("Set min_single_security_weight to 0.0 (wmin*n > 1).")

    sigma = 0.5 * (sigma + sigma.T)
    vals, vecs = np.linalg.eigh(sigma)
    vals = np.maximum(vals, 1e-10)
    sigma = vecs @ np.diag(vals) @ vecs.T
    sigma = 0.5 * (sigma + sigma.T)

    def solve(cfg):
        w = cp.Variable(n)
        constraints, penalty_terms, _ = build_all_constraints(df, w, w0, mu, cfg)
        obj = cp.quad_form(w, sigma)
        for p in penalty_terms:
            obj += p
        prob = cp.Problem(cp.Minimize(obj), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                prob.solve(solver=cp.SCS, verbose=False)
        return prob, w

    def clone_cfg(x):
        import json
        return json.loads(json.dumps(x))

    modes = []

    cfg0 = clone_cfg(fund_cfg)
    modes.append(("strict", cfg0))

    cfg1 = clone_cfg(fund_cfg)
    tc = cfg1.get("optimization", {}).get("turnover_control", {})
    if "max_one_way_turnover" in tc:
        tc.pop("max_one_way_turnover", None)
    modes.append(("no_turnover_cap", cfg1))

    cfg2 = clone_cfg(cfg1)
    rt = cfg2.get("optimization", {}).get("return_target", {})
    if "target_return" in rt:
        rt.pop("target_return", None)
    modes.append(("no_return_target", cfg2))

    cfg3 = clone_cfg(cfg2)
    sc = cfg3.get("constraints", {}).get("sector_caps", {})
    if sc.get("enabled", False):
        sc["enabled"] = False
    modes.append(("no_sector_caps", cfg3))

    cfg4 = clone_cfg(cfg3)
    mc = cfg4.get("constraints", {}).get("market_cap_constraints", {})
    if mc.get("enabled", False):
        mc["enabled"] = False
    modes.append(("no_market_cap_constraints", cfg4))

    cfg5 = clone_cfg(cfg4)
    inc = cfg5.get("constraints", {}).get("income_constraints", {})
    if inc.get("enabled", False):
        inc["enabled"] = False
    modes.append(("no_income_constraints", cfg5))

    cfg6 = clone_cfg(cfg5)
    cfg6.get("constraints", {}).pop("min_single_security_weight", None)
    modes.append(("only_base", cfg6))

    last_status = None
    last_prob = None
    last_w = None
    used = None

    for tag, cfg in modes:
        prob, w = solve(cfg)
        last_status = prob.status
        last_prob = prob
        last_w = w
        if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            used = tag
            break

    if used is None:
        raise ValueError(f"Optimization failed. Status: {last_status}")

    fund_cfg["_runtime_warnings"].append(f"Solved in mode: {used}")

    w_opt = np.array(last_w.value).reshape(-1)
    w_opt[w_opt < 0] = 0
    s = float(w_opt.sum())
    if s <= 0:
        raise ValueError("Optimization produced non-positive weight sum.")
    w_opt = w_opt / s

    wmax2 = fund_cfg.get("constraints", {}).get("max_single_security_weight", None)
    if wmax2 is not None:
        wmax2 = float(wmax2)
        w_opt = np.minimum(w_opt, wmax2)
        w_opt = w_opt / w_opt.sum()

    return {
        "weights": w_opt,
        "status": last_prob.status,
        "objective_value": float(last_prob.value) if last_prob.value is not None else np.nan,
        "diagnostics": {
            "achieved_return": float(np.expm1(mu @ w_opt)),
            "one_way_turnover": float(np.sum(np.abs(w_opt - w0)))
        }
    }
