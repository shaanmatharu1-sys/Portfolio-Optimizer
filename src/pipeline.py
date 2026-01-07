import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.validate import validate_holdings
from src.data import fetch_price_data, compute_log_returns
from src.risk import estimate_mu_sigma
from src.optimize import optimize_portfolio
from src.analytics import build_analytics


def run_pipeline(
    fund_config_path: str,
    holdings_csv_path: str,
    *,
    start_date: str,
    end_date: str,
    output_dir: str = "runs",
):
    """
    Runs full pipeline:
      validate -> download -> risk -> optimize -> analytics -> save artifacts

    Returns:
      dict with paths + key objects
    """
    # -----------------------
    # Load config
    # -----------------------
    with open(fund_config_path, "r") as f:
        fund_cfg = json.load(f)

    fund_name = fund_cfg.get("fund_name", "Fund").replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{fund_name}_{ts}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    holdings_df, validation_report = validate_holdings(holdings_csv_path, fund_cfg)

    tickers = holdings_df["ticker"].tolist()

    w0 = holdings_df["weight"].astype(float).values
    w0 = w0 / w0.sum()

    pd.DataFrame({"warnings": validation_report["warnings"]}).to_csv(
        os.path.join(run_dir, "validation_warnings.csv"), index=False
    )
    holdings_df.to_csv(os.path.join(run_dir, "holdings_validated.csv"), index=False)

    prices = fetch_price_data(tickers, start_date, end_date)
    returns = compute_log_returns(prices)

    returns = returns[tickers]

    prices.to_csv(os.path.join(run_dir, "prices.csv"))
    returns.to_csv(os.path.join(run_dir, "returns.csv"))

    rm = fund_cfg.get("risk_model", {})
    mu, sigma, assets = estimate_mu_sigma(
        returns,
        trading_days=252,
        cov_shrink_alpha=float(rm.get("shrinkage_alpha", 0.15)),
        ensure_psd=bool(rm.get("ensure_psd", True)),
        mu_shrink_beta=0.50,
        mu_clip=0.50
    )

    if assets != tickers:
        raise ValueError("Asset ordering mismatch between holdings and returns.")

    np.save(os.path.join(run_dir, "mu.npy"), mu)
    np.save(os.path.join(run_dir, "sigma.npy"), sigma)

    result = optimize_portfolio(holdings_df, mu, sigma, fund_cfg)
    w1 = result["weights"]

    target_df = pd.DataFrame({"ticker": tickers, "target_weight": w1})
    target_df.to_csv(os.path.join(run_dir, "target_weights.csv"), index=False)

    analytics = build_analytics(holdings_df, mu, sigma, w0, w1, fund_cfg)

    analytics["trades"].to_csv(os.path.join(run_dir, "trades.csv"), index=False)
    analytics["risk_contributions_target"].to_csv(os.path.join(run_dir, "risk_contributions_target.csv"), index=False)
    analytics["compliance_target"].to_csv(os.path.join(run_dir, "compliance_target.csv"), index=False)

    if analytics["sector_exposure_current"] is not None:
        analytics["sector_exposure_current"].to_csv(os.path.join(run_dir, "sector_exposure_current.csv"), index=False)
        analytics["sector_exposure_target"].to_csv(os.path.join(run_dir, "sector_exposure_target.csv"), index=False)

    if analytics["asset_class_exposure_current"] is not None:
        analytics["asset_class_exposure_current"].to_csv(os.path.join(run_dir, "asset_class_exposure_current.csv"), index=False)
        analytics["asset_class_exposure_target"].to_csv(os.path.join(run_dir, "asset_class_exposure_target.csv"), index=False)

    summary = {
        "run_id": run_id,
        "fund_name": fund_cfg.get("fund_name"),
        "status": result["status"],
        "objective_value": result.get("objective_value"),
        "diagnostics": result.get("diagnostics", {}),
        "stats_current": analytics["stats_current"],
        "stats_target": analytics["stats_target"],
        "validation_warnings": validation_report["warnings"],
        "start_date": start_date,
        "end_date": end_date
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "run_dir": run_dir,
        "summary": summary,
        "result": result
    }
