import pandas as pd
import numpy as np
import yfinance as yf


def validate_holdings(csv_path: str, fund_cfg: dict, consolidate_duplicates: bool = True, auto_enrich: bool = True):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    c0 = None
    if len(df.columns) == 1:
        c0 = str(df.columns[0])
    if c0 and "," in c0 and "ticker" in c0.lower():
        header = [x.strip().lower().replace(" ", "_") for x in c0.strip().strip('"').split(",")]
        s = df.iloc[:, 0].astype(str).str.strip()
        s = s.str.strip().str.strip('"')
        split = s.str.split(",", expand=True)
        if split.shape[1] == len(header):
            split.columns = header
            df = split

    df.columns = [str(c).replace("\ufeff", "").strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^unnamed", case=False, regex=True)]
    df = df.dropna(axis=1, how="all")

    alias = {
        "symbol": "ticker",
        "symbols": "ticker",
        "security": "ticker",
        "security_name": "ticker",
        "ticker_symbol": "ticker",
        "position": "ticker",

        "wt": "weight",
        "weights": "weight",
        "portfolio_weight": "weight",
        "percent_of_account": "weight",
        "pct_of_account": "weight",
        "percent_of_portfolio": "weight",
        "pct_of_portfolio": "weight",
        "percent": "weight",
        "pct": "weight",

        "market_value": "market_value",
        "current_value": "market_value",
        "value": "market_value",
        "position_value": "market_value",

        "market_cap": "market_cap_usd",
        "marketcap": "market_cap_usd",
        "mkt_cap": "market_cap_usd",
        "mktcap": "market_cap_usd",
        "market_capitalization": "market_cap_usd",

        "div_yield": "dividend_yield",
        "dividend_yield_%": "dividend_yield",
        "dividend_yield_percent": "dividend_yield",
        "yield": "dividend_yield",

        "assetclass": "asset_class",
        "asset_class_name": "asset_class",
        "class": "asset_class",

        "gics_sector": "sector",
        "industry_sector": "sector",
    }

    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    if "ticker" not in df.columns:
        raise ValueError(f"Missing required column: ticker. Found columns: {list(df.columns)}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"].str.len() > 0].copy()

    warnings = []

    if "weight" not in df.columns:
        if "market_value" in df.columns:
            mv = pd.to_numeric(df["market_value"], errors="coerce")
            if mv.isna().any():
                raise ValueError("market_value contains non-numeric entries and weight is missing.")
            if float(mv.sum()) <= 0:
                raise ValueError("market_value sum must be > 0 to compute weights.")
            df["weight"] = mv / mv.sum()
            warnings.append("weight missing; computed from market_value.")
        else:
            raise ValueError(f"Missing required column: weight (or market_value to derive weights). Found columns: {list(df.columns)}")

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any():
        rows = df.index[df["weight"].isna()].tolist()
        raise ValueError(f"Non-numeric weight at rows: {rows}")

    wsum = float(df["weight"].sum())
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0.")
    if not (0.95 <= wsum <= 1.05):
        warnings.append(f"Sum of weights is {wsum:.6f}, which is outside the acceptable range (0.95, 1.05). Normalizing weights.")
    df["weight"] = df["weight"] / df["weight"].sum()

    if "asset_class" not in df.columns:
        df["asset_class"] = "Equity"
        warnings.append("asset_class missing; defaulted to Equity for all rows.")

    df["asset_class"] = df["asset_class"].astype(str).str.strip()

    if "sector" not in df.columns:
        df["sector"] = "Unknown"
        warnings.append("sector missing; defaulted to Unknown for all rows.")

    df["sector"] = df["sector"].astype(str).str.strip()

    if "market_cap_usd" not in df.columns:
        df["market_cap_usd"] = np.nan

    if "dividend_yield" not in df.columns:
        df["dividend_yield"] = np.nan

    df["market_cap_usd"] = pd.to_numeric(df["market_cap_usd"], errors="coerce")
    df["dividend_yield"] = pd.to_numeric(df["dividend_yield"], errors="coerce")

    if auto_enrich:
        need_mc = df["market_cap_usd"].isna()
        need_dy = df["dividend_yield"].isna()
        need_sector = df["sector"].eq("Unknown") | df["sector"].isna()

        if need_mc.any() or need_dy.any() or need_sector.any():
            tickers = df["ticker"].tolist()
            tks = yf.Tickers(" ".join(tickers)).tickers

            mc_fill = []
            dy_fill = []
            sec_fill = []
            for t in tickers:
                tk = tks.get(t)
                info = {}
                try:
                    info = tk.info if tk is not None else {}
                except Exception:
                    info = {}

                mc = info.get("marketCap", None)
                dy = info.get("dividendYield", None)
                sec = info.get("sector", None)

                mc_fill.append(float(mc) if mc is not None else np.nan)
                dy_fill.append(float(dy) if dy is not None else 0.0)
                sec_fill.append(str(sec).strip() if sec else "Unknown")

            mc_fill = np.array(mc_fill, dtype=float)
            dy_fill = np.array(dy_fill, dtype=float)

            if need_mc.any():
                df.loc[need_mc, "market_cap_usd"] = mc_fill[need_mc.values]
                warnings.append("market_cap_usd missing for some rows; attempted to enrich from Yahoo Finance.")

            if need_dy.any():
                df.loc[need_dy, "dividend_yield"] = dy_fill[need_dy.values]
                warnings.append("dividend_yield missing for some rows; attempted to enrich from Yahoo Finance (defaults to 0.0 if unavailable).")

            if need_sector.any():
                df.loc[need_sector, "sector"] = np.array(sec_fill, dtype=object)[need_sector.values]
                warnings.append("sector missing for some rows; attempted to enrich from Yahoo Finance (defaults to Unknown if unavailable).")

    if df["market_cap_usd"].isna().any():
        rows = df.index[df["market_cap_usd"].isna()].tolist()
        raise ValueError(f"market_cap_usd still missing after enrichment at rows: {rows}")

    if df["dividend_yield"].isna().any():
        df["dividend_yield"] = df["dividend_yield"].fillna(0.0)

    if consolidate_duplicates:
        df = (
            df.groupby(["ticker", "sector", "market_cap_usd", "dividend_yield", "asset_class"], as_index=False)
              .agg({"weight": "sum"})
        )
        df["weight"] = df["weight"] / df["weight"].sum()

    ip = fund_cfg.get("instrument_policy", {})
    allowed = ip.get("allowed_asset_classes", None)
    if allowed is not None:
        allowed_set = {str(x).strip() for x in allowed}
        bad_ac = ~df["asset_class"].isin(allowed_set)
        if bad_ac.any():
            bad_vals = sorted(df.loc[bad_ac, "asset_class"].unique().tolist())
            raise ValueError(f"Invalid asset_class values: {bad_vals}. Allowed: {sorted(list(allowed_set))}")

    return df.reset_index(drop=True), {"warnings": warnings}
