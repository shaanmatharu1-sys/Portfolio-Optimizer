import pandas as pd
import numpy as np
import yfinance as yf


def fetch_price_data(tickers, start_date, end_date, return_missing=False):
    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))

    if len(tickers) == 0:
        raise ValueError("No tickers provided.")

    px = _download_bulk(tickers, start_date, end_date)

    if px is None or px.empty or px.shape[1] == 0:
        px = pd.DataFrame()

    keep = []
    missing = []

    if not px.empty:
        px.columns = [str(c).upper().strip() for c in px.columns]
        for t in tickers:
            if t in px.columns and not px[t].dropna().empty:
                keep.append(t)
            else:
                missing.append(t)
        px = px[keep].dropna(how="all")

    if px.empty or px.shape[1] == 0:
        px = pd.DataFrame()
        keep = []
        missing = []
        for t in tickers:
            one = _download_one(t, start_date, end_date)
            if one is None or one.empty:
                missing.append(t)
                continue
            s = one.dropna()
            if s.empty:
                missing.append(t)
                continue
            px[t] = s
            keep.append(t)

        px = px.dropna(how="all")
        if px.empty or px.shape[1] == 0:
            raise ValueError(f"No price data available for any tickers in the given date range. Tickers: {tickers}")

    px = px.sort_index().replace([np.inf, -np.inf], np.nan)

    if return_missing:
        return px, missing
    return px


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets


def _download_bulk(tickers, start_date, end_date):
    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True
        )
    except Exception:
        return None

    if data is None or len(data) == 0:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(data.columns.get_level_values(0))
        if "Adj Close" in lvl0:
            return data["Adj Close"].copy()
        if "Close" in lvl0:
            return data["Close"].copy()
        for k in ["Open", "High", "Low", "Volume"]:
            if k in lvl0:
                return data[k].copy()
        return None

    if "Adj Close" in data.columns:
        return data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    if "Close" in data.columns:
        return data[["Close"]].rename(columns={"Close": tickers[0]})
    return None


def _download_one(ticker, start_date, end_date):
    try:
        h = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    except Exception:
        return None
    if h is None or h.empty:
        return None
    if "Close" in h.columns:
        return h["Close"].copy()
    return None
