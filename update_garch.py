#!/usr/bin/env python3
"""
Daily GARCH Model Update Script
Runs at 10 PM to update volatility forecasts with latest market data
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import logging

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.volatility_forecasting import DynamicCovarianceMatrix
from src.data import fetch_price_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'garch_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CACHE_DIR = project_root / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

def load_holdings():
    """Load holdings from CSV"""
    holdings_path = project_root / 'holdings.csv'
    if not holdings_path.exists():
        logger.error(f"Holdings file not found: {holdings_path}")
        return None
    
    df = pd.read_csv(holdings_path)
    if 'ticker' not in df.columns:
        logger.error("Holdings CSV must have 'ticker' column")
        return None
    
    return df['ticker'].tolist()

def update_garch_model(tickers, lookback_years=3):
    """Update GARCH model with latest data"""
    logger.info(f"Starting GARCH update for {len(tickers)} tickers")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    try:
        logger.info(f"Fetching price data from {start_date.date()} to {end_date.date()}")
        prices = fetch_price_data(tickers, start_date, end_date)
        
        if prices.empty:
            logger.error("No price data retrieved")
            return False
        
        logger.info(f"Retrieved {len(prices)} days of data for {len(prices.columns)} tickers")
        
        returns = prices.pct_change().dropna()
        logger.info(f"Calculated returns: {len(returns)} observations")
        
        logger.info("Fitting DCC-GARCH model...")
        dcm = DynamicCovarianceMatrix(method='dcc')
        dcm.fit(returns)
        
        forecast_days = 21
        logger.info(f"Generating {forecast_days}-day forecast...")
        mu_forecast, cov_forecast = dcm.forecast_covariance(returns, forecast_days)
        
        cache_data = {
            'timestamp': datetime.now(),
            'tickers': tickers,
            'mu_forecast': mu_forecast,
            'cov_forecast': cov_forecast,
            'model': dcm,
            'last_returns': returns.tail(252)
        }
        
        cache_file = CACHE_DIR / 'garch_forecast.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"GARCH model updated successfully and cached to {cache_file}")
        logger.info(f"Forecast mu range: [{mu_forecast.min():.4f}, {mu_forecast.max():.4f}]")
        logger.info(f"Forecast volatility range: [{np.sqrt(np.diag(cov_forecast)).min():.4f}, {np.sqrt(np.diag(cov_forecast)).max():.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating GARCH model: {str(e)}", exc_info=True)
        return False

def main():
    logger.info("="*60)
    logger.info("Starting daily GARCH update job")
    logger.info("="*60)
    
    tickers = load_holdings()
    if not tickers:
        logger.error("Failed to load holdings")
        sys.exit(1)
    
    success = update_garch_model(tickers)
    
    if success:
        logger.info("GARCH update completed successfully")
        sys.exit(0)
    else:
        logger.error("GARCH update failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
