"""
Portfolio Optimizer - Streamlit Web Application
Web interface for advanced portfolio optimization with enhanced features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io

# Import portfolio optimizer modules
from src.validate import validate_holdings
from src.data import fetch_price_data, compute_log_returns
from src.risk import estimate_mu_sigma
from src.volatility_forecasting import forecast_mu_sigma_forward_looking
from src.optimization_objectives import optimize_sharpe_ratio, optimize_risk_parity, optimize_minimum_volatility, optimize_maximum_return
from src.black_litterman import BlackLittermanModel
from src.transaction_costs import TransactionCostModel, optimize_with_transaction_costs
from src.regime_detection import RegimeDetector
from src.stress_testing import HistoricalStressScenarios, SensitivityAnalysis
from src.factor_model import FactorModel


# Page config
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Force white background and modern UI
st.markdown("""
    <style>
    /* Force white background universally */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
    }
    
    /* Modern card design */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
        color: #4a4a4a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Positive/Negative indicators */
    .positive { 
        color: #10b981; 
        font-weight: 700;
        font-size: 16px;
    }
    .negative { 
        color: #ef4444; 
        font-weight: 700;
        font-size: 16px;
    }
    
    /* Improved buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Modern tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8F9FA;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        padding: 0px 24px;
        font-weight: 600;
        color: #4a4a4a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Enhanced dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Section headers */
    h1 {
        color: #1a1a1a;
        font-weight: 800;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 700;
        margin-top: 24px;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed #cbd5e0;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 10px;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid;
    }
    </style>
    """, unsafe_allow_html=True)


def load_sample_data():
    """Create sample holdings data"""
    sample_data = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JNJ', 'XOM', 'PG'],
        'weight': [0.15, 0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.04],
        'market_cap_usd': [3e12, 2.8e12, 1.7e12, 1.6e12, 800e9, 1.2e12, 600e9, 400e9, 450e9, 350e9],
        'sector': ['Technology', 'Technology', 'Technology', 'Consumer', 'Automotive', 
                   'Technology', 'Technology', 'Healthcare', 'Energy', 'Consumer'],
        'dividend_yield': [0.004, 0.007, 0.0, 0.007, 0.0, 0.0, 0.0, 0.027, 0.035, 0.028],
        'asset_class': ['Equity']*10
    }
    return pd.DataFrame(sample_data)


@st.cache_data
def fetch_and_process_data(tickers, start_date, end_date):
    """Fetch price data and compute returns"""
    try:
        with st.spinner("Fetching price data..."):
            prices = fetch_price_data(tickers, start_date, end_date)
            returns = compute_log_returns(prices)
        return prices, returns, None
    except Exception as e:
        return None, None, str(e)


def create_default_config():
    """Create default configuration"""
    return {
        "fund_name": "Portfolio",
        "instrument_policy": {"long_only": True},
        "constraints": {
            "max_single_security_weight": 0.15,
            "min_single_security_weight": 0.0,
            "sector_caps": {"enabled": False, "max_sector_weight": 0.30},
            "market_cap_constraints": {"enabled": False, "cap_limit_usd": 2e9},
            "income_constraints": {"enabled": False}
        },
        "optimization": {
            "return_target": {},
            "turnover_control": {
                "max_one_way_turnover": 0.30,
                "turnover_penalty_l1": 0.0,
                "use_turnover_penalty": True
            }
        },
        "risk_model": {
            "shrinkage_alpha": 0.15,
            "ensure_psd": True
        }
    }


def main():
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 30px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin-bottom: 30px; color: white;'>
            <h1 style='font-size: 48px; margin: 0; border: none; color: white;'>📊 Portfolio Optimizer</h1>
            <p style='font-size: 18px; margin-top: 10px; color: rgba(255,255,255,0.9);'>
                Advanced portfolio optimization with AI-powered analytics
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            end_date = st.date_input("End Date", datetime.now())
        with col2:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*3))
        
        # Data source
        data_source = st.radio("Data Source", ["Upload CSV", "Manual Entry", "CSV Builder", "Use Sample Data"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload Holdings CSV", type=['csv'])
            if uploaded_file:
                holdings_df = pd.read_csv(uploaded_file)
            else:
                holdings_df = None
        
        elif data_source == "Manual Entry":
            st.markdown("#### Enter Holdings Manually")
            st.caption("Enter ticker and number of shares. Market cap will be fetched automatically.")
            
            manual_holdings = []
            num_entries = st.number_input("Number of Holdings", min_value=1, max_value=20, value=5)
            
            cols = st.columns([1, 2])
            with cols[0]:
                st.write("**Ticker**")
            with cols[1]:
                st.write("**Shares**")
            
            for i in range(num_entries):
                col1, col2 = st.columns([1, 2])
                with col1:
                    ticker = st.text_input(f"Ticker {i+1}", key=f"ticker_{i}", placeholder="AAPL")
                with col2:
                    shares = st.number_input(f"Shares {i+1}", min_value=0.0, value=100.0, key=f"shares_{i}")
                
                if ticker:
                    manual_holdings.append({"ticker": ticker.upper(), "shares": shares})
            
            # Button to fetch data and calculate weights
            if st.button("📊 Fetch Data & Calculate Weights", key="fetch_manual"):
                if manual_holdings:
                    try:
                        import yfinance as yf
                        
                        # Fetch current prices and market caps
                        tickers_list = [h["ticker"] for h in manual_holdings]
                        data = yf.download(tickers_list, period="1d", progress=False, auto_adjust=True)
                        
                        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                            prices = data['Close']
                            if isinstance(prices, pd.Series):
                                prices = prices.to_frame().T
                        else:
                            prices = data
                        
                        # Get current prices
                        current_prices = prices.iloc[-1] if len(prices) > 0 else None
                        
                        holdings_list = []
                        total_value = 0
                        
                        for holding in manual_holdings:
                            ticker = holding["ticker"]
                            shares = holding["shares"]
                            
                            # Get market cap from yfinance
                            info = yf.Ticker(ticker).info
                            market_cap = info.get('marketCap', 0)
                            sector = info.get('sector', 'Unknown')
                            dividend_yield = info.get('dividendYield', 0.0) or 0.0
                            
                            # Current price
                            if current_prices is not None:
                                try:
                                    price = current_prices[ticker] if ticker in current_prices.index else 0
                                except:
                                    price = 0
                            else:
                                price = 0
                            
                            position_value = shares * price if price > 0 else 0
                            total_value += position_value
                            
                            holdings_list.append({
                                "ticker": ticker,
                                "shares": shares,
                                "price": price,
                                "value": position_value,
                                "market_cap_usd": market_cap,
                                "sector": sector,
                                "dividend_yield": dividend_yield,
                                "asset_class": "Equity"
                            })
                        
                        # Calculate weights
                        if total_value > 0:
                            for h in holdings_list:
                                h["weight"] = h["value"] / total_value
                        
                        # Create DataFrame
                        holdings_df = pd.DataFrame(holdings_list)[["ticker", "weight", "sector", "market_cap_usd", "dividend_yield", "asset_class"]]
                        
                        st.success(f"✅ Fetched {len(holdings_df)} holdings")
                        st.dataframe(holdings_df, use_container_width=True)
                        
                        # Store in session for later use
                        st.session_state.manual_holdings_df = holdings_df
                        
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)[:100]}")
                        holdings_df = None
            
            # Use previously fetched data if available
            if "manual_holdings_df" in st.session_state:
                holdings_df = st.session_state.manual_holdings_df
            else:
                holdings_df = None
        
        elif data_source == "CSV Builder":
            st.markdown("#### 📋 Build Your CSV")
            st.caption("Enter tickers and shares - market data fetched automatically")
            
            builder_holdings = []
            num_builder_entries = st.number_input("Number of Holdings", min_value=1, max_value=20, value=3, key="csv_builder_num")
            
            st.markdown("**Enter tickers and number of shares:**")
            
            cols = st.columns([2, 2])
            with cols[0]:
                st.write("**Ticker**")
            with cols[1]:
                st.write("**Shares**")
            
            for i in range(num_builder_entries):
                cols = st.columns([2, 2])
                with cols[0]:
                    ticker = st.text_input(f"Ticker {i+1}", key=f"csv_ticker_{i}", placeholder="AAPL")
                with cols[1]:
                    shares = st.number_input(f"Shares {i+1}", min_value=0.0, value=100.0, key=f"csv_shares_{i}")
                
                if ticker:
                    builder_holdings.append({"ticker": ticker.upper(), "shares": shares})
            
            if st.button("🔄 Fetch Market Data & Build CSV", key="csv_builder_fetch"):
                if builder_holdings:
                    try:
                        import yfinance as yf
                        
                        st.info("Fetching market data...")
                        
                        # Fetch current prices
                        tickers_list = [h["ticker"] for h in builder_holdings]
                        data = yf.download(tickers_list, period="1d", progress=False, auto_adjust=True)
                        
                        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                            prices = data['Close']
                            if isinstance(prices, pd.Series):
                                prices = prices.to_frame().T
                        else:
                            prices = data
                        
                        current_prices = prices.iloc[-1] if len(prices) > 0 else None
                        
                        holdings_list = []
                        total_value = 0
                        
                        for holding in builder_holdings:
                            ticker = holding["ticker"]
                            shares = holding["shares"]
                            
                            try:
                                # Get market data from yfinance
                                ticker_obj = yf.Ticker(ticker)
                                info = ticker_obj.info
                                
                                market_cap = info.get('marketCap', 0) or 0
                                sector = info.get('sector', 'Unknown')
                                dividend_yield = info.get('dividendYield', 0.0) or 0.0
                                
                                # Current price
                                if current_prices is not None:
                                    try:
                                        price = current_prices[ticker] if ticker in current_prices.index else 0
                                    except:
                                        price = info.get('currentPrice', 0) or 0
                                else:
                                    price = info.get('currentPrice', 0) or 0
                                
                                position_value = shares * price if price > 0 else 0
                                total_value += position_value
                                
                                holdings_list.append({
                                    "ticker": ticker,
                                    "shares": shares,
                                    "price": price,
                                    "value": position_value,
                                    "market_cap_usd": market_cap,
                                    "sector": sector,
                                    "dividend_yield": dividend_yield,
                                    "asset_class": "Equity"
                                })
                            except Exception as e:
                                st.warning(f"Error fetching data for {ticker}: {str(e)[:50]}")
                        
                        # Calculate weights
                        if total_value > 0:
                            for h in holdings_list:
                                h["weight"] = h["value"] / total_value
                        
                        # Create CSV DataFrame (only required columns)
                        csv_df = pd.DataFrame(holdings_list)[["ticker", "weight", "sector", "market_cap_usd", "dividend_yield", "asset_class"]]
                        
                        st.success(f"✅ Built portfolio from {len(csv_df)} holdings")
                        
                        # Show details
                        st.markdown("**Portfolio Holdings:**")
                        details_df = pd.DataFrame(holdings_list)[["ticker", "shares", "price", "sector", "value", "weight"]]
                        details_df['value'] = details_df['value'].apply(lambda x: f"${x:,.0f}")
                        details_df['weight'] = details_df['weight'].apply(lambda x: f"{x*100:.1f}%")
                        details_df['price'] = details_df['price'].apply(lambda x: f"${x:.2f}")
                        st.dataframe(details_df, use_container_width=True)
                        
                        # Download button
                        csv_data = csv_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="csv_download"
                        )
                        
                        # Use this data
                        st.session_state.csv_builder_df = csv_df
                        holdings_df = csv_df
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)[:200]}")
                        holdings_df = None
            
            # Use previously built data if available
            if "csv_builder_df" in st.session_state:
                holdings_df = st.session_state.csv_builder_df
            else:
                holdings_df = None
        
        else:  # Use Sample Data
            holdings_df = load_sample_data()
        
        st.divider()
        st.subheader("Portfolio Constraints")
        
        # Market cap constraint - NEW
        use_market_cap = st.checkbox("Limit by Market Cap per Equity", value=False)
        if use_market_cap:
            market_cap_limit = st.number_input(
                "Max Market Cap per Equity ($B)",
                value=2.0,
                min_value=0.1,
                step=0.5
            )
        else:
            market_cap_limit = None
        
        # Sector constraints - NEW
        use_sector_caps = st.checkbox("Limit by Sector", value=False)
        if use_sector_caps:
            sector_cap = st.slider(
                "Max Sector Weight (%)",
                min_value=5,
                max_value=100,
                value=30,
                step=5
            )
        else:
            sector_cap = None
        
        # Single security constraint
        max_security_weight = st.slider(
            "Max Single Security Weight (%)",
            min_value=1,
            max_value=50,
            value=15,
            step=1
        )
        
        # Optimization options
        st.divider()
        st.subheader("Optimization Options")
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["Maximum Return", "Sharpe Ratio", "Risk Parity", "Minimum Volatility"]
        )
        
        use_garch = st.checkbox("Use GARCH Volatility Forecasting", value=False)
        use_black_litterman = st.checkbox("Use Black-Litterman", value=False)
        use_transaction_costs = st.checkbox("Account for Transaction Costs", value=True)
        
        if use_transaction_costs:
            commission_bps = st.number_input("Commission (bps)", value=5, min_value=0)
            spread_bps = st.number_input("Spread (bps)", value=2, min_value=0)
        
        # Return estimation tuning
        st.divider()
        st.subheader("Return Estimation Tuning")
        st.caption("Adjust shrinkage to control return forecasts")
        
        mu_shrink_beta = st.slider(
            "Return Shrinkage (0=historic, 1=zero)",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
            help="Lower = more trust in historical returns, Higher = more conservative"
        )
        
        mu_clip = st.slider(
            "Max Return Bound (±%)",
            min_value=0.2,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Caps annual returns at ±X. Use 1.0+ for tech stocks"
        )
        
        # Run optimization
        run_optimization = st.button("🚀 Run Optimization", type="primary")
    
    # Main content area
    if holdings_df is None:
        st.warning("Please upload a CSV or select sample data")
        return
    
    # Display holdings
    st.subheader("📋 Portfolio Holdings")
    st.dataframe(holdings_df, use_container_width=True)
    
    if not run_optimization:
        st.info("Configure settings in the sidebar and click 'Run Optimization' to begin")
        return
    
    # Validate holdings
    try:
        # Save DataFrame to temporary CSV and validate
        temp_csv = "/tmp/temp_holdings.csv"
        holdings_df.to_csv(temp_csv, index=False)
        
        validated_df, validation_report = validate_holdings(
            temp_csv,
            {"fund_name": "Portfolio"}
        )
    except Exception as e:
        st.error(f"Validation error: {e}")
        return
    
    # Fetch data
    tickers = validated_df['ticker'].tolist()
    prices, returns, error = fetch_and_process_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if error:
        st.error(f"Data fetch error: {error}")
        return
    
    # Filter out holdings with very small weights or problematic data
    min_weight_threshold = 0.01  # 1% minimum weight
    valid_holdings = validated_df[validated_df['weight'] >= min_weight_threshold].copy()
    
    if len(valid_holdings) < len(validated_df):
        removed_count = len(validated_df) - len(valid_holdings)
        removed_tickers = validated_df[validated_df['weight'] < min_weight_threshold]['ticker'].tolist()
        st.warning(
            f"Removed {removed_count} holdings with weight < {min_weight_threshold*100}%: {', '.join(removed_tickers)}. "
            f"These were too small to optimize effectively."
        )
        validated_df = valid_holdings.copy()
        # Renormalize weights
        validated_df['weight'] = validated_df['weight'] / validated_df['weight'].sum()
    
    if len(validated_df) < 2:
        st.error("Need at least 2 holdings to optimize. Please add more holdings.")
        return
    
    # Create configuration
    config = create_default_config()
    config["constraints"]["max_single_security_weight"] = max_security_weight / 100
    
    if use_market_cap and market_cap_limit:
        config["constraints"]["market_cap_constraints"] = {
            "enabled": True,
            "cap_limit_usd": market_cap_limit * 1e9,
            "enforcement": {"method": "weighted_avg_log_market_cap"}
        }
    
    if use_sector_caps and sector_cap:
        config["constraints"]["sector_caps"] = {
            "enabled": True,
            "max_sector_weight": sector_cap / 100
        }
    
    # Filter returns to match validated holdings
    valid_tickers = validated_df['ticker'].tolist()
    returns_filtered = returns[[col for col in returns.columns if col in valid_tickers]]
    
    if returns_filtered.empty or len(returns_filtered.columns) == 0:
        st.error("No valid price data found for the holdings. Please check the tickers.")
        return
    
    # Estimate risk parameters
    if use_garch:
        mu, sigma, assets = forecast_mu_sigma_forward_looking(returns_filtered, use_garch=True)
    else:
        mu, sigma, assets = estimate_mu_sigma(returns_filtered, trading_days=252, mu_shrink_beta=mu_shrink_beta, mu_clip=mu_clip)
    
    # Black-Litterman adjustment (optional)
    if use_black_litterman:
        # Check if market_cap_usd column exists and is numeric
        if 'market_cap_usd' in validated_df.columns:
            market_caps = pd.to_numeric(validated_df['market_cap_usd'], errors='coerce')
            if market_caps.notna().sum() == len(validated_df):  # All values are numeric
                market_weights = market_caps.values / market_caps.sum()
                
                # Ensure dimensions match
                if len(market_weights) == len(mu):
                    bl = BlackLittermanModel(market_weights, sigma, risk_aversion=2.5)
                    mu = bl.posterior_returns()
                else:
                    st.warning(f"Market cap data mismatch: {len(market_weights)} weights vs {len(mu)} assets. Skipping Black-Litterman.")
            else:
                st.warning("Market cap data incomplete. Skipping Black-Litterman.")
                use_black_litterman = False
        else:
            st.warning("Market cap column not found. Skipping Black-Litterman.")
            use_black_litterman = False
    
    # Validate data before optimization
    import numpy as np
    if np.isnan(mu).any() or np.isnan(sigma).any():
        st.error("Invalid data detected (NaN values). Please check your holdings.")
        return
    
    if np.any(np.diag(sigma) <= 0):
        st.error("Invalid covariance matrix (non-positive diagonal). Data quality issue.")
        return
    
    # Run optimization
    try:
        current_weights = validated_df['weight'].astype(float).values
        current_weights = current_weights / current_weights.sum()
        
        result = None
        optimization_attempted = optimization_method
        
        try:
            if optimization_method == "Maximum Return":
                result = optimize_maximum_return(validated_df, mu, sigma, config)
            elif optimization_method == "Sharpe Ratio":
                result = optimize_maximum_return(validated_df, mu, sigma, config)
            elif optimization_method == "Sharpe Ratio":
                result = optimize_sharpe_ratio(validated_df, mu, sigma, config)
            elif optimization_method == "Risk Parity":
                result = optimize_risk_parity(validated_df, sigma, config)
            else:  # Minimum Volatility
                result = optimize_minimum_volatility(validated_df, sigma, config)
        except Exception as e:
            # Fall back to minimum volatility if other methods fail
            if optimization_method != "Minimum Volatility":
                st.warning(f"{optimization_method} failed: {str(e)[:60]}... Falling back to Minimum Volatility.")
                result = optimize_minimum_volatility(validated_df, sigma, config)
            else:
                raise
        
        if result is None:
            raise ValueError("All optimization methods failed")
        
        target_weights = result['weights']
        
        # Apply transaction costs
        if use_transaction_costs:
            cost_model = TransactionCostModel(
                commission_bps=commission_bps,
                spread_bps=spread_bps
            )
            aum = 1_000_000  # Default AUM
            
            cost_result = optimize_with_transaction_costs(
                target_weights,
                current_weights,
                aum,
                cost_model,
                rebalance_threshold_bps=50
            )
            
            target_weights = cost_result['adjusted_weights']
            cost_impact = cost_result['cost_bps']
        else:
            cost_impact = 0
        
    except Exception as e:
        st.error(f"Optimization error: {e}")
        st.write(str(e))
        return
    
    # Display results tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📈 Summary", "🎯 Recommendations", "📊 Analysis", "⚠️ Stress Tests", "📉 Factor Attribution", "🔮 ML Predictions", "📋 Backtest"]
    )
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        port_return_current = np.sum(mu * current_weights)
        port_return_target = np.sum(mu * target_weights)
        port_vol_current = np.sqrt(current_weights @ sigma @ current_weights)
        port_vol_target = np.sqrt(target_weights @ sigma @ target_weights)
        
        with col1:
            st.metric(
                "Expected Return (Current)",
                f"{port_return_current*100:.2f}%"
            )
        with col2:
            st.metric(
                "Expected Return (Target)",
                f"{port_return_target*100:.2f}%",
                delta=f"{(port_return_target - port_return_current)*100:+.2f}%"
            )
        with col3:
            st.metric(
                "Portfolio Vol (Current)",
                f"{port_vol_current*100:.2f}%"
            )
        with col4:
            st.metric(
                "Portfolio Vol (Target)",
                f"{port_vol_target*100:.2f}%",
                delta=f"{(port_vol_target - port_vol_current)*100:+.2f}%"
            )
        
        # Sharpe ratio
        rf = 0.02
        sharpe_current = (port_return_current - rf) / max(port_vol_current, 1e-10)
        sharpe_target = (port_return_target - rf) / max(port_vol_target, 1e-10)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sharpe Ratio (Current)", f"{sharpe_current:.3f}")
        with col2:
            st.metric(
                "Sharpe Ratio (Target)",
                f"{sharpe_target:.3f}",
                delta=f"{sharpe_target - sharpe_current:+.3f}"
            )
        with col3:
            if use_transaction_costs:
                st.metric("Transaction Cost", f"{cost_impact:.1f} bps")
    
    with tab2:
        st.subheader("Portfolio Rebalancing")
        
        # Identify toxic holdings (negative returns + above-average weight)
        st.markdown("#### 🚨 Toxic Holdings to Remove")
        
        toxic_threshold = 0.0  # Returns below 0%
        holdings_df_analysis = validated_df.copy()
        holdings_df_analysis['Expected Return'] = mu * 100
        holdings_df_analysis['Current Weight'] = current_weights * 100
        holdings_df_analysis['Target Weight'] = target_weights * 100
        
        toxic_holdings = holdings_df_analysis[holdings_df_analysis['Expected Return'] < toxic_threshold].sort_values('Expected Return')
        
        if len(toxic_holdings) > 0:
            st.warning(f"Found {len(toxic_holdings)} toxic holdings dragging down portfolio returns:")
            
            toxic_display = pd.DataFrame({
                'Ticker': toxic_holdings['ticker'],
                'Sector': toxic_holdings['sector'],
                'Expected Return': toxic_holdings['Expected Return'].round(2),
                'Current Weight': toxic_holdings['Current Weight'].round(2),
                'Target Weight': toxic_holdings['Target Weight'].round(2),
                'Market Cap ($B)': (toxic_holdings['market_cap_usd'] / 1e9).round(2)
            })
            st.dataframe(toxic_display, use_container_width=True)
            
            st.info(
                f"**Recommendation:** Remove or significantly reduce the above holdings. "
                f"The optimizer is already minimizing them (Target Weight < Current Weight), "
                f"but eliminating them entirely would improve portfolio returns."
            )
        else:
            st.success("✅ No toxic holdings found. All positions have positive expected returns.")
        
        st.divider()
        st.markdown("#### 📊 Full Rebalancing Table")
        
        # Create recommendations table
        rebalance_df = pd.DataFrame({
            'Ticker': validated_df['ticker'],
            'Sector': validated_df['sector'],
            'Expected Return': mu * 100,
            'Current Weight': (current_weights * 100).round(2),
            'Target Weight': (target_weights * 100).round(2),
            'Change': ((target_weights - current_weights) * 100).round(2)
        })
        rebalance_df = rebalance_df.sort_values('Expected Return', ascending=False)
        
        st.dataframe(rebalance_df, use_container_width=True)
        
        # Download recommendations
        csv = rebalance_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations (CSV)",
            data=csv,
            file_name=f"rebalancing_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Portfolio Analysis")
        
        # Expected returns by asset
        st.markdown("#### Expected Annual Returns by Asset")
        returns_df = pd.DataFrame({
            'Ticker': validated_df['ticker'],
            'Sector': validated_df['sector'],
            'Expected Return': mu * 100,
            'Current Weight': current_weights * 100,
            'Target Weight': target_weights * 100
        }).sort_values('Expected Return', ascending=False)
        
        st.dataframe(
            returns_df.style.format({'Expected Return': '{:.2f}%', 'Current Weight': '{:.2f}%', 'Target Weight': '{:.2f}%'}),
            use_container_width=True
        )
        
        st.caption(f"Return Estimation Settings: Shrinkage={mu_shrink_beta:.2f}, Max Bound=±{mu_clip:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Allocation")
            current_alloc = pd.DataFrame({
                'Ticker': validated_df['ticker'],
                'Weight': current_weights * 100
            })
            fig_current = px.pie(current_alloc, values='Weight', names='Ticker', title='Current Allocation')
            st.plotly_chart(fig_current, use_container_width=True)
        
        with col2:
            st.markdown("#### Target Allocation")
            target_alloc = pd.DataFrame({
                'Ticker': validated_df['ticker'],
                'Weight': target_weights * 100
            })
            fig_target = px.pie(target_alloc, values='Weight', names='Ticker', title='Target Allocation')
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Sector breakdown
        st.markdown("#### Sector Exposure")
        
        sector_current = validated_df.groupby('sector').apply(lambda x: (current_weights[x.index] * 100).sum())
        sector_target = validated_df.groupby('sector').apply(lambda x: (target_weights[x.index] * 100).sum())
        
        sector_df = pd.DataFrame({
            'Sector': sector_current.index,
            'Current': sector_current.values,
            'Target': sector_target.values
        })
        
        fig_sector = go.Figure(data=[
            go.Bar(name='Current', x=sector_df['Sector'], y=sector_df['Current']),
            go.Bar(name='Target', x=sector_df['Sector'], y=sector_df['Target'])
        ])
        fig_sector.update_layout(barmode='group', title='Sector Exposure Comparison')
        st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab4:
        st.subheader("Stress Testing")
        
        # Historical crises
        st.markdown("#### Historical Crisis Scenarios")
        
        scenarios = ['2008_financial_crisis', '2020_covid_crash', '2022_rate_shock']
        scenario_results = []
        
        for scenario in scenarios:
            impact = HistoricalStressScenarios.apply_scenario(
                1_000_000,
                target_weights,
                validated_df['asset_class'].tolist(),
                scenario
            )
            scenario_results.append({
                'Scenario': impact['scenario'],
                'Portfolio Loss': f"{impact['loss_pct']:.2f}%",
                'Loss Amount': f"${impact['loss_dollars']/1000:.0f}K"
            })
        
        stress_df = pd.DataFrame(scenario_results)
        st.dataframe(stress_df, use_container_width=True)
        
        # Sensitivity analysis
        st.markdown("#### Sensitivity Analysis")
        
        sensitivity = SensitivityAnalysis.volatility_sensitivity(mu, sigma, target_weights)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=sensitivity['vol_shift_pct'],
            y=sensitivity['sharpe_ratio'],
            mode='lines+markers',
            name='Sharpe Ratio'
        ))
        fig_sens.update_layout(
            title='Sharpe Ratio vs Volatility Change',
            xaxis_title='Volatility Change (%)',
            yaxis_title='Sharpe Ratio'
        )
        st.plotly_chart(fig_sens, use_container_width=True)
    
    with tab5:
        st.subheader("Factor Attribution")
        
        try:
            fm = FactorModel(returns)
            fm.fit()
            
            risk_decomp = fm.decompose_risk(target_weights)
            
            fig_factors = px.bar(
                risk_decomp,
                x='factor',
                y='pct_of_total',
                title='Risk Contribution by Factor',
                labels={'factor': 'Factor', 'pct_of_total': 'Risk Contribution (%)'}
            )
            st.plotly_chart(fig_factors, use_container_width=True)
            
            st.dataframe(risk_decomp, use_container_width=True)
        except Exception as e:
            st.warning(f"Factor model not available: {str(e)[:100]}")
    
    with tab6:
        st.subheader("🔮 ML Predictions: Current vs Optimized")
        st.caption("Projected portfolio growth path over next 12 months")
        
        try:
            # Generate 252 trading days projection (1 year)
            days_ahead = 252
            dates_future = pd.date_range(start=datetime.now(), periods=days_ahead, freq='B')
            
            # Current portfolio projection
            daily_return_current = (port_return_current / 100) / 252
            cumulative_current = [(1 + daily_return_current) ** i for i in range(days_ahead)]
            
            # Optimized portfolio projection
            daily_return_target = (port_return_target / 100) / 252
            cumulative_target = [(1 + daily_return_target) ** i for i in range(days_ahead)]
            
            # Create projection dataframe
            projection_df = pd.DataFrame({
                'Date': dates_future,
                'Current Portfolio': cumulative_current,
                'Optimized Portfolio': cumulative_target
            })
            
            # Plot
            fig_projection = px.line(
                projection_df,
                x='Date',
                y=['Current Portfolio', 'Optimized Portfolio'],
                title='Portfolio Growth Projection (12 months)',
                labels={'value': 'Portfolio Value (multiplier)', 'Date': 'Date'},
                markers=True
            )
            
            fig_projection.update_layout(
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_projection, use_container_width=True)
            
            # Projection metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_ending = cumulative_current[-1]
                st.metric(
                    "Current Portfolio (12m)",
                    f"{(current_ending - 1) * 100:.1f}%",
                    f"Value: {current_ending:.2f}x"
                )
            
            with col2:
                target_ending = cumulative_target[-1]
                st.metric(
                    "Optimized Portfolio (12m)",
                    f"{(target_ending - 1) * 100:.1f}%",
                    f"Value: {target_ending:.2f}x"
                )
            
            with col3:
                improvement = ((target_ending - current_ending) / current_ending) * 100
                st.metric(
                    "Expected Improvement",
                    f"{improvement:.1f}%",
                    f"+{(target_ending - current_ending):.2f}x"
                )
            
            st.info(
                f"**Projection assumes:** Constant daily returns of {daily_return_current*252*100:.2f}% (current) "
                f"and {daily_return_target*252*100:.2f}% (optimized). Actual results will vary."
            )
            
        except Exception as e:
            st.warning(f"Projection not available: {str(e)[:100]}")
    
    with tab7:
        st.subheader("📋 Backtest: Historical Forecast Accuracy")
        st.caption("How well did past return predictions match actual market performance?")
        
        try:
            # Backtest: compare predicted returns vs actual returns for held assets
            backtest_periods = []
            
            for ticker in validated_df['ticker']:
                try:
                    # Get historical data for this ticker
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(period="2y")
                    
                    if len(hist) < 252:
                        continue
                    
                    # Split into train (1y) and test (1y)
                    train_end = len(hist) // 2
                    train_data = hist.iloc[:train_end]
                    test_data = hist.iloc[train_end:]
                    
                    # Calculate returns
                    train_returns = train_data['Close'].pct_change().dropna()
                    test_returns = test_data['Close'].pct_change().dropna()
                    
                    predicted_return = train_returns.mean() * 252 * 100
                    actual_return = test_returns.mean() * 252 * 100
                    
                    backtest_periods.append({
                        'Ticker': ticker,
                        'Predicted Return (%)': predicted_return,
                        'Actual Return (%)': actual_return,
                        'Error (%)': actual_return - predicted_return,
                        'Accuracy': 100 - abs(actual_return - predicted_return)
                    })
                except:
                    pass
            
            if backtest_periods:
                backtest_df = pd.DataFrame(backtest_periods)
                backtest_df = backtest_df.sort_values('Accuracy', ascending=False)
                
                st.dataframe(backtest_df, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mean_accuracy = backtest_df['Accuracy'].mean()
                    st.metric("Average Accuracy", f"{mean_accuracy:.1f}%")
                
                with col2:
                    mean_error = backtest_df['Error (%)'].abs().mean()
                    st.metric("Mean Absolute Error", f"{mean_error:.2f}%")
                
                with col3:
                    positive_bias = (backtest_df['Error (%)'] > 0).sum() / len(backtest_df) * 100
                    st.metric("Positive Bias %", f"{positive_bias:.1f}%")
                
                # Chart
                fig_backtest = px.scatter(
                    backtest_df,
                    x='Predicted Return (%)',
                    y='Actual Return (%)',
                    color='Accuracy',
                    size='Accuracy',
                    hover_data=['Ticker', 'Error (%)'],
                    title='Predicted vs Actual Returns (1Y Backtest)',
                    trendline='ols',
                    color_continuous_scale='RdYlGn'
                )
                
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                st.info(
                    "**Interpretation:** Points near the diagonal line indicate accurate predictions. "
                    "Above the line = conservative estimates. Below the line = optimistic estimates."
                )
            else:
                st.warning("Not enough historical data for backtest analysis")
            
        except Exception as e:
            st.warning(f"Backtest not available: {str(e)[:100]}")
    
    # Footer
    st.divider()
    st.markdown("""
    **Portfolio Optimizer - Built By Shaan Matharu, Moravian University Class of 2028** • Advanced portfolio optimization with GARCH, Black-Litterman, and stress testing
    """)


if __name__ == "__main__":
    main()
