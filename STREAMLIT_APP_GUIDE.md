"""
STREAMLIT WEB APP - USER GUIDE
===============================

Run the Portfolio Optimizer web interface locally or deploy to the cloud.

INSTALLATION
─────────────────────────────────────────────────────────────────────────────

1. Install dependencies:
   pip install -r requirements_streamlit.txt

2. Run the app:
   streamlit run app.py

3. Open in browser:
   http://localhost:8501


FEATURES
─────────────────────────────────────────────────────────────────────────────

✨ INPUT
   • Upload holdings CSV or use sample data
   • Set date range for historical data (default: 3 years)
   • Automatic data validation and cleaning

⚙️  CONSTRAINTS (NEW)
   • Max Market Cap per Equity: Limit position size based on market cap
     → Example: Only allow up to $2B market cap per position
     → Useful for: Liquidity management, small-cap bias
   
   • Max Sector Weight: Cap allocation to any single sector
     → Example: Max 30% in Technology
     → Useful for: Sector diversification, regulatory limits
   
   • Max Single Security Weight: General position size limit
     → Example: Max 15% per stock
     → Useful: Concentration limits, risk management

🎯 OPTIMIZATION OPTIONS
   • Sharpe Ratio Maximization (default)
     → Maximize risk-adjusted returns
     → Best for: Most use cases
   
   • Risk Parity
     → Equal risk contribution per asset
     → Best for: Diversified, defensive portfolios
   
   • Minimum Volatility
     → Minimize portfolio volatility
     → Best for: Conservative investors

🔧 ADVANCED OPTIONS
   • GARCH Volatility Forecasting
     → Forward-looking volatility estimates
     → Improves risk modeling during crises
   
   • Black-Litterman
     → Blend market equilibrium with your views
     → Reduces over-concentration
   
   • Transaction Costs
     → Account for commissions and market impact
     → Enables smart rebalancing decisions

📊 OUTPUTS
   • Summary: Key metrics (return, volatility, Sharpe ratio)
   • Recommendations: Rebalancing trades (download CSV)
   • Analysis: Pie charts, sector breakdown
   • Stress Tests: Crisis scenarios + sensitivity analysis
   • Factor Attribution: Risk decomposition by factor


UPLOADING HOLDINGS CSV
─────────────────────────────────────────────────────────────────────────────

Required columns:
  • ticker (string)          - Stock symbol (e.g., "AAPL")
  • weight (float)           - Current portfolio weight (0-1)
  • market_cap_usd (float)   - Market cap in USD
  • sector (string)          - Sector name (e.g., "Technology")
  • dividend_yield (float)   - Dividend yield (0-1)
  • asset_class (string)     - "Equity", "Bond", etc.

Example CSV:
───────────
ticker,weight,market_cap_usd,sector,dividend_yield,asset_class
AAPL,0.15,3000000000000,Technology,0.004,Equity
MSFT,0.15,2800000000000,Technology,0.007,Equity
GOOGL,0.12,1700000000000,Technology,0.0,Equity
AMZN,0.12,1600000000000,Consumer,0.007,Equity
TSLA,0.10,800000000000,Automotive,0.0,Equity


CONSTRAINT EXAMPLES
─────────────────────────────────────────────────────────────────────────────

Example 1: Conservative Portfolio
├─ Max Security Weight: 10%
├─ Max Sector Weight: 25%
├─ Max Market Cap: $500B
└─ Result: Highly diversified, defensive

Example 2: Growth Portfolio
├─ Max Security Weight: 20%
├─ Max Sector Weight: 40% (Technology allowed)
├─ Max Market Cap: No limit
└─ Result: Growth-oriented, higher concentration

Example 3: Dividend Portfolio
├─ Max Security Weight: 15%
├─ Max Sector Weight: 30%
├─ Min Dividend Yield: 2%
└─ Result: Income-focused

Example 4: Small-Cap Tilt
├─ Max Security Weight: 12%
├─ Max Market Cap: $1B
├─ Max Sector Weight: 25%
└─ Result: Small-cap bias, controlled concentration


INTERPRETATION
─────────────────────────────────────────────────────────────────────────────

📈 SUMMARY TAB
   Expected Return (Current) vs (Target)
     → Positive delta = improvement expected
     → Compare to transaction costs to see net benefit
   
   Volatility (Current) vs (Target)
     → Lower is better for risk-averse investors
     → Balance return vs risk reduction
   
   Sharpe Ratio
     → Risk-adjusted return (higher is better)
     → Key metric for comparing portfolios

🎯 RECOMMENDATIONS TAB
   Current Weight vs Target Weight
     → BUY (positive): Increase position
     → SELL (negative): Decrease position
     → Size indicates trade magnitude
   
   Download CSV for execution
     → Import to trading system
     → Use for order generation

📊 ANALYSIS TAB
   Current vs Target Allocation
     → Visual comparison of changes
     → Pie charts show relative sizes
   
   Sector Exposure
     → Bar chart shows sector changes
     → Helps identify concentration shifts

⚠️  STRESS TESTS TAB
   Historical Scenarios
     → How portfolio performs in crisis (2008, 2020, etc.)
     → Portfolio Loss: Expected decline in crisis
     → Better outcome = lower loss %
   
   Sensitivity Analysis
     → Sharpe Ratio vs Volatility change
     → Shows robustness to vol assumptions

📉 FACTOR ATTRIBUTION TAB
   Risk Decomposition
     → What drives portfolio risk?
     → Market, Momentum, Value, Quality, Vol
     → Helps understand exposures


ADVANCED USAGE
─────────────────────────────────────────────────────────────────────────────

Iterative Optimization:

   1. Start with constraints you're comfortable with
   2. Run optimization
   3. Review recommendations
   4. Adjust constraints if needed
   5. Re-run optimization
   6. Repeat until satisfied

   Example workflow:
   ─────────────────
   Run 1: Max Sector 30%, Max Stock 15%
         → Result: Technology 35% (too high)
         → Adjust: Max Sector 25%
   
   Run 2: Max Sector 25%, Max Stock 15%
         → Result: Balanced sectors
         → Review stress tests
   
   Run 3: Enable Stress Tests
         → 2008 crisis: -22% loss
         → Acceptable? If yes, done. If no, add constraints.

Using Advanced Features:

   Enable GARCH:
     • Better volatility estimates during crises
     • Use when you have 2+ years data
     • Runtime increases slightly

   Enable Black-Litterman:
     • More stable, diversified weights
     • Reduces extreme positions
     • Use if you have strong views on specific assets

   Enable Transaction Costs:
     • Prevents bad small rebalances
     • Shows realistic cost impact
     • Always recommended for real trading


TROUBLESHOOTING
─────────────────────────────────────────────────────────────────────────────

Issue: "Validation error: Missing column X"
→ Fix: Ensure CSV has all required columns (see example)

Issue: "Data fetch error: No price data available"
→ Fix: Check tickers are valid, increase date range

Issue: "Optimization failed"
→ Fix: Loosen constraints (higher max weights, fewer restrictions)

Issue: "Stress tests show unrealistic losses"
→ Fix: Review asset_class mapping (Equity vs Bond)

Issue: "Factor Attribution not available"
→ Fix: Insufficient data, try with more history


DEPLOYMENT OPTIONS
─────────────────────────────────────────────────────────────────────────────

Local (Easiest):
  streamlit run app.py
  → Access at http://localhost:8501

Streamlit Cloud (Free):
  1. Push code to GitHub
  2. Go to share.streamlit.io
  3. Connect GitHub repo
  4. Deploy!
  → Access at share.streamlit.io/[your-app]

Docker (Scalable):
  docker build -t portfolio-optimizer .
  docker run -p 8501:8501 portfolio-optimizer

AWS/Azure/GCP:
  - Deploy Docker container
  - Use their app services
  - Scale as needed


KEYBOARD SHORTCUTS
─────────────────────────────────────────────────────────────────────────────

C     → Clear cache
R     → Rerun app
?     → Help


TIPS & BEST PRACTICES
─────────────────────────────────────────────────────────────────────────────

✓ DO:
  • Start with sample data to understand features
  • Review stress tests before deploying
  • Download recommendations as CSV backup
  • Enable transaction costs for realistic results
  • Use market cap constraints for liquidity

✗ DON'T:
  • Trust optimization blindly - review results
  • Use more than 5 years old data
  • Set impossible constraints (e.g., 100+ holdings in 10% max each)
  • Forget to check sector exposure
  • Deploy large positions without stress testing


EXAMPLE WORKFLOW
─────────────────────────────────────────────────────────────────────────────

Monday Morning: Weekly Rebalancing Review

  1. Upload latest holdings CSV
     ↓
  2. Set constraints:
     - Max Stock: 15%
     - Max Sector: 30%
     - Max Market Cap: $2B (liquidity limit)
     ↓
  3. Run optimization:
     - Sharpe Ratio
     - With Transaction Costs
     - With GARCH
     ↓
  4. Review Summary:
     - Return improvement: +50 bps ✓
     - Sharpe ratio improvement ✓
     - Transaction cost: -5 bps ✓
     ↓
  5. Check Recommendations:
     - BUY MSFT: +2%
     - SELL AAPL: -1.5%
     - ... (review all trades)
     ↓
  6. Run Stress Tests:
     - 2008 scenario: -18% (acceptable)
     - Sensitivity: Stable
     ↓
  7. Download CSV:
     - Send to trading desk
     - Execute via trading system
     ↓
  8. Save results:
     - Document performance vs baseline
     - Track actual vs expected returns


ADVANCED CONFIGURATION (Code)
─────────────────────────────────────────────────────────────────────────────

To customize the app, edit these sections of app.py:

1. Change default constraints:
   → Modify create_default_config() function

2. Add new optimization methods:
   → Import from src/optimization_objectives.py
   → Add to selectbox options

3. Add sector-specific constraints:
   → Extend constraints.py
   → Add to config in app

4. Customize stress test scenarios:
   → Edit scenario list in tab4
   → Add/remove from HistoricalStressScenarios


═══════════════════════════════════════════════════════════════════════════════

For questions or issues, see ADVANCED_FEATURES_GUIDE.md or README_ENTERPRISE.md

═══════════════════════════════════════════════════════════════════════════════
"""
