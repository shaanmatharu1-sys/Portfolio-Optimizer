"""
Advanced transaction cost modeling and market impact estimation.
Includes: fixed fees, variable spreads, market impact, timing costs.
Enables realistic optimization accounting for trading friction.
"""

import numpy as np
import pandas as pd


class TransactionCostModel:
    """
    Comprehensive transaction cost modeling including:
    - Fixed commission (bps)
    - Spreads (bid-ask)
    - Market impact (cost increases with order size)
    - Timing costs (alpha decay over execution period)
    - Opportunity costs
    """
    
    def __init__(
        self,
        commission_bps: float = 5.0,
        spread_bps: float = 2.0,
        market_impact_coef: float = 0.1,
        price_impact_exponent: float = 1.5,
        timing_cost_bps_per_day: float = 0.5,
        daily_volume_pct: np.ndarray = None,
        current_positions: np.ndarray = None
    ):
        """
        Args:
            commission_bps: Fixed commission in basis points
            spread_bps: Bid-ask spread in basis points
            market_impact_coef: Coefficient for market impact function
            price_impact_exponent: Power law exponent (1.5 typical)
            timing_cost_bps_per_day: Opportunity cost of time
            daily_volume_pct: Daily volume as % of AUM per asset
            current_positions: Current position sizes
        """
        self.commission_bps = commission_bps
        self.spread_bps = spread_bps
        self.market_impact_coef = market_impact_coef
        self.price_impact_exponent = price_impact_exponent
        self.timing_cost_bps_per_day = timing_cost_bps_per_day
        self.daily_volume_pct = daily_volume_pct
        self.current_positions = current_positions
    
    def fixed_costs(self, dollar_traded: float) -> float:
        """Fixed costs: commission + spread."""
        return (self.commission_bps + self.spread_bps) / 10000 * dollar_traded
    
    def market_impact_cost(self, dollar_traded: np.ndarray, portfolio_value: float) -> np.ndarray:
        """
        Market impact: cost increases non-linearly with order size.
        Formula: impact = coef * (traded_pct)^exponent * spread
        
        Args:
            dollar_traded: Dollar amounts per asset (array)
            portfolio_value: Total portfolio value
        
        Returns:
            Impact costs per asset (array)
        """
        trade_pcts = dollar_traded / portfolio_value
        
        # Power-law market impact
        impact_bps = self.market_impact_coef * 10000 * (trade_pcts ** self.price_impact_exponent)
        
        return impact_bps / 10000 * dollar_traded
    
    def timing_cost(self, dollar_traded: np.ndarray, execution_days: float = 1.0) -> np.ndarray:
        """
        Opportunity cost of execution spread over multiple days.
        
        Args:
            dollar_traded: Dollar amounts per asset (array)
            execution_days: Number of days to execute (default 1)
        
        Returns:
            Timing costs per asset (array)
        """
        if execution_days <= 1:
            return np.zeros_like(dollar_traded)
        
        # Cost accumulates over execution period
        timing_cost_total = self.timing_cost_bps_per_day / 10000 * execution_days
        
        return timing_cost_total * dollar_traded
    
    def compute_total_costs(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float,
        execution_days: float = 1.0
    ) -> dict:
        """
        Comprehensive cost calculation for rebalancing.
        
        Args:
            current_weights: Current portfolio weights (n_assets,)
            target_weights: Target portfolio weights (n_assets,)
            portfolio_value: Total portfolio value
            execution_days: Number of days for execution
        
        Returns:
            dict with:
            - total_cost (float)
            - cost_bps (float)
            - detailed_costs (dict)
            - cost_per_asset (array)
        """
        current_w = np.asarray(current_weights, dtype=float).reshape(-1)
        target_w = np.asarray(target_weights, dtype=float).reshape(-1)
        
        # Dollar amounts to trade
        dollar_positions_current = current_w * portfolio_value
        dollar_positions_target = target_w * portfolio_value
        
        trades = dollar_positions_target - dollar_positions_current
        abs_trades = np.abs(trades)
        
        # Gross notional traded
        gross_notional = np.sum(abs_trades) / 2  # Each trade involves buyer + seller
        
        # Costs by component
        fixed = self.fixed_costs(gross_notional)
        market_impact = np.sum(self.market_impact_cost(abs_trades, portfolio_value))
        timing = np.sum(self.timing_cost(abs_trades, execution_days))
        
        total_cost = fixed + market_impact + timing
        cost_bps = (total_cost / portfolio_value) * 10000
        
        return {
            "total_cost": total_cost,
            "cost_bps": cost_bps,
            "cost_pct": (total_cost / portfolio_value) * 100,
            "detailed_costs": {
                "fixed_commission_spread": fixed,
                "market_impact": market_impact,
                "timing_cost": timing
            },
            "gross_notional": gross_notional,
            "one_way_turnover": np.sum(abs_trades) / (2 * portfolio_value)
        }
    
    def effective_cost_per_unit(self, trade_size_pcts: np.ndarray) -> np.ndarray:
        """
        What is the effective cost (as % of trade) for different trade sizes?
        Useful for understanding cost structure.
        
        Args:
            trade_size_pcts: Trade sizes as % of portfolio (array)
        
        Returns:
            Effective cost in bps per unit traded
        """
        total_cost_bps = (self.commission_bps + self.spread_bps) + \
                         self.market_impact_coef * 10000 * (trade_size_pcts ** self.price_impact_exponent)
        
        return total_cost_bps


class ExecutionCost:
    """
    Execution strategies and their associated costs.
    Different execution approaches (VWAP, TWAP, IS, MOO, etc.) have different costs.
    """
    
    @staticmethod
    def vwap_cost(
        order_size_pct: float,
        volatility: float,
        daily_volume_pct: float = 0.2,
        time_of_day_factor: float = 1.0
    ) -> float:
        """
        VWAP (Volume Weighted Average Price) execution cost.
        Executes in proportion to market volume throughout the day.
        
        Cost formula inspired by research (Almgren, Chriss):
        cost_bps ≈ λ * σ * sqrt(Q/V) + commissions
        
        Args:
            order_size_pct: Order as % of daily volume
            volatility: Daily volatility (annualized)
            daily_volume_pct: Asset's daily volume as % of AUM
            time_of_day_factor: Multiplier (more expensive if rushed)
        
        Returns:
            Execution cost in basis points
        """
        daily_vol = volatility / np.sqrt(252)
        
        # VWAP cost from execution literature
        # λ ≈ 0.5-1.0 (market impact coefficient)
        lam = 0.75
        
        cost_bps = lam * (daily_vol * 10000) * np.sqrt(order_size_pct / daily_volume_pct) * time_of_day_factor
        
        # Add small fixed component
        cost_bps += 5  # 5 bps base
        
        return cost_bps
    
    @staticmethod
    def twap_cost(
        order_size_pct: float,
        volatility: float,
        execution_hours: float = 4.0,
        daily_volume_pct: float = 0.2
    ) -> float:
        """
        TWAP (Time Weighted Average Price) execution cost.
        Executes uniformly over time period.
        
        More predictable but potentially higher market impact.
        
        Args:
            order_size_pct: Order as % of daily volume
            volatility: Daily volatility
            execution_hours: Hours to execute over (4 = market hours, 8 = extended)
            daily_volume_pct: Asset's daily volume as % of AUM
        
        Returns:
            Execution cost in basis points
        """
        daily_vol = volatility / np.sqrt(252)
        
        # TWAP typically more expensive than VWAP due to timing predictability
        lam = 1.0
        
        # Longer execution = more time risk
        time_risk_factor = execution_hours / 6.5  # Adjust for market hours
        
        cost_bps = lam * (daily_vol * 10000) * np.sqrt(order_size_pct / daily_volume_pct) * time_risk_factor
        cost_bps += 5
        
        return cost_bps
    
    @staticmethod
    def market_on_open_cost(
        order_size_pct: float,
        daily_volume_pct: float = 0.2
    ) -> float:
        """
        Market on Open (MOO) execution cost.
        Execute at open, very fast, potentially high spread.
        
        Best for small orders, worst for large orders.
        """
        base_spread = 10  # 10 bps wider spread at open
        
        # Market impact is high due to speed
        impact_bps = 50 * order_size_pct / (daily_volume_pct * 0.1)
        
        return base_spread + impact_bps


def optimize_with_transaction_costs(
    target_weights: np.ndarray,
    current_weights: np.ndarray,
    portfolio_value: float,
    cost_model: TransactionCostModel = None,
    rebalance_threshold_bps: float = 50.0
) -> dict:
    """
    Adjust target weights to account for transaction costs.
    Benefits: avoids costly micro-rebalances, improves after-cost returns.
    
    Args:
        target_weights: Optimal weights (before costs)
        current_weights: Current weights
        portfolio_value: Portfolio value
        cost_model: TransactionCostModel instance
        rebalance_threshold_bps: Only rebalance if benefit > cost (in bps)
    
    Returns:
        dict with:
        - adjusted_weights: Weights accounting for costs
        - rebalance_decision: bool (whether to execute)
        - cost_impact: estimated cost in dollars
    """
    if cost_model is None:
        cost_model = TransactionCostModel()
    
    # Compute cost of rebalancing
    cost_info = cost_model.compute_total_costs(
        current_weights,
        target_weights,
        portfolio_value
    )
    
    cost_bps = cost_info["cost_bps"]
    
    # Decision: is the rebalance worth it?
    rebalance = cost_bps <= rebalance_threshold_bps
    
    # If not rebalancing, stay at current weights
    if not rebalance:
        adjusted_weights = current_weights
    else:
        adjusted_weights = target_weights
    
    return {
        "adjusted_weights": adjusted_weights,
        "rebalance": rebalance,
        "cost_bps": cost_bps,
        "cost_dollars": cost_info["total_cost"],
        "gross_notional": cost_info["gross_notional"],
        "decision_threshold_bps": rebalance_threshold_bps
    }
