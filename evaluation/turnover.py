"""
Turnover Calculator
Measures portfolio position changes and associated costs.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

from config import EvaluationConfig, PortfolioConfig


class TurnoverCalculator:
    """
    Turnover Calculator.
    
    Measures how frequently portfolio positions change.
    High turnover leads to:
    - Higher transaction costs
    - Potential market impact
    - Tax inefficiency
    
    Key metrics:
    - One-way turnover: Sum of absolute weight changes / 2
    - Two-way turnover: Sum of absolute weight changes
    - Average holding period: 1 / turnover
    """
    
    def __init__(
        self,
        commission_rate: float = PortfolioConfig.COMMISSION_RATE,
        slippage: float = PortfolioConfig.SLIPPAGE,
        max_acceptable_turnover: float = EvaluationConfig.MAX_ACCEPTABLE_TURNOVER
    ):
        """
        Initialize TurnoverCalculator.
        
        Args:
            commission_rate: Commission per transaction.
            slippage: Estimated slippage per transaction.
            max_acceptable_turnover: Maximum acceptable monthly turnover.
        """
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.max_acceptable_turnover = max_acceptable_turnover
        self.total_transaction_cost = commission_rate + slippage
    
    def calculate_weights(
        self,
        rankings: pd.DataFrame,
        top_n: int = 5,
        bottom_n: int = 5,
        equal_weight: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate portfolio weights based on rankings.
        
        Args:
            rankings: DataFrame with stock rankings.
            top_n: Number of stocks in long leg.
            bottom_n: Number of stocks in short leg.
            equal_weight: If True, use equal weights.
            
        Returns:
            Tuple of (long_weights, short_weights) DataFrames.
        """
        n_stocks = rankings.count(axis=1)
        
        # Long weights (top ranked)
        is_long = rankings <= top_n
        if equal_weight:
            long_weights = is_long.astype(float) / top_n
        else:
            long_weights = is_long.astype(float).div(is_long.sum(axis=1), axis=0)
        
        # Short weights (bottom ranked)
        is_short = rankings > (n_stocks.values.reshape(-1, 1) - bottom_n)
        if equal_weight:
            short_weights = is_short.astype(float) / bottom_n
        else:
            short_weights = is_short.astype(float).div(is_short.sum(axis=1), axis=0)
        
        return long_weights, short_weights
    
    def calculate_turnover(
        self,
        weights: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate one-way turnover.
        
        Args:
            weights: DataFrame with portfolio weights.
            
        Returns:
            Series with daily turnover.
        """
        weight_changes = weights.diff().abs()
        one_way_turnover = weight_changes.sum(axis=1) / 2
        
        return one_way_turnover
    
    def calculate_two_way_turnover(
        self,
        weights: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate two-way turnover.
        
        Args:
            weights: DataFrame with portfolio weights.
            
        Returns:
            Series with two-way turnover.
        """
        return self.calculate_turnover(weights) * 2
    
    def calculate_net_portfolio_turnover(
        self,
        long_weights: pd.DataFrame,
        short_weights: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate net turnover for long-short portfolio.
        
        Args:
            long_weights: DataFrame with long weights.
            short_weights: DataFrame with short weights.
            
        Returns:
            Series with net portfolio turnover.
        """
        long_turnover = self.calculate_turnover(long_weights)
        short_turnover = self.calculate_turnover(short_weights)
        
        # Net is average of both legs
        net_turnover = (long_turnover + short_turnover) / 2
        
        return net_turnover
    
    def calculate_average_turnover(
        self,
        weights: pd.DataFrame,
        period: str = "monthly"
    ) -> float:
        """
        Calculate average turnover for a period.
        
        Args:
            weights: DataFrame with portfolio weights.
            period: 'daily', 'weekly', 'monthly', 'annual'.
            
        Returns:
            Average turnover.
        """
        daily_turnover = self.calculate_turnover(weights)
        
        if period == "daily":
            return daily_turnover.mean()
        elif period == "weekly":
            return daily_turnover.mean() * 5
        elif period == "monthly":
            return daily_turnover.mean() * 21
        elif period == "annual":
            return daily_turnover.mean() * 252
        else:
            return daily_turnover.mean()
    
    def calculate_holding_period(
        self,
        weights: pd.DataFrame
    ) -> float:
        """
        Calculate average holding period.
        
        Args:
            weights: DataFrame with portfolio weights.
            
        Returns:
            Average holding period in days.
        """
        monthly_turnover = self.calculate_average_turnover(weights, "monthly")
        
        if monthly_turnover == 0:
            return float('inf')
        
        # Average holding period = 1 / monthly turnover * 21 days
        holding_period = (1 / monthly_turnover) * 21
        
        return holding_period
    
    def calculate_transaction_costs(
        self,
        weights: pd.DataFrame,
        capital: float = PortfolioConfig.INITIAL_CAPITAL
    ) -> pd.Series:
        """
        Calculate transaction costs from turnover.
        
        Args:
            weights: DataFrame with portfolio weights.
            capital: Total capital.
            
        Returns:
            Series with daily transaction costs.
        """
        two_way_turnover = self.calculate_two_way_turnover(weights)
        costs = two_way_turnover * capital * self.total_transaction_cost
        
        return costs
    
    def calculate_annual_cost_drag(
        self,
        weights: pd.DataFrame
    ) -> float:
        """
        Calculate annual cost drag from turnover.
        
        Args:
            weights: DataFrame with portfolio weights.
            
        Returns:
            Annual cost as percentage of capital.
        """
        annual_turnover = self.calculate_average_turnover(weights, "annual")
        cost_drag = annual_turnover * self.total_transaction_cost * 2  # Round trip
        
        return cost_drag
    
    def get_turnover_summary(
        self,
        weights: pd.DataFrame,
        factor_name: str = "Factor"
    ) -> dict:
        """
        Get comprehensive turnover summary.
        
        Args:
            weights: DataFrame with portfolio weights.
            factor_name: Name of the factor.
            
        Returns:
            Dictionary with turnover metrics.
        """
        summary = {
            "Factor": factor_name,
            "Daily_Turnover": self.calculate_average_turnover(weights, "daily"),
            "Monthly_Turnover": self.calculate_average_turnover(weights, "monthly"),
            "Annual_Turnover": self.calculate_average_turnover(weights, "annual"),
            "Avg_Holding_Period_Days": self.calculate_holding_period(weights),
            "Annual_Cost_Drag": self.calculate_annual_cost_drag(weights),
        }
        
        # Add assessment
        monthly_turnover = summary["Monthly_Turnover"]
        if monthly_turnover <= self.max_acceptable_turnover:
            summary["Assessment"] = "ACCEPTABLE"
        else:
            summary["Assessment"] = "HIGH"
        
        return summary
    
    def analyze_turnover_by_threshold(
        self,
        composite_scores: pd.DataFrame,
        thresholds: list = [0.05, 0.10, 0.15, 0.20]
    ) -> pd.DataFrame:
        """
        Analyze how turnover changes with different top/bottom thresholds.
        
        Args:
            composite_scores: DataFrame with composite factor scores.
            thresholds: List of threshold percentages to test.
            
        Returns:
            DataFrame with turnover for each threshold.
        """
        results = []
        n_stocks = composite_scores.count(axis=1)
        
        for threshold in thresholds:
            n_top = int(n_stocks.iloc[-1] * threshold)
            if n_top < 1:
                n_top = 1
            
            rankings = composite_scores.rank(axis=1, ascending=False)
            long_weights = (rankings <= n_top).astype(float) / n_top
            
            monthly_turnover = self.calculate_average_turnover(long_weights, "monthly")
            annual_cost = self.calculate_annual_cost_drag(long_weights)
            
            results.append({
                "Threshold": f"Top {int(threshold*100)}%",
                "N_Stocks": n_top,
                "Monthly_Turnover": monthly_turnover,
                "Annual_Cost_Drag": annual_cost,
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    
    # Create rankings
    factor = prices.pct_change(periods=20)
    rankings = factor.rank(axis=1, ascending=False)
    
    turnover_calc = TurnoverCalculator()
    long_weights, short_weights = turnover_calc.calculate_weights(rankings, top_n=2, bottom_n=2)
    
    print("Turnover Summary:")
    print(turnover_calc.get_turnover_summary(long_weights, "Momentum"))
    
    print("\nThreshold Analysis:")
    print(turnover_calc.analyze_turnover_by_threshold(factor))
