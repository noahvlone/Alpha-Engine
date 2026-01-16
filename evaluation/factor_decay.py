"""
Factor Decay Analyzer
Analyzes how factor signal strength changes over time.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from config import EvaluationConfig
from .ic_calculator import ICCalculator


class FactorDecayAnalyzer:
    """
    Factor Decay Analyzer.
    
    Analyzes how quickly a factor's predictive power diminishes over time.
    This helps determine optimal holding periods for portfolio rebalancing.
    
    Key metrics:
    - IC at different lags
    - Signal half-life
    - Cumulative IC curve
    """
    
    def __init__(
        self,
        max_lag: int = EvaluationConfig.DECAY_MAX_LAG,
        ic_calculator: Optional[ICCalculator] = None
    ):
        """
        Initialize FactorDecayAnalyzer.
        
        Args:
            max_lag: Maximum lag to analyze (days).
            ic_calculator: IC calculator instance.
        """
        self.max_lag = max_lag
        self.ic_calculator = ic_calculator or ICCalculator()
        self.lags = list(range(1, max_lag + 1))
    
    def calculate_decay_curve(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Calculate IC at different forward lags.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            DataFrame with IC for each lag.
        """
        ic_by_lag = {}
        
        for lag in self.lags:
            ic_series = self.ic_calculator.calculate_ic_series(
                factor, prices, lag, method
            )
            ic_by_lag[lag] = {
                "Mean_IC": ic_series.mean(),
                "Std_IC": ic_series.std(),
                "IC_IR": ic_series.mean() / (ic_series.std() + 1e-10),
                "Hit_Rate": (ic_series > 0).mean(),
            }
        
        decay_df = pd.DataFrame(ic_by_lag).T
        decay_df.index.name = "Lag_Days"
        
        return decay_df
    
    def estimate_half_life(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> float:
        """
        Estimate factor signal half-life.
        
        Half-life is the number of days for IC to decay to half its initial value.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            Estimated half-life in days.
        """
        decay_curve = self.calculate_decay_curve(factor, prices, method)
        mean_ic = decay_curve["Mean_IC"]
        
        # Find initial IC (lag 1)
        initial_ic = abs(mean_ic.iloc[0])
        half_ic = initial_ic / 2
        
        # Find lag where IC drops below half
        for lag in mean_ic.index:
            if abs(mean_ic.loc[lag]) < half_ic:
                return lag
        
        # If IC never drops below half, return max lag
        return self.max_lag
    
    def calculate_cumulative_ic(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Calculate cumulative IC curve.
        
        Useful for understanding total predictive power over time.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            DataFrame with cumulative IC.
        """
        decay_curve = self.calculate_decay_curve(factor, prices, method)
        decay_curve["Cumulative_IC"] = decay_curve["Mean_IC"].cumsum()
        
        return decay_curve
    
    def get_optimal_holding_period(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> int:
        """
        Estimate optimal holding period based on IC decay.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            Optimal holding period in days.
        """
        decay_curve = self.calculate_decay_curve(factor, prices, method)
        ic_ir = decay_curve["IC_IR"]
        
        # Find lag with highest IC IR (risk-adjusted IC)
        optimal_lag = ic_ir.idxmax()
        
        return optimal_lag
    
    def analyze_decay(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> dict:
        """
        Comprehensive decay analysis.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            Dictionary with full decay analysis.
        """
        decay_curve = self.calculate_decay_curve(factor, prices, method)
        
        analysis = {
            "decay_curve": decay_curve,
            "half_life": self.estimate_half_life(factor, prices, method),
            "optimal_holding_period": self.get_optimal_holding_period(
                factor, prices, method
            ),
            "initial_ic": decay_curve["Mean_IC"].iloc[0],
            "final_ic": decay_curve["Mean_IC"].iloc[-1],
            "decay_ratio": (
                decay_curve["Mean_IC"].iloc[-1] / 
                (decay_curve["Mean_IC"].iloc[0] + 1e-10)
            ),
        }
        
        return analysis
    
    def compare_factors(
        self,
        factors: dict,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Compare decay characteristics of multiple factors.
        
        Args:
            factors: Dictionary of factor DataFrames.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            DataFrame with comparison metrics.
        """
        comparison = {}
        
        for factor_name, factor_df in factors.items():
            analysis = self.analyze_decay(factor_df, prices, method)
            comparison[factor_name] = {
                "Initial_IC": analysis["initial_ic"],
                "Half_Life": analysis["half_life"],
                "Optimal_Holding": analysis["optimal_holding_period"],
                "Decay_Ratio": analysis["decay_ratio"],
            }
        
        return pd.DataFrame(comparison).T
    
    def get_decay_summary(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> str:
        """
        Get human-readable decay analysis summary.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            Summary string.
        """
        analysis = self.analyze_decay(factor, prices, method)
        
        summary = f"""
Factor Decay Analysis Summary
=============================
Initial IC (1-day):     {analysis['initial_ic']:.4f}
Signal Half-Life:       {analysis['half_life']} days
Optimal Holding Period: {analysis['optimal_holding_period']} days
Decay Ratio:           {analysis['decay_ratio']:.2%}

Interpretation:
"""
        
        if analysis['half_life'] < 5:
            summary += "- FAST decay: Signal requires frequent rebalancing (daily/weekly)\n"
        elif analysis['half_life'] < 20:
            summary += "- MODERATE decay: Weekly to bi-weekly rebalancing suggested\n"
        else:
            summary += "- SLOW decay: Monthly rebalancing is sufficient\n"
        
        return summary


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    
    # Create a simple momentum factor
    factor = prices.pct_change(periods=20)
    
    decay_analyzer = FactorDecayAnalyzer(max_lag=10)
    
    print("Decay Curve:")
    print(decay_analyzer.calculate_decay_curve(factor, prices))
    
    print("\n" + decay_analyzer.get_decay_summary(factor, prices))
