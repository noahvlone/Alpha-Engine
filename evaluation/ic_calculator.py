"""
Information Coefficient (IC) Calculator
Measures the correlation between factor values and future returns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple

from config import EvaluationConfig


class ICCalculator:
    """
    Information Coefficient Calculator.
    
    IC measures the correlation between factor values today and
    forward returns (t+n). Higher IC indicates better predictive power.
    
    - Pearson IC: Linear correlation
    - Spearman IC: Rank correlation (more robust to outliers)
    
    Interpretation:
    - IC > 0.05: Strong predictive signal
    - IC > 0.02: Moderate signal
    - IC < 0.02: Weak signal
    """
    
    def __init__(
        self,
        forward_periods: List[int] = None,
        ic_threshold_strong: float = EvaluationConfig.IC_THRESHOLD_STRONG,
        ic_threshold_weak: float = EvaluationConfig.IC_THRESHOLD_WEAK
    ):
        """
        Initialize ICCalculator.
        
        Args:
            forward_periods: List of forward return periods to analyze.
            ic_threshold_strong: IC threshold for strong signal.
            ic_threshold_weak: IC threshold for weak signal.
        """
        self.forward_periods = forward_periods or EvaluationConfig.IC_FORWARD_PERIODS
        self.ic_threshold_strong = ic_threshold_strong
        self.ic_threshold_weak = ic_threshold_weak
    
    def calculate_forward_returns(
        self,
        prices: pd.DataFrame,
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Calculate forward returns.
        
        Args:
            prices: DataFrame with close prices.
            periods: Number of periods ahead.
            
        Returns:
            DataFrame with forward returns.
        """
        forward_returns = prices.pct_change(periods=periods).shift(-periods)
        return forward_returns
    
    def calculate_ic_single(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        method: str = "spearman"
    ) -> float:
        """
        Calculate IC for a single date.
        
        Args:
            factor_values: Factor values for all stocks.
            forward_returns: Forward returns for all stocks.
            method: 'pearson' or 'spearman'.
            
        Returns:
            IC value.
        """
        # Remove NaN values
        valid_mask = ~(factor_values.isna() | forward_returns.isna())
        if valid_mask.sum() < 3:
            return np.nan
        
        factor_clean = factor_values[valid_mask]
        returns_clean = forward_returns[valid_mask]
        
        if method == "spearman":
            ic, _ = stats.spearmanr(factor_clean, returns_clean)
        else:
            ic, _ = stats.pearsonr(factor_clean, returns_clean)
        
        return ic
    
    def calculate_ic_series(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        forward_period: int = 1,
        method: str = "spearman"
    ) -> pd.Series:
        """
        Calculate IC time series.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            forward_period: Forward return period.
            method: 'pearson' or 'spearman'.
            
        Returns:
            Series with IC values for each date.
        """
        forward_returns = self.calculate_forward_returns(prices, forward_period)
        
        ic_values = []
        dates = []
        
        for date in factor.index:
            if date not in forward_returns.index:
                continue
            
            factor_row = factor.loc[date]
            returns_row = forward_returns.loc[date]
            
            ic = self.calculate_ic_single(factor_row, returns_row, method)
            ic_values.append(ic)
            dates.append(date)
        
        return pd.Series(ic_values, index=dates, name=f"IC_{forward_period}d")
    
    def calculate_ic_all_periods(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Calculate IC for all forward periods.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            DataFrame with IC for each period.
        """
        ic_dict = {}
        
        for period in self.forward_periods:
            ic_series = self.calculate_ic_series(factor, prices, period, method)
            ic_dict[f"IC_{period}d"] = ic_series
        
        return pd.DataFrame(ic_dict)
    
    def calculate_ic_summary(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Calculate IC summary statistics.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            method: 'pearson' or 'spearman'.
            
        Returns:
            DataFrame with IC summary statistics.
        """
        ic_all = self.calculate_ic_all_periods(factor, prices, method)
        
        summary = {
            "Mean IC": ic_all.mean(),
            "Std IC": ic_all.std(),
            "IC IR": ic_all.mean() / ic_all.std(),  # IC Information Ratio
            "Hit Rate": (ic_all > 0).mean(),  # % of positive IC
            "Max IC": ic_all.max(),
            "Min IC": ic_all.min(),
            "T-Stat": ic_all.mean() / (ic_all.std() / np.sqrt(ic_all.count())),
        }
        
        return pd.DataFrame(summary)
    
    def get_ic_interpretation(self, mean_ic: float) -> str:
        """
        Get interpretation of IC value.
        
        Args:
            mean_ic: Mean IC value.
            
        Returns:
            Interpretation string.
        """
        abs_ic = abs(mean_ic)
        direction = "positive" if mean_ic > 0 else "negative"
        
        if abs_ic >= self.ic_threshold_strong:
            strength = "STRONG"
        elif abs_ic >= self.ic_threshold_weak:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return f"{strength} {direction} signal (IC = {mean_ic:.4f})"
    
    def calculate_rolling_ic(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        forward_period: int = 1,
        rolling_window: int = 60,
        method: str = "spearman"
    ) -> pd.Series:
        """
        Calculate rolling average IC.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            forward_period: Forward return period.
            rolling_window: Rolling window size.
            method: 'pearson' or 'spearman'.
            
        Returns:
            Series with rolling IC.
        """
        ic_series = self.calculate_ic_series(factor, prices, forward_period, method)
        rolling_ic = ic_series.rolling(window=rolling_window).mean()
        
        return rolling_ic
    
    def calculate_ic_by_decile(
        self,
        factor: pd.DataFrame,
        prices: pd.DataFrame,
        forward_period: int = 1
    ) -> pd.DataFrame:
        """
        Calculate average returns by factor decile.
        
        Args:
            factor: DataFrame with factor values.
            prices: DataFrame with close prices.
            forward_period: Forward return period.
            
        Returns:
            DataFrame with average returns per decile.
        """
        forward_returns = self.calculate_forward_returns(prices, forward_period)
        
        # Assign deciles
        deciles = factor.rank(axis=1, pct=True).apply(
            lambda x: pd.cut(x, bins=10, labels=False) + 1, axis=1
        )
        
        # Calculate average return per decile
        decile_returns = []
        
        for decile in range(1, 11):
            mask = deciles == decile
            decile_return = (forward_returns * mask).sum(axis=1) / mask.sum(axis=1)
            decile_returns.append(decile_return.mean())
        
        return pd.DataFrame({
            "Decile": range(1, 11),
            "Avg_Return": decile_returns
        }).set_index("Decile")


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    
    # Create a simple momentum factor
    factor = prices.pct_change(periods=20)
    
    ic_calc = ICCalculator()
    ic_series = ic_calc.calculate_ic_series(factor, prices, forward_period=5)
    
    print("IC Summary:")
    print(ic_calc.calculate_ic_summary(factor, prices))
    
    print("\nIC Interpretation:")
    print(ic_calc.get_ic_interpretation(ic_series.mean()))
