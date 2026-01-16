"""
Momentum Factor Module
Calculates price momentum as a key alpha signal.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import FactorConfig


class MomentumFactor:
    """
    Momentum Factor Calculator.
    
    Momentum measures the rate of price change over a specified period.
    Formula: Momentum = (Price_t - Price_{t-n}) / Price_{t-n}
    
    Stocks with higher momentum have outperformed recently and may continue
    to outperform (momentum effect).
    """
    
    def __init__(
        self,
        short_lookback: int = FactorConfig.MOMENTUM_LOOKBACK_SHORT,
        medium_lookback: int = FactorConfig.MOMENTUM_LOOKBACK_MEDIUM,
        long_lookback: int = FactorConfig.MOMENTUM_LOOKBACK_LONG
    ):
        """
        Initialize MomentumFactor.
        
        Args:
            short_lookback: Short-term lookback period (default: 20 days).
            medium_lookback: Medium-term lookback period (default: 60 days).
            long_lookback: Long-term lookback period (default: 120 days).
        """
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
    
    def calculate(
        self,
        prices: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate momentum for all stocks.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            lookback: Lookback period. Defaults to medium_lookback.
            
        Returns:
            DataFrame with momentum values.
        """
        if lookback is None:
            lookback = self.medium_lookback
        
        # Calculate simple momentum: (Price_t - Price_{t-n}) / Price_{t-n}
        momentum = prices.pct_change(periods=lookback)
        
        return momentum
    
    def calculate_all(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum for all lookback periods and combine.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            
        Returns:
            DataFrame with combined momentum score.
        """
        # Calculate momentum for each period
        mom_short = self.calculate(prices, self.short_lookback)
        mom_medium = self.calculate(prices, self.medium_lookback)
        mom_long = self.calculate(prices, self.long_lookback)
        
        # Weight shorter periods more heavily
        # Weights: short=0.5, medium=0.3, long=0.2
        combined = (
            0.5 * self._normalize(mom_short) +
            0.3 * self._normalize(mom_medium) +
            0.2 * self._normalize(mom_long)
        )
        
        return combined
    
    def calculate_12_1(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 12-1 momentum (classic momentum factor).
        
        This skips the most recent month to avoid short-term reversal.
        Returns over months 2-12, ignoring month 1.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            
        Returns:
            DataFrame with 12-1 momentum values.
        """
        # 12-month return minus 1-month return
        ret_12m = prices.pct_change(periods=252)  # ~12 months
        ret_1m = prices.pct_change(periods=21)    # ~1 month
        
        momentum_12_1 = ret_12m - ret_1m
        
        return momentum_12_1
    
    def calculate_volume_momentum(
        self,
        volumes: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate volume momentum.
        
        Measures how trading volume is changing.
        
        Args:
            volumes: DataFrame with dates as index and tickers as columns.
            lookback: Lookback period.
            
        Returns:
            DataFrame with volume momentum values.
        """
        if lookback is None:
            lookback = self.short_lookback
        
        # Volume change over the period
        vol_momentum = volumes.pct_change(periods=lookback)
        
        return vol_momentum
    
    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data using z-score (cross-sectional).
        
        Args:
            data: DataFrame to normalize.
            
        Returns:
            Normalized DataFrame.
        """
        # Cross-sectional z-score (normalize across stocks for each date)
        mean = data.mean(axis=1)
        std = data.std(axis=1)
        
        normalized = data.sub(mean, axis=0).div(std, axis=0)
        
        return normalized


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")["Close"]
    
    mom = MomentumFactor()
    result = mom.calculate(data)
    print("Momentum Factor (last 5 rows):")
    print(result.tail())
