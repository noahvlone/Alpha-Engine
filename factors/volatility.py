"""
Volatility Factor Module
Calculates volatility as a risk measure.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import FactorConfig


class VolatilityFactor:
    """
    Volatility Factor Calculator.
    
    Volatility measures the dispersion of returns, typically using
    rolling standard deviation of log returns.
    
    Lower volatility stocks tend to outperform on a risk-adjusted basis
    (low volatility anomaly).
    """
    
    def __init__(
        self,
        short_window: int = FactorConfig.VOLATILITY_WINDOW_SHORT,
        long_window: int = FactorConfig.VOLATILITY_WINDOW_LONG
    ):
        """
        Initialize VolatilityFactor.
        
        Args:
            short_window: Short-term rolling window (default: 20 days).
            long_window: Long-term rolling window (default: 60 days).
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def calculate_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from prices.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            
        Returns:
            DataFrame with log returns.
        """
        log_returns = np.log(prices / prices.shift(1))
        return log_returns
    
    def calculate(
        self,
        prices: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            window: Rolling window size. Defaults to short_window.
            
        Returns:
            DataFrame with volatility values.
        """
        if window is None:
            window = self.short_window
        
        # Calculate log returns
        log_returns = self.calculate_log_returns(prices)
        
        # Rolling standard deviation
        volatility = log_returns.rolling(window=window).std()
        
        # Annualize (assuming 252 trading days)
        volatility_annualized = volatility * np.sqrt(252)
        
        return volatility_annualized
    
    def calculate_all(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility for all windows and combine.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            
        Returns:
            DataFrame with combined volatility score.
        """
        vol_short = self.calculate(prices, self.short_window)
        vol_long = self.calculate(prices, self.long_window)
        
        # Weight shorter period more heavily
        combined = 0.6 * self._normalize(vol_short) + 0.4 * self._normalize(vol_long)
        
        return combined
    
    def calculate_inverse(
        self,
        prices: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate inverse volatility (low vol = high score).
        
        This is useful for factor investing as low volatility
        stocks tend to have better risk-adjusted returns.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            window: Rolling window size.
            
        Returns:
            DataFrame with inverse volatility values.
        """
        volatility = self.calculate(prices, window)
        
        # Inverse volatility (add small epsilon to avoid division by zero)
        inverse_vol = 1.0 / (volatility + 1e-10)
        
        return inverse_vol
    
    def calculate_realized_volatility(
        self,
        prices: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate realized volatility using squared returns.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            window: Rolling window size.
            
        Returns:
            DataFrame with realized volatility.
        """
        log_returns = self.calculate_log_returns(prices)
        squared_returns = log_returns ** 2
        realized_vol = np.sqrt(squared_returns.rolling(window=window).sum())
        
        return realized_vol
    
    def calculate_parkinson(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate Parkinson volatility using high-low range.
        
        This estimator is more efficient than close-to-close volatility.
        
        Args:
            high: DataFrame with high prices.
            low: DataFrame with low prices.
            window: Rolling window size.
            
        Returns:
            DataFrame with Parkinson volatility.
        """
        log_hl = np.log(high / low)
        factor = 1.0 / (4.0 * np.log(2))
        
        parkinson = np.sqrt(
            factor * (log_hl ** 2).rolling(window=window).mean()
        )
        
        # Annualize
        parkinson_annualized = parkinson * np.sqrt(252)
        
        return parkinson_annualized
    
    def calculate_volatility_of_volatility(
        self,
        prices: pd.DataFrame,
        vol_window: int = 20,
        vov_window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate volatility of volatility (vol of vol).
        
        Measures the stability of volatility.
        
        Args:
            prices: DataFrame with dates as index and tickers as columns.
            vol_window: Window for volatility calculation.
            vov_window: Window for vol-of-vol calculation.
            
        Returns:
            DataFrame with vol-of-vol values.
        """
        volatility = self.calculate(prices, vol_window)
        vol_of_vol = volatility.rolling(window=vov_window).std()
        
        return vol_of_vol
    
    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data using z-score (cross-sectional).
        
        Args:
            data: DataFrame to normalize.
            
        Returns:
            Normalized DataFrame.
        """
        mean = data.mean(axis=1)
        std = data.std(axis=1)
        
        normalized = data.sub(mean, axis=0).div(std, axis=0)
        
        return normalized


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")["Close"]
    
    vol = VolatilityFactor()
    result = vol.calculate(data)
    print("Volatility Factor (last 5 rows):")
    print(result.tail())
