"""
Liquidity Factor Module
Calculates liquidity metrics for stock filtering and factor analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import FactorConfig


class LiquidityFactor:
    """
    Liquidity Factor Calculator.
    
    Liquidity measures the ease of trading a stock without significant
    price impact. Uses Average Daily Trading Value (ADTV).
    
    ADTV = Average(Price × Volume) over a rolling window
    
    More liquid stocks are easier to trade and have lower transaction costs.
    """
    
    def __init__(
        self,
        window: int = FactorConfig.LIQUIDITY_WINDOW,
        min_adtv: float = FactorConfig.MIN_ADTV
    ):
        """
        Initialize LiquidityFactor.
        
        Args:
            window: Rolling window for average calculation (default: 20 days).
            min_adtv: Minimum ADTV threshold for filtering (default: 1B IDR).
        """
        self.window = window
        self.min_adtv = min_adtv
    
    def calculate_adtv(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate Average Daily Trading Value (ADTV).
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            window: Rolling window size.
            
        Returns:
            DataFrame with ADTV values.
        """
        if window is None:
            window = self.window
        
        # Daily trading value = Price × Volume
        daily_value = prices * volumes
        
        # Rolling average
        adtv = daily_value.rolling(window=window).mean()
        
        return adtv
    
    def calculate(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate liquidity factor score.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            window: Rolling window size.
            
        Returns:
            DataFrame with liquidity factor scores (normalized).
        """
        adtv = self.calculate_adtv(prices, volumes, window)
        
        # Log transform to handle skewness
        log_adtv = np.log(adtv + 1)
        
        # Normalize cross-sectionally
        normalized = self._normalize(log_adtv)
        
        return normalized
    
    def create_liquidity_mask(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        min_adtv: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Create a mask for filtering illiquid stocks.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            min_adtv: Minimum ADTV threshold.
            
        Returns:
            Boolean DataFrame (True = liquid, False = illiquid).
        """
        if min_adtv is None:
            min_adtv = self.min_adtv
        
        adtv = self.calculate_adtv(prices, volumes)
        mask = adtv >= min_adtv
        
        return mask
    
    def calculate_turnover_ratio(
        self,
        volumes: pd.DataFrame,
        shares_outstanding: Optional[pd.DataFrame] = None,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate turnover ratio (volume / shares outstanding).
        
        Args:
            volumes: DataFrame with trading volumes.
            shares_outstanding: DataFrame with shares outstanding.
            window: Rolling window size.
            
        Returns:
            DataFrame with turnover ratio.
        """
        if shares_outstanding is None:
            # If no shares outstanding data, just return normalized volume
            avg_volume = volumes.rolling(window=window).mean()
            return self._normalize(avg_volume)
        
        turnover = volumes / shares_outstanding
        avg_turnover = turnover.rolling(window=window).mean()
        
        return avg_turnover
    
    def calculate_amihud_illiquidity(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate Amihud illiquidity ratio.
        
        Amihud = Average(|Return| / Dollar Volume)
        
        Higher values indicate less liquid stocks.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            window: Rolling window size.
            
        Returns:
            DataFrame with Amihud illiquidity values.
        """
        # Calculate returns
        returns = prices.pct_change().abs()
        
        # Dollar volume
        dollar_volume = prices * volumes
        
        # Amihud ratio (add small epsilon to avoid division by zero)
        amihud = returns / (dollar_volume + 1e-10)
        
        # Rolling average
        avg_amihud = amihud.rolling(window=window).mean()
        
        # Multiply by 1e6 for better scaling
        avg_amihud = avg_amihud * 1e6
        
        return avg_amihud
    
    def calculate_bid_ask_spread_proxy(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate bid-ask spread proxy using high-low range.
        
        This is an approximation when actual bid-ask data is unavailable.
        
        Args:
            high: DataFrame with high prices.
            low: DataFrame with low prices.
            close: DataFrame with close prices.
            window: Rolling window size.
            
        Returns:
            DataFrame with spread proxy values.
        """
        # High-low spread as percentage
        spread = (high - low) / close
        
        # Rolling average
        avg_spread = spread.rolling(window=window).mean()
        
        return avg_spread
    
    def calculate_volume_concentration(
        self,
        volumes: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate volume concentration (coefficient of variation).
        
        Lower values indicate more stable, predictable liquidity.
        
        Args:
            volumes: DataFrame with trading volumes.
            window: Rolling window size.
            
        Returns:
            DataFrame with volume CV values.
        """
        rolling_mean = volumes.rolling(window=window).mean()
        rolling_std = volumes.rolling(window=window).std()
        
        cv = rolling_std / (rolling_mean + 1e-10)
        
        return cv
    
    def get_top_liquid_stocks(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        n: int = 10,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get the top N most liquid stocks.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            n: Number of top stocks to return.
            as_of_date: Date to use for ranking. Defaults to latest.
            
        Returns:
            DataFrame with top liquid stocks and their ADTV.
        """
        adtv = self.calculate_adtv(prices, volumes)
        
        if as_of_date is None:
            latest_adtv = adtv.iloc[-1]
        else:
            latest_adtv = adtv.loc[as_of_date]
        
        top_stocks = latest_adtv.nlargest(n)
        
        return top_stocks.to_frame(name="ADTV")
    
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
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    volumes = data["Volume"]
    
    liq = LiquidityFactor()
    adtv = liq.calculate_adtv(prices, volumes)
    print("ADTV (last 5 rows):")
    print(adtv.tail())
    
    print("\nTop Liquid Stocks:")
    print(liq.get_top_liquid_stocks(prices, volumes, n=3))
