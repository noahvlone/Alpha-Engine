"""
Data Loader Module
Handles fetching and caching of OHLCV data from Yahoo Finance.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from config import DEFAULT_UNIVERSE, START_DATE, END_DATE, PathConfig


class DataLoader:
    """
    DataLoader class for fetching OHLCV data from Yahoo Finance.
    
    Features:
    - Batch downloading of multiple tickers
    - Local caching to avoid redundant API calls
    - Data validation and cleaning
    - Support for custom date ranges
    """
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize DataLoader.
        
        Args:
            tickers: List of stock tickers. Defaults to DEFAULT_UNIVERSE.
            start_date: Start date for data. Defaults to START_DATE from config.
            end_date: End date for data. Defaults to END_DATE from config.
            cache_dir: Directory for caching data. Defaults to PathConfig.DATA_DIR.
        """
        self.tickers = tickers or DEFAULT_UNIVERSE
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.cache_dir = Path(cache_dir or PathConfig.DATA_DIR)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self._data: Dict[str, pd.DataFrame] = {}
        self._combined_data: Optional[pd.DataFrame] = None
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache file path for a ticker."""
        safe_ticker = ticker.replace(".", "_").replace("-", "_")
        return self.cache_dir / f"{safe_ticker}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.pkl"
    
    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        cache_path = self._get_cache_path(ticker)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(ticker)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not cache data for {ticker}: {e}")
    
    def fetch_single(self, ticker: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol.
            use_cache: Whether to use cached data if available.
            
        Returns:
            DataFrame with OHLCV data.
        """
        # Try cache first
        if use_cache:
            cached_data = self._load_from_cache(ticker)
            if cached_data is not None:
                print(f"  {ticker}: Loaded from cache")
                self._data[ticker] = cached_data
                return cached_data
        
        # Fetch from Yahoo Finance
        try:
            print(f"  {ticker}: Fetching from Yahoo Finance...")
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True  # Adjust for dividends/splits
            )
            
            if data.empty:
                print(f"  {ticker}: No data available")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Keep only OHLCV columns
            required_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [col for col in required_cols if col in data.columns]
            data = data[available_cols]
            
            # Add ticker column
            data["ticker"] = ticker
            
            # Clean data
            data = self._clean_data(data)
            
            # Cache the data
            if use_cache:
                self._save_to_cache(ticker, data)
            
            self._data[ticker] = data
            return data
            
        except Exception as e:
            print(f"  {ticker}: Error fetching data - {e}")
            return pd.DataFrame()
    
    def fetch_all(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all tickers.
        
        Args:
            use_cache: Whether to use cached data if available.
            
        Returns:
            Dictionary mapping tickers to DataFrames.
        """
        print(f"Fetching data for {len(self.tickers)} tickers...")
        print(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print("-" * 50)
        
        for ticker in self.tickers:
            self.fetch_single(ticker, use_cache=use_cache)
        
        print("-" * 50)
        print(f"Successfully loaded: {len(self._data)} tickers")
        
        return self._data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        
        Args:
            data: Raw OHLCV DataFrame.
            
        Returns:
            Cleaned DataFrame.
        """
        if data.empty:
            return data
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Remove rows with zero or negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Remove rows with zero volume
        if "volume" in data.columns:
            data = data[data["volume"] > 0]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def get_combined_data(self) -> pd.DataFrame:
        """
        Get combined data for all tickers in long format.
        
        Returns:
            DataFrame with all tickers combined.
        """
        if not self._data:
            self.fetch_all()
        
        if self._combined_data is not None:
            return self._combined_data
        
        all_data = []
        for ticker, data in self._data.items():
            if not data.empty:
                df = data.copy()
                df["ticker"] = ticker
                all_data.append(df)
        
        if all_data:
            self._combined_data = pd.concat(all_data, axis=0)
            self._combined_data = self._combined_data.sort_index()
        else:
            self._combined_data = pd.DataFrame()
        
        return self._combined_data
    
    def get_price_matrix(self, price_type: str = "close") -> pd.DataFrame:
        """
        Get price matrix with tickers as columns.
        
        Args:
            price_type: Type of price ('open', 'high', 'low', 'close').
            
        Returns:
            DataFrame with dates as index and tickers as columns.
        """
        if not self._data:
            self.fetch_all()
        
        prices = {}
        for ticker, data in self._data.items():
            if not data.empty and price_type in data.columns:
                prices[ticker] = data[price_type]
        
        if prices:
            return pd.DataFrame(prices)
        return pd.DataFrame()
    
    def get_volume_matrix(self) -> pd.DataFrame:
        """
        Get volume matrix with tickers as columns.
        
        Returns:
            DataFrame with dates as index and tickers as columns.
        """
        if not self._data:
            self.fetch_all()
        
        volumes = {}
        for ticker, data in self._data.items():
            if not data.empty and "volume" in data.columns:
                volumes[ticker] = data["volume"]
        
        if volumes:
            return pd.DataFrame(volumes)
        return pd.DataFrame()
    
    def get_returns(self, periods: int = 1) -> pd.DataFrame:
        """
        Calculate returns for all tickers.
        
        Args:
            periods: Number of periods for return calculation.
            
        Returns:
            DataFrame with returns for each ticker.
        """
        prices = self.get_price_matrix("close")
        if prices.empty:
            return pd.DataFrame()
        
        returns = prices.pct_change(periods=periods)
        return returns
    
    def get_log_returns(self, periods: int = 1) -> pd.DataFrame:
        """
        Calculate log returns for all tickers.
        
        Args:
            periods: Number of periods for return calculation.
            
        Returns:
            DataFrame with log returns for each ticker.
        """
        import numpy as np
        
        prices = self.get_price_matrix("close")
        if prices.empty:
            return pd.DataFrame()
        
        log_returns = np.log(prices / prices.shift(periods))
        return log_returns
    
    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """Get loaded data dictionary."""
        return self._data
    
    def clear_cache(self) -> None:
        """Clear all cached data files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("Cache cleared.")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_stock_data(
    tickers: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> DataLoader:
    """
    Convenience function to load stock data.
    
    Args:
        tickers: List of stock tickers.
        start_date: Start date for data.
        end_date: End date for data.
        
    Returns:
        Initialized DataLoader with fetched data.
    """
    loader = DataLoader(tickers, start_date, end_date)
    loader.fetch_all()
    return loader


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    loader.fetch_all()
    
    print("\nPrice Matrix Shape:", loader.get_price_matrix().shape)
    print("Volume Matrix Shape:", loader.get_volume_matrix().shape)
    print("\nSample Returns:")
    print(loader.get_returns().tail())
