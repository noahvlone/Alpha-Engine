"""
Stock Ranking Module
Cross-sectional ranking system for factor-based investing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

from config import PortfolioConfig


class StockRanker:
    """
    Stock Ranker - Cross-sectional ranking system.
    
    Ranks stocks based on factor scores and classifies them into
    quantiles (deciles, quintiles, etc.) for portfolio construction.
    """
    
    def __init__(
        self,
        top_pct: float = PortfolioConfig.TOP_DECILE,
        bottom_pct: float = PortfolioConfig.BOTTOM_DECILE,
        n_quantiles: int = 10
    ):
        """
        Initialize StockRanker.
        
        Args:
            top_pct: Percentage for top quantile (e.g., 0.1 for decile).
            bottom_pct: Percentage for bottom quantile.
            n_quantiles: Number of quantiles (default: 10 for deciles).
        """
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.n_quantiles = n_quantiles
    
    def rank(
        self,
        factor_scores: pd.DataFrame,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank stocks cross-sectionally.
        
        Args:
            factor_scores: DataFrame with factor scores.
            ascending: If True, lower scores rank higher.
            
        Returns:
            DataFrame with rankings (1 = best).
        """
        rankings = factor_scores.rank(axis=1, ascending=ascending, method="average")
        return rankings
    
    def percentile_rank(
        self,
        factor_scores: pd.DataFrame,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get percentile rankings (0-100 scale).
        
        Args:
            factor_scores: DataFrame with factor scores.
            ascending: If True, lower scores = higher percentile.
            
        Returns:
            DataFrame with percentile rankings.
        """
        rankings = self.rank(factor_scores, ascending)
        n_stocks = factor_scores.count(axis=1)
        
        # Convert to percentile (0-100)
        percentiles = (rankings - 1) / (n_stocks.values.reshape(-1, 1) - 1) * 100
        
        return percentiles
    
    def quantile_rank(
        self,
        factor_scores: pd.DataFrame,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get quantile rankings (1 to n_quantiles).
        
        Args:
            factor_scores: DataFrame with factor scores.
            ascending: If True, lower scores = higher quantile.
            
        Returns:
            DataFrame with quantile rankings.
        """
        percentiles = self.percentile_rank(factor_scores, ascending)
        
        # Convert percentiles to quantiles
        quantiles = np.ceil(percentiles / (100 / self.n_quantiles))
        quantiles = quantiles.replace(0, 1)  # Handle edge case
        
        return quantiles
    
    def get_top_stocks(
        self,
        factor_scores: pd.DataFrame,
        n: Optional[int] = None,
        pct: Optional[float] = None,
        as_of_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get top ranked stocks.
        
        Args:
            factor_scores: DataFrame with factor scores.
            n: Number of top stocks (takes precedence).
            pct: Percentage of top stocks.
            as_of_date: Date to use. Defaults to latest.
            
        Returns:
            Series with top stock tickers and scores.
        """
        if as_of_date is None:
            scores = factor_scores.iloc[-1].dropna()
        else:
            scores = factor_scores.loc[as_of_date].dropna()
        
        if n is None:
            n = int(len(scores) * (pct or self.top_pct))
            n = max(n, 1)
        
        top_stocks = scores.nlargest(n)
        return top_stocks
    
    def get_bottom_stocks(
        self,
        factor_scores: pd.DataFrame,
        n: Optional[int] = None,
        pct: Optional[float] = None,
        as_of_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get bottom ranked stocks (for short leg).
        
        Args:
            factor_scores: DataFrame with factor scores.
            n: Number of bottom stocks.
            pct: Percentage of bottom stocks.
            as_of_date: Date to use.
            
        Returns:
            Series with bottom stock tickers and scores.
        """
        if as_of_date is None:
            scores = factor_scores.iloc[-1].dropna()
        else:
            scores = factor_scores.loc[as_of_date].dropna()
        
        if n is None:
            n = int(len(scores) * (pct or self.bottom_pct))
            n = max(n, 1)
        
        bottom_stocks = scores.nsmallest(n)
        return bottom_stocks
    
    def get_long_short_stocks(
        self,
        factor_scores: pd.DataFrame,
        n_long: Optional[int] = None,
        n_short: Optional[int] = None,
        as_of_date: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Get stocks for long and short legs.
        
        Args:
            factor_scores: DataFrame with factor scores.
            n_long: Number of long stocks.
            n_short: Number of short stocks.
            as_of_date: Date to use.
            
        Returns:
            Tuple of (long_stocks, short_stocks).
        """
        long_stocks = self.get_top_stocks(
            factor_scores, n=n_long, as_of_date=as_of_date
        )
        short_stocks = self.get_bottom_stocks(
            factor_scores, n=n_short, as_of_date=as_of_date
        )
        
        return long_stocks, short_stocks
    
    def create_selection_mask(
        self,
        factor_scores: pd.DataFrame,
        top_pct: Optional[float] = None,
        bottom_pct: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create boolean masks for top and bottom stocks.
        
        Args:
            factor_scores: DataFrame with factor scores.
            top_pct: Top percentage cutoff.
            bottom_pct: Bottom percentage cutoff.
            
        Returns:
            Tuple of (long_mask, short_mask) DataFrames.
        """
        top_pct = top_pct or self.top_pct
        bottom_pct = bottom_pct or self.bottom_pct
        
        percentiles = self.percentile_rank(factor_scores)
        
        # Top stocks (high percentile)
        long_mask = percentiles >= (100 - top_pct * 100)
        
        # Bottom stocks (low percentile)
        short_mask = percentiles <= (bottom_pct * 100)
        
        return long_mask, short_mask
    
    def get_ranking_history(
        self,
        factor_scores: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Get ranking history for a specific stock.
        
        Args:
            factor_scores: DataFrame with factor scores.
            ticker: Stock ticker.
            
        Returns:
            DataFrame with ranking history.
        """
        if ticker not in factor_scores.columns:
            raise ValueError(f"Ticker {ticker} not found in factor scores")
        
        score = factor_scores[ticker]
        ranking = self.rank(factor_scores)[ticker]
        percentile = self.percentile_rank(factor_scores)[ticker]
        quantile = self.quantile_rank(factor_scores)[ticker]
        
        history = pd.DataFrame({
            "Score": score,
            "Rank": ranking,
            "Percentile": percentile,
            "Quantile": quantile,
        })
        
        return history
    
    def get_daily_selections(
        self,
        factor_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get daily long/short selections as a summary.
        
        Args:
            factor_scores: DataFrame with factor scores.
            
        Returns:
            DataFrame with date, long stocks, short stocks.
        """
        long_mask, short_mask = self.create_selection_mask(factor_scores)
        
        results = []
        for date in factor_scores.index:
            long_stocks = long_mask.loc[date][long_mask.loc[date]].index.tolist()
            short_stocks = short_mask.loc[date][short_mask.loc[date]].index.tolist()
            
            results.append({
                "Date": date,
                "Long_Stocks": ", ".join(long_stocks),
                "Short_Stocks": ", ".join(short_stocks),
                "N_Long": len(long_stocks),
                "N_Short": len(short_stocks),
            })
        
        return pd.DataFrame(results).set_index("Date")


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    
    # Create factor scores
    factor = prices.pct_change(periods=20)
    
    ranker = StockRanker(top_pct=0.2, bottom_pct=0.2)
    
    print("Top Stocks:")
    print(ranker.get_top_stocks(factor))
    
    print("\nBottom Stocks:")
    print(ranker.get_bottom_stocks(factor))
    
    print("\nQuantile Rankings (latest):")
    print(ranker.quantile_rank(factor).iloc[-1])
