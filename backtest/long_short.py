"""
Long-Short Portfolio Module
Implements market-neutral long-short strategy.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

from config import PortfolioConfig
from .ranking import StockRanker


class LongShortPortfolio:
    """
    Long-Short Portfolio Constructor.
    
    Implements a market-neutral strategy:
    - Long Leg: Buy top-ranked stocks
    - Short Leg: Sell bottom-ranked stocks
    - Net exposure: Approximately zero (market neutral)
    
    Profits from the spread between winners and losers,
    regardless of overall market direction.
    """
    
    def __init__(
        self,
        top_pct: float = PortfolioConfig.TOP_DECILE,
        bottom_pct: float = PortfolioConfig.BOTTOM_DECILE,
        equal_weight: bool = True,
        max_position: float = PortfolioConfig.MAX_POSITION_SIZE
    ):
        """
        Initialize LongShortPortfolio.
        
        Args:
            top_pct: Percentage of stocks for long leg.
            bottom_pct: Percentage of stocks for short leg.
            equal_weight: If True, use equal weights within each leg.
            max_position: Maximum position size per stock.
        """
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.equal_weight = equal_weight
        self.max_position = max_position
        
        self.ranker = StockRanker(top_pct=top_pct, bottom_pct=bottom_pct)
    
    def construct_weights(
        self,
        factor_scores: pd.DataFrame,
        long_capital: float = 0.5,
        short_capital: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construct portfolio weights for long and short legs.
        
        Args:
            factor_scores: DataFrame with factor scores.
            long_capital: Capital allocation to long leg (as fraction).
            short_capital: Capital allocation to short leg (as fraction).
            
        Returns:
            Tuple of (long_weights, short_weights) DataFrames.
        """
        long_mask, short_mask = self.ranker.create_selection_mask(factor_scores)
        
        # Count stocks in each leg per day
        n_long = long_mask.sum(axis=1)
        n_short = short_mask.sum(axis=1)
        
        if self.equal_weight:
            # Equal weight within each leg
            long_weights = long_mask.astype(float).div(n_long, axis=0) * long_capital
            short_weights = short_mask.astype(float).div(n_short, axis=0) * short_capital
        else:
            # Score-weighted
            long_scores = factor_scores.where(long_mask, 0)
            short_scores = (-factor_scores).where(short_mask, 0)  # Negative for shorts
            
            long_weights = long_scores.div(long_scores.sum(axis=1), axis=0) * long_capital
            short_weights = short_scores.div(short_scores.sum(axis=1), axis=0) * short_capital
        
        # Apply position limits
        long_weights = long_weights.clip(upper=self.max_position)
        short_weights = short_weights.clip(upper=self.max_position)
        
        # Fill NaN with 0
        long_weights = long_weights.fillna(0)
        short_weights = short_weights.fillna(0)
        
        return long_weights, short_weights
    
    def calculate_returns(
        self,
        factor_scores: pd.DataFrame,
        prices: pd.DataFrame,
        rebalance_freq: str = "daily"
    ) -> pd.DataFrame:
        """
        Calculate portfolio returns.
        
        Args:
            factor_scores: DataFrame with factor scores.
            prices: DataFrame with stock prices.
            rebalance_freq: Rebalancing frequency.
            
        Returns:
            DataFrame with returns for long, short, and combined.
        """
        long_weights, short_weights = self.construct_weights(factor_scores)
        
        # Calculate stock returns
        stock_returns = prices.pct_change()
        
        # Calculate weighted returns
        # Long: positive returns when stocks go up
        long_returns = (long_weights.shift(1) * stock_returns).sum(axis=1)
        
        # Short: positive returns when stocks go down
        short_returns = -(short_weights.shift(1) * stock_returns).sum(axis=1)
        
        # Combined long-short return
        combined_returns = long_returns + short_returns
        
        returns_df = pd.DataFrame({
            "Long": long_returns,
            "Short": short_returns,
            "Long_Short": combined_returns,
        })
        
        return returns_df
    
    def calculate_cumulative_returns(
        self,
        factor_scores: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate cumulative returns (equity curve).
        
        Args:
            factor_scores: DataFrame with factor scores.
            prices: DataFrame with stock prices.
            
        Returns:
            DataFrame with cumulative returns.
        """
        returns = self.calculate_returns(factor_scores, prices)
        cumulative = (1 + returns).cumprod()
        
        return cumulative
    
    def get_portfolio_summary(
        self,
        factor_scores: pd.DataFrame,
        prices: pd.DataFrame,
        trading_days: int = 252
    ) -> dict:
        """
        Get comprehensive portfolio summary.
        
        Args:
            factor_scores: DataFrame with factor scores.
            prices: DataFrame with stock prices.
            trading_days: Number of trading days per year.
            
        Returns:
            Dictionary with portfolio metrics.
        """
        returns = self.calculate_returns(factor_scores, prices)
        cumulative = self.calculate_cumulative_returns(factor_scores, prices)
        
        ls_returns = returns["Long_Short"].dropna()
        
        # Calculate metrics
        total_return = cumulative["Long_Short"].iloc[-1] - 1
        annual_return = (1 + total_return) ** (trading_days / len(ls_returns)) - 1
        annual_volatility = ls_returns.std() * np.sqrt(trading_days)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = cumulative["Long_Short"].expanding().max()
        drawdown = (cumulative["Long_Short"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio (downside deviation)
        negative_returns = ls_returns[ls_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(trading_days)
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
        
        # Win rate
        win_rate = (ls_returns > 0).mean()
        
        summary = {
            "Total Return": f"{total_return:.2%}",
            "Annual Return": f"{annual_return:.2%}",
            "Annual Volatility": f"{annual_volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Trading Days": len(ls_returns),
            "Long Return": f"{(cumulative['Long'].iloc[-1] - 1):.2%}",
            "Short Return": f"{(cumulative['Short'].iloc[-1] - 1):.2%}",
        }
        
        return summary
    
    def get_current_positions(
        self,
        factor_scores: pd.DataFrame,
        as_of_date: Optional[str] = None
    ) -> dict:
        """
        Get current portfolio positions.
        
        Args:
            factor_scores: DataFrame with factor scores.
            as_of_date: Date for positions.
            
        Returns:
            Dictionary with long and short positions.
        """
        long_stocks, short_stocks = self.ranker.get_long_short_stocks(
            factor_scores, as_of_date=as_of_date
        )
        
        return {
            "Long Positions": long_stocks.to_dict(),
            "Short Positions": short_stocks.to_dict(),
            "N_Long": len(long_stocks),
            "N_Short": len(short_stocks),
        }
    
    def get_position_changes(
        self,
        factor_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Track position changes over time.
        
        Args:
            factor_scores: DataFrame with factor scores.
            
        Returns:
            DataFrame with position changes.
        """
        long_weights, short_weights = self.construct_weights(factor_scores)
        
        # Track changes
        long_changes = (long_weights > 0).astype(int).diff()
        short_changes = (short_weights > 0).astype(int).diff()
        
        # Count entries and exits
        results = []
        for date in factor_scores.index[1:]:
            long_entries = (long_changes.loc[date] == 1).sum()
            long_exits = (long_changes.loc[date] == -1).sum()
            short_entries = (short_changes.loc[date] == 1).sum()
            short_exits = (short_changes.loc[date] == -1).sum()
            
            results.append({
                "Date": date,
                "Long_Entries": long_entries,
                "Long_Exits": long_exits,
                "Short_Entries": short_entries,
                "Short_Exits": short_exits,
            })
        
        return pd.DataFrame(results).set_index("Date")


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    
    # Create factor scores (momentum)
    factor = prices.pct_change(periods=20)
    
    portfolio = LongShortPortfolio(top_pct=0.2, bottom_pct=0.2)
    
    print("Portfolio Summary:")
    for key, value in portfolio.get_portfolio_summary(factor, prices).items():
        print(f"  {key}: {value}")
    
    print("\nCurrent Positions:")
    positions = portfolio.get_current_positions(factor)
    print(f"  Long ({positions['N_Long']}): {list(positions['Long Positions'].keys())}")
    print(f"  Short ({positions['N_Short']}): {list(positions['Short Positions'].keys())}")
