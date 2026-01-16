"""
Factor Engine Module
Orchestrates all factor calculations and produces composite scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from config import FactorConfig
from .momentum import MomentumFactor
from .volatility import VolatilityFactor
from .liquidity import LiquidityFactor


class FactorEngine:
    """
    Factor Engine - Orchestrates all factor calculations.
    
    Combines multiple alpha factors into a composite score for ranking.
    
    Features:
    - Calculate individual factors (momentum, volatility, liquidity)
    - Normalize factors cross-sectionally (z-score)
    - Combine factors with configurable weights
    - Produce stock rankings based on composite score
    """
    
    def __init__(
        self,
        momentum_weight: float = FactorConfig.MOMENTUM_WEIGHT,
        volatility_weight: float = FactorConfig.VOLATILITY_WEIGHT,
        liquidity_weight: float = FactorConfig.LIQUIDITY_WEIGHT
    ):
        """
        Initialize FactorEngine.
        
        Args:
            momentum_weight: Weight for momentum factor.
            volatility_weight: Weight for inverse volatility factor.
            liquidity_weight: Weight for liquidity factor.
        """
        self.momentum_weight = momentum_weight
        self.volatility_weight = volatility_weight
        self.liquidity_weight = liquidity_weight
        
        # Normalize weights to sum to 1
        total_weight = momentum_weight + volatility_weight + liquidity_weight
        self.momentum_weight /= total_weight
        self.volatility_weight /= total_weight
        self.liquidity_weight /= total_weight
        
        # Initialize individual factor calculators
        self.momentum = MomentumFactor()
        self.volatility = VolatilityFactor()
        self.liquidity = LiquidityFactor()
        
        # Storage for calculated factors
        self._factors: Dict[str, pd.DataFrame] = {}
        self._composite_score: Optional[pd.DataFrame] = None
        self._rankings: Optional[pd.DataFrame] = None
    
    def calculate_all_factors(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate all factors.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            high: Optional DataFrame with high prices (for volatility).
            low: Optional DataFrame with low prices (for volatility).
            
        Returns:
            Dictionary of factor DataFrames.
        """
        # Calculate momentum factor
        self._factors["momentum"] = self.momentum.calculate(prices)
        self._factors["momentum_normalized"] = self._zscore_normalize(
            self._factors["momentum"]
        )
        
        # Calculate volatility factor (inverse - low vol is good)
        self._factors["volatility"] = self.volatility.calculate(prices)
        self._factors["volatility_inverse"] = self.volatility.calculate_inverse(prices)
        self._factors["volatility_normalized"] = self._zscore_normalize(
            self._factors["volatility_inverse"]
        )
        
        # Calculate liquidity factor
        self._factors["liquidity"] = self.liquidity.calculate_adtv(prices, volumes)
        self._factors["liquidity_normalized"] = self._zscore_normalize(
            np.log(self._factors["liquidity"] + 1)  # Log transform for scaling
        )
        
        return self._factors
    
    def calculate_composite_score(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        high: Optional[pd.DataFrame] = None,
        low: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate composite factor score.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            high: Optional DataFrame with high prices.
            low: Optional DataFrame with low prices.
            
        Returns:
            DataFrame with composite scores.
        """
        # Calculate all factors if not already done
        if not self._factors:
            self.calculate_all_factors(prices, volumes, high, low)
        
        # Combine normalized factors with weights
        composite = (
            self.momentum_weight * self._factors["momentum_normalized"] +
            self.volatility_weight * self._factors["volatility_normalized"] +
            self.liquidity_weight * self._factors["liquidity_normalized"]
        )
        
        self._composite_score = composite
        
        return composite
    
    def get_rankings(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get stock rankings based on composite score.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            ascending: If True, lower scores rank higher.
            
        Returns:
            DataFrame with rankings (1 = best).
        """
        if self._composite_score is None:
            self.calculate_composite_score(prices, volumes)
        
        # Rank cross-sectionally (1 = best by default)
        self._rankings = self._composite_score.rank(axis=1, ascending=ascending)
        
        return self._rankings
    
    def get_percentile_rankings(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get percentile rankings (0-100 scale).
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            
        Returns:
            DataFrame with percentile rankings.
        """
        if self._composite_score is None:
            self.calculate_composite_score(prices, volumes)
        
        # Percentile rank (0-100)
        n_stocks = self._composite_score.count(axis=1)
        rankings = self.get_rankings(prices, volumes)
        
        percentiles = (rankings - 1) / (n_stocks.values.reshape(-1, 1) - 1) * 100
        
        return percentiles
    
    def get_decile_rankings(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get decile rankings (1-10 scale).
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            
        Returns:
            DataFrame with decile rankings (1 = top 10%, 10 = bottom 10%).
        """
        percentiles = self.get_percentile_rankings(prices, volumes)
        
        # Convert to deciles (1-10)
        deciles = np.ceil(percentiles / 10).replace(0, 1)
        
        return deciles
    
    def get_top_stocks(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        n: int = 10,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get top N stocks by composite score.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            n: Number of top stocks to return.
            as_of_date: Date to use. Defaults to latest.
            
        Returns:
            DataFrame with top stocks and their scores.
        """
        if self._composite_score is None:
            self.calculate_composite_score(prices, volumes)
        
        if as_of_date is None:
            latest_scores = self._composite_score.iloc[-1]
        else:
            latest_scores = self._composite_score.loc[as_of_date]
        
        top_stocks = latest_scores.nlargest(n).dropna()
        
        # Create result DataFrame with additional info
        result = pd.DataFrame({
            "Ticker": top_stocks.index,
            "Composite_Score": top_stocks.values,
            "Rank": range(1, len(top_stocks) + 1)
        })
        
        # Add individual factor scores
        result["Momentum"] = [
            self._factors["momentum_normalized"].iloc[-1].get(t, np.nan)
            for t in top_stocks.index
        ]
        result["Volatility"] = [
            self._factors["volatility_normalized"].iloc[-1].get(t, np.nan)
            for t in top_stocks.index
        ]
        result["Liquidity"] = [
            self._factors["liquidity_normalized"].iloc[-1].get(t, np.nan)
            for t in top_stocks.index
        ]
        
        return result.set_index("Ticker")
    
    def get_bottom_stocks(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        n: int = 10,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get bottom N stocks by composite score.
        
        Args:
            prices: DataFrame with close prices.
            volumes: DataFrame with trading volumes.
            n: Number of bottom stocks to return.
            as_of_date: Date to use. Defaults to latest.
            
        Returns:
            DataFrame with bottom stocks and their scores.
        """
        if self._composite_score is None:
            self.calculate_composite_score(prices, volumes)
        
        if as_of_date is None:
            latest_scores = self._composite_score.iloc[-1]
        else:
            latest_scores = self._composite_score.loc[as_of_date]
        
        bottom_stocks = latest_scores.nsmallest(n).dropna()
        
        result = pd.DataFrame({
            "Ticker": bottom_stocks.index,
            "Composite_Score": bottom_stocks.values,
            "Rank": range(len(bottom_stocks), 0, -1)
        })
        
        return result.set_index("Ticker")
    
    def get_factor_exposures(
        self,
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get all factor exposures for a given date.
        
        Args:
            as_of_date: Date to use. Defaults to latest.
            
        Returns:
            DataFrame with all factor exposures.
        """
        if not self._factors:
            raise ValueError("Factors not calculated. Call calculate_all_factors first.")
        
        exposures = {}
        
        for factor_name in ["momentum_normalized", "volatility_normalized", "liquidity_normalized"]:
            factor_df = self._factors[factor_name]
            if as_of_date is None:
                exposures[factor_name.replace("_normalized", "")] = factor_df.iloc[-1]
            else:
                exposures[factor_name.replace("_normalized", "")] = factor_df.loc[as_of_date]
        
        return pd.DataFrame(exposures)
    
    def _zscore_normalize(
        self,
        data: pd.DataFrame,
        winsorize_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Z-score normalize data cross-sectionally.
        
        Args:
            data: DataFrame to normalize.
            winsorize_std: Number of std devs for winsorization.
            
        Returns:
            Normalized DataFrame.
        """
        # Cross-sectional mean and std
        mean = data.mean(axis=1)
        std = data.std(axis=1)
        
        # Z-score
        normalized = data.sub(mean, axis=0).div(std, axis=0)
        
        # Winsorize extreme values
        normalized = normalized.clip(lower=-winsorize_std, upper=winsorize_std)
        
        return normalized
    
    @property
    def factors(self) -> Dict[str, pd.DataFrame]:
        """Get calculated factors."""
        return self._factors
    
    @property
    def composite_score(self) -> Optional[pd.DataFrame]:
        """Get composite score."""
        return self._composite_score
    
    @property
    def rankings(self) -> Optional[pd.DataFrame]:
        """Get rankings."""
        return self._rankings


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    volumes = data["Volume"]
    
    engine = FactorEngine()
    engine.calculate_all_factors(prices, volumes)
    composite = engine.calculate_composite_score(prices, volumes)
    
    print("Top Stocks:")
    print(engine.get_top_stocks(prices, volumes, n=3))
    
    print("\nFactor Exposures:")
    print(engine.get_factor_exposures())
