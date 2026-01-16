"""
Backtester Module
VectorBT-powered backtesting engine for factor strategies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings

from config import PortfolioConfig

# Try to import vectorbt, fall back to basic implementation
try:
    import vectorbt as vbt
    HAS_VECTORBT = True
except ImportError:
    HAS_VECTORBT = False
    warnings.warn("VectorBT not available. Using basic backtesting implementation.")


class Backtester:
    """
    Backtester - Strategy backtesting engine.
    
    Uses VectorBT for high-performance backtesting when available,
    falls back to pandas-based implementation otherwise.
    
    Features:
    - Equity curve generation
    - Performance metrics calculation
    - Drawdown analysis
    - Transaction cost modeling
    """
    
    def __init__(
        self,
        initial_capital: float = PortfolioConfig.INITIAL_CAPITAL,
        commission: float = PortfolioConfig.COMMISSION_RATE,
        slippage: float = PortfolioConfig.SLIPPAGE,
        trading_days: int = 252
    ):
        """
        Initialize Backtester.
        
        Args:
            initial_capital: Starting capital.
            commission: Commission rate per trade.
            slippage: Slippage per trade.
            trading_days: Trading days per year.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trading_days = trading_days
        self.total_cost = commission + slippage
    
    def run_backtest(
        self,
        returns: pd.Series,
        weights: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run backtest and return results.
        
        Args:
            returns: Series of strategy returns.
            weights: Optional DataFrame of portfolio weights.
            
        Returns:
            Dictionary with backtest results.
        """
        # Clean returns
        returns = returns.dropna()
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        equity_curve = cumulative * self.initial_capital
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns, cumulative)
        
        # Drawdown analysis
        drawdown = self._calculate_drawdown(cumulative)
        
        results = {
            "returns": returns,
            "cumulative_returns": cumulative,
            "equity_curve": equity_curve,
            "drawdown": drawdown,
            "metrics": metrics,
        }
        
        return results
    
    def run_vectorbt_backtest(
        self,
        prices: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Run backtest using VectorBT (if available).
        
        Args:
            prices: DataFrame with stock prices.
            entries: Boolean DataFrame with entry signals.
            exits: Boolean DataFrame with exit signals.
            direction: 'long', 'short', or 'both'.
            
        Returns:
            Dictionary with backtest results.
        """
        if not HAS_VECTORBT:
            raise ImportError("VectorBT is required for this method. Install with: pip install vectorbt")
        
        # Create portfolio using VectorBT
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.total_cost,
            direction=direction,
            freq="1D"
        )
        
        # Extract results
        results = {
            "returns": portfolio.returns(),
            "cumulative_returns": portfolio.cumulative_returns(),
            "equity_curve": portfolio.value(),
            "drawdown": portfolio.drawdown(),
            "metrics": {
                "Total Return": portfolio.total_return(),
                "Annual Return": portfolio.annualized_return(),
                "Annual Volatility": portfolio.annualized_volatility(),
                "Sharpe Ratio": portfolio.sharpe_ratio(),
                "Sortino Ratio": portfolio.sortino_ratio(),
                "Max Drawdown": portfolio.max_drawdown(),
                "Win Rate": portfolio.win_rate(),
                "Profit Factor": portfolio.profit_factor(),
            },
            "trades": portfolio.trades.records_readable,
            "portfolio": portfolio,
        }
        
        return results
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        cumulative: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            returns: Series of returns.
            cumulative: Series of cumulative returns.
            
        Returns:
            Dictionary of metrics.
        """
        n_periods = len(returns)
        
        # Total and annual return
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (self.trading_days / n_periods) - 1
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(self.trading_days)
        
        # Risk-adjusted returns
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Sortino (downside risk)
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(self.trading_days)
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            "Total Return": total_return,
            "Annual Return": annual_return,
            "Annual Volatility": annual_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Trading Days": n_periods,
        }
        
        return metrics
    
    def _calculate_drawdown(
        self,
        cumulative: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate drawdown series.
        
        Args:
            cumulative: Series of cumulative returns.
            
        Returns:
            DataFrame with drawdown analysis.
        """
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        # Identify drawdown periods
        is_underwater = drawdown < 0
        
        # Calculate drawdown duration
        # Find consecutive underwater periods
        underwater_groups = is_underwater.ne(is_underwater.shift()).cumsum()
        underwater_lengths = is_underwater.groupby(underwater_groups).transform('sum')
        
        return pd.DataFrame({
            "Drawdown": drawdown,
            "Is_Underwater": is_underwater,
            "Drawdown_Duration": underwater_lengths.where(is_underwater, 0),
        })
    
    def get_monthly_returns(
        self,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Get monthly return table.
        
        Args:
            returns: Series of daily returns.
            
        Returns:
            DataFrame with monthly returns (pivoted by year/month).
        """
        monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        monthly_df = monthly.to_frame(name="Return")
        monthly_df["Year"] = monthly_df.index.year
        monthly_df["Month"] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # Add annual return
        pivot["Annual"] = returns.resample("Y").apply(lambda x: (1 + x).prod() - 1).values
        
        return pivot
    
    def get_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Series of returns.
            window: Rolling window size.
            
        Returns:
            DataFrame with rolling metrics.
        """
        rolling_return = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(self.trading_days)
        rolling_sharpe = rolling_return / rolling_vol
        
        return pd.DataFrame({
            "Rolling_Return": rolling_return,
            "Rolling_Volatility": rolling_vol,
            "Rolling_Sharpe": rolling_sharpe,
        })
    
    def compare_strategies(
        self,
        strategies: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: Dictionary of strategy name -> returns series.
            
        Returns:
            DataFrame comparing strategy metrics.
        """
        comparison = {}
        
        for name, returns in strategies.items():
            cumulative = (1 + returns).cumprod()
            metrics = self._calculate_metrics(returns, cumulative)
            comparison[name] = metrics
        
        return pd.DataFrame(comparison).T
    
    def format_metrics(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Format metrics for display.
        
        Args:
            metrics: Raw metrics dictionary.
            
        Returns:
            Formatted metrics dictionary.
        """
        formatted = {}
        
        for key, value in metrics.items():
            if "Return" in key or "Drawdown" in key or "Rate" in key:
                formatted[key] = f"{value:.2%}"
            elif "Ratio" in key or "Factor" in key:
                formatted[key] = f"{value:.2f}"
            elif "Days" in key:
                formatted[key] = f"{int(value)}"
            else:
                formatted[key] = f"{value:.4f}"
        
        return formatted


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    data = yf.download(tickers, start="2023-01-01", end="2024-01-01")
    prices = data["Close"]
    
    # Create simple momentum strategy returns
    factor = prices.pct_change(periods=20)
    
    # Create long-short returns (simplified)
    rankings = factor.rank(axis=1)
    n_stocks = rankings.count(axis=1)
    long_mask = rankings >= (n_stocks.values.reshape(-1, 1) * 0.8)
    short_mask = rankings <= (n_stocks.values.reshape(-1, 1) * 0.2)
    
    stock_returns = prices.pct_change()
    long_returns = (long_mask.shift(1) * stock_returns).mean(axis=1)
    short_returns = -(short_mask.shift(1) * stock_returns).mean(axis=1)
    strategy_returns = long_returns + short_returns
    
    backtester = Backtester()
    results = backtester.run_backtest(strategy_returns)
    
    print("Backtest Results:")
    print("-" * 40)
    for key, value in backtester.format_metrics(results["metrics"]).items():
        print(f"  {key}: {value}")
    
    print("\nMonthly Returns:")
    print(backtester.get_monthly_returns(strategy_returns))
