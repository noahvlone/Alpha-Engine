"""
Alpha Research Engine - Configuration
Central configuration for stock universe, date ranges, factor parameters, and portfolio settings.
"""

from datetime import datetime, timedelta
from typing import List

# =============================================================================
# STOCK UNIVERSE CONFIGURATION
# =============================================================================

# Indonesian Blue-Chip Stocks (IDX30 Components)
IDX_STOCKS: List[str] = [
    "BBCA.JK",  # Bank Central Asia
    "BBRI.JK",  # Bank Rakyat Indonesia
    "BMRI.JK",  # Bank Mandiri
    "TLKM.JK",  # Telkom Indonesia
    "ASII.JK",  # Astra International
    "UNVR.JK",  # Unilever Indonesia
    "HMSP.JK",  # HM Sampoerna
    "ICBP.JK",  # Indofood CBP
    "KLBF.JK",  # Kalbe Farma
    "PTBA.JK",  # Bukit Asam
    "INDF.JK",  # Indofood Sukses Makmur
    "SMGR.JK",  # Semen Indonesia
    "CPIN.JK",  # Charoen Pokphand
    "ADRO.JK",  # Adaro Energy
    "ANTM.JK",  # Aneka Tambang
    "EXCL.JK",  # XL Axiata
    "MDKA.JK",  # Merdeka Copper Gold
    "ACES.JK",  # Ace Hardware
    "MNCN.JK",  # Media Nusantara Citra
    "PGAS.JK",  # Perusahaan Gas Negara
]

# US Technology Stocks (Alternative Universe)
US_TECH_STOCKS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "INTC", "CRM",
    "ADBE", "NFLX", "PYPL", "UBER", "SQ",
]

# Default Stock Universe
DEFAULT_UNIVERSE: List[str] = IDX_STOCKS

# =============================================================================
# DATE CONFIGURATION
# =============================================================================

# Default date range (3 years of historical data)
END_DATE: datetime = datetime.now()
START_DATE: datetime = END_DATE - timedelta(days=3*365)

# Date format for display
DATE_FORMAT: str = "%Y-%m-%d"

# =============================================================================
# FACTOR PARAMETERS
# =============================================================================

class FactorConfig:
    """Configuration for factor calculations."""
    
    # Momentum parameters
    MOMENTUM_LOOKBACK_SHORT: int = 20   # 1 month
    MOMENTUM_LOOKBACK_MEDIUM: int = 60  # 3 months
    MOMENTUM_LOOKBACK_LONG: int = 120   # 6 months
    
    # Volatility parameters
    VOLATILITY_WINDOW_SHORT: int = 20   # 1 month rolling
    VOLATILITY_WINDOW_LONG: int = 60    # 3 months rolling
    
    # Liquidity parameters
    LIQUIDITY_WINDOW: int = 20          # 1 month average
    MIN_ADTV: float = 1e9               # Minimum 1 billion IDR daily value
    
    # Composite factor weights
    MOMENTUM_WEIGHT: float = 0.4
    VOLATILITY_WEIGHT: float = 0.3      # Inverse volatility (low vol = good)
    LIQUIDITY_WEIGHT: float = 0.3

# =============================================================================
# PORTFOLIO PARAMETERS
# =============================================================================

class PortfolioConfig:
    """Configuration for portfolio construction and backtesting."""
    
    # Ranking configuration
    TOP_DECILE: float = 0.1             # Top 10% for Long
    BOTTOM_DECILE: float = 0.1          # Bottom 10% for Short
    
    # Rebalancing
    REBALANCE_FREQUENCY: str = "weekly"  # 'daily', 'weekly', 'monthly'
    
    # Transaction costs
    COMMISSION_RATE: float = 0.001      # 0.1% per transaction
    SLIPPAGE: float = 0.0005            # 0.05% slippage
    
    # Initial capital (in IDR)
    INITIAL_CAPITAL: float = 1e9        # 1 billion IDR
    
    # Position sizing
    MAX_POSITION_SIZE: float = 0.2      # Max 20% per stock
    
# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================

class EvaluationConfig:
    """Configuration for factor evaluation."""
    
    # Information Coefficient
    IC_FORWARD_PERIODS: List[int] = [1, 5, 10, 20]  # Days ahead
    IC_THRESHOLD_STRONG: float = 0.05   # IC > 0.05 = strong signal
    IC_THRESHOLD_WEAK: float = 0.02     # IC > 0.02 = weak signal
    
    # Factor decay analysis
    DECAY_MAX_LAG: int = 20             # Maximum lag days
    
    # Turnover thresholds
    MAX_ACCEPTABLE_TURNOVER: float = 0.5  # 50% monthly turnover

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

class DashboardConfig:
    """Configuration for Streamlit dashboard."""
    
    PAGE_TITLE: str = "Alpha Research Engine"
    PAGE_ICON: str = "📈"
    LAYOUT: str = "wide"
    
    # Theme colors
    PRIMARY_COLOR: str = "#1f77b4"
    SECONDARY_COLOR: str = "#ff7f0e"
    SUCCESS_COLOR: str = "#2ca02c"
    DANGER_COLOR: str = "#d62728"
    
    # Chart settings
    CHART_HEIGHT: int = 400
    CHART_TEMPLATE: str = "plotly_dark"

# =============================================================================
# FILE PATHS
# =============================================================================

class PathConfig:
    """Configuration for file paths."""
    
    DATA_DIR: str = "data/cache"
    REPORTS_DIR: str = "reports"
    LOGS_DIR: str = "logs"
