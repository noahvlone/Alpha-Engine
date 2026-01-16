# Alpha Research Engine

A quantitative finance portfolio project demonstrating skills in factor-based investing, big data processing, and capital market strategies.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

- **Factor Generation**: Momentum, Volatility, and Liquidity factors with configurable parameters
- **Factor Evaluation**: Information Coefficient (IC), Factor Decay, and Turnover analysis
- **Long-Short Simulation**: Market-neutral strategy backtesting with VectorBT integration
- **Interactive Dashboard**: Modern Streamlit UI with real-time visualizations
- **Report Generation**: Export CSV/Excel reports for top buy recommendations

## 📁 Project Structure

```
Alpha-Engine/
├── app.py                 # Main Streamlit dashboard
├── config.py              # Central configuration
├── requirements.txt       # Dependencies
├── data/
│   ├── __init__.py
│   └── data_loader.py     # Yahoo Finance data ingestion
├── factors/
│   ├── __init__.py
│   ├── momentum.py        # Momentum factor
│   ├── volatility.py      # Volatility factor
│   ├── liquidity.py       # Liquidity factor
│   └── factor_engine.py   # Composite factor scoring
├── evaluation/
│   ├── __init__.py
│   ├── ic_calculator.py   # Information Coefficient
│   ├── factor_decay.py    # Factor decay analysis
│   └── turnover.py        # Portfolio turnover metrics
└── backtest/
    ├── __init__.py
    ├── ranking.py         # Stock ranking system
    ├── long_short.py      # Long-short portfolio
    └── backtester.py      # Backtesting engine
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Alpha-Engine.git
cd Alpha-Engine
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Programmatic Usage

```python
from data.data_loader import DataLoader
from factors.factor_engine import FactorEngine
from backtest.long_short import LongShortPortfolio

# Load data
loader = DataLoader(tickers=["AAPL", "MSFT", "GOOGL"])
loader.fetch_all()

prices = loader.get_price_matrix()
volumes = loader.get_volume_matrix()

# Calculate factors
engine = FactorEngine()
composite = engine.calculate_composite_score(prices, volumes)

# Get top stocks
top_stocks = engine.get_top_stocks(prices, volumes, n=5)
print(top_stocks)

# Run backtest
portfolio = LongShortPortfolio(top_pct=0.1, bottom_pct=0.1)
summary = portfolio.get_portfolio_summary(composite, prices)
print(summary)
```
### Output
![Image](https://github.com/user-attachments/assets/c167be10-b90c-4dd7-b08e-ddfd9fa2f772)

![Image](https://github.com/user-attachments/assets/db880238-873f-47b0-af1d-569a6cbe8904)

![Image](https://github.com/user-attachments/assets/3e44123f-1bd1-468c-bf27-d22e8017c6eb)

![Image](https://github.com/user-attachments/assets/a4b2116c-57ba-4ced-b744-651d41b4d03d)

## 📊 Factor Definitions

### Momentum
```
Momentum = (Price_t - Price_{t-n}) / Price_{t-n}
```
Measures price trends over 20, 60, and 120-day windows.

### Volatility
```
Volatility = Rolling_Std(log_returns, window=20)
```
Uses inverse volatility (low vol = high score).

### Liquidity
```
ADTV = Average(Price × Volume, window=20)
```
Average Daily Trading Value to ensure tradability.

## 📈 Performance Metrics

- **Information Coefficient (IC)**: Correlation between factor and forward returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of positive return days

## 🎯 Stock Universe

Default: Indonesian Blue-Chip Stocks (IDX30)
- BBCA.JK, BBRI.JK, BMRI.JK, TLKM.JK, ASII.JK, etc.

Alternative: US Tech Stocks
- AAPL, MSFT, GOOGL, AMZN, NVDA, etc.

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Farhan Ramadhan**  
IT/Data Science Student  
Portfolio Project demonstrating quantitative finance skills.
